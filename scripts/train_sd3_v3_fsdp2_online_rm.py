#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from ml_collections import config_flags

import csv
import torchvision.transforms as transforms
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP

# --- FSDP2 ---
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule

# 你项目内的 EMA 模块
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

# -----------------------
# 轻量 logger
# -----------------------
def get_logger(name):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(name)
logger = get_logger(__name__)

# -----------------------
# 分布式初始化 / 工具
# -----------------------
def setup_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, device, dist.get_rank(), dist.get_world_size()

def is_main_process():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

@contextlib.contextmanager
def maybe_distributed_barrier():
    if dist.is_initialized():
        dist.barrier()
    yield

def all_gather_concat(t: torch.Tensor) -> torch.Tensor:
    """all_gather 沿 batch 维拼接"""
    if not dist.is_initialized():
        return t
    tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors, t.contiguous())
    return torch.cat(tensors, dim=0)

def gather_str_token_ids(ids: torch.Tensor, tokenizer) -> list:
    """把 token ids（[B, L]）跨卡聚合后 decode 成字符串列表"""
    ids_all = all_gather_concat(ids)
    return tokenizer.batch_decode(ids_all.cpu().numpy(), skip_special_tokens=True)

def unwrap_fsdp(m):
    """拿到被 DDP/FSDP2 包裹下的“可用模型对象”。对 FSDP2 保持原样（它自身就是可用的 proxy）"""
    if hasattr(m, "module"):  # e.g., DDP
        m = m.module
    return m

def disable_adapter_ctx(m):
    """在 FSDP2 包裹下安全地进入 peft 的 disable_adapter() 上下文；若无则空上下文"""
    base = unwrap_fsdp(m)
    if hasattr(base, "disable_adapter"):
        return base.disable_adapter()
    return contextlib.nullcontext()

# -----------------------
# 数据集 & 采样器
# -----------------------
class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
    def __len__(self): return len(self.prompts)
    def __getitem__(self, idx): return {"prompt": self.prompts[idx], "metadata": {}}
    @staticmethod
    def collate_fn(examples):
        prompts = [e["prompt"] for e in examples]
        metas = [e["metadata"] for e in examples]
        return prompts, metas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
    def __len__(self): return len(self.prompts)
    def __getitem__(self, idx): return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}
    @staticmethod
    def collate_fn(examples):
        prompts = [e["prompt"] for e in examples]
        metas = [e["metadata"] for e in examples]
        return prompts, metas

class DistributedKRepeatSampler(Sampler):
    """
    让每个迭代中，全局抽 m 个 unique 样本，每个重复 k 次，打散后再按 replica 切成 n*b 大小；
    本 Sampler **每个 rank** 都实例化一份（num_replicas=world_size, rank=rank）。
    DataLoader 里用 batch_sampler=...
    """
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k({self.k}) must divide n*b({self.total_samples})"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator(); g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated = [idx for idx in indices for _ in range(self.k)]
            order = torch.randperm(len(repeated), generator=g).tolist()
            shuffled = [repeated[i] for i in order]
            # split per replica
            start = self.rank * self.batch_size
            end = start + self.batch_size
            yield shuffled[start:end]
    def set_epoch(self, epoch): self.epoch = epoch

# -----------------------
# 文本编码等工具
# -----------------------
def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    gens = []
    for prompt in prompts:
        h = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(h[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2**31)
        gens.append(torch.Generator().manual_seed(seed))
    return gens

def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_uncond = noise_pred_uncond.detach()
        noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )
    return prev_sample, log_prob, prev_sample_mean, std_dev_t

# -----------------------
# 选择正负样本 & 对比损失
# -----------------------
_gt_prompt2path = None

def init_gt_prompt2path_distributed(gt_data_path: str):
    """
    仅在 rank0 读取 TSV，构建 {prompt: abs_path}，然后广播给所有 rank。
    """
    global _gt_prompt2path
    if _gt_prompt2path is not None:
        return

    mapping = None
    is_dist = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if is_dist else 0

    if rank == 0:
        mapping = {}
        tsv_path = os.path.join(gt_data_path, "prompts.tsv")
        try:
            with open(tsv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    prompt = row.get("prompt")
                    filename = row.get("filename")
                    if not prompt or not filename:
                        continue
                    if prompt not in mapping:
                        mapping[prompt] = os.path.join(gt_data_path, filename)
            print(f"[rank0] GT prompt2path loaded: {len(mapping)} entries from {tsv_path}")
        except Exception as e:
            print(f"[rank0] Error reading TSV file {tsv_path}: {e}")
            mapping = {}

    if is_dist:
        obj_list = [mapping] if rank == 0 else [None]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        mapping = obj_list[0]

    _gt_prompt2path = mapping or {}

def get_gt_image_path(prompt: str):
    """
    返回 prompt 对应的图片绝对路径；不存在则 None。
    需先调用 init_gt_prompt2path_distributed 初始化。
    """
    return _gt_prompt2path.get(prompt) if _gt_prompt2path is not None else None

def select_positive_negative_pairs_GT(images, prompts, rewards, num_pairs=None, gt_data_path="/pfs/yangyuanming/code2/datasets/fluxkrea-data/pickscore-prompt/train_1"):
    batch_size = len(images)
    if batch_size < 2: return None, None, None, None
    prompt_set = list(set(prompts))
    if len(prompt_set) == 0: return None, None, None, None

    # 确保已初始化（rank0 读+广播一次）
    init_gt_prompt2path_distributed(gt_data_path)

    selected_prompt = random.choice(prompt_set)
    indices = [i for i, p in enumerate(prompts) if p == selected_prompt]
    if len(indices) < 2: return None, None, None, None
    neg_idx, _ = random.sample(indices, 2)
    negative_images = images[neg_idx].unsqueeze(0)
    negative_prompts = [selected_prompt]

    # 仅哈希查询：没有就返回 None
    image_path = get_gt_image_path(selected_prompt)
    if image_path is None:
        return None, None, None, None

    try:
        pil = Image.open(image_path).convert('RGB')
        if len(images.shape) == 4:
            target_size = (images.shape[2], images.shape[3])
        else:
            target_size = (1024, 1024)
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        positive_images = transform(pil).unsqueeze(0).to(images.device)
        positive_prompts = [selected_prompt]
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None, None

    return positive_images, negative_images, positive_prompts, negative_prompts

def compute_contrastive_loss(clip_model, clip_processor, positive_images, negative_images, positive_prompts, negative_prompts, device):
    if positive_images is None or len(positive_images) == 0:
        return torch.tensor(0.0, device=device)

    def process_images(images):
        # === FIX: 防止 [-1,1] 范围图像（GT）直接*255 溢出，统一映射到 [0,255] ===
        arr = images.detach().cpu().numpy()
        # 若值域小于0，先反归一化到[0,1]
        if arr.min() < 0.0:
            arr = (arr * 0.5 + 0.5)
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        arr = arr.transpose(0, 2, 3, 1)  # CHW -> HWC
        return [Image.fromarray(img) for img in arr]

    if isinstance(positive_images, list):
        positive_images = torch.cat(positive_images, dim=0)
        negative_images = torch.cat(negative_images, dim=0)
        positive_prompts = [p for L in positive_prompts for p in L]
        negative_prompts = [p for L in negative_prompts for p in L]

    pos_pil = process_images(positive_images)
    neg_pil = process_images(negative_images)

    pos_inputs = clip_processor(text=positive_prompts, images=pos_pil, return_tensors="pt", padding=True, truncation=True).to(device)
    neg_inputs = clip_processor(text=negative_prompts, images=neg_pil, return_tensors="pt", padding=True, truncation=True).to(device)

    pos_outputs = clip_model(**pos_inputs)
    pos_image_embeds, pos_text_embeds = pos_outputs.image_embeds, pos_outputs.text_embeds
    neg_outputs = clip_model(**neg_inputs)
    neg_image_embeds, neg_text_embeds = neg_outputs.image_embeds, neg_outputs.text_embeds

    pos_sim = F.cosine_similarity(pos_image_embeds, pos_text_embeds, dim=-1)
    neg_sim = F.cosine_similarity(neg_image_embeds, neg_text_embeds, dim=-1)
    loss = -torch.log(torch.sigmoid(pos_sim - neg_sim) + 1e-8).mean()
    return loss

def save_reward_model(reward_model, global_step, config):
    if not is_main_process(): return
    save_root = os.path.join(config.train.reward_model_path, "checkpoints", f"checkpoint-{global_step}")
    os.makedirs(save_root, exist_ok=True)
    unwrap_fsdp(reward_model).save_pretrained(save_root)

# -----------------------
# 主流程（FSDP2 + torchrun）
# -----------------------
def main(_):
    config = FLAGS.config
    local_rank, device, rank, world = setup_dist()

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = (config.run_name or "run") + "_" + unique_id

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    if is_main_process():
        wandb.init(project="flow_grpo_v1_fsdp2", mode="offline")
        logger.info(f"\n{config}")

    # 随机种子（按 rank 偏移）
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    random.seed(config.seed + rank)

    # ======= 加载 SD3 pipeline =======
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.safety_checker = None
    # === FIX: 只在 rank0 显示内部进度条，避免多 rank 光标打架 ===
    pipeline.set_progress_bar_config(position=1, disable=(rank != 0), leave=False, desc="Timestep", dynamic_ncols=True)

    # 推理 dtype（非训练权重）
    inference_dtype = torch.float32
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(device, dtype=inference_dtype)

    # 训练对象：transformer（policy）
    pipeline.transformer.requires_grad_(not config.use_lora)
    if config.use_lora:
        target_modules = [
            "attn.add_k_proj","attn.add_q_proj","attn.add_v_proj","attn.to_add_out",
            "attn.to_k","attn.to_out.0","attn.to_q","attn.to_v",
        ]
        lora_cfg = LoraConfig(r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules)
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            unwrap_fsdp(pipeline.transformer).set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, lora_cfg)

    # 把 transformer 放到 device（FSDP 前）
    pipeline.transformer.to(device)
    transformer = pipeline.transformer

    # EMA 只跟踪可训练参数
    transformer_trainable_parameters = [p for p in transformer.parameters() if p.requires_grad]
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=device)

    # TF32（可选）
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 优化器（policy/transformer）
    if config.train.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # ======= Reward 函数 & 模型 =======
    # 包装单一reward函数，使其返回字典格式（与multi_score保持一致）
    _rule_reward_fn = flow_grpo.rewards.ocr_score(device)
    rule_reward_fn = lambda images, prompts, metadata: ({"avg": _rule_reward_fn(images, prompts, metadata)[0]}, {})

    # reward_fn = getattr(flow_grpo.rewards, 'multi_score')(device, config.reward_fn)
    # eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(device, config.reward_fn)

    _pickscore_fn = flow_grpo.rewards.pickscore_score(device)
    reward_fn = lambda images, prompts, metadata: ({"avg": _pickscore_fn(images, prompts, metadata)[0]}, {})
    eval_reward_fn = lambda images, prompts, metadata: ({"avg": _pickscore_fn(images, prompts, metadata)[0]}, {})

    executor = futures.ThreadPoolExecutor(max_workers=8)

    if config.train.update_reward_model:
        reward_key = list(config.reward_fn.keys())
        # reward_model = reward_fn.model
        # reward_processor = reward_fn.processor
        reward_model = _pickscore_fn.model
        reward_processor = _pickscore_fn.processor
        for p in reward_model.parameters():
            p.requires_grad = True
        reward_optimizer = torch.optim.AdamW(
            reward_model.parameters(),
            lr=config.train.reward_model_learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )
        reward_model.to(device)


    # ======= FSDP2 包装（policy & reward）=======
    fsdp_kwargs = {}
    if getattr(config, "mixed_precision", None) == "bf16":
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
    # transformer = fully_shard(transformer, **fsdp_kwargs)
    transformer = DDP(transformer, device_ids=[device], output_device=device, broadcast_buffers=False, find_unused_parameters=True)
    pipeline.transformer = transformer  # 重要：让 pipeline 使用包裹后的模块
    if hasattr(pipeline.transformer, "module"):
        pipeline.transformer.config = pipeline.transformer.module.config

    if config.train.update_reward_model:
        # reward_model = fully_shard(reward_model, **fsdp_kwargs)
        reward_model = DDP(reward_model, device_ids=[device], output_device=device, broadcast_buffers=False, find_unused_parameters=True)

    # ======= 数据 =======
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')
        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=world,
            rank=rank,
            seed=42
        )
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=1,
                                      collate_fn=TextPromptDataset.collate_fn)
        test_dataloader  = DataLoader(test_dataset, batch_size=config.sample.test_batch_size,
                                      collate_fn=TextPromptDataset.collate_fn, shuffle=False, num_workers=8)
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset  = GenevalPromptDataset(config.dataset, 'test')
        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=world,
            rank=rank,
            seed=42
        )
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=1,
                                      collate_fn=GenevalPromptDataset.collate_fn)
        test_dataloader  = DataLoader(test_dataset, batch_size=config.sample.test_batch_size,
                                      collate_fn=GenevalPromptDataset.collate_fn, shuffle=False, num_workers=8)
    else:
        raise NotImplementedError("Only general_ocr or geneval supported")

    # 负提示 embedding 预先算好
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, 128, device)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds  = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds  = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    # AMP & autocast
    use_amp = (device.type == "cuda") and (getattr(config, "mixed_precision", None) in ["fp16", "bf16"])
    amp_dtype = torch.float16 if getattr(config, "mixed_precision", None) == "fp16" else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and getattr(config, "mixed_precision", None) == "fp16"))
    if config.use_lora:
        autocast_ctx = contextlib.nullcontext
    else:
        def _autocast(): return torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)
        autocast_ctx = _autocast

    # per-prompt 统计
    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    samples_per_epoch = config.sample.train_batch_size * world * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world * config.train.gradient_accumulation_steps
    if is_main_process():
        logger.info("***** Running training (FSDP2) *****")
        logger.info(f"  World size = {world}")
        logger.info(f"  Sample batch size per rank = {config.sample.train_batch_size}")
        logger.info(f"  Train batch size per rank = {config.train.batch_size}")
        logger.info(f"  Grad Accumulation = {config.train.gradient_accumulation_steps}")
        logger.info(f"  Total samples/epoch (global) = {samples_per_epoch}")
        logger.info(f"  Total train global batch size = {total_train_batch_size}")
        logger.info(f"  Inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0
    reward_step = 0
    train_iter = iter(train_dataloader)
    global_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps

    # ========= 主训练循环 =========
    while True:
        # 保存（rank0）
        if (epoch % config.save_freq == 0) and (epoch > 0) and is_main_process():
            save_root = os.path.join(config.save_dir, "checkpoints", f"checkpoint-{global_step}")
            save_root_lora = os.path.join(save_root, "lora"); os.makedirs(save_root_lora, exist_ok=True)
            if config.train.ema:
                ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
            unwrap_fsdp(transformer).save_pretrained(save_root_lora)
            if config.train.ema:
                ema.copy_temp_to(transformer_trainable_parameters)
            if config.train.update_reward_model:
                save_reward_model(reward_model, global_step, config)    

        # ---------- SAMPLING ----------
        unwrap_fsdp(transformer).eval()
        if config.train.update_reward_model:
            unwrap_fsdp(reward_model).eval()

        samples = []
        last_local_images = None    # rank 本地“最后一批”图像
        last_local_prompts = None   # === FIX: 记录“最后一批”的 prompts ===
        last_local_rewards_future = None  # === FIX: 记录“最后一批”的 reward future ===
        last_local_rule_rewards_future = None  # === FIX: 记录“最后一批”的 rule reward future ===

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling (rank {rank})",
            disable=(rank != 0),  # === FIX: 只让 rank0 打印 ===
            position=0,
        ):
            # 同步 sampler 的 epoch（让每个 rank 的无限采样器状态一致）
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, 128, device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None

            with autocast_ctx():
                with torch.no_grad():
                    images, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[:len(prompt_embeds)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[:len(prompt_embeds)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator
                    )
            last_local_images = images  # 每个 rank 保留本地最后一批

            latents = torch.stack(latents, dim=1)         # (B, T+1, C, H, W)
            log_probs = torch.stack(log_probs, dim=1)     # (B, T)
            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1)

            # 异步计算 reward
            # rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata)
            time.sleep(0)
            rule_rewards_future = executor.submit(rule_reward_fn, images, prompts, prompt_metadata)
            time.sleep(0)

            # === FIX: 记录“最后一批”的三件套（图/文/reward future）===
            last_local_prompts = prompts
            last_local_rewards_future = rewards_future
            last_local_rule_rewards_future = rule_rewards_future

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompts": prompts,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "rewards": rewards_future,
                    "rule_rewards": rule_rewards_future,
                    "images": images,
                }
            )
            
        # 等所有 reward 完成，并转成 tensor
        for s in tqdm(samples, desc="Waiting for rewards", disable=(rank != 0), position=0):  # === FIX: 非0静默 ===
            rewards, reward_metadata = s["rewards"].result()
            s["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
        
        for s in tqdm(samples, desc="Waiting for rule rewards", disable=(rank != 0), position=0):  # === FIX: 非0静默 ===
            rule_rewards, rule_reward_metadata = s["rule_rewards"].result()
            s["rule_rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rule_rewards.items()}

        # === FIX: 提取“最后一批”的 reward（用于可视化严格对齐）===
        last_res_dict, _ = last_local_rewards_future.result()
        last_local_rewards_avg = torch.as_tensor(last_res_dict["avg"]).float()

        last_rule_res_dict, _ = last_local_rule_rewards_future.result()
        last_local_rule_rewards_avg = torch.as_tensor(last_rule_res_dict["avg"]).float()

        # 把 list[dict] 合并成 dict[tensor or list]
        merged = {}
        for k in samples[0].keys():
            if k == "prompts":
                merged[k] = sum([s[k] for s in samples], [])
            elif not isinstance(samples[0][k], dict):
                merged[k] = torch.cat([s[k] for s in samples], dim=0)
            else:
                merged[k] = {sub_k: torch.cat([s[k][sub_k] for s in samples], dim=0)
                            for sub_k in samples[0][k]}
        samples = merged

        # rank0 可视化：严格使用同一批（最后一批）的图/文/分数，避免错位
        if (epoch % 2 == 0) and is_main_process():
            with tempfile.TemporaryDirectory() as tmpdir:
                imgs = last_local_images
                bs = len(imgs)
                num_samples = min(15, bs)
                idxs = random.sample(range(bs), num_samples)
                for idx, ii in enumerate(idxs):
                    image = imgs[ii]
                    pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                sampled_prompts = [last_local_prompts[i] for i in idxs]  # === FIX: 用最后一批的 prompts ===
                sampled_rewards = [float(last_local_rewards_avg[i].item()) for i in idxs]  # === FIX: 同批 reward ===
                sampled_rule_rewards = [float(last_local_rule_rewards_avg[i].item()) for i in idxs]  # === FIX: 同批 rule reward ===
                wandb.log(
                    {
                        "images": [
                            wandb.Image(os.path.join(tmpdir, f"{idx}.jpg"),
                                        caption=f"{p:.100} | avg: {r:.2f}")
                            for idx, (p, r) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ]
                    },
                    step=global_step,
                )

        # 奖励处理 & 聚合
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)

        # all_gather 聚合（跨 rank）
        rewards_gathered = {k: all_gather_concat(v) for k, v in samples["rewards"].items()}
        rewards_np = {k: v.cpu().numpy() for k, v in rewards_gathered.items()}

        if is_main_process():
            wandb.log({"epoch": epoch, **{f"reward_{k}": v.mean() for k, v in rewards_np.items()
                                          if '_strict_accuracy' not in k and '_accuracy' not in k}},
                      step=global_step)

        # per-prompt 统计 + advantages 计算（全局→本地切片）
        local_total = samples["timesteps"].shape[0]
        start = rank * local_total
        end = start + local_total

        if config.per_prompt_stat_tracking:
            prompt_ids_all = all_gather_concat(samples["prompt_ids"])
            prompts_all = tokenizers[0].batch_decode(prompt_ids_all.cpu().numpy(), skip_special_tokens=True)
            advantages_global = stat_tracker.update(prompts_all, rewards_np['avg'])  # [global_total, T]
            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all, rewards_np)
            if is_main_process():
                wandb.log({"group_size": group_size, "trained_prompt_num": trained_prompt_num,
                           "zero_std_ratio": zero_std_ratio, "reward_std_mean": reward_std_mean},
                          step=global_step)
            stat_tracker.clear()
            # === FIX: 切回本 rank 的片段 ===
            advantages_local = torch.as_tensor(advantages_global[start:end], device=device)
        else:
            adv_all_global = (rewards_np['avg'] - rewards_np['avg'].mean()) / (rewards_np['avg'].std() + 1e-4)
            advantages_local = torch.as_tensor(adv_all_global[start:end], device=device)  # === FIX: 切本地 ===

        samples["advantages"] = advantages_local  # [local_total, T]

        # 清理之后无用字段
        del samples["rewards"]
        # del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps

        # 把samples的数据保存下来（调试）
        # tmpdir = f"./debug_images/epoch_{epoch}_rank_{rank}"
        # os.makedirs(tmpdir, exist_ok=True)
        # for index, prompt in enumerate(samples['prompts']):
        #     advantage = samples['advantages'][index]
        #     image_np = samples['images'][index].cpu().numpy().transpose(1, 2, 0)
        #     image_np = np.clip(image_np, 0.0, 1.0) * 255.0
        #     img_pil = Image.fromarray(image_np.astype(np.uint8))
        #     img_pil = img_pil.resize((config.resolution, config.resolution))
        #     img_pil.save(os.path.join(tmpdir, f"sample_{epoch}_{rank}_{index}.jpg"))
        #     with open(os.path.join(tmpdir, f"sample_{epoch}_{rank}_{index}.json"), "w", encoding="utf-8") as f:
        #         json.dump({"prompt": prompt, "advantage": advantage.tolist()}, f, ensure_ascii=False, indent=2)  # === FIX: tensor->list ===

        # ---------- TRAINING（policy / transformer） ----------
        microstep = 0
        optimizer.zero_grad(set_to_none=True)
        unwrap_fsdp(transformer).train()
        if config.train.update_reward_model:
            unwrap_fsdp(reward_model).eval()

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=device)

            # 打乱
            shuffled = {}
            for k, v in samples.items():
                if isinstance(v, torch.Tensor):
                    shuffled[k] = v[perm]
                elif isinstance(v, list) and (len(v) == total_batch_size) and isinstance(v[0], str):
                    idx = perm.cpu().numpy().tolist()
                    shuffled[k] = [v[i] for i in idx]
                else:
                    shuffled[k] = v  # 其它保持不变（如 images 若不是逐样本同长列表，可跳过）

            # 重组 batch：list of dicts
            reb = {}
            for k, v in shuffled.items():
                if isinstance(v, torch.Tensor):
                    reb[k] = v.reshape(-1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:])
                elif isinstance(v, list):
                    step = total_batch_size // config.sample.num_batches_per_epoch
                    reb[k] = [v[i:i+step] for i in range(0, len(v), step)]
            samples_batched = [dict(zip(reb, x)) for x in zip(*reb.values())]

            info = defaultdict(list)
            for i, sample in tqdm(list(enumerate(samples_batched)),
                                  desc=f"Epoch {epoch}.{inner_epoch}: training (rank {rank})",
                                  position=0, disable=(rank != 0)):
                if config.train.cfg:
                    embeds = torch.cat([train_neg_prompt_embeds[:len(sample["prompt_embeds"])],
                                        sample["prompt_embeds"]])
                    pooled_embeds = torch.cat([train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])],
                                               sample["pooled_prompt_embeds"]])
                else:
                    embeds, pooled_embeds = sample["prompt_embeds"], sample["pooled_prompt_embeds"]

                train_ts = list(range(num_train_timesteps))
                for j in tqdm(train_ts, desc="Timestep", position=1, leave=False, disable=(rank != 0)):
                    with autocast_ctx():
                        prev_sample, log_prob, prev_mean, std_t = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                        if config.train.beta > 0:
                            with torch.no_grad():
                                with disable_adapter_ctx(transformer):
                                    _, _, prev_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)

                    advantages_j = torch.clamp(sample["advantages"][:, j], -config.train.adv_clip_max, config.train.adv_clip_max)
                    ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                    unclipped = -advantages_j * ratio
                    clipped = -advantages_j * torch.clamp(ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range)
                    policy_loss = torch.mean(torch.maximum(unclipped, clipped))
                    if config.train.beta > 0:
                        kl = ((prev_mean - prev_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_t ** 2)
                        kl = torch.mean(kl)
                        loss = policy_loss + config.train.beta * kl
                    else:
                        loss = policy_loss

                    info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                    info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                    info["clipfrac_gt_one"].append(torch.mean(((ratio - 1.0) > config.train.clip_range).float()))
                    info["clipfrac_lt_one"].append(torch.mean(((1.0 - ratio) > config.train.clip_range).float()))
                    info["policy_loss"].append(policy_loss)
                    if config.train.beta > 0:
                        info["kl_loss"].append(kl)
                    info["loss"].append(loss)

                    # backward + 累积
                    loss_to_back = loss / float(global_accum_steps)
                    if scaler.is_enabled():
                        scaler.scale(loss_to_back).backward()
                    else:
                        loss_to_back.backward()

                    microstep += 1
                    if microstep % global_accum_steps == 0:
                        if config.train.max_grad_norm and config.train.max_grad_norm > 0:
                            if scaler.is_enabled(): scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
                        if scaler.is_enabled():
                            scaler.step(optimizer); scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    # 记录（只在 rank0）
                    if is_main_process():
                        info_mean = {k: torch.mean(torch.stack(v)).item() for k, v in info.items()}
                        info_mean.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        wandb.log(info_mean, step=global_step)
                    global_step += 1
                    info = defaultdict(list)

                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
        
        dist.barrier()


        # ---------- REWARD MODEL UPDATE ----------
        if config.train.update_reward_model:
            unwrap_fsdp(transformer).eval()
            unwrap_fsdp(reward_model).train()

            # # 重新分 batch
            # reb = {}
            # for k, v in samples.items():
            #     if isinstance(v, torch.Tensor):
            #         reb[k] = v.reshape(-1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:])
            #     elif isinstance(v, list):
            #         step = total_batch_size // config.sample.num_batches_per_epoch
            #         reb[k] = [v[i:i+step] for i in range(0, len(v), step)]
            # samples_batched_rm = [dict(zip(reb, x)) for x in zip(*reb.values())]

            # pos_imgs_B, neg_imgs_B, pos_prompts_B, neg_prompts_B = [], [], [], []
            # for i_rm, sample in tqdm(list(enumerate(samples_batched_rm)),
            #                          desc=f"Reward Model {epoch}: creating batch (rank {rank})",
            #                          position=0, disable=(rank != 0)):  # === FIX: 非0静默 ===
            #     batch_images, batch_prompts, batch_rewards = [], [], []
            #     bs = sample["timesteps"].shape[0]
            #     for idx in range(min(bs, 16)):
            #         image = sample["images"][idx]
            #         batch_images.append(image)
            #         prompt = sample["prompts"][idx] if idx < len(sample["prompts"]) else sample["prompts"][idx % len(sample["prompts"])]
            #         batch_prompts.append(prompt)
            #         r_value = sample["advantages"][idx, 0].item()
            #         batch_rewards.append(r_value)

            #     if len(batch_images) > 1:
            #         batch_images = torch.stack(batch_images)
            #         batch_rewards = np.array(batch_rewards)
            #         pos_i, neg_i, pos_p, neg_p = select_positive_negative_pairs_GT(batch_images, batch_prompts, batch_rewards, num_pairs=4)
            #         if pos_i is not None:
            #             pos_imgs_B.append(pos_i); neg_imgs_B.append(neg_i)
            #             pos_prompts_B.append(pos_p); neg_prompts_B.append(neg_p)

            # if len(pos_imgs_B) > 0:
            #     contrastive_loss = compute_contrastive_loss(
            #         reward_model, reward_processor,
            #         pos_imgs_B, neg_imgs_B, pos_prompts_B, neg_prompts_B, device
            #     )
            # else:
            #     contrastive_loss = torch.tensor(0.0, device=device)
            # contrastive_loss.backward()
            # reward_optimizer.step(); reward_optimizer.zero_grad()

            # 1. 收集所有 GPU 数据
            images_all = all_gather_concat(samples["images"])
            prompt_ids_all = all_gather_concat(samples["prompt_ids"])
            prompts_all = tokenizers[0].batch_decode(prompt_ids_all.cpu().numpy(), skip_special_tokens=True)
            rule_rewards_all = all_gather_concat(samples["rule_rewards"]["avg"])
            if rule_rewards_all.dim() > 1:
                rule_rewards_all = rule_rewards_all[:, 0]
            rule_rewards_np = rule_rewards_all.cpu().numpy()
            
            # 2. 主进程选择正负样本（每个 prompt 的最高/最低 reward）
            pos_imgs, neg_imgs, pos_prompts = [], [], []
            if rank == 0:
                for prompt in np.unique(prompts_all):
                    idx = np.where(np.array(prompts_all) == prompt)[0]
                    if len(idx) < 2: continue
                    rewards = rule_rewards_np[idx]
                    max_i, min_i = idx[np.argmax(rewards)], idx[np.argmin(rewards)]
                    if max_i != min_i:
                        pos_imgs.append(images_all[max_i])
                        neg_imgs.append(images_all[min_i])
                        pos_prompts.append(prompt)
            
            # 3. 广播样本数量和数据
            num_pairs = torch.tensor(len(pos_imgs) if rank == 0 else 0, device=device)
            dist.broadcast(num_pairs, 0)
            
            if num_pairs.item() > 0:
                # 准备或创建 buffer
                if rank == 0:
                    pos_imgs = torch.stack(pos_imgs)
                    neg_imgs = torch.stack(neg_imgs)
                else:
                    shape = images_all[0].shape
                    pos_imgs = torch.empty(num_pairs, *shape, device=device, dtype=images_all.dtype)
                    neg_imgs = torch.empty(num_pairs, *shape, device=device, dtype=images_all.dtype)
                
                dist.broadcast(pos_imgs, 0)
                dist.broadcast(neg_imgs, 0)
                
                # 广播 prompts：简化处理，只广播 prompt token ids
                if rank == 0:
                    prompt_tokens = tokenizers[0](pos_prompts, padding="max_length", max_length=77,
                                                  truncation=True, return_tensors="pt").input_ids.to(device)
                else:
                    prompt_tokens = torch.empty(num_pairs, 77, dtype=torch.long, device=device)
                dist.broadcast(prompt_tokens, 0)
                pos_prompts_all = tokenizers[0].batch_decode(prompt_tokens.cpu().numpy(), skip_special_tokens=True)
                
                # 4. 分配到各 GPU
                n = num_pairs.item()
                world_size = dist.get_world_size()
                per_gpu = (n + world_size - 1) // world_size
                start, end = rank * per_gpu, min((rank + 1) * per_gpu, n)
                
                # 5. 计算 loss 并更新
                if start < end:
                    pos_list = [pos_imgs[i:i+1] for i in range(start, end)]
                    neg_list = [neg_imgs[i:i+1] for i in range(start, end)]
                    prompt_list = [[pos_prompts_all[i]] for i in range(start, end)]
                    
                    loss = compute_contrastive_loss(reward_model, reward_processor, 
                                                    pos_list, neg_list, prompt_list, prompt_list, device)
                    loss.backward()
                else:
                    loss = torch.tensor(0.0, device=device)
                
                reward_optimizer.step()
                reward_optimizer.zero_grad()
            else:
                loss = torch.tensor(0.0, device=device)
            
            dist.barrier()
            if is_main_process():
                wandb.log({"reward_loss": loss.item(), "reward_pairs": num_pairs.item()}, step=global_step)
            
            # if is_main_process():
            #     wandb.log({"reward_contrastive_loss": float(contrastive_loss.item())}, step=global_step)
            #     # 可视化正/负样本（转 PIL）
            #     def _tensor_to_pil(img_t):
            #         # img_t: [C,H,W] in either [0,1] or [-1,1]
            #         arr = img_t.detach().cpu().numpy()
            #         if arr.min() < 0.0:
            #             arr = (arr * 0.5 + 0.5)
            #         arr = np.clip(arr, 0.0, 1.0)
            #         arr = (arr * 255).astype(np.uint8).transpose(1,2,0)
            #         return Image.fromarray(arr)
            #     if epoch % 10 == 0:
            #         pos_vis = [_tensor_to_pil(t) for t in (torch.cat(pos_imgs_B,0) if isinstance(pos_imgs_B, list) else pos_imgs_B)]
            #         neg_vis = [_tensor_to_pil(t) for t in (torch.cat(neg_imgs_B,0) if isinstance(neg_imgs_B, list) else neg_imgs_B)]
            #         wandb.log(
            #             {
            #                 "positive_images": [wandb.Image(p, caption=pos_prompts_B[i] if i < len(pos_prompts_B) else "") for i,p in enumerate(pos_vis)],
            #                 "negative_images": [wandb.Image(n, caption=neg_prompts_B[i] if i < len(neg_prompts_B) else "") for i,n in enumerate(neg_vis)],
            #             },
            #             step=global_step
            #         )
            reward_step += 1

        epoch += 1
        dist.barrier()

if __name__ == "__main__":
    app.run(main)
