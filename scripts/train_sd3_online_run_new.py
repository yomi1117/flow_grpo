from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
import torchvision.transforms as transforms
import csv
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class JointModel(torch.nn.Module):
    """
    Joint model that combines policy model (transformer) and reward model (CLIP)
    for unified training and accelerate management
    """
    def __init__(self, transformer, clip_model):
        super().__init__()
        self.transformer = transformer
        self.clip_model = clip_model

    def forward(self, *args, **kwargs):
        """
        Forward pass for the joint model
        This is mainly for compatibility with accelerate
        """
        # This method is mainly for compatibility with accelerate
        # The actual forward passes will be handled separately for each model
        raise NotImplementedError("Use forward_policy or forward_reward instead")

    def forward_policy(self, *args, **kwargs):
        """
        Forward pass for the policy model (transformer)
        """
        return self.transformer(*args, **kwargs)

    def forward_reward(self, *args, **kwargs):
        """
        Forward pass for the reward model (CLIP)
        """
        return self.clip_model(*args, **kwargs)

    def train_policy(self):
        """Set policy model to train mode and reward model to eval mode"""
        self.transformer.train()
        self.clip_model.eval()

    def train_reward(self):
        """Set reward model to train mode and policy model to eval mode"""
        self.transformer.eval()
        self.clip_model.train()

    def train_both(self):
        """Set both models to train mode"""
        self.transformer.train()
        self.clip_model.train()

    def parameters_policy(self):
        """Get parameters of the policy model"""
        return self.transformer.parameters()

    def parameters_reward(self):
        """Get parameters of the reward model"""
        return self.clip_model.parameters()

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization

        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()

            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])

            # Return current replica's sample indices
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.

    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.

    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)

    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array,
        return_inverse=True,
        return_counts=True
    )

    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)

    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])

    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)

    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

def select_positive_negative_pairs(images, prompts, rewards, num_pairs=None):
    """
    随机从样本中选择一对正负样本用于对比学习，完全不使用reward，且正负样本的prompt相同
    Args:
        images: 生成的图像 tensor
        prompts: 对应的提示词列表
        rewards: 占位参数，不使用
        num_pairs: 要选择的样本对数量（未用到，只返回一对）
    Returns:
        positive_images, negative_images, positive_prompts, negative_prompts
    """
    batch_size = len(images)
    if batch_size < 2:
        return None, None, None, None

    # 随机选择一个prompt
    prompt_set = list(set(prompts))
    if len(prompt_set) == 0:
        return None, None, None, None

    # 随机选一个prompt
    selected_prompt = random.choice(prompt_set)
    # 找到所有该prompt的索引
    indices = [i for i, p in enumerate(prompts) if p == selected_prompt]
    if len(indices) < 2:
        return None, None, None, None

    # 随机选两个不同的索引
    pos_idx, neg_idx = random.sample(indices, 2)

    positive_images = images[pos_idx].unsqueeze(0)
    negative_images = images[neg_idx].unsqueeze(0)
    positive_prompts = [selected_prompt]
    negative_prompts = [selected_prompt]

    return positive_images, negative_images, positive_prompts, negative_prompts

def select_positive_negative_pairs_GT(images, prompts, rewards, num_pairs=None, gt_data_path="/pfs/yangyuanming/code2/datasets/fluxkrea-data/pickscore-prompt/train_1"):
    """
    选择正负样本用于对比学习，负样本从当前批次抽取，正样本从ground truth数据中获取
    Args:
        images: 生成的图像 tensor
        prompts: 对应的提示词列表
        rewards: 占位参数，不使用
        num_pairs: 要选择的样本对数量（未用到，只返回一对）
        gt_data_path: ground truth数据的路径
    Returns:
        positive_images, negative_images, positive_prompts, negative_prompts
    """
    batch_size = len(images)
    if batch_size < 2:
        return None, None, None, None

    # 负样本抽取逻辑保持不变
    prompt_set = list(set(prompts))
    if len(prompt_set) == 0:
        return None, None, None, None

    # 随机选一个prompt作为负样本的prompt
    selected_prompt = random.choice(prompt_set)
    indices = [i for i, p in enumerate(prompts) if p == selected_prompt]
    if len(indices) < 2:
        return None, None, None, None

    # 随机选两个不同的索引作为负样本
    neg_idx, _ = random.sample(indices, 2)
    negative_images = images[neg_idx].unsqueeze(0)
    negative_prompts = [selected_prompt]

    # 正样本从ground truth数据中获取 - 直接匹配selected_prompt
    prompts_tsv_path = os.path.join(gt_data_path, "prompts.tsv")

    # 读取TSV文件
    gt_data = []
    try:
        with open(prompts_tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                gt_data.append({
                    'index': int(row['index']),
                    'filename': row['filename'],
                    'prompt': row['prompt']
                })
    except Exception as e:
        print(f"Error reading TSV file: {e}")
        return None, None, None, None

    if len(gt_data) == 0:
        return None, None, None, None

    # 直接查找与selected_prompt匹配的样本
    matching_samples = [sample for sample in gt_data if sample['prompt'] == selected_prompt]

    # 如果找到完全匹配的样本，从中随机选择一个
    if matching_samples:
        gt_sample = random.choice(matching_samples)
    else:
        # 如果没有完全匹配的，随机选择一个ground truth样本
        gt_sample = random.choice(gt_data)
        print(f"Warning: No matching samples found for prompt '{selected_prompt}', using random selection")

    image_path = os.path.join(gt_data_path, gt_sample['filename'])

    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 定义图像预处理变换（需要根据images tensor的尺寸调整）
        # 假设images是[batch_size, C, H, W]格式
        if len(images.shape) == 4:
            target_size = (images.shape[2], images.shape[3])  # (H, W)
        else:
            target_size = (1024, 1024)  # SD3 默认尺寸

        # 这里transforms报错，改为直接导入torchvision.transforms
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化到[-1, 1]
        ])

        positive_images = transform(image).unsqueeze(0).to(images.device)
        positive_prompts = [gt_sample['prompt']]

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None, None

    return positive_images, negative_images, positive_prompts, negative_prompts

def compute_contrastive_loss(clip_model, clip_processor, positive_images, negative_images, positive_prompts, negative_prompts, device):
    """
    计算对比学习损失（SigLIP loss 版本），支持多batch输入
    Args:
        clip_model: CLIP模型
        clip_processor: CLIP处理器
        positive_images: 正样本图像 [B, 3, H, W] 或 [total_samples, 3, H, W]
        negative_images: 负样本图像 [B, 3, H, W] 或 [total_samples, 3, H, W]
        positive_prompts: 正样本提示词列表
        negative_prompts: 负样本提示词列表
        device: 设备
    Returns:
        对比学习损失
    """
    if positive_images is None or len(positive_images) == 0:
        return torch.tensor(0.0, device=device)

    # 处理图像（从CHW格式转换为PIL需要的HWC格式）
    def process_images(images):
        # images shape: [batch, 3, H, W], 值范围 [0, 1]
        images_np = (images.cpu().numpy() * 255).astype(np.uint8)
        images_np = images_np.transpose(0, 2, 3, 1)  # CHW -> HWC
        pil_images = [Image.fromarray(img) for img in images_np]
        return pil_images

    # 处理输入数据
    if isinstance(positive_images, list):
        # 如果输入是列表，说明是多batch数据，需要拼接
        positive_images = torch.cat(positive_images, dim=0)
        negative_images = torch.cat(negative_images, dim=0)
        positive_prompts = [item for sublist in positive_prompts for item in sublist]
        negative_prompts = [item for sublist in negative_prompts for item in sublist]

    pos_pil_images = process_images(positive_images)
    neg_pil_images = process_images(negative_images)

    # 处理正样本
    pos_inputs = clip_processor(
        text=positive_prompts,
        images=pos_pil_images,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # 处理负样本
    neg_inputs = clip_processor(
        text=negative_prompts,
        images=neg_pil_images,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # 获取特征（需要计算梯度，所以不用torch.no_grad()）
    # 获取正样本的图像和文本特征
    pos_outputs = clip_model(**pos_inputs)
    pos_image_embeds = pos_outputs.image_embeds  # [B, D]
    pos_text_embeds = pos_outputs.text_embeds    # [B, D]

    # 获取负样本的图像和文本特征
    neg_outputs = clip_model(**neg_inputs)
    neg_image_embeds = neg_outputs.image_embeds  # [B, D]
    neg_text_embeds = neg_outputs.text_embeds    # [B, D]

    # SigLIP loss 实现
    # 1. 计算正样本的相似度
    pos_sim = F.cosine_similarity(pos_image_embeds, pos_text_embeds, dim=-1)  # [B]
    # 2. 计算负样本的相似度
    neg_sim = F.cosine_similarity(neg_image_embeds, neg_text_embeds, dim=-1)  # [B]

    # 3. Sigmoid对比损失
    # SigLIP: -log(sigmoid(pos_sim - neg_sim))
    loss = -torch.log(torch.sigmoid(pos_sim - neg_sim) + 1e-8).mean()

    return loss

def compute_log_prob(joint_model, pipeline, sample, j, embeds, pooled_embeds, config):
    # 确保latent张量使用正确的dtype以匹配模型参数
    latents = sample["latents"][:, j]
    if hasattr(joint_model.transformer, 'dtype'):
        target_dtype = next(joint_model.transformer.parameters()).dtype
        latents = latents.to(dtype=target_dtype)

    if config.train.cfg:
        noise_pred = joint_model.forward_policy(
            hidden_states=torch.cat([latents] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_uncond = noise_pred_uncond.detach()
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = joint_model.forward_policy(
            hidden_states=latents,
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    # compute the log prob of next_latents given latents under the current model
    # 确保所有latent张量使用与模型匹配的dtype
    model_dtype = next(joint_model.transformer.parameters()).dtype
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.to(dtype=model_dtype),
        sample["timesteps"][:, j],
        latents.to(dtype=model_dtype),
        prev_sample=sample["next_latents"][:, j].to(dtype=model_dtype),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=128,
            device=accelerator.device
        )
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, latents, log_probs = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)

    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, joint_model, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(joint_model.transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        # gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
        gradient_accumulation_steps=1,
    )
    if accelerator.is_main_process:
        wandb.init(
            project="flow_grpo_online_RL_sd3_clip",
            mode="offline",
            # mode="disabled"
        )
        # accelerator.init_trackers(
        #     project_name="flow-grpo",
        #     config=config.to_dict(),
        #     init_kwargs={"wandb": {"name": config.run_name}},
        # )
    logger.info(f"\n{config}")
    logger.info(f"Accelerator info: num_processes={accelerator.num_processes}, process_index={accelerator.process_index}, device={accelerator.device}")
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    # >>> NEW: only rank-0 hits HF hub/disk first; others wait to avoid cache lock / IO stampede
    with accelerator.main_process_first():
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            config.pretrained.model
        )
    accelerator.wait_for_everyone()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)

    pipeline.transformer.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers for SD3
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    # Initialize CLIP model for reward model updating
    # 直接从reward_fn中获取clip_model和clip_processor，无需重新加载
    reward_key = list(config.reward_fn.keys())
    clip_model = reward_fn.score_fns[reward_key[0]].model
    clip_processor = reward_fn.score_fns[reward_key[0]].processor

    # CLIP model dtype will be handled after accelerator.prepare()

    # Create joint model combining transformer and CLIP
    joint_model = JointModel(transformer, clip_model)
    # move again to make sure the parameters are on the correct device and dtype
    # to check
    if accelerator.mixed_precision in ["fp16", "bf16"]:
        target_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16
        joint_model.transformer = joint_model.transformer.to(accelerator.device, dtype=target_dtype)
        joint_model.clip_model = joint_model.clip_model.to(accelerator.device, dtype=target_dtype)
    else:
        joint_model.transformer = joint_model.transformer.to(accelerator.device)
        joint_model.clip_model = joint_model.clip_model.to(accelerator.device)

    transformer_device = next(joint_model.transformer.parameters()).device
    clip_device = next(joint_model.clip_model.parameters()).device
    assert transformer_device == clip_device, f"Device mismatch: transformer on {transformer_device}, clip on {clip_device}"

    # 更新可训练参数列表，使用joint model的参数
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, joint_model.parameters_policy()))

    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 只用普通AdamW优化器
    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # Enable gradient computation for CLIP model parameters
    clip_model.train()
    for param in clip_model.parameters():
        param.requires_grad = True

    # Initialize CLIP optimizer
    clip_learning_rate = getattr(config.train, 'clip_learning_rate', 1e-5)
    clip_optimizer = torch.optim.AdamW(
        clip_model.parameters(),
        lr=clip_learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it's controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create a regular DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )

    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    # for deepspeed zero
    if accelerator.state.deepspeed_plugin:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.sample.train_batch_size
    joint_model, optimizer, clip_optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        joint_model, optimizer, clip_optimizer, train_dataloader, test_dataloader
    )

    # Fix dtype consistency after accelerator.prepare() for all models
    if accelerator.mixed_precision in ["fp16", "bf16"]:
        dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16

        # 智能转换模型参数，跳过索引相关的张量
        def smart_dtype_conversion(module, target_dtype):
            """智能转换参数和缓冲区，避免转换索引张量"""
            # 定义需要跳过的缓冲区名称模式（这些通常是索引）
            skip_patterns = [
                'position_ids',  # CLIP位置索引
                'position_embedding',  # 位置嵌入（虽然是参数但用于索引）
                'token_type_ids',  # token类型索引
                'attention_mask',  # 注意力掩码（通常是bool或int）
                'ids',  # 各种ID张量
                'indices',  # 索引张量
                'mask',  # 掩码张量
            ]

            for param in module.parameters():
                if param.dtype != target_dtype:
                    # 跳过索引相关的参数（通常是Long/Int类型）
                    if param.dtype in [torch.long, torch.int, torch.bool]:
                        continue
                    param.data = param.data.to(target_dtype)

            for name, buffer in module.named_buffers():
                if buffer.dtype != target_dtype:
                    # 跳过索引相关的缓冲区
                    should_skip = any(pattern in name.lower() for pattern in skip_patterns)
                    if should_skip or buffer.dtype in [torch.long, torch.int, torch.bool]:
                        continue
                    # 直接修改缓冲区数据，不重新注册
                    buffer.data = buffer.data.to(target_dtype)

        # 确保pipeline的VAE输出与模型输入类型匹配
        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            # VAE通常输出float32，但模型需要bf16输入
            # 我们需要确保VAE的输出被正确转换
            pass

        # 只转换transformer和joint_model的参数，跳过索引参数
        smart_dtype_conversion(joint_model.transformer, dtype)
        smart_dtype_conversion(pipeline.transformer, dtype)

        # 特别处理diffusers transformer中的所有embeddings相关模块
        if hasattr(pipeline.transformer, 'pos_embed'):
            # pos_embed通常包含投影层，可能有类型不匹配问题
            smart_dtype_conversion(pipeline.transformer.pos_embed, dtype)

        # 检查并转换transformer中的所有子模块
        for name, module in pipeline.transformer.named_modules():
            if 'embed' in name.lower() or 'proj' in name.lower():
                # embeddings和projection层可能有类型不匹配问题
                smart_dtype_conversion(module, dtype)

        # 对于CLIP模型，更保守地只转换权重参数
        clip_converted_count = 0
        clip_skipped_count = 0
        for param in joint_model.clip_model.parameters():
            if param.dtype != dtype and param.dtype not in [torch.long, torch.int, torch.bool]:
                param.data = param.data.to(dtype)
                clip_converted_count += 1
            elif param.dtype in [torch.long, torch.int, torch.bool]:
                clip_skipped_count += 1

        print(f"智能转换模型参数为 {dtype} 类型:")
        print(f"  - Transformer参数已转换")
        print(f"  - Embeddings/Projection层已处理")
        print(f"  - CLIP模型: 转换了{clip_converted_count}个参数，跳过了{clip_skipped_count}个索引参数")

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        # pipeline.transformer.eval()
        # if epoch % config.eval_freq == 0:
        #     eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        # if epoch % config.save_freq == 0 and accelerator.is_main_process:
        #     save_ckpt(config.save_dir, joint_model, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    images, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator
                    )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                    "prompts": prompts,  # 保存原始prompts文本用于CLIP模型更新
                    "images": images,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples_collated = {}
        for k in samples[0].keys():
            if k == "prompts":
                # 对于prompts，我们直接扁平化列表
                samples_collated[k] = []
                for s in samples:
                    samples_collated[k].extend(s[k])
            elif not isinstance(samples[0][k], dict):
                samples_collated[k] = torch.cat([s[k] for s in samples], dim=0)
            else:
                samples_collated[k] = {
                    sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                    for sub_key in samples[0][k]
                }
        samples = samples_collated

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]
        # 保留samples["prompts"]用于training循环中的CLIP模型更新

        total_batch_size, num_timesteps = samples["timesteps"].shape

        assert num_timesteps == config.sample.num_steps
        
        # import pdb; pdb.set_trace()
        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples_shuffled = {}
            for k, v in samples.items():
                if isinstance(v, torch.Tensor):
                    samples_shuffled[k] = v[perm]
                elif isinstance(v, list) and isinstance(v[0], str):  # prompt 这种字符串列表
                    idx = perm.cpu().numpy().tolist()
                    samples_shuffled[k] = [v[i] for i in idx]
                else:
                    raise TypeError(f"Unsupported type for key {k}: {type(v)}")

            # rebatch for training
            samples_batched = {}
            for k, v in samples_shuffled.items():
                if isinstance(v, torch.Tensor):
                    samples_batched[k] = v.reshape(-1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:])
                elif isinstance(v, list):  # 字符串不能 reshape
                    step = total_batch_size // config.sample.num_batches_per_epoch
                    samples_batched[k] = [v[i:i+step] for i in range(0, len(v), step)]

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            # pipeline.transformer.train()
            joint_model.train_policy()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(joint_model):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(joint_model, pipeline, sample, j, embeds, pooled_embeds, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with joint_model.transformer.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(joint_model, pipeline, sample, j, embeds, pooled_embeds, config)

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        print("ratio", ratio)
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                joint_model.parameters_policy(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
            
            #################### REWARD MODEL UPDATE ####################
            # Update CLIP reward model using contrastive learning after each batch
            # import pdb; pdb.set_trace()
            samples_batched_rm = {}
            for k, v in samples.items():
                if isinstance(v, torch.Tensor):
                    samples_batched_rm[k] = v.reshape(-1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:])
                elif isinstance(v, list):  # 字符串不能reshape
                    step = total_batch_size // config.sample.num_batches_per_epoch
                    samples_batched_rm[k] = [v[i:i+step] for i in range(0, len(v), step)]
                else:
                    raise TypeError(f"Unsupported type for key {k}: {type(v)}")
            # 转换为list of dicts，方便后续遍历
            samples_batched_rm = [
                dict(zip(samples_batched_rm, x)) for x in zip(*samples_batched_rm.values())
            ]
            # 更新reward model
            if config.train.update_reward_model:
                joint_model.train_reward()
                for param in joint_model.clip_model.parameters():
                    param.requires_grad = True

            positive_images_batched, negative_images_batched, positive_prompts_batched, negative_prompts_batched = [], [], [], []

            for i_rm, sample in tqdm(
                list(enumerate(samples_batched_rm)),
                desc=f"Reward Model {epoch}.{inner_epoch}: creating batch",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                # 从当前batch样本生成图像用于reward model更新
                batch_images = []
                batch_prompts = []
                batch_rewards = []

                # 从当前样本batch中获取数据
                with torch.no_grad():
                    # 获取当前batch的样本数量
                    batch_size = sample["timesteps"].shape[0]

                    for idx in range(min(batch_size, 16)):  # to check: batch_size is the Group size
                        image = sample["images"][idx]
                        batch_images.append(image)

                        # 获取对应的prompt - 从当前sample batch中的prompts获取
                        if idx < len(sample["prompts"]):
                            prompt = sample["prompts"][idx]
                        else:
                            # fallback: 使用模运算获取
                            prompt = sample["prompts"][idx % len(sample["prompts"])]
                        batch_prompts.append(prompt)

                        # 使用当前样本的优势值作为奖励代理
                        reward_value = sample["advantages"][idx, 0].item()  # to check: 取第一个时间步的优势值
                        batch_rewards.append(reward_value)

                if len(batch_images) > 1:  # 至少需要2个样本才能做对比学习
                    batch_images = torch.stack(batch_images)
                    batch_rewards = np.array(batch_rewards)

                    # 选择正负样本对
                    positive_images, negative_images, positive_prompts, negative_prompts = select_positive_negative_pairs_GT(
                        batch_images, batch_prompts, batch_rewards, num_pairs=4
                    )
                    if positive_images is not None:
                        positive_images_batched.append(positive_images)
                        negative_images_batched.append(negative_images)
                        positive_prompts_batched.append(positive_prompts)
                        negative_prompts_batched.append(negative_prompts)

            if len(positive_images_batched) > 0:
                # 计算对比学习损失
                contrastive_loss = compute_contrastive_loss(
                    joint_model.clip_model, clip_processor,
                    positive_images_batched, negative_images_batched,
                    positive_prompts_batched, negative_prompts_batched,
                    accelerator.device
                )

                # 反向传播和优化
                clip_optimizer.zero_grad()
                accelerator.backward(contrastive_loss)
                # clip_optimizer.step()

                # 将对比损失添加到info中一起记录
                if accelerator.is_main_process:
                    wandb.log({"clip_contrastive_loss": contrastive_loss.item()}, step=global_step)

            if config.train.ema:
                ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients

        epoch+=1

if __name__ == "__main__":
    app.run(main)
