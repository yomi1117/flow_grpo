import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from diffusers import FluxPipeline


# 你的路径
lora_path = "/pfs/yangyuanming/code2/models/logs/pickscore/flux-group24-8gpu-online/checkpoints/checkpoint-3780/lora"

# 读取 LoRA 配置
config = PeftConfig.from_pretrained(lora_path)

# 加载FluxPipeline
pipeline = FluxPipeline.from_pretrained(
    "/pfs/yangyuanming/code2/models/FLUX.1-dev/model/black-forest-labs__FLUX.1-dev/main",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
# 套lora
pipeline.transformer = PeftModel.from_pretrained(
    pipeline.transformer,
    lora_path
)

model = pipeline.transformer
# 遍历所有 LoRA 层，检查 B 矩阵是否全为 0
all_zero = True
for name, param in model.named_parameters():
    if "lora_B" in name:  # 关键：LoRA 的 B 矩阵
        tensor = param.detach().cpu()
        if torch.count_nonzero(tensor) > 0:
            print(f"[更新检测] {name} 非零，共 {torch.count_nonzero(tensor).item()} 个非零元素")
            all_zero = False
        else:
            print(f"[未更新] {name} 仍为全零")

if all_zero:
    print("\n>>> 所有 LoRA B 矩阵仍然为零，LoRA 没有生效！")
else:
    print("\n>>> LoRA 已经生效，B 矩阵不为零。")
