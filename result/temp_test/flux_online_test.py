import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



import torch
from diffusers import FluxPipeline
from peft import PeftModel

# 设置模型和lora路径
model_path = "/pfs/yangyuanming/code2/models/FLUX.1-dev/model/black-forest-labs__FLUX.1-dev/main"
lora_path = "/pfs/yangyuanming/code2/models/logs/pickscore/flux-group24-8gpu-online/checkpoints/checkpoint-3660/lora"
device = "cuda"
dtype = torch.float16

# 加载FluxPipeline
pipeline = FluxPipeline.from_pretrained(
    model_path,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
# # 套lora
# pipeline.transformer = PeftModel.from_pretrained(
#     pipeline.transformer,
#     lora_path
# )
pipeline.safety_checker = None
pipeline.vae.requires_grad_(False)
pipeline.text_encoder.requires_grad_(False)
pipeline.text_encoder_2.requires_grad_(False)
pipeline.transformer.requires_grad_(False)
pipeline = pipeline.to(device)
pipeline.transformer.eval()

# 生图
prompt = "a family of roman soldiers on holiday, mallorca, 1 9 8 3, polaroid photography by andrei tarkovsky"
with torch.no_grad():
    result = pipeline(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=30,
        guidance_scale=3.5,
        output_type="pil"
    )
image = result.images[0]
image.save("flux_lora_sample.png")
print("图片已保存为 flux_lora_sample.png")
