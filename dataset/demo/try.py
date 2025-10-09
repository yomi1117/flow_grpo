import torch
import numpy as np
from PIL import Image

# 生成512x512的噪音图
def generate_noise_image(size=512, noise_type='gaussian'):
    """
    生成指定尺寸的噪音图
    
    Args:
        size: 图像尺寸，默认512
        noise_type: 噪音类型，'gaussian'或'uniform'
    
    Returns:
        PIL Image对象
    """
    if noise_type == 'gaussian':
        # 生成高斯噪音
        noise = torch.randn(3, size, size)
    elif noise_type == 'uniform':
        # 生成均匀分布噪音
        noise = torch.rand(3, size, size)
    else:
        raise ValueError("noise_type must be 'gaussian' or 'uniform'")
    
    # 将tensor转换为numpy数组并调整到0-255范围
    noise_np = noise.numpy()
    noise_np = (noise_np - noise_np.min()) / (noise_np.max() - noise_np.min()) * 255
    noise_np = noise_np.astype(np.uint8)
    
    # 转换为PIL Image
    noise_image = Image.fromarray(noise_np.transpose(1, 2, 0))
    
    return noise_image

# 生成噪音图
noise_img = generate_noise_image(512, 'gaussian')

# 保存噪音图
noise_img.save('noise_512x512.png')
print("已生成512x512的噪音图并保存为 noise_512x512.png")

# 显示图像信息
print(f"图像尺寸: {noise_img.size}")
print(f"图像模式: {noise_img.mode}")
