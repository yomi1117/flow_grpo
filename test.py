from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取原图
img_path = "/pfs/yangyuanming/code2/datasets/sd3_5m-data/capybara.png"
img = Image.open(img_path).convert('RGB')
img_np = np.array(img).astype(np.float32) / 255.0  # [0,1]归一化

beta = 0.9  # 可调参数
noise = np.random.randn(*img_np.shape).astype(np.float32)
noise = (noise - noise.min()) / (noise.max() - noise.min())  # [0,1]归一化噪声
noisy_img_np = (1-beta) * img_np + beta * noise
noisy_img_np = np.clip(noisy_img_np, 0, 1)

noisy_img = Image.fromarray((noisy_img_np * 255).astype(np.uint8))

# 保存原图和加噪图
img.save("original.jpg")
noisy_img.save(f"noisy_beta_{beta}.jpg")
