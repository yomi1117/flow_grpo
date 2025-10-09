# CLIP Reward Model更新功能

本文档说明了在`train_flux_online.py`中添加的CLIP reward model对比学习更新功能。

## 功能概述

在每轮GRPO训练完成后，使用生成的样本对CLIP reward model进行对比学习更新，以改善reward model的性能。

## 主要修改

### 1. 新增导入
```python
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
```

### 2. 新增函数

#### `select_positive_negative_pairs(images, prompts, rewards, num_pairs=None)`
- 从GRPO样本中选择正负样本对
- 基于奖励值确定正负样本
- 随机选择不同的prompt样本作为对比对

#### `compute_contrastive_loss(clip_model, clip_processor, positive_images, negative_images, positive_prompts, negative_prompts, device)`
- 计算对比学习损失
- 使用margin ranking loss最大化正样本相似度，最小化负样本相似度
- 参考clipscore的实现方式

### 3. CLIP模型初始化
```python
# Initialize CLIP model for reward model updating
clip_model = CLIPModel.from_pretrained("/pfs/yangyuanming/code2/models/clip-vit-large-patch14/model/openai__clip-vit-large-patch14/main").to(accelerator.device)
clip_processor = CLIPProcessor.from_pretrained("/pfs/yangyuanming/code2/models/clip-vit-large-patch14/model/openai__clip-vit-large-patch14/main")

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
```

### 4. 训练循环内更新
在每个GRPO训练步骤的梯度同步点添加了reward model更新步骤：
- 从当前batch的latents解码生成图像
- 使用优势值作为奖励代理选择正负样本对
- 计算改进的对比学习损失（包含margin ranking loss + MSE正则化）
- 反向传播更新CLIP模型
- 将损失记录到wandb与其他训练指标一起

## 配置参数

### 启用功能
在配置文件中添加：
```python
config.train.update_reward_model = True  # 启用reward model更新
config.train.clip_learning_rate = 1e-5   # CLIP学习率（可选，默认1e-5）
```

### 关键参数
- `num_pairs=4`: 每次更新使用的正负样本对数量（在训练循环内减少以节省计算）
- `margin=0.5`: margin ranking loss的边界值（减小以避免过度惩罚）
- 每次更新最多使用16个样本（限制GPU内存使用）
- `pos_target=0.8`: 正样本目标相似度
- `neg_target=0.2`: 负样本目标相似度
- MSE正则化权重：0.1

## 使用方法

1. 在配置文件中启用功能：
```python
config.train.update_reward_model = True
```

2. 运行训练脚本：
```bash
python train_flux_online.py --config=your_config.py
```

3. 监控wandb中的`clip_contrastive_loss`指标

## 注意事项

- 该功能会增加训练时间和GPU内存使用（每个梯度更新步骤都会进行）
- 需要至少2个不同的样本才能进行对比学习
- CLIP模型路径硬编码，如需修改请调整代码中的路径
- 正负样本基于优势值（advantages）和prompt差异进行选择
- 更新频率：每个梯度同步点（`accelerator.sync_gradients`为True时）
- 使用改进的对比损失：margin ranking loss + MSE正则化

## 代码位置

- 正负样本选择：`select_positive_negative_pairs()` 函数
- 对比损失计算：`compute_contrastive_loss()` 函数（包含改进的margin ranking loss）
- 模型初始化：main函数中CLIP模型加载部分
- 训练更新：training循环内部的"REWARD MODEL UPDATE"部分（在梯度同步点）
- Prompts保存：sampling阶段在samples中保存原始prompts文本

## 改进的Margin Ranking Loss

新的对比损失函数包含三个组件：

1. **Margin Ranking Loss**: 确保正样本相似度高于负样本相似度
2. **正样本MSE Loss**: 鼓励正样本达到目标相似度(0.8)
3. **负样本MSE Loss**: 惩罚负样本超过目标相似度(0.2)

```python
total_loss = margin_ranking_loss + 0.1 * (pos_mse_loss + neg_mse_loss)
```

这种设计能更好地指导CLIP模型学习，避免单纯的margin loss可能导致的梯度问题。
