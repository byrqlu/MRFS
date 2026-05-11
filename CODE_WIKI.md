# MRFS 项目代码文档 (Code Wiki)

## 1. 项目概述

### 1.1 项目简介

**MRFS (Mutually Reinforcing Image Fusion and Segmentation)** 是 CVPR 2024 的论文实现，核心思想是通过mutually reinforcing mechanism实现图像融合与语义分割的相互促进。该项目包含三个数据集对应的子项目：FMB、MFNet 和 PST900。

### 1.2 项目结构

```
/workspace/
├── FMB/                    # FMB数据集实验配置
│   ├── train.py           # 训练入口
│   ├── eval.py            # 测试入口
│   ├── models/            # 模型定义
│   ├── dataloader/        # 数据加载
│   ├── engine/            # 训练引擎
│   └── utils/             # 工具函数
├── MFNet/                 # MFNet数据集实验配置
├── PST900/                # PST900数据集实验配置
├── README.md
└── LICENSE
```

### 1.3 核心模块依赖关系

```
train.py / eval.py
    ├── models/model.py (MRFS主模型)
    │       ├── models/encoder_agg.py (RGBXTransformer编码器)
    │       ├── models/Seg_head.py (DecoderHead分割头)
    │       └── models/Fusion_head.py (CNNHead融合头)
    ├── dataloader/dataloader.py (数据加载)
    ├── engine/engine.py (训练引擎)
    ├── utils/loss_utils.py (损失函数)
    └── utils/metric.py (评估指标)
```

---

## 2. 环境配置

### 2.1 推荐依赖环境

| 依赖包 | 版本要求 |
|--------|----------|
| Python | 3.8 |
| PyTorch | 1.8.1+cu111 |
| timm | 0.9.8 |
| numpy | 1.24.4 |
| scipy | 1.10.1 |
| pillow | 10.1.0 |
| tqdm | 4.66.1 |
| tensorboardX | 2.6.2.2 |
| opencv-python | 4.8.1.78 |

---

## 3. 数据集结构

每个子项目的数据集需按以下结构组织：

```
dataset/{dataset_name}/
├── Visible/              # RGB可见光图像
├── Infrared/             # 红外/深度/其他模态图像
├── Label/                # 语义分割标签
├── train.txt             # 训练集图像名称列表
├── val.txt               # 验证集图像名称列表
└── test.txt              # 测试集图像名称列表
```

---

## 4. 核心模块详解

### 4.1 模型模块 (models/)

#### 4.1.1 model.py - MRFS主模型

**类: `MRFS(nn.Module)`**

主模型类，整合编码器、解码头和融合头。

| 属性 | 说明 |
|------|------|
| `backbone` | RGBXTransformer编码器 |
| `decode_head` | DecoderHead分割解码器 |
| `aux_head` | CNNHead图像融合头 |
| `criterion` | MakeLoss损失计算器 |
| `channels` | [64, 128, 320, 512] 特征通道数 |

**关键方法:**

| 方法 | 说明 |
|------|------|
| `encode_decode(rgb, modal_x)` | 编码-解码流程，返回分割结果和融合图像 |
| `forward(rgb, modal_x, Mask, label)` | 前向传播，训练时返回损失，测试时返回预测 |

**支持的Backbone:**

- `mit_b0` - Segformer-B0 (通道: [32, 64, 160, 256])
- `mit_b1` ~ `mit_b5` - Segformer-B1~B5 (通道: [64, 128, 320, 512])

---

#### 4.1.2 encoder_agg.py - RGB-X双分支Transformer编码器

**核心类: `RGBXTransformer`**

双分支Transformer编码器，包含RGB和X(红外/深度等)两个独立分支。

**网络结构:**

| 组件 | 说明 |
|------|------|
| `patch_embed1~4` | RGB分支的Patch Embedding |
| `extra_patch_embed1~4` | X模态分支的Patch Embedding |
| `block1~4` | RGB分支的Transformer Block |
| `extra_block1~4` | X模态分支的Transformer Block |
| `IGMAVCs` | 交互式门控混合注意力模块列表 |
| `PCASCs` | 渐进式循环注意力模块列表 |

**方法: `forward_features(x_rgb, x_e)`**

返回两个列表:
- `outs_vision`: [RGB特征, X特征] * 4个阶段
- `outs_semantic`: 融合特征 * 4个阶段

---

#### 4.1.3 modules.py - 核心注意力模块

**类: `IGMAVC` (交互式门控混合注意力)**

用于视觉补全的两模态交互注意力机制。

```
输入: x1, x2 (两个模态的特征)
输出: out_x1, out_x2 (增强后的两个模态)
```

子模块:
- `ChannelAttention`: 通道注意力
- `SpatialAttention`: 空间注意力
- `MixAttention`: 混合注意力

**类: `PCASC` (渐进式循环注意力)**

用于语义补全的跨模态注意力机制。

```
输入: x1, x2 (两个模态的特征)
输出: Fuse_out (融合特征)
```

子模块:
- `SelfAttention`: 自注意力
- `CrossAttention`: 交叉注意力

---

#### 4.1.4 Seg_head.py - 分割解码头

**类: `DecoderHead`**

MLP-based分割解码器，将多尺度特征融合并预测分割结果。

```
输入: [C1, C2, C3, C4] 多尺度特征 (1/4, 1/8, 1/16, 1/32)
输出: 分割预测图 (原始分辨率)
```

结构:
```
Linear_c4 → 上采样 → 
Linear_c3 → 上采样 → 
Linear_c2 → 上采样 → 
Linear_c1 → 
→ Concat → Linear_fuse → Dropout → Linear_pred → 输出
```

---

#### 4.1.5 Fusion_head.py - 图像融合头

**类: `CNNHead`**

CNN-based图像融合模块，将多尺度双模态特征融合为融合图像。

```
输入: 
  - inputs: [F1_rgb, F1_ir, F2_rgb, F2_ir, F3_rgb, F3_ir, F4_rgb, F4_ir]
  - original_input: [input_rgb, input_ir]

输出: 融合RGB图像 (3通道)
```

---

### 4.2 数据加载模块 (dataloader/)

#### 4.2.1 RGBXDataset.py

**类: `RGBXDataset(data.Dataset)`**

处理RGB-X双模态数据集的PyTorch Dataset类。

| 方法 | 说明 |
|------|------|
| `__len__` | 返回数据集长度 |
| `__getitem__` | 返回单个样本 (data, label, modal_x, Mask, fn) |

**图像读取模式:**
- `mode="RGB"`: 读取RGB彩色图像 (BGR→RGB转换)
- `mode="Gray"`: 读取灰度图并复制为3通道

---

#### 4.2.2 dataloader.py

**类: `TrainPre`**

训练数据预处理:
1. 随机镜像翻转
2. 随机尺度变换
3. 图像归一化
4. 随机裁剪并生成Mask

**函数:**

| 函数 | 说明 |
|------|------|
| `get_train_loader(config, engine, dataset)` | 创建训练DataLoader |
| `get_val_loader(config, engine, dataset)` | 创建验证DataLoader |

---

### 4.3 训练引擎模块 (engine/)

#### 4.3.1 engine.py - Engine类

**类: `Engine`**

统一的训练引擎，管理分布式训练、检查点和状态。

| 属性 | 说明 |
|------|------|
| `state` | State对象，包含epoch、iteration、dataloader等 |
| `distributed` | 是否分布式训练 |
| `world_size` | GPU数量 |

**关键方法:**

| 方法 | 说明 |
|------|------|
| `save_checkpoint(path)` | 保存检查点 |
| `restore_checkpoint()` | 恢复检查点 |
| `save_and_link_checkpoint(dir)` | 保存并链接最新检查点 |

---

#### 4.3.2 evaluator.py - Evaluator类

**类: `Evaluator`**

模型评估基类，支持单进程和多进程评估。

**关键方法:**

| 方法 | 说明 |
|------|------|
| `run(model_path, model_indice, log_file, link_log_file)` | 运行评估 |
| `sliding_eval_rgbX(img, modal_x, crop_size, stride_rate, device)` | 滑动窗口评估 |
| `scale_process_rgbX(...)` | 多尺度处理 |

---

### 4.4 工具模块 (utils/)

#### 4.4.1 loss_utils.py - 损失函数

**类: `FusionLoss`**

图像融合损失，包含:
- `con_loss`: 对比度损失 (L1)
- `gradient_loss`: 梯度损失 (Sobel边缘)
- `color_loss`: 颜色损失 (YCbCr空间)

**类: `MakeLoss`**

总损失计算:
```python
total_loss = 1.0 * semantic_loss + 0.1 * fusion_loss
```

---

#### 4.4.2 metric.py - 评估指标

| 函数 | 说明 |
|------|------|
| `hist_info(n_cl, pred, gt)` | 计算混淆矩阵 |
| `compute_metric(results, class_num)` | 计算IoU和mIoU |
| `compute_score(hist, correct, labeled)` | 计算多指标 (IoU, 像素精度, 类别精度) |

---

#### 4.4.3 lr_policy.py - 学习率策略

| 类 | 说明 |
|------|------|
| `PolyLR` | 多项式衰减 |
| `WarmUpPolyLR` | Warmup + 多项式衰减 |
| `MultiStageLR` | 多阶段阶梯衰减 |
| `LinearIncreaseLR` | 线性 warmup |

**默认使用: `WarmUpPolyLR`**

```python
lr = base_lr * ((1 - cur_iter / total_iters) ** lr_power)
if cur_iter < warmup_steps:
    lr = base_lr * (cur_iter / warmup_steps)
```

---

#### 4.4.4 pyt_utils.py - 通用工具

| 函数 | 说明 |
|------|------|
| `load_model(model, model_file, is_restore)` | 加载模型权重 |
| `parse_devices(devices)` | 解析GPU设备字符串 |
| `all_reduce_tensor(tensor, world_size)` | 分布式张量同步 |
| `reduce_value(value, average)` | 分布式值同步 |
| `ensure_dir(path)` | 确保目录存在 |
| `link_file(src, target)` | 创建符号链接 |

---

### 4.5 transforms.py

图像变换工具，包括:
- `generate_random_crop_pos`: 随机裁剪位置
- `random_crop_pad_to_shape`: 随机裁剪并填充
- `normalize`: 图像归一化
- `pad_image_to_shape`: 图像填充

---

## 5. 关键算法说明

### 5.1 互增强机制 (Mutually Reinforcing)

MRFS的核心创新在于图像融合与语义分割的相互促进:

1. **RGB分支 + X分支** → 通过IGMAVC进行交互
2. **交互特征** → 通过PCASC进行语义融合
3. **融合特征** → 同时用于分割预测和图像融合
4. **分割结果** → 提供语义指导优化融合质量

### 5.2 训练流程

```
for epoch in epochs:
    for batch in train_loader:
        imgs, modal_xs, Mask, gts = batch
        
        # 前向传播
        loss, seg_loss, fus_loss = model(imgs, modal_xs, Mask, gts)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 学习率更新
        lr = lr_policy.get_lr(iteration)
        optimizer.param_groups[i]['lr'] = lr
```

### 5.3 测试流程

```
for img, modal_x in test_loader:
    # 滑动窗口评估
    pred = sliding_eval_rgbX(img, modal_x, crop_size=[600, 800], stride_rate=2/3)
    
    # 多尺度融合
    # 翻转增强 (可选)
    
    # 计算指标
    hist += hist_info(num_classes, pred, label)

mIoU = compute_metric(hist, num_classes)
```

---

## 6. 项目运行方式

### 6.1 训练

**单GPU训练:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

**多GPU分布式训练:**
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### 6.2 测试

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py -e=epoch_number
```

**示例:**
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py -e=MRFS
```

### 6.3 主要训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backbone` | mit_b4 | 骨干网络 |
| `--batch_size` | 6 | 批次大小 |
| `--lr` | 6e-5 | 学习率 |
| `--nepochs` | 5000 | 训练轮数 |
| `--image_height` | 480 | 训练图像高度 |
| `--image_width` | 640 | 训练图像宽度 |
| `--num_classes` | 15 | 分割类别数 |
| `--checkpoint_start_epoch` | 150 | 开始保存检查点的轮数 |
| `--checkpoint_step` | 5 | 检查点保存间隔 |

---

## 7. 各子项目配置

### 7.1 FMB 数据集

- **类别数**: 15
- **类别**: background, Road, Sidewalk, Building, Lamp, Sign, Vegetation, Sky, Person, Car, Truck, Bus, Motorcycle, Bicycle, Pole
- **训练样本**: 1220
- **测试样本**: 280
- **图像尺寸**: 480x640

### 7.2 MFNet 数据集

- **类别数**: 15 (与FMB类似)
- **训练样本**: 约900
- **测试样本**: 约200

### 7.3 PST900 数据集

- **类别数**: 15 (与FMB类似)
- **特点**: 主要用于RGB-thermal融合任务

---

## 8. 预训练模型

预训练权重可从以下链接获取:
- [FMB模型](https://drive.google.com/drive/folders/17LroKnuEWttvtcbuv-G-DXaXYdua52cN)
- [MFNet模型](https://drive.google.com/drive/folders/1txDn-U04KEKA6gbSUjSsn-QqaN4nFw0Y)
- [PST900模型](https://drive.google.com/drive/folders/1iEu3QZSV-q18u28X4cB7GK8UpLoRhXTX)

---

## 9. 文件清单

### 9.1 入口文件

| 文件 | 路径 | 说明 |
|------|------|------|
| train.py | FMB/ | FMB训练入口 |
| eval.py | FMB/ | FMB测试入口 |

### 9.2 模型文件

| 文件 | 路径 | 说明 |
|------|------|------|
| model.py | models/ | MRFS主模型 |
| modules.py | models/ | IGMAVC/PCASC模块 |
| encoder_agg.py | models/ | 编码器 |
| Seg_head.py | models/ | 分割头 |
| Fusion_head.py | models/ | 融合头 |

### 9.3 数据文件

| 文件 | 路径 | 说明 |
|------|------|------|
| dataloader.py | dataloader/ | 数据加载器 |
| RGBXDataset.py | dataloader/ | 数据集类 |

### 9.4 引擎文件

| 文件 | 路径 | 说明 |
|------|------|------|
| engine.py | engine/ | 训练引擎 |
| evaluator.py | engine/ | 评估器 |
| logger.py | engine/ | 日志系统 |

### 9.5 工具文件

| 文件 | 路径 | 说明 |
|------|------|------|
| loss_utils.py | utils/ | 损失函数 |
| metric.py | utils/ | 评估指标 |
| lr_policy.py | utils/ | 学习率策略 |
| pyt_utils.py | utils/ | 通用工具 |
| transforms.py | utils/ | 图像变换 |
| visualize.py | utils/ | 可视化工具 |

---

## 10. 注意事项

1. **数据格式**: 图像使用OpenCV读取，RGB图像会自动从BGR转换为RGB
2. **单通道X模态**: 会自动复制为3通道以匹配RGB格式
3. **分布式训练**: 需要设置`WORLD_SIZE`环境变量
4. **检查点恢复**: 使用`--continue_fpath`参数指定检查点路径
5. **类别忽略**: 背景类别(255)不参与损失计算

---

## 11. 论文引用

```bibtex
@inproceedings{zhang2024mrfs,
  title={MRFS: Mutually Reinforcing Image Fusion and Segmentation},
  author={Zhang, Hao and Zuo, Xuhui and Jiang, Jie and Guo, Chunchao and Ma, Jiayi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26974--26983},
  year={2024}
}
```
