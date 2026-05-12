# MRFS 项目复现全流程指南

## 第1步：环境配置

### 1.1 检查服务器环境

首先检查当前Python和CUDA版本：
```bash
python --version
nvcc --version
nvidia-smi
```

### 1.2 创建虚拟环境

#### 选项A：使用 conda（推荐）
```bash
# 创建Python 3.8环境
conda create -n mrfs python=3.8 -y
conda activate mrfs

# 安装PyTorch 1.8.1 + CUDA 11.1
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# 安装其他依赖
pip install timm==0.9.8 numpy==1.24.4 scipy==1.10.1 pillow==10.1.0 tqdm==4.66.1 tensorboardX==2.6.2.2 opencv-python==4.8.1.78
```

#### 选项B：使用 venv
```bash
# 创建虚拟环境
python3.8 -m venv mrfs_env
source mrfs_env/bin/activate  # Linux/Mac
# 或 mrfs_env\Scripts\activate  # Windows

# 安装依赖（同上）
```

### 1.3 验证环境安装
```bash
python -c "
import torch
import timm
import numpy
import cv2
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('All dependencies installed successfully!')
"
```

---

## 第2步：数据集准备

### 2.1 数据集下载

项目支持三个数据集，根据你的需求选择一个或多个：

#### FMB 数据集
- 下载链接：（请参考原论文或联系作者获取）
- 类别：15个（道路、人行道、建筑、灯、标志、植被、天空、人、车等）
- 包含：RGB可见光 + 红外图像 + 语义标签

#### MFNet 数据集
- 下载链接：（请参考原论文或联系作者获取）
- 类别：15个（与FMB类似）

#### PST900 数据集
- 下载链接：（请参考原论文或联系作者获取）
- 特点：RGB-thermal融合任务

### 2.2 数据集组织

在项目根目录创建 `dataset/` 文件夹：
```bash
mkdir -p dataset/FMB
mkdir -p dataset/MFNet
mkdir -p dataset/PST900
```

以FMB为例，按以下结构组织：
```
dataset/FMB/
├── Visible/              # 所有RGB图像
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── Infrared/             # 所有红外图像
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── Label/                # 所有语义标签
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── train.txt             # 训练集文件名列表（不含扩展名）
├── val.txt               # 验证集文件名列表
└── test.txt              # 测试集文件名列表
```

**注意**：图像文件名需保持一致（RGB、红外、标签同名）。

---

## 第3步：预训练模型准备

### 3.1 下载Segformer预训练权重

项目使用Segformer作为骨干网络，需要下载mit_b*预训练权重：

```bash
# 在项目根目录创建checkpoints/pretrained/
mkdir -p checkpoints/pretrained
```

然后从以下地址下载mit_b4.pth：
- Segformer官方仓库：https://github.com/NVlabs/SegFormer
- 或百度网盘/Google Drive搜索"Segformer mit_b4 pre-trained"

### 3.2 下载MRFS预训练权重（可选，用于直接测试）

如果你想直接测试而不重新训练，可以从以下链接下载预训练权重：
- FMB: https://drive.google.com/drive/folders/17LroKnuEWttvtcbuv-G-DXaXYdua52cN
- MFNet: https://drive.google.com/drive/folders/1txDn-U04KEKA6gbSUjSsn-QqaN4nFw0Y
- PST900: https://drive.google.com/drive/folders/1iEu3QZSV-q18u28X4cB7GK8UpLoRhXTX

下载后，将权重放到对应的目录下：
```bash
mkdir -p FMB/checkpoints/log_FMB_mit_b4/weights/
mv your_downloaded_fmb_weights.pth FMB/checkpoints/log_FMB_mit_b4/weights/
```

---

## 第4步：训练模型

### 4.1 单GPU训练

以FMB数据集为例：
```bash
cd FMB

# 编辑train.py中的参数（可选，根据需要修改）
# - 数据集路径：--dataset_path
# - 类别数：--num_classes
# - 图像尺寸：--image_height, --image_width
# - 预训练权重路径：--pretrained_backbone

# 开始训练
CUDA_VISIBLE_DEVICES=0 python train.py
```

### 4.2 多GPU分布式训练

```bash
# 使用2个GPU训练
cd FMB
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### 4.3 训练参数说明

关键参数（在[train.py](file:///workspace/FMB/train.py)中配置）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --dataset_path | ./dataset/FMB | 数据集路径 |
| --backbone | mit_b4 | 骨干网络 |
| --pretrained_backbone | ./checkpoints/pretrained/mit_b4.pth | 预训练权重路径 |
| --batch_size | 6 | 批次大小 |
| --lr | 6e-5 | 学习率 |
| --nepochs | 5000 | 训练轮数 |
| --checkpoint_start_epoch | 150 | 开始保存检查点 |
| --checkpoint_step | 5 | 检查点保存间隔 |

### 4.4 训练监控

训练过程中：
- TensorBoard日志保存在 `FMB/checkpoints/log_FMB_mit_b4/tb/`
- 检查点保存在 `FMB/checkpoints/log_FMB_mit_b4/weights/`
- 训练日志保存在 `FMB/checkpoints/log_FMB_mit_b4/train_*.log`

启动TensorBoard监控：
```bash
cd FMB
tensorboard --logdir=checkpoints/log_FMB_mit_b4/tb/
```

---

## 第5步：测试模型

### 5.1 测试预训练模型

```bash
cd FMB

# 使用epoch-xxx.pth测试
CUDA_VISIBLE_DEVICES=0 python eval.py -e=150

# 或使用作者提供的权重
CUDA_VISIBLE_DEVICES=0 python eval.py -e=MRFS
```

### 5.2 测试参数说明

关键参数（在[eval.py](file:///workspace/FMB/eval.py)中配置）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| -e/--epochs | MRFS | 要测试的epoch数或模型文件名 |
| --save_path | ./results/FMB | 结果保存路径 |
| --eval_crop_size | [600, 800] | 评估时的裁剪尺寸 |
| --eval_stride_rate | 0.666 | 滑动窗口步长率 |

### 5.3 可视化结果

测试完成后，分割结果会保存在 `./results/FMB/` 目录下，使用彩色标签可视化。

---

## 第6步：常见问题解决

### 6.1 CUDA内存不足

如果遇到CUDA OOM错误：
1. 减小 `--batch_size`
2. 使用更小的backbone：`--backbone mit_b2` 或 `mit_b0`
3. 减小图像尺寸：`--image_height 256 --image_width 320`

### 6.2 数据集路径错误

确保：
1. 数据集路径与 `--dataset_path` 一致
2. 三个文件夹（Visible/Infrared/Label）存在
3. train.txt/val.txt/test.txt 文件格式正确（每行一个文件名，无扩展名）

### 6.3 预训练权重加载失败

1. 检查 `--pretrained_backbone` 路径是否正确
2. 确保权重文件名与backbone对应
3. 如无法下载，可以修改代码不使用预训练权重

---

## 快速开始示例（完整流程）

```bash
# 1. 创建环境
conda create -n mrfs python=3.8 -y
conda activate mrfs
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.9.8 numpy==1.24.4 scipy==1.10.1 pillow==10.1.0 tqdm==4.66.1 tensorboardX==2.6.2.2 opencv-python==4.8.1.78

# 2. 准备数据集
mkdir -p dataset/FMB/Visible dataset/FMB/Infrared dataset/FMB/Label
# ... 复制你的数据集到对应文件夹 ...

# 3. 准备预训练权重
mkdir -p checkpoints/pretrained
# ... 下载mit_b4.pth到checkpoints/pretrained/ ...

# 4. 开始训练
cd FMB
CUDA_VISIBLE_DEVICES=0 python train.py

# 5. 测试（训练完成后）
CUDA_VISIBLE_DEVICES=0 python eval.py -e=best
```
