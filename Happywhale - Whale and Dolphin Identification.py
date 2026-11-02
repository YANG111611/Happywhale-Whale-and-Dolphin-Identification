# -*- coding: utf-8 -*-
"""
===================================================================================
项目名称: Happywhale - Whale and Dolphin Identification (鲸鱼与海豚个体身份识别)
代码功能: 基于深度度量学习 (ResNet34 + ArcFace) 的端到端鲸鱼海豚个体重识别方案
涵盖流程:
  1. 数据清洗与类别编码 (修正官方物种拼写错误、建立个体ID映射表)
  2. 数据增强与数据集构建 (使用 Albumentations 配合 PyTorch DataLoader)
  3. 模型架构搭建 (预训练 Backbone + 全局池化 + 特征投影层 + ArcMarginProduct)
  4. 混合精度训练 (AMP) 与 Cosine 退火学习率调度
  5. 特征向量提取 (L2 归一化 Embedding)
  6. 相似度检索 (Cosine Similarity kNN) 与未见个体 (new_individual) 阈值判定
  7. Kaggle 官方评价指标 MAP@5 验证与生成 submission.csv 预测文件
  8. 全方位数据与模型结果可视化 (物种分布、长尾特征、训练曲线、近邻检索比对、t-SNE聚类)
===================================================================================
"""

import os
import gc
import math
import time
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')  # 确保无 GUI 环境或后台运行时安全绘图
import matplotlib.pyplot as plt

# 配置中文字体，确保图像上的中文注释正常渲染
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =================================================================================
# 1. 全局配置类 (Configuration)
# =================================================================================
class CFG:
    """全局超参数及路径配置类"""
    # 路径配置: 兼容绝对路径与相对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "Happywhale - Whale and Dolphin Identification")
    train_csv = os.path.join(data_dir, "train.csv")
    sample_sub_csv = os.path.join(data_dir, "sample_submission.csv")
    train_img_dir = os.path.join(data_dir, "train_images")
    test_img_dir = os.path.join(data_dir, "test_images")
    output_dir = os.path.join(base_dir, "output")
    checkpoint_path = os.path.join(output_dir, "happywhale_resnet34_best.pth")
    submission_path = os.path.join(output_dir, "submission.csv")

    # 可视化图表保存路径
    vis_data_dist_path = os.path.join(output_dir, "data_distribution.png")
    vis_training_curves_path = os.path.join(output_dir, "training_curves.png")
    vis_retrieval_path = os.path.join(output_dir, "retrieval_visualization.png")
    vis_tsne_path = os.path.join(output_dir, "embeddings_tsne.png")

    # 随机种子
    seed = 42

    # 模型与特征维度
    model_name = "resnet34"      # 骨干网络名称 (轻量高效，适合 8GB 显存)
    pretrained = True            # 加载官方 ImageNet 预训练权重
    embedding_size = 512         # 特征投影空间维度 (Embedding Dim)
    
    # ArcFace 超参数
    arcface_s = 30.0             # 特征尺度因子 (Scale s)
    arcface_m = 0.50             # 角度边缘裕度 (Margin m)
    
    # 图像与数据加载参数
    image_size = 256             # 输入模型分辨率 (512 原图下采样至 256x256 兼顾速度与精度)
    batch_size = 32              # 批处理大小
    num_workers = 0              # Windows 平台推荐设为 0，防止多进程卡死
    
    # 训练优化参数
    epochs = 10                   # 训练轮次
    learning_rate = 3e-4         # 初始最大学习率
    min_lr = 1e-6                # 最小学习率
    weight_decay = 1e-4          # 权重衰减系数
    max_grad_norm = 10.0         # 梯度裁剪阈值
    use_amp = True               # 是否启用自动混合精度 (AMP) 加速训练
    
    # 开集识别 (new_individual) 检索阈值
    # 当最近邻的最大余弦相似度小于此阈值时，认为该图像属于训练集中未出现过的新个体
    new_individual_threshold = 0.55
    
    # 调试模式配置
    debug = False                # 是否开启 Debug 模式 (仅抽取少量样本快速跑通全流程)
    debug_train_samples = 600    # Debug 模式下的训练集样本量
    debug_test_samples = 150     # Debug 模式下的测试集样本量


# =================================================================================
# 2. 基础辅助工具函数 (Utilities)
# =================================================================================
def seed_everything(seed=42):
    """固定所有随机种子，保证实验结果的可复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Info] 随机种子已设置为: {seed}")


def clean_species_typos(df):
    """
    修正比赛官方 train.csv 中已知存在的物种名称拼写错误
    例如: 'bottlenose_dolpin' -> 'bottlenose_dolphin', 'kiler_whale' -> 'killer_whale'
    """
    corrections = {
        'bottlenose_dolpin': 'bottlenose_dolphin',
        'kiler_whale': 'killer_whale',
        'globis': 'short_finned_pilot_whale',
        'pilot_whale': 'short_finned_pilot_whale',
    }
    df['species'] = df['species'].replace(corrections)
    return df


# =================================================================================
# 3. 数据增强与数据集类 (Dataset & Transforms)
# =================================================================================
def get_train_transforms(image_size):
    """
    训练阶段数据增强流水线:
    包含尺寸重塑、水平翻转、平移缩放旋转、随机色彩对比度抖动与标准归一化
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_valid_transforms(image_size):
    """
    验证/测试阶段数据处理流水线:
    仅执行严格的尺寸调整与标准归一化，保持输入确定性
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


class HappywhaleDataset(Dataset):
    """
    Happywhale 自定义数据集类
    - 支持加载训练集 (有标签)、验证集以及无标签测试集
    - 使用 PIL 读取并转换为 RGB NumPy 矩阵，再送入 Albumentations 变换
    """
    def __init__(self, df, img_dir, transforms=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self.is_test = is_test
        self.image_names = self.df['image'].values
        
        # 若不是测试模式，则提取目标个体类别索引 (label)
        if not self.is_test:
            self.labels = self.df['label'].values

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 安全读取图像 (若个别图像读取异常则创建空白图兜底)
        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            print(f"[Warning] 图像读取失败: {img_path}, 错误: {e}")
            image = np.zeros((CFG.image_size, CFG.image_size, 3), dtype=np.uint8)

        # 应用数据增强
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        if self.is_test:
            return image, img_name
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label


# =================================================================================
# 4. 核心网络模块: ArcMarginProduct (ArcFace 边缘度量学习头)
# =================================================================================
class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) 核心实现
    论文参考: ArcFace: Additive Angular Margin Loss for Deep Face Recognition (CVPR 2019)
    原理说明:
      标准 Softmax 只考虑了欧氏距离可分，但在大规模细粒度识别/开集重识别中，
      ArcFace 将特征向量 x 与类别中心权重 W 均进行 L2 归一化，
      使得两者的点积等于夹角余弦值 cos(theta)。
      对于真实类别目标标签 y_i，在其夹角上增加固定的正角度裕度 m:
        cos(theta + m) = cos(theta) * cos(m) - sin(theta) * sin(m)
      从而强力约束同类别特征在超球面上高度紧凑，并显著推大异类之间的间距。
    """
    def __init__(self, in_features, out_classes, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_classes = out_classes
        self.s = s             # 半径缩放系数 (Scale)
        self.m = m             # 角度边缘间隔 (Margin)
        self.ls_eps = ls_eps   # 标签平滑系数
        
        # 初始化超球面类别权重矩阵 W
        self.weight = nn.Parameter(torch.FloatTensor(out_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # 预计算常数避免在 forward 中重复计算
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels=None):
        """
        参数:
          features: [Batch_Size, in_features] 输入特征向量
          labels:   [Batch_Size] 真实类别标签 (训练阶段需要，推理特征提取阶段可为 None)
        返回:
          输出 Logits: [Batch_Size, out_classes]
        """
        # 1. 分别对特征输入向量和类别权重进行 L2 归一化
        # cosine: [B, num_classes] 矩阵乘法计算各类别夹角余弦值
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        
        # 若未提供标签 (如纯推理模式提取分类分数)，直接返回缩放后的余弦得分
        if labels is None:
            return cosine * self.s

        # 2. 计算 sin(theta) = sqrt(1 - cos^2(theta))，添加 clamp 保证数值稳定性
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0.0, 1.0))

        # 3. 利用三角恒等变换展开: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # 4. 处理极端角度范围 (当 theta + m > pi 时保持单调性)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 5. 构建 one-hot 掩码，仅对样本对应的真实类别应用角度裕度 phi，其余负样本保持原始 cosine
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # 6. 标签平滑处理 (可选)
        if self.ls_eps > 0:
            one_hot = (1.0 - self.ls_eps) * one_hot + self.ls_eps / self.out_classes

        # 7. 合并正负类别分数并乘以尺度因子 s
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# =================================================================================
# 5. 整体识别网络模型 (HappywhaleModel)
# =================================================================================
class HappywhaleModel(nn.Module):
    """
    鲸鱼与海豚个体识别完整网络架构:
      Backbone (ResNet34) -> AdaptiveAvgPool2d -> Linear(Embedding) -> BatchNorm1d -> ArcMarginProduct
    """
    def __init__(self, num_classes, model_name="resnet34", embedding_size=512, pretrained=True, s=30.0, m=0.5):
        super(HappywhaleModel, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size

        # 加载 ResNet34 骨干网络并剥离原始全连接层
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet34(weights=weights)
        
        # 提取卷积特征抽取器
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1])  # 输出 shape: [B, 512, 1, 1]
        backbone_dim = base_resnet.fc.in_features                          # 512 维

        # 特征降维/投影瓶颈层 (Bottleneck Projection Layer)
        self.embedding_layer = nn.Linear(backbone_dim, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

        # ArcFace 角度度量分类头
        self.arcface_head = ArcMarginProduct(
            in_features=embedding_size,
            out_classes=num_classes,
            s=s,
            m=m
        )

    def extract_features(self, x):
        """
        提取 L2 归一化后的个体特征向量 (用于测试检索与余弦相似度计算)
        输入: [B, 3, H, W]
        输出: [B, embedding_size] (模长为 1 的单位向量)
        """
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        features = self.embedding_layer(features)
        features = self.bn(features)
        normalized_features = F.normalize(features, p=2, dim=1)
        return normalized_features

    def forward(self, x, labels=None):
        """
        前向传播:
        若处于训练阶段且提供 labels，则输出经过 ArcFace 惩罚的分类 logits
        若处于纯特征提取或无标签模式，则输出 L2 归一化特征
        """
        embeddings = self.extract_features(x)
        if labels is not None:
            logits = self.arcface_head(embeddings, labels)
            return logits
        return embeddings


# =================================================================================
# 6. Kaggle 官方评估指标: MAP@5 (Mean Average Precision @ 5)
# =================================================================================
def compute_map_per_image(actual_label, predicted_ids, k=5):
    """
    单张图像的 Average Precision @ 5 计算:
    在预测的前 k 个个体 ID 中，若真实标签命中于第 rank 位 (1-based)，得分为 1 / rank；
    若前 5 个均未命中，则得分为 0.0。
    """
    top_k_preds = predicted_ids[:k]
    if actual_label in top_k_preds:
        rank = top_k_preds.index(actual_label) + 1
        return 1.0 / rank
    return 0.0


def compute_map5(actual_labels, predictions_list):
    """
    全体验证集的 Mean Average Precision @ 5 (MAP@5)
    """
    scores = [
        compute_map_per_image(act, preds, k=5)
        for act, preds in zip(actual_labels, predictions_list)
    ]
    return float(np.mean(scores))


# =================================================================================
# 7. 模型训练与特征提取 (Training & Embedding Extraction)
# =================================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, scheduler, device, epoch):
    """单轮模型混合精度训练循环"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        optimizer.zero_grad()

        # 启用自动混合精度 (AMP) 加快训练并节省显存
        use_cuda_amp = (CFG.use_amp and device.type == 'cuda')
        amp_ctx = torch.amp.autocast('cuda', enabled=use_cuda_amp) if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(enabled=use_cuda_amp)
        
        with amp_ctx:
            logits = model(images, labels)
            loss = criterion(logits, labels)

        if use_cuda_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            optimizer.step()

        # 统计指标
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct / total:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })

    # 更新余弦退火学习率
    if scheduler is not None:
        scheduler.step()

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def extract_all_embeddings(model, dataloader, device):
    """
    遍历数据加载器，提取所有样本的 L2 归一化 Embedding 特征
    返回:
      embeddings: [N, embedding_size] NumPy 数组
    """
    model.eval()
    embeddings_list = []
    
    pbar = tqdm(dataloader, desc="Extracting Embeddings", leave=False)
    for batch in pbar:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        images = images.to(device, non_blocking=True)
        
        # 提取特征
        embs = model.extract_features(images)
        embeddings_list.append(embs.cpu().numpy())

    return np.vstack(embeddings_list)


# =================================================================================
# 8. 最近邻检索与未见个体 (new_individual) 预测算法
# =================================================================================
def predict_with_knn(query_embeddings, gallery_embeddings, gallery_individual_ids,
                     threshold=0.55, top_k=5, chunk_size=2000, return_details=False):
    """
    基于余弦相似度的分批近邻匹配与未见个体 (new_individual) 自适应插入算法
    
    参数说明:
      query_embeddings:       [N_query, Dim] 查询集特征 (测试集或验证集)
      gallery_embeddings:     [N_gallery, Dim] 底库图集特征 (训练集)
      gallery_individual_ids: [N_gallery] 底库对应的真实个体 ID 列表
      threshold:              相似度阈值 (若最近邻最高相似度低于此值，将 new_individual 排首位)
      top_k:                  最终每个样本保留的推荐个体数 (默认为 5)
      chunk_size:             显存/内存分块大小，防止大矩阵相乘爆显存
      return_details:         是否返回最相似样本的索引和相似度，用于检索可视化

    返回:
      predictions: list of list of str, 每个查询样本对应的 top-5 预测个体 ID
      details (可选): 包含 topk 匹配索引和相似度分数的列表
    """
    query_tensor = torch.from_numpy(query_embeddings).float()
    gallery_tensor = torch.from_numpy(gallery_embeddings).float().t()  # [Dim, N_gallery]

    num_queries = query_tensor.size(0)
    all_predictions = []
    all_details = []

    # 统计训练库中最常见的个体 (用于候选数量不足 5 个时兜底补充)
    popular_ids = list(pd.Series(gallery_individual_ids).value_counts().index[:10])

    print(f"[Info] 开始执行近邻相似度匹配 (查询集: {num_queries}, 底库: {gallery_tensor.size(1)})...")

    for start_idx in range(0, num_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, num_queries)
        q_chunk = query_tensor[start_idx:end_idx]

        # 计算批量余弦相似度矩阵: [chunk_size, N_gallery]
        sim_matrix = torch.mm(q_chunk, gallery_tensor)

        # 检索前 50 个最相似的底库样本索引与余弦得分
        topk_scores, topk_indices = torch.topk(sim_matrix, k=min(50, gallery_tensor.size(1)), dim=1)
        topk_scores = topk_scores.numpy()
        topk_indices = topk_indices.numpy()

        for row_scores, row_indices in zip(topk_scores, topk_indices):
            # 获取对应的个体 ID 列表
            candidate_ids = [gallery_individual_ids[i] for i in row_indices]

            # 去重并保持相似度排序顺序
            seen = set()
            unique_candidates = []
            for cid in candidate_ids:
                if cid not in seen:
                    seen.add(cid)
                    unique_candidates.append(cid)

            # 核心策略: 判断是否判定为 new_individual (新个体)
            max_sim = row_scores[0] if len(row_scores) > 0 else 0.0
            
            # 若最高相似度低于设定阈值，说明与训练集库中所有个体都不相似，将 new_individual 设为首选
            if max_sim < threshold:
                preds = ['new_individual'] + [cid for cid in unique_candidates if cid != 'new_individual']
            else:
                # 否则置于第 2 位或第 3 位 (应对开集分类的经典 Kaggle 方案)
                preds = []
                inserted_new = False
                for i, cid in enumerate(unique_candidates):
                    # 当相似度出现明显断崖式下跌或遍历到第 2 个位置时插入 new_individual
                    if (not inserted_new) and (i == 1 or (i < len(row_scores) and row_scores[i] < threshold)):
                        preds.append('new_individual')
                        inserted_new = True
                    preds.append(cid)
                if not inserted_new:
                    preds.append('new_individual')

            # 若候选仍不足 5 个，使用常见个体填充兜底
            for pop_id in popular_ids:
                if len(preds) >= top_k:
                    break
                if pop_id not in preds:
                    preds.append(pop_id)

            all_predictions.append(preds[:top_k])
            if return_details:
                all_details.append({
                    'top_indices': row_indices[:top_k],
                    'top_scores': row_scores[:top_k],
                    'top_candidates': candidate_ids[:top_k]
                })

    if return_details:
        return all_predictions, all_details
    return all_predictions


# =================================================================================
# 9. 数据与模型结果全方位可视化模块 (Visualization Module)
# =================================================================================
def visualize_dataset_distribution(train_df, output_path):
    """
    可视化 1: 训练集物种与个体数量长尾分布
    生成包含:
      1. Top 15 物种样本数水平柱状图
      2. 每个个体的样本数长尾分布饼图/分布区间柱状图
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 子图 1: 物种出现频率 Top 15
    species_counts = train_df['species'].value_counts().head(15)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(species_counts)))
    bars = axes[0].barh(species_counts.index[::-1], species_counts.values[::-1], color=colors[::-1])
    axes[0].set_title("训练集前 15 种物种样本频次分布 (Top 15 Species)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("样本数量 (Counts)", fontsize=11)
    axes[0].grid(axis='x', linestyle='--', alpha=0.6)
    
    # 标注数值
    for bar in bars:
        width = bar.get_width()
        axes[0].text(width + 50, bar.get_y() + bar.get_height()/2, f"{int(width):,}",
                     va='center', fontsize=9, color='#333333')

    # 子图 2: 鲸豚个体样本数的极端长尾分布 (Long-Tail Distribution)
    individual_counts = train_df['individual_id'].value_counts()
    c1 = (individual_counts == 1).sum()
    c2 = (individual_counts == 2).sum()
    c3_5 = ((individual_counts >= 3) & (individual_counts <= 5)).sum()
    c6_10 = ((individual_counts >= 6) & (individual_counts <= 10)).sum()
    c_gt10 = (individual_counts > 10).sum()

    categories = ['仅 1 张图片\n(单样本个体)', '2 张图片', '3 ~ 5 张', '6 ~ 10 张', '> 10 张图片']
    counts = [c1, c2, c3_5, c6_10, c_gt10]
    palette = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']

    axes[1].bar(categories, counts, color=palette, edgecolor='black', alpha=0.85)
    axes[1].set_title("个体样本拥有量长尾分布 (体现开集识别/小样本挑战)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("个体类别数量 (Number of Individuals)", fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    for i, count in enumerate(counts):
        pct = (count / len(individual_counts)) * 100
        axes[1].text(i, count + max(counts)*0.02, f"{count:,}\n({pct:.1f}%)",
                     ha='center', fontsize=10, fontweight='bold')

    plt.suptitle("Happywhale 数据集全景统计与长尾分布分析", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] 数据分布可视化已保存至: {output_path}")


def visualize_training_curves(history, output_path):
    """
    可视化 2: 模型训练 Loss 与 Accuracy 收敛曲线
    """
    if not history or 'loss' not in history or len(history['loss']) == 0:
        return

    epochs = range(1, len(history['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    ax1.plot(epochs, history['loss'], marker='o', color='#d35400', linewidth=2.2, label='Train ArcFace Loss')
    ax1.set_title("训练损失收敛曲线 (ArcFace Margin Loss)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("轮次 (Epoch)", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')

    # 准确率曲线
    ax2.plot(epochs, [acc * 100 for acc in history['acc']], marker='s', color='#27ae60', linewidth=2.2, label='Train Accuracy (%)')
    ax2.set_title("训练分类准确率曲线 (Training Accuracy)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("轮次 (Epoch)", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right')

    plt.suptitle("Happywhale ResNet34 + ArcFace 训练历程", fontsize=15, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] 训练曲线可视化已保存至: {output_path}")


def visualize_retrieval_predictions(query_df, query_img_dir, gallery_df, gallery_img_dir,
                                    details_list, predictions_list, output_path, num_samples=3):
    """
    可视化 3: 图像检索比对图 (Query 图像与 Top-5 检索结果及余弦相似度)
    每行展示一个查询样本及其对应的 Top-5 候选检索结果
    若底库匹配与查询图像个体一致，使用绿色高亮边框；若不一致显示红框或灰框；包含 new_individual 说明
    """
    num_samples = min(num_samples, len(query_df))
    if num_samples == 0:
        return

    fig, axes = plt.subplots(num_samples, 6, figsize=(20, 3.8 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(num_samples):
        q_row = query_df.iloc[row]
        q_name = q_row['image']
        q_label = q_row.get('individual_id', 'Unknown')
        q_species = q_row.get('species', '')
        q_path = os.path.join(query_img_dir, q_name)

        # 显示 Query 查询图像
        try:
            q_img = Image.open(q_path).convert("RGB")
            axes[row, 0].imshow(q_img)
        except Exception:
            axes[row, 0].imshow(np.zeros((100, 100, 3), dtype=np.uint8))
        axes[row, 0].set_title(f"[查询图 Query #{row+1}]\n真实个体: {q_label[:8]}...\n({q_species})",
                                fontsize=10, fontweight='bold', color='#2980b9')
        axes[row, 0].axis('off')

        # 显示 Top-5 检索结果
        preds = predictions_list[row]
        details = details_list[row]
        top_indices = details['top_indices']
        top_scores = details['top_scores']

        for col in range(5):
            ax = axes[row, col + 1]
            if col < len(preds):
                pred_id = preds[col]
                if pred_id == 'new_individual':
                    # new_individual 使用醒目的特色提示卡片
                    placeholder = np.ones((120, 120, 3), dtype=np.uint8) * 240
                    ax.imshow(placeholder)
                    ax.text(60, 60, "新个体\nnew_individual\n(相似度低于阈值)",
                            ha='center', va='center', fontsize=9, fontweight='bold', color='#8e44ad')
                    ax.set_title(f"Rank {col+1}: new_individual", fontsize=10, color='#8e44ad')
                else:
                    # 找到对应的 gallery 图像
                    match_idx = top_indices[col] if col < len(top_indices) else 0
                    score = top_scores[col] if col < len(top_scores) else 0.0
                    g_name = gallery_df.iloc[match_idx]['image']
                    g_label = gallery_df.iloc[match_idx]['individual_id']
                    g_path = os.path.join(gallery_img_dir, g_name)
                    try:
                        g_img = Image.open(g_path).convert("RGB")
                        ax.imshow(g_img)
                    except Exception:
                        ax.imshow(np.zeros((100, 100, 3), dtype=np.uint8))

                    # 判断是否命中真实个体
                    is_correct = (g_label == q_label)
                    border_color = '#27ae60' if is_correct else '#7f8c8d'
                    tag = "✓ 命中" if is_correct else "候选"

                    for spine in ax.spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(3 if is_correct else 1)

                    ax.set_title(f"Rank {col+1}: {tag}\nSim: {score:.3f} | {g_label[:8]}...",
                                 fontsize=9, color=border_color, fontweight='bold')
            ax.axis('off')

    plt.suptitle("鲸豚个体身份重识别 (kNN Cosine Retrieval & Rank 1~5 Matches)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] 检索比对可视化已保存至: {output_path}")


def visualize_embeddings_pca(embeddings, individual_ids, output_path, max_samples=400):
    """
    可视化 4: ArcFace 学习到的超球面特征空间降维投影 (PCA 2D 散点图)
    展示同类别个体在特征空间的聚拢与异类别个体的区分度
    """
    if len(embeddings) == 0:
        return

    # 选取样本数最多的前 8 个个体进行重点着色可视化
    df_temp = pd.DataFrame({'ind': individual_ids})
    top_inds = df_temp['ind'].value_counts().head(8).index.tolist()

    mask = df_temp['ind'].isin(top_inds)
    selected_indices = np.where(mask)[0]
    if len(selected_indices) < 20:
        # 若样本过少，则选取前 max_samples 个样本
        selected_indices = np.arange(min(len(embeddings), max_samples))

    sub_embs = embeddings[selected_indices]
    sub_labels = [individual_ids[i] for i in selected_indices]

    # 执行 PCA 降维到 2D
    pca = PCA(n_components=2, random_state=CFG.seed)
    embs_2d = pca.fit_transform(sub_embs)

    fig, ax = plt.subplots(figsize=(10, 7))
    unique_selected = sorted(list(set(sub_labels)))
    cmap = plt.colormaps['tab10']

    for idx, ind in enumerate(unique_selected):
        ind_mask = [l == ind for l in sub_labels]
        ax.scatter(
            embs_2d[ind_mask, 0],
            embs_2d[ind_mask, 1],
            label=f"个体: {ind[:8]}...",
            s=45,
            alpha=0.85,
            edgecolors='white',
            linewidths=0.5
        )

    ax.set_title("ArcFace 潜空间个体特征嵌入降维分布 (PCA 2D Projection)", fontsize=13, fontweight='bold')
    ax.set_xlabel(f"主成分 1 (解释方差比: {pca.explained_variance_ratio_[0]*10:.1f}%)", fontsize=11)
    ax.set_ylabel(f"主成分 2 (解释方差比: {pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] 特征空间 PCA 投影可视化已保存至: {output_path}")


# =================================================================================
# 10. 完整流水线执行 (Pipeline Execution)
# =================================================================================
def run_pipeline(args):
    """
    运行 Happywhale 完整项目全流程
    """
    seed_everything(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] 当前运算设备: {device} "
          f"{'(' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")

    os.makedirs(CFG.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 步骤 A: 加载原始 CSV 并执行数据清洗
    # -------------------------------------------------------------------------
    print("\n--- 步骤 1/5: 加载与清洗数据 ---")
    if not os.path.exists(CFG.train_csv):
        raise FileNotFoundError(f"找不到训练标注文件: {CFG.train_csv}")
    
    train_df = pd.read_csv(CFG.train_csv)
    train_df = clean_species_typos(train_df)
    print(f"原始训练集样本数: {len(train_df)}, 独特物种数: {train_df['species'].nunique()}")

    # 生成数据探索与长尾分布可视化图表
    if args.visualize:
        visualize_dataset_distribution(train_df, CFG.vis_data_dist_path)

    # 建立类别编码字典: individual_id -> label (0 ~ num_classes-1)
    unique_individuals = sorted(train_df['individual_id'].unique())
    num_classes = len(unique_individuals)
    individual_to_label = {ind: idx for idx, ind in enumerate(unique_individuals)}
    label_to_individual = {idx: ind for idx, ind in enumerate(unique_individuals)}
    train_df['label'] = train_df['individual_id'].map(individual_to_label)
    print(f"训练集独特个体总数 (总类别数): {num_classes}")

    # 若开启调试模式，抽取部分数据以快速验证流程
    if args.debug or CFG.debug:
        print(f"[Debug 模式] 截取前 {CFG.debug_train_samples} 条训练数据进行快速验证...")
        train_df = train_df.iloc[:CFG.debug_train_samples].copy()
        # 重新映射调试样本的局部类别
        debug_individuals = sorted(train_df['individual_id'].unique())
        num_classes = len(debug_individuals)
        individual_to_label = {ind: idx for idx, ind in enumerate(debug_individuals)}
        label_to_individual = {idx: ind for idx, ind in enumerate(debug_individuals)}
        train_df['label'] = train_df['individual_id'].map(individual_to_label)
        print(f"[Debug 模式] 调试类别数: {num_classes}")

    # -------------------------------------------------------------------------
    # 步骤 B: 划分训练集与验证集 (分层策略处理单样本与多样本)
    # -------------------------------------------------------------------------
    id_counts = train_df['individual_id'].value_counts()
    singletons = id_counts[id_counts == 1].index.tolist()
    multi_samples = id_counts[id_counts > 1].index.tolist()

    # 单样本个体直接放入训练集，多样本个体按 8:2 分层划分
    multi_df = train_df[train_df['individual_id'].isin(multi_samples)].copy()
    singleton_df = train_df[train_df['individual_id'].isin(singletons)].copy()

    if len(multi_df) > 0:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
        train_idx, val_idx = next(skf.split(multi_df, multi_df['individual_id']))
        train_split = pd.concat([multi_df.iloc[train_idx], singleton_df], axis=0).reset_index(drop=True)
        val_split = multi_df.iloc[val_idx].reset_index(drop=True)
    else:
        train_split = train_df.copy()
        val_split = train_df.iloc[:min(20, len(train_df))].copy()

    print(f"训练集划分样本数: {len(train_split)}, 验证集划分样本数: {len(val_split)}")

    # -------------------------------------------------------------------------
    # 步骤 C: 构建 Dataset 与 DataLoader
    # -------------------------------------------------------------------------
    print("\n--- 步骤 2/5: 构建数据加载器 ---")
    train_dataset = HappywhaleDataset(
        train_split,
        CFG.train_img_dir,
        transforms=get_train_transforms(CFG.image_size),
        is_test=False
    )
    val_dataset = HappywhaleDataset(
        val_split,
        CFG.train_img_dir,
        transforms=get_valid_transforms(CFG.image_size),
        is_test=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True if len(train_dataset) > args.batch_size else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # -------------------------------------------------------------------------
    # 步骤 D: 初始化 ResNet34 + ArcFace 模型
    # -------------------------------------------------------------------------
    print("\n--- 步骤 3/5: 构建网络模型与优化器 ---")
    model = HappywhaleModel(
        num_classes=num_classes,
        model_name=CFG.model_name,
        embedding_size=CFG.embedding_size,
        pretrained=CFG.pretrained,
        s=CFG.arcface_s,
        m=CFG.arcface_m
    ).to(device)

    # 若已存在预训练权重且用户未指定纯重新训练，可选择恢复
    if os.path.exists(CFG.checkpoint_path) and not args.force_retrain and args.mode in ['eval', 'infer']:
        print(f"[Info] 加载已存在的检查点权重: {CFG.checkpoint_path}")
        checkpoint = torch.load(CFG.checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=CFG.min_lr
    )
    use_cuda_amp = (CFG.use_amp and device.type == 'cuda')
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_cuda_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    # -------------------------------------------------------------------------
    # 步骤 E: 执行模型训练
    # -------------------------------------------------------------------------
    training_history = {'loss': [], 'acc': []}
    if args.mode in ['all', 'train']:
        print(f"\n--- 步骤 4/5: 开始模型训练 (共 {args.epochs} 轮) ---")
        best_acc = 0.0
        start_time = time.time()

        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                device=device,
                epoch=epoch
            )
            elapsed = time.time() - t0
            training_history['loss'].append(train_loss)
            training_history['acc'].append(train_acc)

            print(f"[Epoch {epoch+1:02d}/{args.epochs:02d}] "
                  f"耗时: {elapsed:.1f}s | 损失 Loss: {train_loss:.4f} | 准确率 Acc: {train_acc*100:.2f}%")

            # 保存最优模型检查点
            if train_acc >= best_acc:
                best_acc = train_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'num_classes': num_classes,
                    'best_acc': best_acc,
                    'individual_to_label': individual_to_label,
                    'label_to_individual': label_to_individual
                }, CFG.checkpoint_path)
                print(f"  --> 已保存当前最优模型检查点至: {CFG.checkpoint_path}")

        print(f"[Info] 训练完成，总耗时: {(time.time() - start_time) / 60:.2f} 分钟")

        # 绘制训练收敛曲线
        if args.visualize:
            visualize_training_curves(training_history, CFG.vis_training_curves_path)

    # -------------------------------------------------------------------------
    # 步骤 F: 验证集特征检索、MAP@5 评估与检索可视化
    # -------------------------------------------------------------------------
    if args.mode in ['all', 'eval']:
        print("\n--- 步骤 5A: 验证集特征提取与 MAP@5 评估 ---")
        # 提取训练集与验证集特征
        full_train_dataset = HappywhaleDataset(
            train_split,
            CFG.train_img_dir,
            transforms=get_valid_transforms(CFG.image_size),
            is_test=False
        )
        full_train_loader = DataLoader(
            full_train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers
        )

        train_embs = extract_all_embeddings(model, full_train_loader, device)
        val_embs = extract_all_embeddings(model, val_loader, device)

        # 相似度近邻检索匹配 (获取详细匹配信息用于可视化)
        val_predictions, val_details = predict_with_knn(
            query_embeddings=val_embs,
            gallery_embeddings=train_embs,
            gallery_individual_ids=train_split['individual_id'].values,
            threshold=CFG.new_individual_threshold,
            top_k=5,
            return_details=True
        )

        # 计算并展示 MAP@5 得分
        map5_score = compute_map5(val_split['individual_id'].values, val_predictions)
        print(f"\n=======================================================")
        print(f" [评估结果] 验证集 MAP@5 得分: {map5_score:.5f}")
        print(f"=======================================================\n")

        # 绘制特征空间 PCA 投影与检索结果匹配比对图
        if args.visualize:
            visualize_retrieval_predictions(
                query_df=val_split,
                query_img_dir=CFG.train_img_dir,
                gallery_df=train_split,
                gallery_img_dir=CFG.train_img_dir,
                details_list=val_details,
                predictions_list=val_predictions,
                output_path=CFG.vis_retrieval_path,
                num_samples=3
            )
            visualize_embeddings_pca(
                embeddings=val_embs,
                individual_ids=val_split['individual_id'].values,
                output_path=CFG.vis_tsne_path,
                max_samples=400
            )

    # -------------------------------------------------------------------------
    # 步骤 G: 测试集推理与生成 submission.csv 提交文件
    # -------------------------------------------------------------------------
    if args.mode in ['all', 'infer']:
        print("\n--- 步骤 5B: 测试集特征提取与生成预测文件 ---")
        if not os.path.exists(CFG.sample_sub_csv):
            print(f"[Warning] 找不到 sample_submission.csv: {CFG.sample_sub_csv}，跳过测试集推理")
            return

        sub_df = pd.read_csv(CFG.sample_sub_csv)
        if args.debug or CFG.debug:
            sub_df = sub_df.iloc[:CFG.debug_test_samples].copy()
            print(f"[Debug 模式] 截取前 {CFG.debug_test_samples} 个测试样本生成预测文件...")

        test_dataset = HappywhaleDataset(
            sub_df,
            CFG.test_img_dir,
            transforms=get_valid_transforms(CFG.image_size),
            is_test=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers
        )

        # 若此前尚未提取训练集特征，在此补提
        if 'train_embs' not in locals():
            full_train_dataset = HappywhaleDataset(
                train_split,
                CFG.train_img_dir,
                transforms=get_valid_transforms(CFG.image_size),
                is_test=False
            )
            full_train_loader = DataLoader(
                full_train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=CFG.num_workers
            )
            train_embs = extract_all_embeddings(model, full_train_loader, device)

        # 提取测试集特征
        test_embs = extract_all_embeddings(model, test_loader, device)

        # 执行近邻预测匹配
        test_predictions = predict_with_knn(
            query_embeddings=test_embs,
            gallery_embeddings=train_embs,
            gallery_individual_ids=train_split['individual_id'].values,
            threshold=CFG.new_individual_threshold,
            top_k=5
        )

        # 拼接为空格分隔格式 (例如: "id1 id2 new_individual id3 id4")
        formatted_preds = [" ".join(p) for p in test_predictions]
        sub_df['predictions'] = formatted_preds

        # 保存为符合比赛要求的 submission.csv
        sub_df.to_csv(CFG.submission_path, index=False)
        print(f"[Info] 预测文件已成功保存至: {CFG.submission_path}")
        print("预测结果前 5 行预览:")
        print(sub_df.head(5))


# =================================================================================
# 11. 命令行主入口 (Main CLI Interface)
# =================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Happywhale - Whale and Dolphin Identification 训练与推理脚本")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "train", "eval", "infer"],
        help="运行模式: all (全流程), train (仅训练), eval (仅评估), infer (仅推理)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CFG.epochs,
        help="模型训练轮数 (Epochs)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=CFG.batch_size,
        help="数据加载批次大小 (Batch Size)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=CFG.learning_rate,
        help="模型初始学习率 (Learning Rate)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="是否启用 Debug 模式 (抽取微量样本极速走通全流程)"
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="是否忽略现有 checkpoint 强制重新训练"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="是否自动生成数据分布、训练曲线、近邻检索与特征聚类可视化图表"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
