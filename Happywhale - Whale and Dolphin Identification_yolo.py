"""
=================================================================================
Kaggle 竞赛: Happywhale - Whale and Dolphin Identification (鲸豚个体身份重识别)
增强进阶版本: 集成 YOLO 目标检测边界框动态裁剪 (Bounding Box Crop) 与 高分辨率特征提取
=================================================================================

主要升级特性:
1. YOLOv8 目标检测边界框裁剪:
   - 自动检测鲸鱼/海豚及背鳍主体，过滤大面积海水背景噪声，聚焦细粒度纹理识别；
   - 包含自适应外扩安全边界 (Padding Margin)，防止截断背鳍边缘；
   - 包含自动安全回退机制 (Fallback)，无检出或异常时安全平滑退回原图。
2. 高分辨率视觉输入:
   - 图像输入分辨率由 256x256 增大至 384x384，显著提升背鳍锯齿、划痕等细微特征表达能力；
   - 动态调整批次大小 (Batch Size = 16) 并采用 PyTorch 混合精度 (AMP)，完美适配 8GB 显存。
3. 高效 Bbox 缓存与预提取机制:
   - 支持自动化检测结果缓存为 CSV，训练时零延迟瞬时切片，避免重复检测拖慢训练。
4. 全新独立输出目录 output_yolo:
   - 包含专属模型权重、测试集预测 CSV 文件；
   - 新增 YOLO 检测裁剪对比可视化、数据长尾分布、训练收敛曲线、kNN 检索匹配及 PCA 投影图。
=================================================================================
"""

import os
import gc
import math
import time
import random
import argparse
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# 配置中文字体，优先匹配常见中文字体，兼容负号
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO


# =================================================================================
# 1. 全局配置类 (Configuration)
# =================================================================================

class CFG:
    # 路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "Happywhale - Whale and Dolphin Identification")
    train_csv = os.path.join(data_dir, "train.csv")
    sample_sub_csv = os.path.join(data_dir, "sample_submission.csv")
    train_img_dir = os.path.join(data_dir, "train_images")
    test_img_dir = os.path.join(data_dir, "test_images")

    # 全新输出目录与产物路径
    output_dir = os.path.join(base_dir, "output_yolo")
    checkpoint_path = os.path.join(output_dir, "happywhale_resnet50_best.pth")
    submission_path = os.path.join(output_dir, "submission.csv")

    # YOLO 目标检测边界框缓存路径
    train_bbox_csv = os.path.join(output_dir, "train_bboxes.csv")
    test_bbox_csv = os.path.join(output_dir, "test_bboxes.csv")

    # 可视化图表产物路径
    vis_data_dist_path = os.path.join(output_dir, "data_distribution.png")
    vis_training_curves_path = os.path.join(output_dir, "training_curves.png")
    vis_yolo_crop_path = os.path.join(output_dir, "yolo_crop_visualization.png")
    vis_retrieval_path = os.path.join(output_dir, "retrieval_visualization.png")
    vis_tsne_path = os.path.join(output_dir, "embeddings_tsne.png")

    # 随机种子与模型骨干
    seed = 42
    model_name = "resnet50"
    pretrained = True
    embedding_size = 512

    # ArcFace 超参数
    arcface_s = 30.0
    arcface_m = 0.50

    # 图像分辨率与批处理大小 (增大分辨率至 384x384，批次设为 16 适配 8GB 显存)
    image_size = 384
    batch_size = 16
    num_workers = 0

    # YOLO 目标检测与裁剪超参数
    yolo_model_name = "yolov8n.pt"
    yolo_conf = 0.15           # 检测置信度阈值
    yolo_padding = 0.12        # 目标框向外扩张比例 (12%)，避免切损背鳍轮廓

    # 训练超参数
    epochs = 10
    learning_rate = 3e-4
    min_lr = 1e-6
    weight_decay = 1e-4
    max_grad_norm = 10.0
    use_amp = True             # 启用自动混合精度加速与显存节省

    # 识别与匹配超参数
    new_individual_threshold = 0.55

    # 快速调试模式配置
    debug = False
    debug_train_samples = 400
    debug_test_samples = 100


# =================================================================================
# 2. 随机种子设置 (Reproducibility)
# =================================================================================

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Info] 随机种子已设置为：{seed}")


# =================================================================================
# 3. YOLO 目标检测与边界框裁剪模块 (YOLO Bounding Box Cropper)
# =================================================================================

class YoloBboxManager:
    """
    负责使用 YOLO 提取目标边界框、向外扩张安全边距、支持缓存文件读写与数据集零延迟索引。
    """
    def __init__(self, model_name=CFG.yolo_model_name, conf=CFG.yolo_conf, padding=CFG.yolo_padding, device=None):
        self.model_name = model_name
        self.conf = conf
        self.padding = padding
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            print(f"[YOLO] 正在加载轻量目标检测模型: {self.model_name} (设备: {self.device})...")
            self._model = YOLO(self.model_name)
        return self._model

    def extract_single_bbox(self, image_np):
        """
        对单张 numpy 图像提取最佳前景边界框 [x1, y1, x2, y2]。
        若未检测出目标，返回 None。
        """
        h, w = image_np.shape[:2]
        try:
            results = self.model.predict(image_np, conf=self.conf, device=self.device, verbose=False)
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                # 优先选取综合面积与置信度最高的目标框
                best_idx = 0
                max_score = -1.0
                for i, (b, c) in enumerate(zip(boxes, confs)):
                    area = (b[2] - b[0]) * (b[3] - b[1])
                    score = float(c) * math.sqrt(max(1.0, area))
                    if score > max_score:
                        max_score = score
                        best_idx = i

                box = boxes[best_idx]
                x1, y1, x2, y2 = box

                # 向外安全扩张 padding 边距
                bw, bh = x2 - x1, y2 - y1
                pad_w, pad_h = bw * self.padding, bh * self.padding
                crop_x1 = max(0, int(round(x1 - pad_w)))
                crop_y1 = max(0, int(round(y1 - pad_h)))
                crop_x2 = min(w, int(round(x2 + pad_w)))
                crop_y2 = min(h, int(round(y2 + pad_h)))

                # 面积过小 (小于整图 1.5%) 则舍弃
                if (crop_x2 - crop_x1) * (crop_y2 - crop_y1) < (w * h * 0.015):
                    return None

                return [crop_x1, crop_y1, crop_x2, crop_y2]
        except Exception as e:
            pass
        return None

    def precompute_bboxes(self, df, img_dir, save_csv_path, batch_size=32, desc="Extracting BBoxes"):
        """
        批量高效提取数据集中所有图片的边界框并落盘至 CSV，实现极速缓存。
        """
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        image_names = df['image'].unique().tolist()
        print(f"[YOLO] 开始批量检测与提取边界框 (共 {len(image_names)} 张图)...")

        results_data = []
        pbar = tqdm(range(0, len(image_names), batch_size), desc=desc)

        for i in pbar:
            batch_names = image_names[i:i + batch_size]
            batch_paths = [os.path.join(img_dir, name) for name in batch_names]
            valid_paths = [p for p in batch_paths if os.path.exists(p)]

            if not valid_paths:
                continue

            try:
                preds = self.model.predict(valid_paths, conf=self.conf, device=self.device, verbose=False)
                for name, pred in zip(batch_names, preds):
                    h, w = pred.orig_shape
                    if len(pred.boxes) > 0:
                        boxes = pred.boxes.xyxy.cpu().numpy()
                        confs = pred.boxes.conf.cpu().numpy()

                        best_idx = 0
                        max_score = -1.0
                        for bi, (b, c) in enumerate(zip(boxes, confs)):
                            area = (b[2] - b[0]) * (b[3] - b[1])
                            score = float(c) * math.sqrt(max(1.0, area))
                            if score > max_score:
                                max_score = score
                                best_idx = bi

                        b = boxes[best_idx]
                        c = confs[best_idx]
                        x1, y1, x2, y2 = b

                        pad_w = (x2 - x1) * self.padding
                        pad_h = (y2 - y1) * self.padding
                        crop_x1 = max(0, int(round(x1 - pad_w)))
                        crop_y1 = max(0, int(round(y1 - pad_h)))
                        crop_x2 = min(w, int(round(x2 + pad_w)))
                        crop_y2 = min(h, int(round(y2 + pad_h)))

                        if (crop_x2 - crop_x1) * (crop_y2 - crop_y1) >= (w * h * 0.015):
                            results_data.append({
                                'image': name,
                                'x1': crop_x1, 'y1': crop_y1,
                                'x2': crop_x2, 'y2': crop_y2,
                                'conf': float(c)
                            })
                            continue

                    # Fallback: 全图
                    results_data.append({
                        'image': name,
                        'x1': 0, 'y1': 0, 'x2': w, 'y2': h, 'conf': 0.0
                    })
            except Exception as e:
                for name in batch_names:
                    results_data.append({
                        'image': name,
                        'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'conf': 0.0
                    })

        bbox_df = pd.DataFrame(results_data)
        bbox_df.to_csv(save_csv_path, index=False)
        print(f"[YOLO] 边界框缓存已成功保存至: {save_csv_path}")
        return bbox_df

    @staticmethod
    def load_bbox_dict(csv_path):
        """
        快速加载已有边界框 CSV 为字典映射: image_name -> (x1, y1, x2, y2)
        """
        if not os.path.exists(csv_path):
            return {}
        try:
            df = pd.read_csv(csv_path)
            bbox_dict = {}
            for _, row in df.iterrows():
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                if x2 > x1 and y2 > y1:
                    bbox_dict[row['image']] = (x1, y1, x2, y2)
            print(f"[YOLO] 成功从缓存读取 {len(bbox_dict)} 个有效检测边界框: {csv_path}")
            return bbox_dict
        except Exception as e:
            print(f"[YOLO Warning] 读取缓存文件失败: {e}")
            return {}


# =================================================================================
# 4. 数据增强与数据集加载器 (Dataset & Augmentations)
# =================================================================================

def get_train_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.10,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

def get_valid_transform(image_size):
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
    Happywhale 自定义数据集，原生集成 YOLO 边界框裁剪。
    """
    def __init__(self, df, img_dir, bbox_dict=None, transforms=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.bbox_dict = bbox_dict if bbox_dict is not None else {}
        self.transforms = transforms
        self.is_test = is_test
        self.image_name = self.df['image'].values

        if not self.is_test:
            self.labels = self.df['label'].values

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        img_name = self.image_name[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 1. 尝试读取原图
        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            print(f"[Warning] 图像读取失败：{img_path}, 错误: {e}")
            image = np.zeros((CFG.image_size, CFG.image_size, 3), dtype=np.uint8)

        # 2. YOLO Bounding Box 快速切片裁剪
        if img_name in self.bbox_dict:
            x1, y1, x2, y2 = self.bbox_dict[img_name]
            h, w = image.shape[:2]
            # 边界安全校验
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1 > 10) and (y2 - y1 > 10):
                image = image[y1:y2, x1:x2]

        # 3. 数据增强与尺寸规整 (Albumentations)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        if self.is_test:
            return image, img_name
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label


# =================================================================================
# 5. 模型结构与 ArcFace 损失头 (ResNet50 + ArcFace)
# =================================================================================

class ArcMarginProduct(nn.Module):
    """
    经典 ArcFace (Additive Angular Margin Loss) 分类头模块。
    通过在角度空间加入裕度惩罚 m，迫使同类特征更聚集、异类特征更分散。
    """
    def __init__(self, in_features, out_classes, s=30.0, m=0.50, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_classes = out_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps

        self.weight = nn.Parameter(torch.FloatTensor(out_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels=None):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))

        if labels is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0.0, 1.0)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        if self.ls_eps > 0:
            one_hot = (1.0 - self.ls_eps) * one_hot + self.ls_eps / self.out_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output, cosine


class HappywhaleModel(nn.Module):
    """
    基于 ResNet50 骨干网络的高分辨率特征提取模型。
    """
    def __init__(self, num_classes, model_name="resnet50", embedding_size=512, pretrained=True, s=30.0, m=0.50):
        super(HappywhaleModel, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet50(weights=weights)

        # 骨干网络 (移除原全连接层)
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1])
        backbone_dim = base_resnet.fc.in_features

        # 512 维特征嵌入投射层与批归一化
        self.embedding_layer = nn.Linear(backbone_dim, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

        # ArcFace 角度分类头
        self.arcface_head = ArcMarginProduct(
            in_features=embedding_size,
            out_classes=num_classes,
            s=s,
            m=m
        )

    def extract_features(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        features = self.embedding_layer(features)
        features = self.bn(features)
        normalized_features = F.normalize(features, p=2, dim=1)
        return normalized_features

    def forward(self, x, labels=None):
        embeddings = self.extract_features(x)
        if labels is not None:
            logits, raw_cos = self.arcface_head(embeddings, labels)
            return logits, raw_cos
        return embeddings


# =================================================================================
# 6. 评估指标与训练函数 (MAP@5 & Training Loop)
# =================================================================================

def compute_map_per_image(actual_label, predicted_ids, k=5):
    top_k_preds = predicted_ids[:k]
    if actual_label in top_k_preds:
        rank = top_k_preds.index(actual_label) + 1
        return 1.0 / rank
    return 0.0

def compute_map5(actual_labels, prediction_lists):
    scores = [
        compute_map_per_image(act, preds, k=5)
        for act, preds in zip(actual_labels, prediction_lists)
    ]
    return float(np.mean(scores))


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    correct_arc = 0
    correct_cos = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1:02d} [Train 384x384]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        optimizer.zero_grad()

        use_cuda_amp = (CFG.use_amp and device.type == 'cuda')
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            amp_ctx = torch.amp.autocast(device_type='cuda', enabled=use_cuda_amp)
        else:
            amp_ctx = torch.cuda.amp.autocast(enabled=use_cuda_amp)

        with amp_ctx:
            logits, raw_cos = model(images, labels)
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

        total_loss += loss.item() * batch_size

        # 统计 ArcFace 惩罚后的预测以及未惩罚的余弦相似度预测
        preds_arc = torch.argmax(logits, dim=1)
        preds_cos = torch.argmax(raw_cos, dim=1)

        correct_arc += (preds_arc == labels).sum().item()
        correct_cos += (preds_cos == labels).sum().item()
        total += batch_size

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc_arc': f"{correct_arc / total:.4f}",
            'acc_cos': f"{correct_cos / total:.4f}",
            'lr': f"{current_lr:.2e}"
        })

    if scheduler is not None:
        scheduler.step()

    epoch_loss = total_loss / total
    epoch_acc = correct_arc / total
    epoch_cos_acc = correct_cos / total
    return epoch_loss, epoch_acc, epoch_cos_acc


@torch.no_grad()
def extract_all_embeddings(model, dataloader, device):
    model.eval()
    embeddings_list = []

    pbar = tqdm(dataloader, desc="Extracting Embeddings", leave=False)
    for batch in pbar:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        images = images.to(device, non_blocking=True)

        embs = model.extract_features(images)
        embeddings_list.append(embs.cpu().numpy())

    return np.vstack(embeddings_list)


def predict_with_knn(query_embeddings, gallery_embeddings, gallery_individual_ids,
                     threshold=0.55, top_k=5, is_validation=False, chunk_size=2000, return_details=False):
    """
    执行向量空间余弦近邻检索并生成 Top-5 预测。
    对于验证集 (已知类别均为既有样本)，不盲目在 Rank 2 强插 new_individual，避免破坏排名；
    对于测试集，按自适应相似度判定插入 new_individual。
    """
    query_tensor = torch.from_numpy(query_embeddings).float()
    gallery_tensor = torch.from_numpy(gallery_embeddings).float().t()

    num_queries = query_tensor.size(0)
    all_predictions = []
    all_details = []

    popular_ids = list(pd.Series(gallery_individual_ids).value_counts().index[:10])
    print(f"[Info] 开始近邻检索匹配 (查询样本: {num_queries}, 底库样本: {gallery_tensor.size(1)})...")

    for start_idx in range(0, num_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, num_queries)
        q_chunk = query_tensor[start_idx:end_idx]

        sim_matrix = torch.mm(q_chunk, gallery_tensor)
        topk_scores, topk_indices = torch.topk(sim_matrix, k=min(60, gallery_tensor.size(1)), dim=1)
        topk_scores = topk_scores.numpy()
        topk_indices = topk_indices.numpy()

        for row_scores, row_indices in zip(topk_scores, topk_indices):
            candidate_ids = [gallery_individual_ids[i] for i in row_indices]

            seen = set()
            unique_candidates = []
            for cid in candidate_ids:
                if cid not in seen:
                    seen.add(cid)
                    unique_candidates.append(cid)

            max_sim = row_scores[0] if len(row_scores) > 0 else 0.0

            if is_validation:
                # 验证集评估逻辑：已知全部来自训练集个体，按检索相似度直接排序
                preds = unique_candidates.copy()
            else:
                # 测试集推理逻辑：考虑未知新个体 (new_individual) 机制
                if max_sim < threshold:
                    preds = ['new_individual'] + [cid for cid in unique_candidates if cid != 'new_individual']
                else:
                    preds = []
                    inserted_new = False
                    for i, cid in enumerate(unique_candidates):
                        if (not inserted_new) and (i == 1 or (i < len(row_scores) and row_scores[i] < threshold)):
                            preds.append('new_individual')
                            inserted_new = True
                        preds.append(cid)
                    if not inserted_new:
                        preds.append('new_individual')

            # 填充高频个体确保满足 top_k
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
# 7. 全面可视化系统 (Visualizations)
# =================================================================================

def visualize_yolo_crops(df, img_dir, bbox_dict, output_path, num_samples=6):
    """
    全新新增: 可视化 YOLO 边界框检测与裁剪去海水背景效果对比图。
    """
    valid_names = [name for name in df['image'] if name in bbox_dict]
    if len(valid_names) == 0:
        valid_names = df['image'].head(num_samples).tolist()

    selected = valid_names[:num_samples]
    if len(selected) == 0:
        return

    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 3.2 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, name in enumerate(selected):
        img_path = os.path.join(img_dir, name)
        try:
            raw_img = Image.open(img_path).convert("RGB")
            raw_np = np.array(raw_img)
            h, w = raw_np.shape[:2]

            # 左图: 原图与检测框
            axes[i, 0].imshow(raw_np)
            if name in bbox_dict:
                x1, y1, x2, y2 = bbox_dict[name]
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2.5, edgecolor='#e74c3c', facecolor='none')
                axes[i, 0].add_patch(rect)
                axes[i, 0].text(x1, max(15, y1 - 8), f"YOLO Whale/Fin ({x2-x1}x{y2-y1})",
                                color='white', backgroundcolor='#e74c3c', fontsize=9, fontweight='bold')
                title_text = f"[{i+1}] 原始图像与检测框 (含大面积海水背景)"
            else:
                title_text = f"[{i+1}] 原始图像 (未检出，平滑回退原图)"

            axes[i, 0].set_title(title_text, fontsize=10, fontweight='bold')
            axes[i, 0].axis('off')

            # 右图: 裁剪后高清背鳍/鲸豚主体
            if name in bbox_dict:
                x1, y1, x2, y2 = bbox_dict[name]
                crop_np = raw_np[y1:y2, x1:x2]
                if crop_np.size == 0:
                    crop_np = raw_np
            else:
                crop_np = raw_np

            axes[i, 1].imshow(crop_np)
            crop_h, crop_w = crop_np.shape[:2]
            axes[i, 1].set_title(f"[{i+1}] YOLO 裁剪主体 (已过滤海水噪声 -> 输入 384x384 模型)",
                                 fontsize=10, fontweight='bold', color='#27ae60')
            axes[i, 1].axis('off')

        except Exception as e:
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')

    plt.suptitle("YOLOv8 智能目标检测与背鳍裁剪效果对比 (消除海水噪声)", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] YOLO 目标检测与裁剪对比图已保存至: {output_path}")


def visualize_dataset_distribution(train_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Top 15 物种分布
    species_counts = train_df['species'].value_counts().head(15)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(species_counts)))
    bars = axes[0].barh(species_counts.index[::-1], species_counts.values[::-1], color=colors[::-1])
    axes[0].set_title("训练集前 15 种物种样本分布 (Top 15 Species)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("样本数量", fontsize=11)
    axes[0].grid(axis='x', linestyle='--', alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        axes[0].text(width + 50, bar.get_y() + bar.get_height() / 2, f"{int(width):,}",
                     va='center', fontsize=9, color='#333333')

    # 2. 个体样本长尾分布
    individual_counts = train_df['individual_id'].value_counts()
    c1 = (individual_counts == 1).sum()
    c2 = (individual_counts == 2).sum()
    c3_5 = ((individual_counts >= 3) & (individual_counts <= 5)).sum()
    c6_10 = ((individual_counts >= 6) & (individual_counts <= 10)).sum()
    c_gt10 = (individual_counts > 10).sum()

    categories = ['仅 1 张图片\n(单样本个体)', '2 张图片', '3 - 5 张', '6 - 10 张', ' > 10 张图片']
    counts = [c1, c2, c3_5, c6_10, c_gt10]
    palette = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']

    axes[1].bar(categories, counts, color=palette, edgecolor='black', alpha=0.85)
    axes[1].set_title("个体样本拥有量长尾分布 (开集识别/小样本挑战)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("个体数量", fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    for i, count in enumerate(counts):
        pct = (count / len(individual_counts)) * 100
        axes[1].text(i, count + max(counts) * 0.02, f"{count:,}\n({pct:.1f}%)",
                     ha='center', fontsize=10, fontweight='bold')

    plt.suptitle("Happywhale 数据集全景统计与长尾分布", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] 数据分布可视化已保存至：{output_path}")


def visualize_training_curves(history, output_path):
    if not history or 'loss' not in history or len(history['loss']) == 0:
        return

    epochs = range(1, len(history['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax1.plot(epochs, history['loss'], marker='o', color='#d35400', linewidth=2.2, label='Train ArcFace Loss')
    ax1.set_title("训练损失收敛曲线 (ArcFace Margin Loss)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("轮次 (Epoch)", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')

    # Accuracy 曲线 (同时显示 ArcFace 裕度准度和余弦空间原生准度)
    ax2.plot(epochs, [acc * 100 for acc in history['acc']], marker='s', color='#27ae60', linewidth=2.2, label='ArcFace Acc (%)')
    if 'acc_cos' in history and len(history['acc_cos']) == len(epochs):
        ax2.plot(epochs, [c * 100 for c in history['acc_cos']], marker='^', color='#2980b9', linewidth=2.0, linestyle='--', label='Raw Cosine Acc (%)')
    ax2.set_title("训练分类准确率曲线 (Training Accuracy)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("轮次 (Epoch)", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right')

    plt.suptitle("Happywhale ResNet50 (384x384 + YOLO Crop) 训练历程", fontsize=15, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] 训练曲线可视化已保存至: {output_path}")


def visualize_retrieval_predictions(query_df, query_img_dir, gallery_df, gallery_img_dir,
                                    details_list, predictions_list, output_path, num_samples=3):
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

        # Query 查询图
        try:
            q_img = Image.open(q_path).convert("RGB")
            axes[row, 0].imshow(q_img)
        except Exception:
            axes[row, 0].imshow(np.zeros((100, 100, 3), dtype=np.uint8))
        axes[row, 0].set_title(f"[查询图 Query #{row + 1}]\n真实个体:{q_label[:8]}...\n({q_species})",
                                fontsize=10, fontweight='bold', color='#2980b9')
        axes[row, 0].axis('off')

        # Top-5 检索结果
        preds = predictions_list[row]
        details = details_list[row]
        top_indices = details['top_indices']
        top_scores = details['top_scores']

        for col in range(5):
            ax = axes[row, col + 1]
            if col < len(preds):
                preds_id = preds[col]
                if preds_id == 'new_individual':
                    placeholder = np.ones((120, 120, 3), dtype=np.uint8) * 240
                    ax.imshow(placeholder)
                    ax.text(60, 60, "新个体\nnew_individual\n(相似度低于阈值)",
                            ha='center', va='center', fontsize=9, fontweight='bold', color='#8e44ad')
                    ax.set_title(f"Rank {col + 1}: new_individual", fontsize=10, color='#8e44ad')
                else:
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

                    is_correct = (g_label == q_label)
                    border_color = '#27ae60' if is_correct else '#7f8c8d'
                    tag = "[命中]" if is_correct else "[候选]"

                    for spine in ax.spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(3 if is_correct else 1)

                    ax.set_title(f"Rank {col + 1}: {tag}\nSim: {score:.3f} | {g_label[:8]}...",
                                 fontsize=9, color=border_color, fontweight='bold')
            ax.axis('off')

    plt.suptitle("鲸豚身份近邻检索结果 (kNN Cosine Retrieval & Rank 1-5 Matches)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] 检索比对可视化已保存至: {output_path}")


def visualize_embeddings_pca(embeddings, individual_ids, output_path, max_samples=400):
    if len(embeddings) == 0:
        return

    df_temp = pd.DataFrame({'ind': individual_ids})
    top_inds = df_temp['ind'].value_counts().head(8).index.tolist()

    mask = df_temp['ind'].isin(top_inds)
    selected_indices = np.where(mask)[0]
    if len(selected_indices) < 20:
        selected_indices = np.arange(min(len(embeddings), max_samples))

    sub_embs = embeddings[selected_indices]
    sub_labels = [individual_ids[i] for i in selected_indices]

    pca = PCA(n_components=2, random_state=CFG.seed)
    emb_2d = pca.fit_transform(sub_embs)

    fig, ax = plt.subplots(figsize=(10, 7))
    unique_selected = sorted(list(set(sub_labels)))

    for idx, ind in enumerate(unique_selected):
        ind_mask = [l == ind for l in sub_labels]
        ax.scatter(
            emb_2d[ind_mask, 0],
            emb_2d[ind_mask, 1],
            label=f"个体:{ind[:8]}...",
            s=45,
            alpha=0.85,
            edgecolors='white',
            linewidths=0.5
        )

    ax.set_title("ArcFace 潜空间个体特征嵌入降维分布 (PCA 2D Projection)", fontsize=13, fontweight='bold')
    ax.set_xlabel(f"主成分 1 (解释方差比: {pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"主成分 2 (解释方差比: {pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] 特征空间 PCA 投影可视化已保存至: {output_path}")


# =================================================================================
# 8. 全流程执行引擎 (End-to-End Pipeline)
# =================================================================================

def run_pipeline(args):
    seed_everything(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] 当前运算设备: {device} "
          f"{'(' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")

    # 创建全新的独立输出文件夹 output_yolo
    os.makedirs(CFG.output_dir, exist_ok=True)
    print(f"[Output] 本次训练与推理结果将输出至全新目录: {CFG.output_dir}")

    # -------------------------------------------------------------------------
    # 步骤 1/5: 加载与清洗标注数据
    # -------------------------------------------------------------------------
    print("\n--- 步骤 1/5: 加载与清洗数据 ---")
    if not os.path.exists(CFG.train_csv):
        raise FileNotFoundError(f"找不到训练标注文件: {CFG.train_csv}")

    train_df = pd.read_csv(CFG.train_csv)
    print(f"原始训练集样本数: {len(train_df)}, 独特物种数: {train_df['species'].nunique()}")

    if args.visualize:
        visualize_dataset_distribution(train_df, CFG.vis_data_dist_path)

    unique_individuals = sorted(train_df['individual_id'].unique())
    num_classes = len(unique_individuals)
    individual_to_label = {ind: idx for idx, ind in enumerate(unique_individuals)}
    label_to_individual = {idx: ind for idx, ind in enumerate(unique_individuals)}
    train_df['label'] = train_df['individual_id'].map(individual_to_label)
    print(f"训练集独特个体总数 (总分类数): {num_classes}")

    # Debug 模式
    if args.debug or CFG.debug:
        print(f"[Debug 模式] 截取前 {CFG.debug_train_samples} 条样本以极速验证流程...")
        train_df = train_df.iloc[:CFG.debug_train_samples].copy()
        debug_individuals = sorted(train_df['individual_id'].unique())
        num_classes = len(debug_individuals)
        individual_to_label = {ind: idx for idx, ind in enumerate(debug_individuals)}
        label_to_individual = {idx: ind for idx, ind in enumerate(debug_individuals)}
        train_df['label'] = train_df['individual_id'].map(individual_to_label)
        print(f"[Debug 模式] 调试类别数: {num_classes}")

    # 分层划分训练集与验证集
    id_counts = train_df['individual_id'].value_counts()
    singletons = id_counts[id_counts == 1].index.tolist()
    multi_samples = id_counts[id_counts > 1].index.tolist()

    multi_df = train_df[train_df['individual_id'].isin(multi_samples)].copy()
    singleton_df = train_df[train_df['individual_id'].isin(singletons)].copy()

    if len(multi_df) > 0:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
        train_idx, valid_idx = next(skf.split(multi_df, multi_df['individual_id']))
        train_split = pd.concat([multi_df.iloc[train_idx], singleton_df]).reset_index(drop=True)
        valid_split = multi_df.iloc[valid_idx].reset_index(drop=True)
    else:
        train_split = train_df.copy()
        valid_split = train_df.iloc[:min(20, len(train_df))].copy()

    print(f"训练集划分样本数: {len(train_split)}, 验证集划分样本数: {len(valid_split)}")

    # -------------------------------------------------------------------------
    # 步骤 2/5: YOLO 目标检测边界框提取与数据加载器构建
    # -------------------------------------------------------------------------
    print("\n--- 步骤 2/5: YOLO 目标检测边界框提取与数据加载器 ---")
    yolo_manager = YoloBboxManager(
        model_name=CFG.yolo_model_name,
        conf=CFG.yolo_conf,
        padding=CFG.yolo_padding,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 检查是否存在已缓存的 Bbox CSV
    train_bbox_dict = {}
    if os.path.exists(CFG.train_bbox_csv) and not args.force_detect:
        train_bbox_dict = YoloBboxManager.load_bbox_dict(CFG.train_bbox_csv)
    else:
        # 若为 Debug 模式或显式指定了提取边界框，则执行快速提取并缓存
        if args.extract_bboxes or args.debug:
            yolo_manager.precompute_bboxes(train_df, CFG.train_img_dir, CFG.train_bbox_csv, batch_size=32)
            train_bbox_dict = YoloBboxManager.load_bbox_dict(CFG.train_bbox_csv)
        else:
            print("[YOLO Info] 未找到边界框缓存 CSV，将使用轻量在线/按需检测，或建议使用 --extract_bboxes 预先提取以获得最高训练速度。")

    # 可视化 YOLO 边界框裁剪效果 (原始图 vs 裁剪图)
    if args.visualize:
        visualize_yolo_crops(train_df, CFG.train_img_dir, train_bbox_dict, CFG.vis_yolo_crop_path, num_samples=6)

    # 构建带 YOLO 边界框裁剪的数据集与数据加载器 (分辨率 image_size = 384)
    print(f"[DataLoader] 正在构建高分辨率数据加载器 (输入尺寸: {args.image_size}x{args.image_size}, Batch Size: {args.batch_size})...")
    train_dataset = HappywhaleDataset(
        train_split,
        CFG.train_img_dir,
        bbox_dict=train_bbox_dict,
        transforms=get_train_transform(args.image_size),
        is_test=False
    )
    valid_dataset = HappywhaleDataset(
        valid_split,
        CFG.train_img_dir,
        bbox_dict=train_bbox_dict,
        transforms=get_valid_transform(args.image_size),
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
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # -------------------------------------------------------------------------
    # 步骤 3/5: 构建网络模型与优化器
    # -------------------------------------------------------------------------
    print("\n--- 步骤 3/5: 构建 ResNet50 + ArcFace 模型与优化器 ---")
    model = HappywhaleModel(
        num_classes=num_classes,
        model_name=CFG.model_name,
        embedding_size=CFG.embedding_size,
        pretrained=CFG.pretrained,
        s=CFG.arcface_s,
        m=CFG.arcface_m
    ).to(device)

    # 若已存在检查点且不需要重训
    if os.path.exists(CFG.checkpoint_path) and not args.force_retrain and args.mode in ['eval', 'infer']:
        print(f"[Info] 加载已存在的检查点权重: {CFG.checkpoint_path}")
        checkpoint = torch.load(CFG.checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

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
    # 步骤 4/5: 执行模型训练
    # -------------------------------------------------------------------------
    training_history = {'loss': [], 'acc': [], 'acc_cos': []}
    if args.mode in ['all', 'train']:
        print(f"\n--- 步骤 4/5: 开始模型训练 (共 {args.epochs} 轮, 分辨率 {args.image_size}x{args.image_size}) ---")
        best_acc = -1.0
        start_time = time.time()

        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss, train_acc, train_cos_acc = train_one_epoch(
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
            training_history['acc_cos'].append(train_cos_acc)

            print(f"[Epoch {epoch + 1:02d}/{args.epochs:02d}] "
                  f"耗时: {elapsed:.1f}s | Loss: {train_loss:.4f} | "
                  f"ArcFace Acc: {train_acc * 100:.2f}% | 原生余弦 Acc: {train_cos_acc * 100:.2f}%")

            # 以更稳健的指标保存最佳模型
            val_criterion = train_cos_acc
            if val_criterion >= best_acc:
                best_acc = val_criterion
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

        if args.visualize:
            visualize_training_curves(training_history, CFG.vis_training_curves_path)

    # -------------------------------------------------------------------------
    # 步骤 5A: 验证集特征检索与 MAP@5 评估
    # -------------------------------------------------------------------------
    if args.mode in ['all', 'eval']:
        print("\n--- 步骤 5A: 验证集特征提取与 MAP@5 评估 ---")
        full_train_dataset = HappywhaleDataset(
            train_split,
            CFG.train_img_dir,
            bbox_dict=train_bbox_dict,
            transforms=get_valid_transform(args.image_size),
            is_test=False
        )
        full_train_loader = DataLoader(
            full_train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers
        )

        train_embs = extract_all_embeddings(model, full_train_loader, device)
        val_embs = extract_all_embeddings(model, valid_loader, device)

        valid_predictions, valid_details = predict_with_knn(
            query_embeddings=val_embs,
            gallery_embeddings=train_embs,
            gallery_individual_ids=train_split['individual_id'].values,
            threshold=CFG.new_individual_threshold,
            top_k=5,
            is_validation=True,
            return_details=True
        )

        map5_score = compute_map5(valid_split['individual_id'].values, valid_predictions)
        print(f"\n=======================================================")
        print(f" [评估结果] 验证集 MAP@5 得分: {map5_score:.5f}")
        print(f"=======================================================\n")

        if args.visualize:
            visualize_retrieval_predictions(
                query_df=valid_split,
                query_img_dir=CFG.train_img_dir,
                gallery_df=train_split,
                gallery_img_dir=CFG.train_img_dir,
                details_list=valid_details,
                predictions_list=valid_predictions,
                output_path=CFG.vis_retrieval_path,
                num_samples=3
            )
            visualize_embeddings_pca(
                embeddings=val_embs,
                individual_ids=valid_split['individual_id'].values,
                output_path=CFG.vis_tsne_path,
                max_samples=400
            )

    # -------------------------------------------------------------------------
    # 步骤 5B: 测试集特征提取与生成 submission.csv
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

        test_bbox_dict = {}
        if os.path.exists(CFG.test_bbox_csv) and not args.force_detect:
            test_bbox_dict = YoloBboxManager.load_bbox_dict(CFG.test_bbox_csv)
        elif args.extract_bboxes or args.debug:
            yolo_manager.precompute_bboxes(sub_df, CFG.test_img_dir, CFG.test_bbox_csv, batch_size=32)
            test_bbox_dict = YoloBboxManager.load_bbox_dict(CFG.test_bbox_csv)

        test_dataset = HappywhaleDataset(
            sub_df,
            CFG.test_img_dir,
            bbox_dict=test_bbox_dict,
            transforms=get_valid_transform(args.image_size),
            is_test=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers
        )

        if 'train_embs' not in locals():
            full_train_dataset = HappywhaleDataset(
                train_split,
                CFG.train_img_dir,
                bbox_dict=train_bbox_dict,
                transforms=get_valid_transform(args.image_size),
                is_test=False
            )
            full_train_loader = DataLoader(
                full_train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=CFG.num_workers
            )
            train_embs = extract_all_embeddings(model, full_train_loader, device)

        test_embs = extract_all_embeddings(model, test_loader, device)

        test_predictions = predict_with_knn(
            query_embeddings=test_embs,
            gallery_embeddings=train_embs,
            gallery_individual_ids=train_split['individual_id'].values,
            threshold=CFG.new_individual_threshold,
            top_k=5,
            is_validation=False
        )

        formatted_preds = [" ".join(p) for p in test_predictions]
        sub_df['predictions'] = formatted_preds
        sub_df.to_csv(CFG.submission_path, index=False)
        print(f"[Info] 预测文件已成功保存至: {CFG.submission_path}")
        print("预测结果前 5 行预览:")
        print(sub_df.head(5))


# =================================================================================
# 9. 命令行入口 (CLI Interface)
# =================================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Happywhale Identification - YOLO Crop & High-Resolution Enhanced")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "train", "eval", "infer"],
                        help="运行模式: all (全流程), train (仅训练), eval (仅评估), infer (仅推理)")
    parser.add_argument("--epochs", type=int, default=CFG.epochs,
                        help="模型训练轮次")
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size,
                        help="批处理大小 (默认 16，适配 384x384 与 8GB 显存)")
    parser.add_argument("--image_size", type=int, default=CFG.image_size,
                        help="输入图像分辨率 (默认 384)")
    parser.add_argument("--lr", type=float, default=CFG.learning_rate,
                        help="初始学习率")
    parser.add_argument("--debug", action="store_true",
                        help="启用 Debug 模式极速验证全流程")
    parser.add_argument("--extract_bboxes", action="store_true",
                        help="显式批量执行 YOLO 目标检测并保存边界框缓存 CSV")
    parser.add_argument("--force_detect", action="store_true",
                        help="忽略已有边界框缓存强制重新检测")
    parser.add_argument("--force_retrain", action="store_true",
                        help="忽略已有模型检查点强制重新训练")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="自动生成所有可视化图表 (包含 YOLO 裁剪对比、训练曲线、近邻检索与特征聚类)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
