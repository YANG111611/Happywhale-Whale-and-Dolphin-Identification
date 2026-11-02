# 🐋 Happywhale - Whale and Dolphin Identification (海洋鲸豚个体细粒度身份重识别)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.15%2B-red.svg?logo=pytorch&logoColor=white)](https://pytorch.org/vision/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![Albumentations](https://img.shields.io/badge/Albumentations-Data%20Aug-green.svg)](https://albumentations.ai/)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Happywhale%202022-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/c/happywhale-whale-and-dolphin-identification)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

本项目针对 Kaggle 顶级计算机视觉赛事 [Happywhale - Whale and Dolphin Identification (鲸鱼与海豚个体身份识别)](https://www.kaggle.com/c/happywhale-whale-and-dolphin-identification) 构建了一套工业级的**细粒度图像检索与开集重识别（Re-ID）系统**。

通过结合**深度度量学习（ArcFace）**与**目标检测智能裁剪（YOLOv8）**，旨在根据野生鲸鱼和海豚的背鳍形状、尾叶边缘、切口凹槽、体表色素与划痕等微细特征，在全球跨海域数万张图像中精准识别出 15,000+ 个独特个体，并有效甄别测试集中未曾记录的新个体（`new_individual`）。

---

## 📌 核心挑战与方案演进 (Challenges & Solutions)

### 1. 核心业务痛点
* **极端长尾分布 (Long-Tail Distribution)**：全数据集涵盖 30 个物种、15,587 个独立个体，大量个体仅有 1~2 张历史观测记录。
* **开集检索判定 (Open-Set Recognition)**：测试集中有相当比例的个体在训练集中从未出现过，系统必须能准确推断 `new_individual`。
* **背景高频噪声 (Environmental Clutter)**：野生拍摄图像中存在大面积海浪、反光、水雾、阴影干扰，严重弱化了鳍部微小纹理特征。
* **官方物种标签脏乱 (Data Label Noise)**：官方原始标注中存在物种名拼写错误（如 `bottlenose_dolpin`）及同种异名冗余。

### 2. 双版本算法演进体系
本项目提供由浅入深、兼顾速度与精度的两套工程化实现方案：

| 方案版本 | 核心脚本 | 特性概述 | 推荐场景 |
| :--- | :--- | :--- | :--- |
| **标准基线方案 (Baseline)** | `Happywhale - Whale and Dolphin Identification.py` | 官方标签清洗 + 全图预处理 + **ResNet-34 + ArcFace** 度量学习 + Cosine 检索 | 显存资源受限 (<= 6GB)、轻量快速实验与基线对齐 |
| **高阶进阶方案 (YOLO+ReID)** | `Happywhale - Whale and Dolphin Identification_yolo.py` | **YOLOv8** 目标检测智能外扩裁剪 + 消除海浪噪声 + **ResNet-50 + ArcFace** 高分辨率 (384×384) 深度特征提取 | 高精度竞赛冲榜、微细锯齿划痕识别 |

---

## 🚀 关键技术亮点 (Key Highlights)

1. **工业级数据清洗与同义物种合并 (Data Sanitization)**：
   - 自动映射并修正官方标注中的拼写错误（如 `globicephala_macrorhynchus`、`kogia_sima` 规范化为通用物种名称）；
   - 构建 `image_id -> individual_id` 整数编码哈希索引，建立稳健的无泄漏分层交叉验证（Stratified K-Fold）。
2. **两阶段主体检测与自适应边界裁剪 (YOLOv8 Dynamic Cropping)**：
   - 部署 YOLOv8 目标检测定位鲸豚及背鳍主体，引入**外扩保护比例 (Margin Padding)**，杜绝鳍尖与边缘特征被截断；
   - 包含自适应回退机制（Fallback），无检出或低置信度时平滑降级使用原图；
   - 支持检测边界框离线缓存为 CSV，彻底消除训练时的重复检测开销。
3. **基于超球面边缘约束的深度度量学习 (ArcFace Loss)**：
   - 在特征向量空间（512 维）强制执行 L2 归一化；
   - 引入加性角度边界损失（Additive Angular Margin Loss, $s=30.0, m=0.50$），在超球流形上最大化类间角度分离度，极大压缩同个体特征方差。
4. **自适应开集判定与余弦近邻检索 (Cosine kNN & Adaptive Threshold)**：
   - 训练结束后批量抽取数据库与查询图的 L2-Embedding；
   - 依据余弦相似度计算 Top-5 匹配；当最大置信度低于动态阈值 $\tau$（如 0.55）时，智能切入并插入 `new_individual` 预测候选。
5. **Kaggle 官方评价协议 MAP@5 离线对齐**：
   - 完整复现了 Kaggle 官方的 Mean Average Precision @ 5 (MAP@5) 评估指标，离线严密验证召回效果。
6. **全维度图表可视化分析系统**：
   - 物种与个体样本长尾分布直方图 (`data_distribution.png`)
   - 训练与验证损失动态收敛曲线 (`training_curves.png`)
   - Top-5 最近邻检索样本图解比对 (`retrieval_visualization.png`)
   - 隐空间高维特征 PCA / t-SNE 降维聚类投影 (`embeddings_tsne.png`)
   - YOLO 检测定位前后对比效果切片 (`yolo_crop_visualization.png`)

---

## 📂 项目结构 (Project Directory)

```text
Happywhale - Whale and Dolphin Identification/
├── Happywhale - Whale and Dolphin Identification/  # 原始数据集目录
│   ├── train_images/                              # 训练图像集 (~51,000 张)
│   ├── test_images/                               # 测试图像集 (~27,900 张)
│   ├── train.csv                                  # 官方训练标签 (含 image_id, species, individual_id)
│   └── sample_submission.csv                      # 官方提交规范模板
├── output/                                        # 标准基线方案输出目录
│   ├── happywhale_resnet34_best.pth               # 训练收敛的最优 ResNet-34 权重
│   ├── data_distribution.png                      # 数据分布可视化
│   ├── training_curves.png                        # 训练收敛曲线
│   ├── retrieval_visualization.png                # Top-5 检索比对图
│   ├── embeddings_tsne.png                        # 隐空间聚类投影图
│   └── submission.csv                             # 标准提交预测文件
├── output_yolo/                                   # YOLO 进阶裁剪方案输出目录
│   ├── happywhale_resnet50_best.pth               # 训练收敛的最优 ResNet-50 权重
│   ├── train_bboxes.csv / test_bboxes.csv         # YOLO 检测缓存边界框
│   ├── yolo_crop_visualization.png                # 检测裁剪对比效果图
│   └── submission.csv                             # YOLO 进阶预测文件
├── Happywhale - Whale and Dolphin Identification.py       # 基线方案核心源码 (ResNet34 + ArcFace)
├── Happywhale - Whale and Dolphin Identification_yolo.py  # 进阶方案核心源码 (YOLOv8 + ResNet50 + ArcFace)
├── yolov8n.pt                                     # YOLOv8 预训练检测权重
├── requirements.txt                               # 项目依赖清单
├── .gitignore                                     # Git 大文件与临时文件过滤配置
└── README.md                                      # 项目说明文档
```

---

## 🏗️ 算法架构流程 (Architecture Pipeline)

```
[输入原始图像 (Wild Whales & Dolphins)]
       │
       ▼ (方案二：YOLOv8 检测定位)
 [YOLOv8 Target Detector] ──> [自适应外扩边界 Margin Crop] ──> [过滤大面积海浪背景]
       │
       ▼ (统一分辨率归一化)
 [Albumentations 数据增强管道 (翻转/抖动/仿射)]
       │
       ▼
 [CNN 骨干网络 (ResNet-34 / ResNet-50)]
       │
       ▼
 [全局平均池化 (GAP) + Linear 投影 + BatchNorm]
       │
       ▼
 [512 维 L2 归一化特征向量 Embedding]
       │
       ├───► [训练阶段] ──> [ArcFace Margin Product (s=30, m=0.5)] ──> [Cross-Entropy Loss]
       │
       └───► [推理阶段] ──> [特征库构建 & Cosine 相似度计算]
                                  │
                                  ▼
                         [Top-5 候选检索排序]
                                  │
                                  ▼ (相似度 < 阈值 τ)
                         [自适应插入 "new_individual"]
                                  │
                                  ▼
                         [输出 Kaggle MAP@5 提交结果]
```

---

## ⚙️ 环境依赖与安装 (Installation)

### 1. 克隆代码仓库
```bash
git clone https://github.com/YANG111611/Happywhale-Whale-and-Dolphin-Identification.git
cd Happywhale-Whale-and-Dolphin-Identification
```

### 2. 配置 Python 虚拟环境 (推荐 Conda)
```bash
conda create -n happywhale python=3.10 -y
conda activate happywhale
```

### 3. 安装 PyTorch 与核心库
根据本地显卡环境安装对应 CUDA 版本的 PyTorch：
```bash
# 以 CUDA 11.8 为例
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装目标检测与数据增强等第三方依赖
pip install -r requirements.txt
```

> **核心依赖清单 (`requirements.txt`)**：
> ```text
> torch>=2.0.0
> torchvision>=0.15.0
> ultralytics>=8.0.0
> albumentations>=1.3.0
> pandas>=1.5.0
> numpy>=1.23.0
> scikit-learn>=1.2.0
> Pillow>=9.0.0
> matplotlib>=3.6.0
> tqdm>=4.64.0
> ```

---

## 🚀 快速开始 (Quick Start)

### 1. 运行基线模型 (ResNet-34 + ArcFace)
直接启动全流程端到端训练与评估：
```bash
python "Happywhale - Whale and Dolphin Identification.py"
```

可选参数调节示例：
```bash
# 使用 debug 模式微量样本快速测试全流程
python "Happywhale - Whale and Dolphin Identification.py" --debug

# 自定义超参数训练
python "Happywhale - Whale and Dolphin Identification.py" --epochs 12 --batch_size 32 --lr 3e-4 --backbone resnet34
```

### 2. 运行进阶 YOLO 裁剪方案 (YOLOv8 + ResNet-50)
在显卡算力充裕时，一键启用检测裁剪与高清特征提取：
```bash
python "Happywhale - Whale and Dolphin Identification_yolo.py" --epochs 10 --batch_size 16 --image_size 384
```

---

## 📊 评估指标与可视化效果 (Visualizations)

训练运行完成后，所有可视化监控图表与预测结果将自动归档至 `output/` 与 `output_yolo/` 目录中：

| 图表类型 | 保存路径 | 核心展示内容 |
| :--- | :--- | :--- |
| **长尾分布图** | `output/data_distribution.png` | 30 种物种分布比例、前 50 种高频个体样本量直方图与长尾累计贡献曲线 |
| **训练收敛图** | `output/training_curves.png` | 各轮次 ArcFace Loss 收敛轨迹、验证集 MAP@5 动态提升曲线与学习率退火过程 |
| **近邻检索比对** | `output/retrieval_visualization.png` | 查询测试图（Query）与 Top-5 检索候选个体的左右对比图（附带余弦相似度） |
| **隐空间投影** | `output/embeddings_tsne.png` | 512 维个体 Embedding 降维散点图，直观展现同类聚集、异类分散的超球面流形分布 |
| **YOLO 裁剪图** | `output_yolo/yolo_crop_visualization.png` | 原始大视野水面照片与 YOLO 裁剪后背鳍关键区域的对比样张 |

---

## 💡 单张图像个体检索示例 (Inference Demo)

训练完成后，可通过以下代码加载保存的模型对任意单张鲸豚照片进行个体识别预测：

```python
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 重新构建轻量推理模型 (只保留 Backbone 与 Embedding 映射层)
class InferenceModel(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        backbone = models.resnet34(pretrained=False)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        features = self.backbone(x)
        embed = self.embedding(features)
        return F.normalize(embed, p=2, dim=1)

# 2. 加载最优权重
model = InferenceModel().to(device)
ckpt = torch.load("output/happywhale_resnet34_best.pth", map_location=device)
model.load_state_dict({k: v for k, v in ckpt.items() if 'arcface' not in k}, strict=False)
model.eval()

# 3. 预处理待测样本并提取 512 维特征
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

img = Image.open("Happywhale - Whale and Dolphin Identification/test_images/000188782.jpg").convert("RGB")
tensor_img = transform(image=np.array(img))['image'].unsqueeze(0).to(device)

with torch.no_grad():
    feat = model(tensor_img).cpu().numpy()

print(f"成功提取归一化特征向量，形状: {feat.shape} | 向量模长: {np.linalg.norm(feat):.4f}")
```

---

## 🛠️ 后续进阶路线 (Roadmap)

- [ ] **视觉主干升级**：引入 ConvNeXt、Swin-Transformer 或 EVA-02 作为 Backbone，进一步提高特征表达上限。
- [ ] **分部位联合检索**：针对背鳍（Dorsal Fin）与尾叶（Fluke）分别训练独立检测头与度量头并做 Late Fusion。
- [ ] **度量损失升级**：探索 SubCenter-ArcFace（支持单类多中心，适应同个体不同拍摄姿态）。
- [ ] **高维向量数据库检索**：集成 Facebook FAISS 库，实现千万级特征库的毫秒级近似近邻检索（ANN）。

---

## 📜 许可证 (License)

本项目采用 [MIT License](LICENSE) 开源许可协议。

## 🤝 致谢 (Acknowledgments)

- 感谢 [Happywhale.com](https://happywhale.com/) 与 Kaggle 官方提供极具科学价值与挑战性的海洋生物开源数据集。
- 感谢 Ultralytics YOLOv8 社区及 PyTorch 团队提供的优秀深度学习算法框架支持。
