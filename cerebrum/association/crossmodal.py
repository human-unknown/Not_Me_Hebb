"""
stage2_crossmodal.py — Stage 2: 跨模态 Hebb 绑定
自由能原理智能体

目标: 文本和视觉同时喂入 → Hebb 共激活 → 集群同时包含文本和视觉特征 → 跨模态双向检索

机制:
  每对 (image, text_description):
    s[0:64]    = text_embedding      ← 文本通道
    s[64:160]  = V1 Gabor (96d)      ← 视觉通道
    s[160:224] = V2 Gabor (64d)      ← 视觉通道
    s[224:288] = V4 Gabor (64d)      ← 视觉通道
    s[288:330] = Color opponent (42d) ← 视觉通道

  → ClusterNetwork.learn(s)
  → 集群 centroid 同时编码文本和视觉模式

跨模态检索 (已有 recall(mask=...) 原生支持):
  文本 → 视觉: mask[0:64]=True, recall → centroid[64:330] = 视觉特征
  视觉 → 文本: mask[64:330]=True, recall → centroid[0:64] = 文本特征

验收标准:
  [ ] 文本→视觉: 输入"猫"，recall 返回的视觉特征与真实猫图像 cosine > 0.5
  [ ] 视觉→文本: 输入猫图像，recall 返回的文本 centroid 最近邻包含"猫"
  [ ] 目盲测试: 只给视觉通道，Agent 能区分不同类别

Usage:
  python stage2_crossmodal.py --n 2000 --mode train
  python stage2_crossmodal.py --n 2000 --mode eval
"""

import os
import sys
import argparse
import time
import pickle
import numpy as np
from collections import defaultdict

from cns.data_types import D, Theta
from cerebrum.limbic_system.hippocampus import ClusterNetwork, sleep_cycle, sleep_replay, _masked_cosine, _auto_mask
from brainstem_cerebellum.neuromodulatory.meta_learning import create_default_theta


# ================================================================
# Sensory Layout (D=330)
# ================================================================

TEXT_WIDTH  = 64
V1_WIDTH    = 96
V2_WIDTH    = 64
V4_WIDTH    = 45       # reduced from 64 → 45 to fit gestalt
GESTALT_WIDTH = 19     # Module A: Gestalt grouping features
COLOR_WIDTH = 42       # truncated from 64

TEXT_START,    TEXT_END    = 0,   TEXT_WIDTH                          # s[0:64]
V1_START,      V1_END      = 64,  64 + V1_WIDTH                       # s[64:160]
V2_START,      V2_END      = 160, 160 + V2_WIDTH                      # s[160:224]
V4_START,      V4_END      = 224, 224 + V4_WIDTH                      # s[224:269]
GESTALT_START, GESTALT_END = 269, 269 + GESTALT_WIDTH                 # s[269:288]
COLOR_START,   COLOR_END   = 288, 288 + COLOR_WIDTH                   # s[288:330]

assert COLOR_END == D, f"Layout ends at {COLOR_END}, expected {D}"


# ================================================================
# 中文类别描述 (每类 12 条 → 120 条足够 PCA)
# ================================================================

IMAGENETTE_DESCRIPTIONS = {
    0: [  # tench — 丁鱥鱼
        "一条丁鱥鱼", "淡水鱼", "有鳞片的鱼", "绿色的鱼",
        "在水中游动的鱼", "一条鲤鱼", "淡水湖泊中的鱼", "有胡须的鱼",
        "亚洲常见的淡水鱼", "鳞片闪亮的鱼", "棕色和绿色的鱼", "河里的鱼",
    ],
    1: [  # springer — 英国斯普林格犬
        "一只英国斯普林格犬", "棕色和白色的狗", "长耳朵的猎犬",
        "在草地上奔跑的狗", "一只精力充沛的狗", "黑白相间的猎犬",
        "一只中等体型的狗", "友善的宠物狗", "长毛的猎犬", "有斑点花纹的狗",
        "一只敏捷的狗", "忠诚的家庭犬",
    ],
    2: [  # cassette_player — 卡带播放器
        "一台卡带播放器", "老式音乐播放器", "复古的录音机",
        "八十年代的音响设备", "可以放磁带的机器", "便携式收音机",
        "带按钮的音频设备", "一个磁带机", "有扬声器的播放器",
        "银色和黑色的电子设备", "可以录音和播放的机器", "怀旧的音乐设备",
    ],
    3: [  # chain_saw — 链锯
        "一把链锯", "用于切割木材的工具", "带有旋转链条的机器",
        "伐木工人使用的电锯", "危险的切割工具", "橙色的链锯",
        "带锯齿的机器", "可以锯断树木的工具", "一把汽油驱动的锯",
        "噪音很大的工具", "户外使用的电动工具", "有锋利刀片的机器",
    ],
    4: [  # church — 教堂
        "一座教堂", "有尖塔的建筑", "宗教礼拜的场所",
        "哥特式风格的教堂", "有彩色玻璃窗的建筑", "古老的石制教堂",
        "乡村里的小教堂", "有钟楼的建筑", "基督教教堂",
        "庄严的宗教建筑", "有十字架的建筑", "有拱形门的教堂",
    ],
    5: [  # french_horn — 法国号
        "一把法国号", "铜管乐器", "金色的圆号",
        "交响乐团使用的乐器", "可以吹奏的铜管", "螺旋形的乐器",
        "一把金色的法国号", "音色柔和的管乐器", "有活塞的圆号",
        "古典音乐中常见的乐器", "闪亮的铜管乐器", "手工制作的金色圆号",
    ],
    6: [  # garbage_truck — 垃圾车
        "一辆垃圾车", "大型绿色卡车", "城市里收垃圾的车",
        "有压缩装置的卡车", "市政清洁车辆", "可以举起垃圾桶的车",
        "一辆重型卡车", "环卫工人使用的车辆", "有巨大车厢的卡车",
        "在街道上收垃圾的车", "白色和绿色的清洁车", "有液压装置的垃圾车",
    ],
    7: [  # gas_pump — 加油泵
        "一个加油泵", "加油站设备", "给汽车加油的机器",
        "红色的加油泵", "有计量表的加油设备", "可以加汽油的泵",
        "一个汽油泵", "加油站的设施", "有数字显示屏的油泵",
        "路边加油站的设备", "黑色和红色的加油机", "给车辆加油的装置",
    ],
    8: [  # golf_ball — 高尔夫球
        "一个高尔夫球", "白色带凹痕的球", "高尔夫运动用的球",
        "在草地上的小白球", "有纹理的白色球", "一个硬质塑料球",
        "带凹坑的白色球体", "可以打得很远的小球", "专业高尔夫球",
        "放在球座上的白球", "有弹性的运动球", "表面粗糙的白色球",
    ],
    9: [  # parachute — 降落伞
        "一个降落伞", "空中使用的伞状物", "跳伞运动的装备",
        "彩色的降落伞", "可以在空中打开的伞", "减速下降的装置",
        "尼龙制成的降落伞", "极限运动用的伞", "圆形的大伞",
        "空降兵使用的装备", "在蓝天中展开的降落伞", "可以折叠的伞",
    ],
}


# ================================================================
# COCO 2017 Loader
# ================================================================

COCO_VAL2017_URL = 'http://images.cocodataset.org/zips/val2017.zip'
COCO_ANNOTATIONS_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'


def _download_coco(data_dir: str):
    """下载 COCO val2017 (5k 图像, 25k 标注) 到 data_dir."""
    import zipfile
    from urllib.request import urlretrieve

    os.makedirs(data_dir, exist_ok=True)

    # 下载 val2017 图像
    img_zip = os.path.join(data_dir, 'val2017.zip')
    img_dir = os.path.join(data_dir, 'val2017')
    if not os.path.exists(img_dir):
        print(f"  Downloading COCO val2017 (1.0 GB)...")
        print(f"  This may take 5-10 minutes...")
        urlretrieve(COCO_VAL2017_URL, img_zip)
        print(f"  Extracting val2017...")
        with zipfile.ZipFile(img_zip, 'r') as zf:
            zf.extractall(data_dir)
        os.remove(img_zip)
        print(f"  val2017 ready: {img_dir}")
    else:
        print(f"  val2017 already exists: {img_dir}")

    # 下载标注
    ann_zip = os.path.join(data_dir, 'annotations_trainval2017.zip')
    ann_file = os.path.join(data_dir, 'annotations', 'captions_val2017.json')
    if not os.path.exists(ann_file):
        print(f"  Downloading COCO annotations (250 MB)...")
        urlretrieve(COCO_ANNOTATIONS_URL, ann_zip)
        print(f"  Extracting annotations...")
        with zipfile.ZipFile(ann_zip, 'r') as zf:
            zf.extractall(data_dir)
        os.remove(ann_zip)
        print(f"  Annotations ready: {ann_file}")
    else:
        print(f"  Annotations already exist: {ann_file}")

    return img_dir, ann_file


# ================================================================
# IT Layer — 下颞叶视觉原型群体编码
class COCOLoader:
    """COCO 2017 数据加载器 — 图像 + 英文标注。

    管线:
      1. 下载 val2017 (如需要)
      2. 加载图像 + captions JSON
      3. Gabor 编码全部图像 (V1+V2+V4+Color)
      4. Sentence-transformer 编码全部 captions → PCA → 64d
      5. 提供 get_pair(idx) → (visual_features, text_embedding, caption, image_id)
    """

    def __init__(self, n_images: int = 5000, image_size: int = 128,
                 use_categories: bool = True):
        import json
        from PIL import Image

        base = os.path.dirname(__file__)
        data_dir = os.path.join(base, '.cache', 'coco')
        self.img_dir, self.ann_file = _download_coco(data_dir)
        self.use_categories = use_categories

        # 加载实例标注 (类别)
        instances_file = os.path.join(data_dir, 'annotations',
                                       'instances_val2017.json')
        with open(instances_file, 'r') as f:
            instances_data = json.load(f)

        # 类别映射: id → name
        self.category_names = {}
        for cat in instances_data['categories']:
            self.category_names[cat['id']] = cat['name']

        # 每张图像的类别列表
        self.image_to_categories = defaultdict(set)
        for ann in instances_data['annotations']:
            self.image_to_categories[ann['image_id']].add(ann['category_id'])

        if use_categories:
            print(f"  COCO: {len(self.category_names)} categories "
                  f"(using category labels for text)")
            self.image_to_captions = {}  # Not used
            # Still need images list — load from captions file
            with open(self.ann_file, 'r') as f:
                self.coco_data = json.load(f)
        else:
            # 加载 caption 标注
            print(f"  Loading COCO captions...")
            with open(self.ann_file, 'r') as f:
                self.coco_data = json.load(f)
            self.image_to_captions = defaultdict(list)
            for ann in self.coco_data['annotations']:
                self.image_to_captions[ann['image_id']].append(ann['caption'])

        # 图像列表
        self.images = self.coco_data['images'][:n_images]
        self.n_images = len(self.images)

        # ---- 视觉特征编码 (Gabor V1+V2+V4+Gestalt+Color) ----
        from layer0_visual import GaborFilterBank
        from layer0_gestalt import compute_gestalt_from_image
        self._gabor = GaborFilterBank(image_size=image_size, grid_size=4)

        cache_dir = os.path.join(base, '.cache')
        cache_key = (f'coco_val2017_{self.n_images}_sz{image_size}'
                     f'_gabor_v1v2v4gestaltcolor')
        cache_path = os.path.join(cache_dir, f'{cache_key}.npz')

        if os.path.exists(cache_path):
            print(f"  Loading cached Gabor encodings...")
            cached = np.load(cache_path)
            self.encodings_v1 = cached['v1']
            self.encodings_v2 = cached['v2']
            self.encodings_v4 = cached['v4']
            self.encodings_gestalt = cached.get('gestalt', None)
            self.encodings_color = cached['color']
            if self.encodings_gestalt is None:
                # Old cache without gestalt → compute on the fly below
                self.encodings_gestalt = np.zeros(
                    (self.n_images, GESTALT_WIDTH), dtype=np.float32)
        else:
            print(f"  Encoding {self.n_images} COCO images [Gabor V1+V2+V4+Gestalt+Color]...")
            self.encodings_v1 = np.zeros((self.n_images, V1_WIDTH), dtype=np.float32)
            self.encodings_v2 = np.zeros((self.n_images, V2_WIDTH), dtype=np.float32)
            self.encodings_v4 = np.zeros((self.n_images, V4_WIDTH), dtype=np.float32)
            self.encodings_gestalt = np.zeros((self.n_images, GESTALT_WIDTH), dtype=np.float32)
            self.encodings_color = np.zeros((self.n_images, 64), dtype=np.float32)

            for i, img_info in enumerate(self.images):
                if (i + 1) % 500 == 0:
                    print(f"    Encoding {i+1}/{self.n_images}...")
                img_path = os.path.join(self.img_dir, img_info['file_name'])
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_np = np.array(img, dtype=np.uint8)
                    v1_raw = self._gabor.encode(img_np, learn=True)
                    v2_raw = self._gabor.encode_v2(img_np)
                    v4_raw = self._gabor.encode_v4(img_np)
                    color_raw = self._gabor.encode_color(img_np)
                    # Module A: gestalt features
                    gestalt_raw = compute_gestalt_from_image(
                        img_np, self._gabor)

                    self.encodings_v1[i, :] = v1_raw[:V1_WIDTH] if len(v1_raw) >= V1_WIDTH else np.pad(v1_raw, (0, V1_WIDTH - len(v1_raw)))
                    self.encodings_v2[i, :] = v2_raw[:V2_WIDTH] if len(v2_raw) >= V2_WIDTH else np.pad(v2_raw, (0, V2_WIDTH - len(v2_raw)))
                    self.encodings_v4[i, :] = v4_raw[:V4_WIDTH] if len(v4_raw) >= V4_WIDTH else np.pad(v4_raw, (0, V4_WIDTH - len(v4_raw)))
                    self.encodings_gestalt[i, :] = gestalt_raw[:GESTALT_WIDTH] if len(gestalt_raw) >= GESTALT_WIDTH else np.pad(gestalt_raw, (0, GESTALT_WIDTH - len(gestalt_raw)))
                    self.encodings_color[i, :] = color_raw[:64] if len(color_raw) >= 64 else np.pad(color_raw, (0, 64 - len(color_raw)))
                except Exception:
                    pass

            for arr in [self.encodings_v1, self.encodings_v2,
                         self.encodings_v4, self.encodings_gestalt,
                         self.encodings_color]:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                arr[:] = arr / (norms + 1e-8)

            np.savez_compressed(cache_path,
                               v1=self.encodings_v1, v2=self.encodings_v2,
                               v4=self.encodings_v4,
                               gestalt=self.encodings_gestalt,
                               color=self.encodings_color)
            print(f"  Cached: {cache_path}")

        print(f"  V1: {self.encodings_v1.shape}, V2: {self.encodings_v2.shape}, "
              f"V4: {self.encodings_v4.shape}, "
              f"Gestalt: {self.encodings_gestalt.shape}, "
              f"Color: {self.encodings_color.shape}")

        # 每张图像分配 captions/类别 索引列表
        self.pairs = []  # [(image_idx, caption_text), ...]

        if use_categories:
            # 类别模式: 每张图 × 每个类别 → 一对
            # 文本: "a photo of a {category}" 或 "a photo containing {category}"
            for i, img_info in enumerate(self.images):
                cats = self.image_to_categories.get(img_info['id'], set())
                for cat_id in cats:
                    cat_name = self.category_names.get(cat_id, 'object')
                    # 生成多条描述来模拟 5 captions 的多样性
                    texts = [
                        f"a photo of a {cat_name}",
                        f"a {cat_name} in the scene",
                        f"an image containing a {cat_name}",
                    ]
                    for text in texts:
                        self.pairs.append((i, text))

            # 统计
            unique_texts = set(cap for _, cap in self.pairs)
            print(f"  Category mode: {len(unique_texts)} unique texts, "
                  f"{len(self.pairs)} total pairs")
        else:
            for i, img_info in enumerate(self.images):
                captions = self.image_to_captions.get(img_info['id'], [])
                for cap in captions[:5]:
                    self.pairs.append((i, cap))

        self.n_pairs = len(self.pairs)
        print(f"  COCO: {self.n_images} images, "
              f"{self.n_pairs} text pairs, "
              f"{len(set(cap for _, cap in self.pairs))} unique texts")

    def get_visual(self, idx: int) -> dict:
        """获取第 idx 张图像的视觉特征"""
        result = {
            'v1': self.encodings_v1[idx],
            'v2': self.encodings_v2[idx],
            'v4': self.encodings_v4[idx],
            'color': self.encodings_color[idx],
        }
        if hasattr(self, 'encodings_gestalt') and self.encodings_gestalt is not None:
            result['gestalt'] = self.encodings_gestalt[idx]
        return result

    def get_all_captions(self) -> list[str]:
        """获取所有文本标签 (用于 PCA 拟合)"""
        return [cap for _, cap in self.pairs]

    def get_image_texts(self, img_idx: int) -> list[str]:
        """获取某张图像的所有文本标签"""
        img_id = self.images[img_idx]['id']
        if self.use_categories:
            cats = self.image_to_categories.get(img_id, set())
            result = []
            for cat_id in cats:
                cat_name = self.category_names.get(cat_id, 'object')
                result.append(f"a photo of a {cat_name}")
            return result if result else ['an image']
        else:
            return self.image_to_captions.get(img_id, ['an image'])


# ================================================================
# Text Encoder
# ================================================================

class TextEncoder:
    """文本编码器: sentence-transformer → PCA → 64d 语义向量。

    使用 all-MiniLM-L6-v2 (384d) → PCA → TEXT_WIDTH (64d)
    PCA 拟合在类别描述上 (120 条)。
    """

    def __init__(self, descriptions: dict[int, list[str]],
                 pca_components: int = TEXT_WIDTH):
        from sentence_transformers import SentenceTransformer

        self.pca_components = pca_components
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # 收集所有描述 + 类别名称
        all_texts = []
        for class_id, descs in descriptions.items():
            all_texts.extend(descs)

        print(f"  Encoding {len(all_texts)} text descriptions...")
        full = self.model.encode(all_texts, show_progress_bar=False,
                                 batch_size=32)  # (N, 384)

        # PCA 降维到 64d
        from sklearn.decomposition import PCA
        n_pca = min(pca_components, len(all_texts), full.shape[1])
        self.pca = PCA(n_components=n_pca, random_state=42)
        self.embeddings = self.pca.fit_transform(full).astype(np.float32)

        # L2 归一化
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings /= (norms + 1e-8)

        print(f"  Text PCA: {full.shape[1]}d → {n_pca}d, "
              f"explained={self.pca.explained_variance_ratio_.sum():.1%}")

        # 建立文本 → 嵌入 映射
        self.text_to_emb = {}
        for i, text in enumerate(all_texts):
            self.text_to_emb[text] = self.embeddings[i]

        self.all_texts = all_texts

    def encode(self, text: str) -> np.ndarray:
        """单条文本 → 64d 语义向量"""
        if text in self.text_to_emb:
            return self.text_to_emb[text].copy()

        full = self.model.encode([text])[0]
        emb = self.pca.transform(full.reshape(1, -1))[0].astype(np.float32)
        if len(emb) < self.pca_components:
            p = np.zeros(self.pca_components, dtype=np.float32)
            p[:len(emb)] = emb
            emb = p
        # L2 归一化
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb /= norm
        return emb

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """批量编码 → (N, 64)"""
        full = self.model.encode(texts, show_progress_bar=False,
                                 batch_size=32)
        emb = self.pca.transform(full).astype(np.float32)
        if emb.shape[1] < self.pca_components:
            p = np.zeros((emb.shape[0], self.pca_components), dtype=np.float32)
            p[:, :emb.shape[1]] = emb
            emb = p
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb /= (norms + 1e-8)
        return emb


# ================================================================
# Sensory Vector Construction
# ================================================================

def build_crossmodal_sensory(text_emb: np.ndarray,
                               v1_feat: np.ndarray,
                               v2_feat: np.ndarray = None,
                               v4_feat: np.ndarray = None,
                               color_feat: np.ndarray = None,
                               gestalt_feat: np.ndarray = None) -> np.ndarray:
    """构建跨模态感知向量 (D=330)。

    s[0:64]    = text embedding
    s[64:160]  = V1 Gabor
    s[160:224] = V2 Gabor
    s[224:269] = V4 Gabor
    s[269:288] = Gestalt grouping (Module A)
    s[288:330] = Color opponent
    """
    s = np.zeros(D, dtype=np.float32)

    # Text
    flen = min(len(text_emb), TEXT_WIDTH)
    s[TEXT_START:TEXT_START + flen] = text_emb[:flen]

    # V1
    if v1_feat is not None:
        flen = min(len(v1_feat), V1_WIDTH)
        s[V1_START:V1_START + flen] = v1_feat[:flen]

    # V2
    if v2_feat is not None:
        flen = min(len(v2_feat), V2_WIDTH)
        s[V2_START:V2_START + flen] = v2_feat[:flen]

    # V4
    if v4_feat is not None:
        flen = min(len(v4_feat), V4_WIDTH)
        s[V4_START:V4_START + flen] = v4_feat[:flen]

    # Gestalt grouping (Module A)
    if gestalt_feat is not None:
        flen = min(len(gestalt_feat), GESTALT_WIDTH)
        s[GESTALT_START:GESTALT_START + flen] = gestalt_feat[:flen]

    # Color
    if color_feat is not None:
        flen = min(len(color_feat), COLOR_WIDTH)
        s[COLOR_START:COLOR_START + flen] = color_feat[:flen]

    return s


# ================================================================
# Mask Helpers
# ================================================================

def make_text_mask() -> np.ndarray:
    """文本通道 mask: s[0:64]=True, 其余=False"""
    mask = np.zeros(D, dtype=bool)
    mask[TEXT_START:TEXT_END] = True
    return mask


def make_visual_mask() -> np.ndarray:
    """视觉通道 mask: s[64:330]=True, 其余=False"""
    mask = np.zeros(D, dtype=bool)
    mask[V1_START:COLOR_END] = True
    return mask


def _get_visual_parts(vis: dict) -> tuple:
    """从 vis dict 提取 (v1, v2, v4, gestalt, color) 元组"""
    return (
        vis.get('v1'),
        vis.get('v2'),
        vis.get('v4'),
        vis.get('gestalt'),
        vis.get('color'),
    )


def _build_s_from_vis(text_emb: np.ndarray, vis: dict) -> np.ndarray:
    """从 vis dict 构建 crossmodal sensory vector (便捷函数)"""
    v1, v2, v4, gestalt, color = _get_visual_parts(vis)
    return build_crossmodal_sensory(text_emb, v1, v2, v4, color, gestalt)


def make_full_mask() -> np.ndarray:
    """全通道 mask"""
    return np.ones(D, dtype=bool)


# ================================================================
# Masked Recall (包装层, 使用 auto_mask 机制)
# ================================================================

def masked_recall(net: ClusterNetwork, s: np.ndarray,
                   channel_mask: np.ndarray,
                   threshold_override: float = None) -> tuple:
    """使用指定通道 mask 进行 recall。

    策略: 将非查询通道置零 → auto_mask 自动忽略零维度 →
          只在指定通道上进行余弦匹配。

    Args:
        net: ClusterNetwork
        s: (D,) full sensory vector (查询通道有值，其余可为任意值)
        channel_mask: (D,) bool — True=参与匹配，False=忽略
        threshold_override: 覆盖默认阈值 (视觉查询用更低阈值)

    Returns:
        (matched_cluster_or_None, best_similarity)
    """
    if net.n_clusters == 0:
        return None, 0.0

    threshold = threshold_override if threshold_override is not None \
        else net.theta.cluster_threshold

    # 创建查询向量: 非查询通道置零
    s_masked = s.copy()
    s_masked[~channel_mask] = 0.0

    h = np.tanh(s_masked + 1e-8)
    mask = _auto_mask(s_masked)

    best_sim, best_c = -1.0, None
    for c in net.clusters:
        sim = _masked_cosine(h, c.centroid, mask)
        if sim > best_sim:
            best_sim, best_c = sim, c

    if best_c is not None and best_sim >= threshold:
        return best_c, best_sim
    return None, best_sim


# ================================================================
# Training
# ================================================================

def train_crossmodal(venv, text_encoder: TextEncoder,
                      n_images: int, seed: int = 42) -> dict:
    """训练跨模态 Hebb 绑定。

    每张图像:
      1. 随机选一条该类别的中文描述
      2. 编码文本 → 64d
      3. 加载视觉特征 → V1+V2+V4+Color
      4. 拼接 → s (330d)
      5. ClusterNetwork.learn(s) → Hebb 共激活

    Returns:
        {'net': ClusterNetwork, 'pairs': list, 'F_ema': float}
    """
    rng = np.random.default_rng(seed)

    # 创建单个 ClusterNetwork (跨模态绑定只需要一层)
    theta = create_default_theta()
    theta.cluster_threshold = 0.45  # 更高阈值 → 更大集群 → 更平滑的视觉质心
    theta.learn_rate_l0 = 0.02
    theta.decay_rate = 0.003
    net = ClusterNetwork(theta, hash_offset=0)

    order = rng.permutation(n_images)
    t0 = time.perf_counter()

    # 为每张图像配多个描述 (每张图 5 条，模拟 COCO 5 captions)
    class_to_descs = {}
    for class_id in range(venv.n_classes):
        class_to_descs[class_id] = IMAGENETTE_DESCRIPTIONS.get(
            class_id, [venv.label_names[class_id]])

    # 每个图像重复 N_CAPTIONS 次，每次配不同描述
    N_CAPTIONS = 5  # 模拟 COCO: 每张图 5 句描述

    pairs = []  # 记录训练配对
    n_train_pairs = n_images * N_CAPTIONS

    print(f"\n{'='*64}")
    print(f"  Stage 2: Cross-modal Hebb Binding")
    print(f"{'='*64}")
    print(f"  n_images={n_images}, n_captions_per_image={N_CAPTIONS}")
    print(f"  Total training pairs: {n_train_pairs}")
    print(f"  threshold={theta.cluster_threshold}")
    print(f"  Layout: text[0:64] + V1[64:160] + V2[160:224] + "
          f"V4[224:288] + Color[288:330]")

    # 扩展训练顺序：每张图像重复 N_CAPTIONS 次
    expanded_order = np.repeat(order, N_CAPTIONS)
    rng.shuffle(expanded_order)  # 打乱避免连续相同图像

    for step, idx in enumerate(expanded_order):
        label = int(venv.labels[idx])

        # 选描述 (不重复选择以避免同一图像配不同描述时冲突)
        descs = class_to_descs[label]
        desc = descs[step % len(descs)]  # 循环选择不同描述
        text_emb = text_encoder.encode(desc)

        # 视觉特征
        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)
        color_feat = None
        if getattr(venv, 'encodings_color', None) is not None:
            color_feat = venv.encodings_color[idx].copy()

        # 构建跨模态感知向量
        s = build_crossmodal_sensory(
            text_emb, v1_feat, v2_feat, v4_feat, color_feat)

        # Hebb 学习
        net.learn(s)

        pairs.append({
            'image_idx': idx,
            'label': label,
            'class_name': venv.label_names[label],
            'text': desc,
        })

        # Sleep
        if (step + 1) % 500 == 0:
            n_removed = sleep_cycle(net, net.theta)
            if n_removed > 0:
                print(f"  [Sleep] removed {n_removed}, "
                      f"{net.n_clusters} remain")

        # Logging
        if (step + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            ips = (step + 1) / max(elapsed, 0.001)
            print(f"  {step+1}/{n_images} ({ips:.0f} img/s) | "
                  f"clusters={net.n_clusters}")

    elapsed = time.perf_counter() - t0
    print(f"  Training complete: {elapsed:.1f}s, "
          f"{net.n_clusters} clusters formed")

    return {
        'net': net,
        'pairs': pairs,
        'n_clusters': net.n_clusters,
    }


# ================================================================
# Evaluation
# ================================================================

def evaluate_crossmodal(result: dict, venv, text_encoder: TextEncoder,
                          n_test: int = 500) -> dict:
    """评估跨模态双向检索。

    1. 文本 → 视觉: 文本查询 → 检索集群 → 比较视觉特征
    2. 视觉 → 文本: 视觉查询 → 检索集群 → 比较文本特征
    3. 目盲测试: 仅视觉通道 → 分类准确率
    """
    net = result['net']
    if net.n_clusters == 0:
        print("  No clusters — cannot evaluate")
        return {}

    print(f"\n{'='*64}")
    print(f"  Cross-modal Evaluation")
    print(f"{'='*64}")
    print(f"  Clusters: {net.n_clusters}")
    print(f"  Test samples: {n_test}")

    # ================================================================
    # 准备: 为每个集群标注主类别
    # ================================================================
    cluster_labels = {}  # cluster_id → label
    cluster_texts = {}   # cluster_id → [texts]
    cluster_visuals = defaultdict(list)  # cluster_id → [visual_features]

    text_mask = make_text_mask()
    visual_mask = make_visual_mask()

    # 对训练数据重新 recall 以标注集群
    for idx in range(min(1000, venv.n_images)):
        label = int(venv.labels[idx])
        descs = IMAGENETTE_DESCRIPTIONS.get(
            label, [venv.label_names[label]])
        desc = descs[np.random.default_rng(idx).integers(0, len(descs))]
        text_emb = text_encoder.encode(desc)

        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)
        color_feat = None
        if getattr(venv, 'encodings_color', None) is not None:
            color_feat = venv.encodings_color[idx].copy()

        s_full = build_crossmodal_sensory(
            text_emb, v1_feat, v2_feat, v4_feat, color_feat)

        # 全通道 recall 找匹配集群
        c, sim = masked_recall(net, s_full, make_full_mask())
        if c is not None:
            cid = id(c)
            cluster_labels[cid] = cluster_labels.get(cid, []) + [label]
            cluster_texts[cid] = cluster_texts.get(cid, []) + [desc]
            # 存储视觉部分
            vis_part = s_full[V1_START:COLOR_END].copy()
            cluster_visuals[cid].append(vis_part)

    # 为主集群确定类别
    cluster_to_class = {}
    for cid, labels in cluster_labels.items():
        if len(labels) < 3:
            continue
        class_counts = defaultdict(int)
        for l in labels:
            class_counts[l] += 1
        majority_class = max(class_counts, key=class_counts.get)
        purity = class_counts[majority_class] / len(labels)
        if purity >= 0.3:  # 至少 30% 纯度
            cluster_to_class[cid] = {
                'class': majority_class,
                'class_name': venv.label_names[majority_class],
                'purity': purity,
                'n_samples': len(labels),
            }

    n_labeled = len(cluster_to_class)
    print(f"  Labeled clusters: {n_labeled}/{net.n_clusters} "
          f"({n_labeled/max(1,net.n_clusters):.0%})")

    # ================================================================
    # Test 1: 文本 → 视觉检索
    # ================================================================
    print(f"\n  --- Test 1: Text → Visual Retrieval ---")

    t2v_sims = defaultdict(list)  # class → [cosine: centroid vs best-matching image]
    t2v_hits = defaultdict(int)   # class → correct hits
    t2v_total = defaultdict(int)

    rng = np.random.default_rng(99)
    n_t2v = min(n_test, venv.n_images)

    for _ in range(n_t2v):
        idx = rng.integers(0, venv.n_images)
        true_label = int(venv.labels[idx])
        true_class = venv.label_names[true_label]

        # 随机选一条该类别的描述作为文本查询
        descs = IMAGENETTE_DESCRIPTIONS.get(true_label, [true_class])
        query_text = descs[rng.integers(0, len(descs))]
        query_emb = text_encoder.encode(query_text)

        # 构建查询: 只有文本通道有值
        s_query = build_crossmodal_sensory(
            query_emb,
            np.zeros(V1_WIDTH, dtype=np.float32),
            np.zeros(V2_WIDTH, dtype=np.float32),
            np.zeros(V4_WIDTH, dtype=np.float32),
            np.zeros(COLOR_WIDTH, dtype=np.float32),
        )

        # 文本 mask recall
        c, sim = masked_recall(net, s_query, text_mask)
        if c is not None:
            # 检索到的视觉特征 → 在数据集中找最匹配的图像
            retrieved_visual = c.centroid[V1_START:COLOR_END]

            # 在 200 张随机图像中找最高 cosine
            n_search = min(200, venv.n_images)
            search_indices = rng.choice(venv.n_images, n_search, replace=False)
            best_cos = -1.0
            best_match_label = -1
            for si in search_indices:
                v1_f = venv.encodings[si]
                v2_f = (venv.encodings_v2[si].copy()
                        if venv.encodings_v2 is not None else None)
                v4_f = (venv.encodings_v4[si].copy()
                        if venv.encodings_v4 is not None else None)
                col_f = (venv.encodings_color[si].copy()
                         if getattr(venv, 'encodings_color', None) is not None
                         else None)
                s_vis = build_crossmodal_sensory(
                    np.zeros(TEXT_WIDTH, dtype=np.float32),
                    v1_f, v2_f, v4_f, col_f)
                gt_vis = s_vis[V1_START:COLOR_END]
                cos = float(np.dot(retrieved_visual, gt_vis) / (
                    np.linalg.norm(retrieved_visual) * np.linalg.norm(gt_vis) + 1e-8))
                if cos > best_cos:
                    best_cos = cos
                    best_match_label = int(venv.labels[si])

            t2v_sims[true_class].append(best_cos)

            # 检查检索到的集群是否匹配正确类别
            cid = id(c)
            if cid in cluster_to_class:
                t2v_total[true_class] += 1
                if cluster_to_class[cid]['class'] == true_label:
                    t2v_hits[true_class] += 1

    # 汇总
    all_t2v_sims = [s for sims in t2v_sims.values() for s in sims]
    avg_t2v_sim = float(np.mean(all_t2v_sims)) if all_t2v_sims else 0.0

    t2v_accuracy = {}
    for cls_name in venv.label_names:
        total = t2v_total.get(cls_name, 0)
        if total > 0:
            t2v_accuracy[cls_name] = t2v_hits.get(cls_name, 0) / total
        else:
            t2v_accuracy[cls_name] = 0.0
    avg_t2v_acc = float(np.mean(list(t2v_accuracy.values())))

    print(f"  Avg cosine (retrieved vs GT visual): {avg_t2v_sim:.4f}")
    print(f"  Avg retrieval accuracy: {avg_t2v_acc:.3f}")
    print(f"  Per-class accuracy:")
    for name in venv.label_names:
        bar = '█' * int(t2v_accuracy[name] * 20)
        print(f"    {name:<16}: {t2v_accuracy[name]:.3f} {bar}")

    # ================================================================
    # Test 2: 视觉 → 文本检索
    # ================================================================
    print(f"\n  --- Test 2: Visual → Text Retrieval ---")

    v2t_top1_hits = defaultdict(int)
    v2t_top5_hits = defaultdict(int)
    v2t_total = defaultdict(int)
    VISUAL_THRESHOLD = 0.25  # 视觉查询用更低的阈值 (主阈值 0.45 的 ~55%)

    for _ in range(n_t2v):
        idx = rng.integers(0, venv.n_images)
        true_label = int(venv.labels[idx])
        true_class = venv.label_names[true_label]

        # 获取视觉特征
        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)
        color_feat = (venv.encodings_color[idx].copy()
                      if getattr(venv, 'encodings_color', None) is not None
                      else None)

        # 构建查询: 只有视觉通道有值
        s_query = build_crossmodal_sensory(
            np.zeros(TEXT_WIDTH, dtype=np.float32),
            v1_feat, v2_feat, v4_feat, color_feat)

        # 视觉 mask recall — 使用更低阈值
        c, sim = masked_recall(net, s_query, visual_mask,
                                threshold_override=VISUAL_THRESHOLD)
        if c is not None:
            # 检索到的文本特征
            retrieved_text = c.centroid[TEXT_START:TEXT_END]

            # 找 top-5 最近邻文本
            nearest = []
            for text, emb in text_encoder.text_to_emb.items():
                cos = float(np.dot(retrieved_text, emb) / (
                    np.linalg.norm(retrieved_text) * np.linalg.norm(emb) + 1e-8))
                nearest.append((text, cos))
            nearest.sort(key=lambda x: -x[1])

            v2t_total[true_class] += 1

            # Top-1: 最近文本是否属于正确类别
            top1_text = nearest[0][0]
            for cid, descs in IMAGENETTE_DESCRIPTIONS.items():
                if top1_text in descs:
                    if cid == true_label:
                        v2t_top1_hits[true_class] += 1
                    break

            # Top-5: 前 5 个文本中是否有正确类别
            for text, _ in nearest[:5]:
                found = False
                for cid, descs in IMAGENETTE_DESCRIPTIONS.items():
                    if text in descs and cid == true_label:
                        v2t_top5_hits[true_class] += 1
                        found = True
                        break
                if found:
                    break

    # 汇总 v2t
    v2t_top1_accuracy = {}
    v2t_top5_accuracy = {}
    for cls_name in venv.label_names:
        total = v2t_total.get(cls_name, 0)
        if total > 0:
            v2t_top1_accuracy[cls_name] = v2t_top1_hits.get(cls_name, 0) / total
            v2t_top5_accuracy[cls_name] = v2t_top5_hits.get(cls_name, 0) / total
        else:
            v2t_top1_accuracy[cls_name] = 0.0
            v2t_top5_accuracy[cls_name] = 0.0
    avg_v2t_top1 = float(np.mean(list(v2t_top1_accuracy.values())))
    avg_v2t_top5 = float(np.mean(list(v2t_top5_accuracy.values())))

    print(f"  Avg retrieval accuracy (top-1 / top-5): "
          f"{avg_v2t_top1:.3f} / {avg_v2t_top5:.3f}")
    print(f"  Per-class accuracy (top-1 | top-5):")
    for name in venv.label_names:
        bar1 = '█' * int(v2t_top1_accuracy[name] * 20)
        bar5 = '█' * int(v2t_top5_accuracy[name] * 10)
        print(f"    {name:<16}: {v2t_top1_accuracy[name]:.3f} {bar1} | "
              f"top5={v2t_top5_accuracy[name]:.3f} {bar5}")

    # ================================================================
    # Test 3: 目盲测试 (仅视觉通道分类)
    # ================================================================
    print(f"\n  --- Test 3: Blind Test (Vision-only classification) ---")

    blind_hits = defaultdict(int)
    blind_total = defaultdict(int)

    n_blind = min(n_test, venv.n_images)
    indices = rng.choice(venv.n_images, n_blind, replace=False)

    for idx in indices:
        true_label = int(venv.labels[idx])
        true_class = venv.label_names[true_label]

        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)
        color_feat = (venv.encodings_color[idx].copy()
                      if getattr(venv, 'encodings_color', None) is not None
                      else None)

        s_query = build_crossmodal_sensory(
            np.zeros(TEXT_WIDTH, dtype=np.float32),
            v1_feat, v2_feat, v4_feat, color_feat)

        c, sim = masked_recall(net, s_query, visual_mask,
                                threshold_override=VISUAL_THRESHOLD)
        if c is not None:
            cid = id(c)
            blind_total[true_class] += 1
            if cid in cluster_to_class:
                if cluster_to_class[cid]['class'] == true_label:
                    blind_hits[true_class] += 1

    blind_accuracy = {}
    for cls_name in venv.label_names:
        total = blind_total.get(cls_name, 0)
        if total > 0:
            blind_accuracy[cls_name] = blind_hits.get(cls_name, 0) / total
        else:
            blind_accuracy[cls_name] = 0.0
    avg_blind_acc = float(np.mean(list(blind_accuracy.values())))

    print(f"  Avg blind accuracy: {avg_blind_acc:.3f}")
    print(f"  Per-class accuracy:")
    for name in venv.label_names:
        bar = '█' * int(blind_accuracy[name] * 20)
        print(f"    {name:<16}: {blind_accuracy[name]:.3f} {bar}")

    # ================================================================
    # Test 4: 集群示例展示
    # ================================================================
    print(f"\n  --- Test 4: Cluster Examples ---")

    sorted_clusters = sorted(cluster_to_class.items(),
                             key=lambda x: -x[1]['purity'])

    for rank, (cid, info) in enumerate(sorted_clusters[:10]):
        texts = cluster_texts.get(cid, [])[:3]
        print(f"  [{rank+1}] {info['class_name']:<16} "
              f"purity={info['purity']:.2f} n={info['n_samples']} "
              f"texts={texts}")

    return {
        't2v_avg_cosine': avg_t2v_sim,
        't2v_avg_accuracy': avg_t2v_acc,
        't2v_per_class': t2v_accuracy,
        'v2t_top1_accuracy': avg_v2t_top1,
        'v2t_top5_accuracy': avg_v2t_top5,
        'v2t_per_class_top1': v2t_top1_accuracy,
        'v2t_per_class_top5': v2t_top5_accuracy,
        'blind_avg_accuracy': avg_blind_acc,
        'blind_per_class': blind_accuracy,
        'n_labeled_clusters': n_labeled,
        'top_clusters': sorted_clusters[:10],
    }


# ================================================================
# 交互式跨模态检索 Demo
# ================================================================

def interactive_demo(result: dict, venv, text_encoder: TextEncoder):
    """交互式跨模态检索演示"""
    net = result['net']
    if net.n_clusters == 0:
        print("No clusters to demo.")
        return

    text_mask = make_text_mask()
    visual_mask = make_visual_mask()

    print(f"\n{'='*64}")
    print(f"  Interactive Cross-modal Demo")
    print(f"{'='*64}")
    print(f"  Type a Chinese description to retrieve visual features")
    print(f"  Type 'img N' to retrieve text for image N")
    print(f"  Type 'list' to see available descriptions")
    print(f"  Type 'quit' to exit")
    print()

    while True:
        try:
            cmd = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd.lower() == 'quit':
            break

        if cmd.lower() == 'list':
            print("  Available descriptions:")
            for class_id, descs in IMAGENETTE_DESCRIPTIONS.items():
                name = venv.label_names[class_id]
                print(f"    [{name}]")
                for d in descs[:3]:
                    print(f"      - {d}")
            continue

        if cmd.lower().startswith('img '):
            try:
                img_idx = int(cmd.split()[1])
                if 0 <= img_idx < venv.n_images:
                    v1_feat = venv.encodings[img_idx]
                    v2_feat = (venv.encodings_v2[img_idx].copy()
                               if venv.encodings_v2 is not None else None)
                    v4_feat = (venv.encodings_v4[img_idx].copy()
                               if venv.encodings_v4 is not None else None)
                    color_feat = (venv.encodings_color[img_idx].copy()
                                  if getattr(venv, 'encodings_color', None) is not None
                                  else None)
                    s_query = build_crossmodal_sensory(
                        np.zeros(TEXT_WIDTH, dtype=np.float32),
                        v1_feat, v2_feat, v4_feat, color_feat)
                    c, sim = masked_recall(net, s_query, visual_mask)
                    if c is not None:
                        retrieved_text = c.centroid[TEXT_START:TEXT_END]
                        # 找最近邻文本
                        nearest = []
                        for text, emb in text_encoder.text_to_emb.items():
                            cos = float(np.dot(retrieved_text, emb) / (
                                np.linalg.norm(retrieved_text) * np.linalg.norm(emb) + 1e-8))
                            nearest.append((text, cos))
                        nearest.sort(key=lambda x: -x[1])
                        print(f"  Image {img_idx} ({venv.label_names[int(venv.labels[img_idx])]}):")
                        print(f"    Match similarity: {sim:.3f}")
                        print(f"    Top-5 nearest texts:")
                        for text, cos in nearest[:5]:
                            print(f"      {cos:.3f} | {text}")
                    else:
                        print(f"  No cluster matched (best sim={sim:.3f})")
                else:
                    print(f"  Invalid index. Range: [0, {venv.n_images-1}]")
            except (ValueError, IndexError):
                print("  Usage: img <index>")
            continue

        # Text → Visual query
        query_emb = text_encoder.encode(cmd)
        s_query = build_crossmodal_sensory(
            query_emb,
            np.zeros(V1_WIDTH, dtype=np.float32),
            np.zeros(V2_WIDTH, dtype=np.float32),
            np.zeros(V4_WIDTH, dtype=np.float32),
            np.zeros(COLOR_WIDTH, dtype=np.float32),
        )
        c, sim = masked_recall(net, s_query, text_mask)
        if c is not None:
            retrieved_visual = c.centroid[V1_START:COLOR_END]
            print(f"  Query: '{cmd}'")
            print(f"    Match similarity: {sim:.3f}")

            # 在真实图像中找最近邻视觉特征
            n_search = min(200, venv.n_images)
            best_vis_sim = -1.0
            best_vis_idx = -1
            for i in range(n_search):
                v1_f = venv.encodings[i]
                v2_f = (venv.encodings_v2[i].copy()
                        if venv.encodings_v2 is not None else None)
                v4_f = (venv.encodings_v4[i].copy()
                        if venv.encodings_v4 is not None else None)
                col_f = (venv.encodings_color[i].copy()
                         if getattr(venv, 'encodings_color', None) is not None
                         else None)
                gt_vis = build_crossmodal_sensory(
                    np.zeros(TEXT_WIDTH, dtype=np.float32),
                    v1_f, v2_f, v4_f, col_f)[V1_START:COLOR_END]
                cos = float(np.dot(retrieved_visual, gt_vis) / (
                    np.linalg.norm(retrieved_visual) * np.linalg.norm(gt_vis) + 1e-8))
                if cos > best_vis_sim:
                    best_vis_sim = cos
                    best_vis_idx = i

            if best_vis_idx >= 0:
                print(f"    Best visual match: image {best_vis_idx} "
                      f"({venv.label_names[int(venv.labels[best_vis_idx])]}), "
                      f"cosine={best_vis_sim:.3f}")
        else:
            print(f"  Query: '{cmd}' → No cluster matched "
                  f"(best sim={sim:.3f})")


# ================================================================
# COCO Training & Evaluation
# ================================================================

def _quick_eval_epoch(net, coco: COCOLoader, text_encoder,
                       n_sample: int = 100, rng=None) -> dict:
    """Per-epoch quick eval: sample 100 Visual→Text queries, track GT rank."""
    if rng is None:
        rng = np.random.default_rng(42)

    visual_mask = make_visual_mask()
    VISUAL_THRESHOLD = 0.20  # Lower threshold during training to get more hits

    # Build text corpus for ranking (all unique texts from category mode)
    all_texts = list(dict.fromkeys(cap for _, cap in coco.pairs[:2000]))
    if len(all_texts) > 500:
        all_texts = all_texts[:500]

    cap_embs = text_encoder.encode_batch(all_texts)

    ranks = []
    hits = 0
    for _ in range(n_sample):
        img_idx = rng.integers(0, coco.n_images)
        vis = coco.get_visual(img_idx)
        s_query = _build_s_from_vis(
            np.zeros(TEXT_WIDTH, dtype=np.float32), vis)

        c, sim = masked_recall(net, s_query, visual_mask,
                                threshold_override=VISUAL_THRESHOLD)
        if c is not None:
            hits += 1
            retrieved_text = c.centroid[TEXT_START:TEXT_END]
            sims = np.dot(cap_embs, retrieved_text) / (
                np.linalg.norm(cap_embs, axis=1) * np.linalg.norm(retrieved_text) + 1e-8)

            gt_texts = coco.get_image_texts(img_idx)
            gt_cap = gt_texts[0] if gt_texts else ''
            if gt_cap in all_texts:
                rank = int(np.sum(sims > sims[all_texts.index(gt_cap)])) + 1
                ranks.append(rank)

    avg_rank = float(np.mean(ranks)) if ranks else 999.0
    top5 = sum(1 for r in ranks if r <= 5) / max(len(ranks), 1)
    hit_rate = hits / n_sample

    return {
        'hit_rate': hit_rate,
        'avg_rank': avg_rank,
        'top5': top5,
        'n_hits': hits,
        'n_queries': n_sample,
    }


def train_crossmodal_coco(coco: COCOLoader, text_encoder,
                           seed: int = 42, n_epochs: int = 5,
                           lr_decay: float = 0.7,
                           threshold_anneal: float = 0.02,
                           pe_lr_scale: float = 2.0) -> dict:
    """训练 COCO 跨模态 Hebb 绑定。

    打乱全部 (image, text) pair → ClusterNetwork.learn(s)
    → PE-driven LR decay + threshold annealing → sleep 巩固
    """
    rng = np.random.default_rng(seed)

    theta = create_default_theta()
    theta.cluster_threshold = 0.45
    theta.learn_rate_l0 = 0.02
    theta.decay_rate = 0.003
    theta.pe_lr_scale = pe_lr_scale
    net = ClusterNetwork(theta, hash_offset=0)

    n_pairs = coco.n_pairs
    n_images = coco.n_images

    print(f"\n{'='*64}")
    print(f"  Stage 2: COCO Cross-modal Hebb Binding")
    print(f"{'='*64}")
    print(f"  {n_images} images, {n_pairs} pairs")
    n_unique_texts = len(set(cap for _, cap in coco.pairs))
    print(f"  {n_unique_texts} unique texts")
    print(f"  threshold={theta.cluster_threshold:.2f}, "
          f"lr={theta.learn_rate_l0:.3f}, "
          f"pe_lr_scale={pe_lr_scale:.1f}")

    epoch_metrics = []
    t_global = time.perf_counter()

    # PE EMA 状态 (用于去噪 + 趋势检测)
    pe_ema = 0.2   # 初始 rank=100 基准
    pe_prev = 0.2

    for epoch in range(1, n_epochs + 1):
        t_epoch = time.perf_counter()

        order = np.arange(n_pairs)
        rng.shuffle(order)

        sleep_interval = max(500, n_pairs // 50)
        log_interval = max(5000, n_pairs // 10)

        for step, pair_idx in enumerate(order):
            img_idx, caption = coco.pairs[pair_idx]
            text_emb = text_encoder.encode(caption)
            vis = coco.get_visual(img_idx)

            s = _build_s_from_vis(text_emb, vis)

            net.learn(s)

            if (step + 1) % sleep_interval == 0:
                n_removed = sleep_cycle(net, net.theta)
                if n_removed > 0:
                    pass  # Suppress per-sleep spam; report at epoch end

            if (step + 1) % log_interval == 0:
                elapsed = time.perf_counter() - t_epoch
                ips = (step + 1) / max(elapsed, 0.001)
                print(f"  Epoch {epoch}: {step+1}/{n_pairs} ({ips:.0f} p/s) "
                      f"| clusters={net.n_clusters}")

        # ---- Epoch end: sleep replay consolidate ----
        # 海马重放 + 跨簇关联: 主动强化近期记忆，而非仅衰减
        sr_stats = sleep_replay(net, net.theta,
                                replay_lr=0.04,
                                n_replay_cycles=1,
                                cross_link_strength=0.005)
        total_removed = sr_stats['n_removed']

        epoch_time = time.perf_counter() - t_epoch

        # ---- Quick eval ----
        qe = _quick_eval_epoch(net, coco, text_encoder, n_sample=100,
                                   rng=rng)

        # ---- Epoch-level PE-driven LR decay + Threshold (FEP-native) ----
        # 预测误差高 (rank ↑) → 少衰减 (保持学习) + 缓升/降阈值 (扩大匹配)
        # 预测误差低 (rank ↓) → 多衰减 (巩固) + 快升阈值 (专精化)
        if pe_lr_scale > 1e-6:
            pe_raw = qe['avg_rank'] / 500.0  # 0=perfect, ~1=random

            # EMA 平滑 (减少 100-sample quick eval 噪声)
            pe_ema = 0.6 * pe_ema + 0.4 * pe_raw
            pe_smooth = pe_ema

            # 非对称调制: 恶化时反应更激烈, 改善时温和退场
            if pe_smooth > pe_prev:
                asym_scale = 1.3  # 恶化: 放大响应
            else:
                asym_scale = 0.7  # 改善: 温和退出
            pe_prev = pe_smooth

            # 以 rank=100 (norm=0.2) 为基准中心
            pe_mod = 1.0 - pe_lr_scale * asym_scale * (pe_smooth - 0.2)
            pe_mod = max(-1.5, min(pe_mod, 2.0))

            # LR decay 调制
            effective_decay = lr_decay * pe_mod
            effective_decay = max(0.10, min(effective_decay, 1.5))

            # Threshold 退火调制 (同源信号)
            effective_anneal = threshold_anneal * pe_mod
            effective_anneal = max(-0.04, min(effective_anneal, 0.06))
        else:
            effective_decay = lr_decay
            effective_anneal = threshold_anneal

        theta.learn_rate_l0 *= effective_decay
        theta.cluster_threshold = min(0.65, max(0.35,
            theta.cluster_threshold + effective_anneal))

        pe_label = f"decay={effective_decay:.3f} anneal={effective_anneal:+.3f}" if pe_lr_scale > 1e-6 else ""
        print(f"  ── Epoch {epoch} done: {epoch_time:.1f}s | "
              f"clusters={net.n_clusters} (removed {total_removed}) | "
              f"replay={sr_stats['n_replayed']} sep={sr_stats['n_linked']} | "
              f"thr={theta.cluster_threshold:.2f} lr={theta.learn_rate_l0:.4f} "
              f"{pe_label} | "
              f"V→T hit={qe['hit_rate']:.2f} rank={qe['avg_rank']:.0f} "
              f"top5={qe['top5']:.3f}")

        epoch_metrics.append({
            'epoch': epoch,
            'n_clusters': net.n_clusters,
            'epoch_time_s': epoch_time,
            'replay_n': sr_stats['n_replayed'],
            'replay_sep': sr_stats['n_linked'],  # 模式分离对数
            'replay_boost': sr_stats['replay_boost'],
            **qe,
        })

    elapsed = time.perf_counter() - t_global
    print(f"\n  Training complete: {elapsed:.1f}s, "
          f"{net.n_clusters} clusters formed")

    return {
        'net': net,
        'n_clusters': net.n_clusters,
        'epoch_metrics': epoch_metrics,
    }


def evaluate_crossmodal_coco(result: dict, coco: COCOLoader,
                               text_encoder, n_test: int = 500) -> dict:
    """评估 COCO 跨模态检索 (无类别标签，使用 caption 相似度)。"""
    net = result['net']
    if net.n_clusters == 0:
        print("  No clusters — cannot evaluate")
        return {}

    print(f"\n{'='*64}")
    print(f"  COCO Cross-modal Evaluation")
    print(f"{'='*64}")
    print(f"  Clusters: {net.n_clusters}")

    text_mask = make_text_mask()
    visual_mask = make_visual_mask()
    VISUAL_START = V1_START
    VISUAL_END = COLOR_END

    VISUAL_THRESHOLD = 0.25
    rng = np.random.default_rng(99)

    def _build_vis_query(vis):
        txt = np.zeros(TEXT_WIDTH, dtype=np.float32)
        return _build_s_from_vis(txt, vis)

    def _build_text_query(text_emb):
        return build_crossmodal_sensory(
            text_emb,
            np.zeros(V1_WIDTH, dtype=np.float32),
            np.zeros(V2_WIDTH, dtype=np.float32),
            np.zeros(V4_WIDTH, dtype=np.float32),
            np.zeros(COLOR_WIDTH, dtype=np.float32),
        )

    # ---- Test 1: Text → Visual Retrieval ----
    print(f"\n  --- Test 1: Text → Visual Retrieval ---")
    t2v_cosines = []
    t2v_correct = 0
    n_t2v = min(n_test, coco.n_images)

    for _ in range(n_t2v):
        img_idx = rng.integers(0, coco.n_images)
        texts = coco.get_image_texts(img_idx)
        query_text = texts[rng.integers(0, len(texts))]

        query_emb = text_encoder.encode(query_text)
        s_query = _build_text_query(query_emb)

        c, sim = masked_recall(net, s_query, text_mask)
        if c is not None:
            retrieved_visual = c.centroid[VISUAL_START:VISUAL_END]

            # 在 200 张随机图像中找最佳匹配
            n_search = min(200, coco.n_images)
            search_indices = rng.choice(coco.n_images, n_search, replace=False)
            best_cos = -1.0
            best_idx = -1
            for si in search_indices:
                vis = coco.get_visual(si)
                s_vis = _build_vis_query(vis)
                gt_vis = s_vis[VISUAL_START:VISUAL_END]
                cos = float(np.dot(retrieved_visual, gt_vis) / (
                    np.linalg.norm(retrieved_visual) * np.linalg.norm(gt_vis) + 1e-8))
                if cos > best_cos:
                    best_cos = cos
                    best_idx = si

            t2v_cosines.append(best_cos)
            if best_idx == img_idx:
                t2v_correct += 1

    avg_t2v_cos = float(np.mean(t2v_cosines)) if t2v_cosines else 0.0
    t2v_acc = t2v_correct / n_t2v if n_t2v > 0 else 0.0
    print(f"  Avg cosine (retrieved vs best-match image): {avg_t2v_cos:.4f}")
    print(f"  Exact image retrieval rate: {t2v_acc:.3f} "
          f"({t2v_correct}/{n_t2v})")

    # ---- Test 2: Visual → Text Retrieval ----
    print(f"\n  --- Test 2: Visual → Text Retrieval ---")
    v2t_cosines = []
    v2t_ranks = []

    for _ in range(n_t2v):
        img_idx = rng.integers(0, coco.n_images)
        vis = coco.get_visual(img_idx)
        s_query = _build_vis_query(vis)

        c, sim = masked_recall(net, s_query, visual_mask,
                                threshold_override=VISUAL_THRESHOLD)
        if c is not None:
            retrieved_text = c.centroid[TEXT_START:TEXT_END]

            # 对所有唯一文本计算相似度
            all_texts = list(set(cap for _, cap in coco.pairs[:2000]))
            if len(all_texts) > 500:
                all_texts = all_texts[:500]  # 限制数量以加速

            # Encode all unique texts
            cap_embs = text_encoder.encode_batch(all_texts)
            sims = np.dot(cap_embs, retrieved_text) / (
                np.linalg.norm(cap_embs, axis=1) * np.linalg.norm(retrieved_text) + 1e-8)

            # 找 ground truth texts 在 ranked list 中的位置
            gt_texts = coco.get_image_texts(img_idx)
            gt_cap = gt_texts[0] if gt_texts else ''

            if gt_cap in all_texts:
                gt_rank = np.sum(sims > sims[all_texts.index(gt_cap)]) + 1
                v2t_ranks.append(gt_rank)
                v2t_cosines.append(float(sims[all_texts.index(gt_cap)]))

    avg_v2t_cos = float(np.mean(v2t_cosines)) if v2t_cosines else 0.0
    avg_v2t_rank = float(np.mean(v2t_ranks)) if v2t_ranks else 0.0
    v2t_top1 = sum(1 for r in v2t_ranks if r == 1) / max(len(v2t_ranks), 1)
    v2t_top5 = sum(1 for r in v2t_ranks if r <= 5) / max(len(v2t_ranks), 1)
    print(f"  Avg cosine (retrieved text vs GT caption): {avg_v2t_cos:.4f}")
    print(f"  Avg GT caption rank: {avg_v2t_rank:.1f} (out of 1000)")
    print(f"  Top-1 / Top-5 retrieval: {v2t_top1:.3f} / {v2t_top5:.3f}")

    # ---- Test 3: Image-to-Image via Text Bridge ----
    print(f"\n  --- Test 3: Image-to-Image (via Text Bridge) ---")
    i2i_cosines = []

    for _ in range(min(200, n_test)):
        img_idx = rng.integers(0, coco.n_images)
        vis = coco.get_visual(img_idx)
        s_query = _build_vis_query(vis)

        c, sim = masked_recall(net, s_query, visual_mask,
                                threshold_override=VISUAL_THRESHOLD)
        if c is not None:
            # 用集群的视觉质心在其他图像中找匹配
            retrieved_visual = c.centroid[VISUAL_START:VISUAL_END]

            n_search = min(100, coco.n_images)
            search_indices = rng.choice(coco.n_images, n_search, replace=False)
            best_cos = -1.0
            best_idx = -1
            for si in search_indices:
                if si == img_idx:
                    continue
                vis2 = coco.get_visual(si)
                s_vis2 = _build_vis_query(vis2)
                cos = float(np.dot(retrieved_visual, s_vis2[VISUAL_START:VISUAL_END]) / (
                    np.linalg.norm(retrieved_visual) * np.linalg.norm(s_vis2[VISUAL_START:VISUAL_END]) + 1e-8))
                if cos > best_cos:
                    best_cos = cos
                    best_idx = si

            i2i_cosines.append(best_cos)

    avg_i2i_cos = float(np.mean(i2i_cosines)) if i2i_cosines else 0.0
    print(f"  Avg image-to-image cosine (via cluster bridge): {avg_i2i_cos:.4f}")

    return {
        't2v_avg_cosine': avg_t2v_cos,
        't2v_accuracy': t2v_acc,
        'v2t_avg_cosine': avg_v2t_cos,
        'v2t_avg_rank': avg_v2t_rank,
        'v2t_top1': v2t_top1,
        'v2t_top5': v2t_top5,
        'i2i_avg_cosine': avg_i2i_cos,
        'n_clusters': net.n_clusters,
    }


# ================================================================
# Main
# ================================================================

def run_stage2(n_images: int = 2000, dataset: str = 'imagenette',
                mode: str = 'train', seed: int = 42,
                interactive: bool = False, n_epochs: int = 5,
                lr_decay: float = 0.7,
                threshold_anneal: float = 0.02,
                pe_lr_scale: float = 2.0):
    """Stage 2: Cross-modal Hebb binding.

    Args:
        n_images: number of images
        dataset: 'cifar10', 'imagenette', or 'coco'
        mode: 'train', 'eval', or 'all'
        seed: random seed
        interactive: launch interactive demo after training
        n_epochs: number of training epochs (COCO only)
        lr_decay: per-epoch learning rate multiplier
        threshold_anneal: per-epoch threshold increase
        pe_lr_scale: prediction error-driven LR decay modulation (0=off)
    """
    np.random.seed(seed)

    print("=" * 72)
    print("  Stage 2: Cross-modal Hebb Binding")
    print("=" * 72)
    print(f"  Dataset: {dataset}, {n_images} images")
    print()

    cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(
        cache_dir,
        f'stage2_crossmodal_{dataset}_{n_images}_s{seed}.pkl')

    # ================================================================
    # COCO Path
    # ================================================================
    if dataset == 'coco':
        # 1. Load COCO data (download if needed)
        print("[1/3] Loading COCO 2017 val...")
        t0 = time.perf_counter()
        coco = COCOLoader(n_images=n_images, image_size=128)
        print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

        # 2. Initialize Text Encoder on COCO captions
        print(f"\n[2/3] Initializing text encoder (on COCO captions)...")
        t0 = time.perf_counter()
        all_captions = coco.get_all_captions()
        # Fit PCA on a subset for speed
        sample_captions = all_captions[:min(5000, len(all_captions))]
        from collections import OrderedDict
        unique_captions = list(OrderedDict.fromkeys(sample_captions))
        print(f"  Fitting PCA on {len(unique_captions)} unique captions...")

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        full_embs = model.encode(unique_captions, show_progress_bar=True,
                                 batch_size=64)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=TEXT_WIDTH, random_state=42)
        pca.fit(full_embs)
        print(f"  Text PCA: {full_embs.shape[1]}d → {TEXT_WIDTH}d, "
              f"explained={pca.explained_variance_ratio_.sum():.1%}")

        # Build TextEncoder-compatible interface
        class COCOTextEncoder:
            def __init__(self, model, pca, captions_sample):
                self.model = model
                self.pca_obj = pca
                self.text_to_emb = {}
                # Pre-encode sample captions
                embs = pca.transform(model.encode(captions_sample, batch_size=64))
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs /= (norms + 1e-8)
                for i, cap in enumerate(captions_sample):
                    self.text_to_emb[cap] = embs[i].astype(np.float32)
                self.all_texts = captions_sample

            def encode(self, text):
                if text in self.text_to_emb:
                    return self.text_to_emb[text].copy()
                emb = self.model.encode([text])[0]
                emb = self.pca_obj.transform(emb.reshape(1, -1))[0]
                emb = emb.astype(np.float32)
                if len(emb) < TEXT_WIDTH:
                    p = np.zeros(TEXT_WIDTH, dtype=np.float32)
                    p[:len(emb)] = emb
                    emb = p
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    emb /= norm
                return emb

            def encode_batch(self, texts):
                full = self.model.encode(texts, batch_size=64)
                emb = self.pca_obj.transform(full).astype(np.float32)
                if emb.shape[1] < TEXT_WIDTH:
                    p = np.zeros((emb.shape[0], TEXT_WIDTH), dtype=np.float32)
                    p[:, :emb.shape[1]] = emb
                    emb = p
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                emb /= (norms + 1e-8)
                return emb

        text_encoder = COCOTextEncoder(model, pca, unique_captions)
        print(f"  Initialized in {time.perf_counter() - t0:.1f}s")
        print(f"  Vocabulary: {len(text_encoder.all_texts)} unique captions")

        # 3. Train & Evaluate
        if mode in ('train', 'all') or not os.path.exists(model_path):
            print(f"\n[3/3] Training cross-modal binding on COCO...")
            result = train_crossmodal_coco(coco, text_encoder, seed=seed,
                                           n_epochs=n_epochs,
                                           lr_decay=lr_decay,
                                           threshold_anneal=threshold_anneal,
                                           pe_lr_scale=pe_lr_scale)
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'net': result['net'],
                    'n_clusters': result['n_clusters'],
                    'n_images': n_images,
                    'epoch_metrics': result.get('epoch_metrics', []),
                }, f)
            print(f"  Model saved: {model_path}")
        else:
            print(f"\n[3/3] Loading cached model...")
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
            result = {'net': saved['net'], 'n_clusters': saved['n_clusters'],
                       'epoch_metrics': saved.get('epoch_metrics', [])}
            print(f"  Loaded: {result['n_clusters']} clusters")

        print(f"\n[Eval] Evaluating cross-modal retrieval...")
        eval_results = evaluate_crossmodal_coco(result, coco, text_encoder,
                                                  n_test=min(500, n_images))

        # ---- Epoch Metrics Summary ----
        if result.get('epoch_metrics'):
            print(f"\n  ── Epoch-by-Epoch Improvement ──")
            print(f"  {'Epoch':>5s} | {'Clusters':>8s} | {'V→T hit':>8s} | "
                  f"{'V→T rank':>9s} | {'V→T top5':>9s}")
            print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+--{'-'*9}-+-{'-'*9}")
            for em in result['epoch_metrics']:
                print(f"  {em['epoch']:5d} | {em['n_clusters']:8d} | "
                      f"{em['hit_rate']:8.2f} | {em['avg_rank']:9.1f} | "
                      f"{em['top5']:9.3f}")

        # ---- Acceptance Checks ----
        print(f"\n{'='*48}")
        print(f"  ACCEPTANCE CHECK (COCO)")
        print(f"{'='*48}")
        checks = []
        c1 = eval_results.get('t2v_avg_cosine', 0) > 0.4
        checks.append(('Text→Visual cosine > 0.4', c1,
                       f"{eval_results.get('t2v_avg_cosine', 0):.4f}"))
        c2 = eval_results.get('v2t_top5', 0) > 0.20
        checks.append(('Visual→Text top-5 > 0.20', c2,
                       f"{eval_results.get('v2t_top5', 0):.3f}"))
        c3 = eval_results.get('t2v_accuracy', 0) > 0.02
        checks.append(('Exact image retrieval > 2%', c3,
                       f"{eval_results.get('t2v_accuracy', 0):.3f}"))
        c4 = eval_results.get('v2t_avg_rank', 999) < 300
        checks.append(('Avg GT caption rank < 300', c4,
                       f"{eval_results.get('v2t_avg_rank', 0):.0f}"))
        c5 = eval_results.get('n_clusters', 0) >= 10
        checks.append(('At least 10 clusters', c5,
                       str(eval_results.get('n_clusters', 0))))

        for desc, passed, value in checks:
            status = '[PASS]' if passed else '[FAIL]'
            print(f"    {status} {desc}: {value}")

        all_pass = all(c[1] for c in checks)
        if all_pass:
            print(f"\n  *** ALL CHECKS PASSED ***")
        else:
            print(f"\n  {sum(1 for _, p, _ in checks if p)}/{len(checks)} checks passed")

        return {'result': result, 'eval': eval_results,
                'all_checks_passed': all_pass}

    # ================================================================
    # ImageNette/CIFAR-10 Path
    # ================================================================
    # 1. Load Visual Encodings
    print("[1/4] Loading visual encodings...")
    t0 = time.perf_counter()
    from visual_interface import VisualEnvironment

    venv = VisualEnvironment(
        dataset=dataset, n_images=n_images,
        pca_components=V1_WIDTH,
        v2_components=V2_WIDTH,
        v4_components=V4_WIDTH,
        color_components=64,
        use_v2=True, use_v4=True,
        use_color=True,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  V1: {venv.encodings.shape}")
    if venv.encodings_v2 is not None:
        print(f"  V2: {venv.encodings_v2.shape}")
    if venv.encodings_v4 is not None:
        print(f"  V4: {venv.encodings_v4.shape}")
    if getattr(venv, 'encodings_color', None) is not None:
        print(f"  Color: {venv.encodings_color.shape}")
    n_actual = venv.n_images

    # 2. Initialize Text Encoder
    print(f"\n[2/4] Initializing text encoder...")
    t0 = time.perf_counter()
    text_encoder = TextEncoder(IMAGENETTE_DESCRIPTIONS)
    print(f"  Initialized in {time.perf_counter() - t0:.1f}s")
    print(f"  Vocabulary: {len(text_encoder.all_texts)} descriptions")

    # 3. Train & Evaluate
    # 3. Train Cross-modal Binding
    # ================================================================
    cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(
        cache_dir,
        f'stage2_crossmodal_{dataset}_{n_actual}_s{seed}.pkl')

    if mode in ('train', 'all') or not os.path.exists(model_path):
        print(f"\n[3/4] Training cross-modal binding...")
        result = train_crossmodal(venv, text_encoder, n_actual, seed=seed)

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump({
                'net': result['net'],
                'n_clusters': result['n_clusters'],
                'n_images': n_actual,
            }, f)
        print(f"  Model saved: {model_path}")
    else:
        print(f"\n[3/4] Loading cached model...")
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)
        result = {'net': saved['net'], 'n_clusters': saved['n_clusters']}
        print(f"  Loaded: {result['n_clusters']} clusters")

    # ================================================================
    # 4. Evaluate
    # ================================================================
    print(f"\n[4/4] Evaluating cross-modal retrieval...")
    eval_results = evaluate_crossmodal(result, venv, text_encoder,
                                        n_test=min(500, n_actual))

    # ================================================================
    # Acceptance Checks
    # ================================================================
    print(f"\n{'='*48}")
    print(f"  ACCEPTANCE CHECK")
    print(f"{'='*48}")

    checks = []

    # Check 1: Text→Visual accuracy > 0.4 (4x random)
    t2v_acc = eval_results.get('t2v_avg_accuracy', 0)
    c1 = t2v_acc > 0.4
    checks.append(('Text→Visual accuracy > 0.4', c1, f"{t2v_acc:.3f}"))

    # Check 2: Visual→Text top-5 accuracy > 0.35
    v2t_top5 = eval_results.get('v2t_top5_accuracy', 0)
    c2 = v2t_top5 > 0.35
    checks.append(('Visual→Text top-5 accuracy > 0.35', c2, f"{v2t_top5:.3f}"))

    # Check 3: Blind test accuracy > 0.18
    blind_acc = eval_results.get('blind_avg_accuracy', 0)
    c3 = blind_acc > 0.18
    checks.append(('Blind (vision-only) accuracy > 0.18', c3, f"{blind_acc:.3f}"))

    # Check 4: At least 5 labeled clusters
    n_labeled = eval_results.get('n_labeled_clusters', 0)
    c4 = n_labeled >= 5
    checks.append(('At least 5 labeled clusters', c4, str(n_labeled)))

    # Check 5: Text→Visual cosine vs class centroid > 0.3
    t2v_cos = eval_results.get('t2v_avg_cosine', 0)
    c5 = t2v_cos > 0.3
    checks.append(('Text→Visual cosine vs class centroid > 0.3', c5, f"{t2v_cos:.4f}"))

    for desc, passed, value in checks:
        status = '[PASS]' if passed else '[FAIL]'
        print(f"    {status} {desc}: {value}")

    all_pass = all(c[1] for c in checks)
    if all_pass:
        print(f"\n  *** ALL CHECKS PASSED ***")
    else:
        n_pass = sum(1 for _, p, _ in checks if p)
        print(f"\n  {n_pass}/{len(checks)} checks passed")

    # ================================================================
    # Interactive Demo
    # ================================================================
    if interactive:
        interactive_demo(result, venv, text_encoder)

    return {
        'result': result,
        'eval': eval_results,
        'all_checks_passed': all_pass,
    }


# ================================================================
# CLI
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage 2: Cross-modal Hebb Binding')
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--dataset', type=str, default='imagenette',
                       choices=['cifar10', 'imagenette', 'coco'])
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'eval', 'all'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--interactive', action='store_true',
                       help='Launch interactive demo after evaluation')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (COCO only)')
    parser.add_argument('--lr-decay', type=float, default=0.7,
                       help='Per-epoch learning rate multiplier')
    parser.add_argument('--threshold-anneal', type=float, default=0.02,
                       help='Per-epoch threshold increase')
    parser.add_argument('--pe-lr-scale', type=float, default=2.0,
                       help='Prediction error-driven LR decay (0=off, 2.0 default)')
    args = parser.parse_args()

    run_stage2(
        n_images=args.n,
        dataset=args.dataset,
        mode=args.mode,
        seed=args.seed,
        interactive=args.interactive,
        n_epochs=args.epochs,
        lr_decay=args.lr_decay,
        threshold_anneal=args.threshold_anneal,
        pe_lr_scale=args.pe_lr_scale,
    )
