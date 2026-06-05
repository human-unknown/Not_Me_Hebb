"""
visual_interface.py —— 视觉环境 (阶段 0: 视觉基础)
自由能原理智能体

⚠️ v4.1 废弃说明: 本模块的视觉编码功能已被以下模块取代:
  - cerebrum/occipital_lobe/retina_lgn.py  → ImageEncoder, Gabor 多尺度编码
  - cerebrum/occipital_lobe/visual_pathway.py → V1+V2+V4+Color 视觉通路
  - cerebrum/occipital_lobe/gestalt.py        → Gestalt 知觉分组层
  - cerebrum/association/crossmodal.py        → 跨模态 Hebb 绑定

保留本文件用于现有缓存和向后兼容。新代码请使用 cerebrum/ 层级模块。

将图像数据集编码为 64-dim 视觉向量，填入 s[64:128] vision 通道。
Gabor 滤波器组 = 生物"视网膜 + V1"，纯数学信号处理，零 ML 训练。

管线:
  图像 → GaborFilterBank (32 filters × 4×4 grid × 2 stats = 1024d)
       → PCA → 64d 视觉特征向量

数据集: CIFAR-10 (60,000 张 32×32 RGB 图像, 10 类)
"""

import os
import pickle
import tarfile
import hashlib
import numpy as np
from urllib.request import urlretrieve

from cerebrum.occipital_lobe.gestalt import compute_gestalt_from_image

CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

IMAGENETTE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
IMAGENETTE_CLASSES = [
    'tench', 'springer', 'cassette_player', 'chain_saw', 'church',
    'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute',
]


class VisualEnvironment:
    """视觉环境 — 图像数据导航。

    Gabor 滤波器组 → V1 (4x4 grid) + V2 (2x2 grid) → PCA → 视觉通道。

    管线:
      V1: 32 filters x 16 cells x 2 stats = 1024d → PCA → pca_components d
      V2: 32 filters x 4 cells x 2 stats + cross-orient = ~276d → PCA → v2_components d
    """

    def __init__(self, dataset: str = 'cifar10', n_images: int = 10000,
                 image_size: int = None, grid_size: int = 4,
                 pca_components: int = 128, v2_components: int = 64,
                 v4_components: int = 0, color_components: int = 64,
                 use_v2: bool = True, use_v4: bool = True,
                 use_color: bool = False):
        self.dataset = dataset
        self.n_images_limit = n_images
        # Auto-select image size: 64 for CIFAR-10, 128 for ImageNette
        if image_size is None:
            image_size = 128 if dataset == 'imagenette' else 64
        self.image_size = image_size
        self.grid_size = grid_size
        self.pca_components = pca_components    # V1 PCA dims
        self.v2_components = v2_components      # V2 PCA dims
        self.v4_components = v4_components      # V4 PCA dims (0 = keep raw)
        self.color_components = color_components # Color opponent PCA dims
        self.use_v2 = use_v2
        self.use_v4 = use_v4
        self.use_color = use_color
        base = os.path.dirname(__file__)

        # ---- Gabor V1 ----
        from cerebrum.occipital_lobe.visual_pathway import GaborFilterBank
        self._gabor = GaborFilterBank(image_size=image_size,
                                       grid_size=grid_size)

        # ---- 加载数据集 ----
        if dataset == 'cifar10':
            self.images, self.labels, self.label_names = \
                self._load_cifar10(n_images)
        elif dataset == 'imagenette':
            self.images, self.labels, self.label_names = \
                self._load_imagenette(n_images)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.n_images = len(self.images)
        print(f"  VisualEnvironment: {self.n_images} images loaded "
              f"({self.n_images // 1000}k)")

        # ---- 编码 + PCA ----
        cache_dir = os.path.join(base, '.cache')
        os.makedirs(cache_dir, exist_ok=True)

        # 哈希: 处理 list (ImageNette) 和 ndarray (CIFAR-10)
        if isinstance(self.images, list):
            img_sample = np.stack([img.ravel()[:256] for img in self.images[:50]
                                  if img.size >= 256], axis=0)
            img_hash = hashlib.md5(img_sample.tobytes()).hexdigest()[:8]
        else:
            img_hash = hashlib.md5(
                self.images[:100].tobytes()).hexdigest()[:8]
        cache_key = (f'vision_{dataset}_{self.n_images}_'
                     f'sz{image_size}_g{grid_size}_'
                     f'p{pca_components}_v2{int(use_v2)}v4{int(use_v4)}_'
                     f'v4c{v4_components}_col{int(use_color)}c{color_components}_'
                     f'{img_hash}')
        cache_path = os.path.join(cache_dir, f'{cache_key}.npy')
        pca_path = os.path.join(cache_dir, f'{cache_key}_pca.pkl')
        v2_cache_path = os.path.join(cache_dir, f'{cache_key}_v2.npy')
        v4_cache_path = os.path.join(cache_dir, f'{cache_key}_v4.npy')
        color_cache_path = os.path.join(cache_dir, f'{cache_key}_color.npy')
        gestalt_cache_path = os.path.join(cache_dir,
                                          f'{cache_key}_gestalt.npy')

        if os.path.exists(cache_path) and os.path.exists(pca_path):
            print(f"  Loading cached vision encodings + PCA...")
            self.encodings = np.load(cache_path)
            with open(pca_path, 'rb') as f:
                self.pca = pickle.load(f)
            print(f"  V1 cached: {self.encodings.shape}")
            if use_v2 and os.path.exists(v2_cache_path):
                self.encodings_v2 = np.load(v2_cache_path)
                print(f"  V2 cached: {self.encodings_v2.shape}")
            elif use_v2:
                self.encodings_v2 = None
            if use_v4 and os.path.exists(v4_cache_path):
                self.encodings_v4 = np.load(v4_cache_path)
                print(f"  V4 cached: {self.encodings_v4.shape}")
            elif use_v4:
                self.encodings_v4 = None
            if use_color and os.path.exists(color_cache_path):
                self.encodings_color = np.load(color_cache_path)
                print(f"  Color cached: {self.encodings_color.shape}")
            elif use_color:
                self.encodings_color = None
            # Module A: Gestalt — load cache or compute lazily
            if os.path.exists(gestalt_cache_path):
                self.encodings_gestalt = np.load(gestalt_cache_path)
                print(f"  Gestalt cached: {self.encodings_gestalt.shape}")
            else:
                self.encodings_gestalt = None  # computed lazily
        else:
            if use_v2 or use_v4 or use_color:
                self.encodings, self.encodings_v2, self.encodings_v4, \
                    self.encodings_color, self.encodings_gestalt, self.pca = (
                    self._encode_with_pca_v2(cache_path, v2_cache_path,
                                             v4_cache_path, color_cache_path,
                                             gestalt_cache_path, pca_path))
            else:
                self.encodings, self.pca = self._encode_with_pca(
                    cache_path, pca_path)
                self.encodings_v2 = None
                self.encodings_v4 = None
                self.encodings_color = None
                self.encodings_gestalt = None

        # 导航状态
        self.cursor: int = 0

    # ================================================================
    # CIFAR-10 加载
    # ================================================================

    def _load_cifar10(self, n_images: int
                      ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """加载 CIFAR-10 数据集。

        自动下载 (如需要) → 解压 → 加载 batch 文件 → 合并。

        Returns:
            (images, labels, label_names)
            images: (N, 32, 32, 3) uint8
            labels: (N,) int64
            label_names: list[str] (长度 10)
        """
        base = os.path.dirname(__file__)
        data_dir = os.path.join(base, '.cache', 'cifar-10-batches-py')

        # ---- 下载 + 解压 ----
        if not os.path.exists(data_dir):
            print("  Downloading CIFAR-10 (162 MB)...")
            os.makedirs(data_dir, exist_ok=True)
            tar_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')

            try:
                urlretrieve(CIFAR10_URL, tar_path)
            except Exception:
                # 备用 URL
                mirror = 'https://github.com/knifecake/cifar10/raw/main/cifar-10-python.tar.gz'
                urlretrieve(mirror, tar_path)

            print("  Extracting...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=os.path.dirname(data_dir))
            os.remove(tar_path)
            print("  CIFAR-10 ready.")

        # ---- 加载数据 batch ----
        images_list = []
        labels_list = []

        batch_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith('data_batch_'))

        for bf in batch_files:
            if len(images_list) >= n_images:
                break
            with open(os.path.join(data_dir, bf), 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']  # (10000, 3072)
            labels = batch[b'labels']  # (10000,)

            # 重塑: (N, 3072) → (N, 32, 32, 3) RGB
            n_batch = data.shape[0]
            # CIFAR-10 存储格式: R 通道 (1024) + G 通道 (1024) + B 通道 (1024)
            images_r = data[:, :1024].reshape(n_batch, 32, 32)
            images_g = data[:, 1024:2048].reshape(n_batch, 32, 32)
            images_b = data[:, 2048:3072].reshape(n_batch, 32, 32)
            images = np.stack([images_r, images_g, images_b], axis=-1)

            images_list.append(images)
            labels_list.append(np.array(labels, dtype=np.int64))

        all_images = np.concatenate(images_list, axis=0)[:n_images]
        all_labels = np.concatenate(labels_list, axis=0)[:n_images]

        print(f"  CIFAR-10: {len(all_images)} images, "
              f"{len(set(all_labels))} classes")
        for i, name in enumerate(CIFAR10_CLASSES):
            count = int(np.sum(all_labels == i))
            print(f"    [{i}] {name}: {count}")

        return all_images, all_labels, CIFAR10_CLASSES

    # ================================================================
    # ImageNette 加载
    # ================================================================

    def _load_imagenette(self, n_images: int
                         ) -> tuple[list, np.ndarray, list[str]]:
        """加载 ImageNette 数据集 (10 类 ImageNet 子集, ~13k 张全尺寸图像)。

        自动下载 (1.5GB) → 解压 → 扫描类别文件夹 → 加载图像。

        Images are returned as a list of numpy arrays (variable sizes, RGB).
        They will be resized during Gabor encoding via _preprocess().

        Returns:
            (images_list, labels, label_names)
            images_list: list[np.ndarray] (variable H, W, 3) uint8
            labels: (N,) int64
            label_names: list[str] (length 10)
        """
        base = os.path.dirname(__file__)
        data_dir = os.path.join(base, '.cache', 'imagenette2')

        # ---- 下载 + 解压 ----
        if not os.path.exists(data_dir):
            print("  Downloading ImageNette (1.5 GB)...")
            print("  This may take a few minutes...")
            os.makedirs(data_dir, exist_ok=True)
            tar_path = os.path.join(data_dir, 'imagenette2.tgz')

            try:
                urlretrieve(IMAGENETTE_URL, tar_path)
            except Exception as e:
                print(f"  Download failed: {e}")
                print(f"  Trying mirror...")
                mirror = ('https://github.com/fastai/imagenette/raw/master/'
                         'imagenette2.tgz')
                urlretrieve(mirror, tar_path)

            print("  Extracting...")
            import tarfile as tarfile_mod
            with tarfile_mod.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=os.path.dirname(data_dir))
            os.remove(tar_path)
            print("  ImageNette ready.")

        # ---- 加载图像 ----
        # ImageNette 结构: imagenette2/train/{class_folder}/*.JPEG
        train_dir = os.path.join(data_dir, 'train')
        if not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"ImageNette train dir not found: {train_dir}")

        # 获取类别文件夹 (按 ImageNette 标准顺序)
        class_folders = sorted(os.listdir(train_dir))
        class_folders = [f for f in class_folders
                        if os.path.isdir(os.path.join(train_dir, f))]
        if len(class_folders) == 0:
            raise FileNotFoundError(
                f"No class folders in {train_dir}")

        print(f"  ImageNette: {len(class_folders)} classes found")

        # 映射文件夹到标签
        folder_to_label = {}
        for i, folder in enumerate(class_folders):
            folder_to_label[folder] = i
            print(f"    [{i}] {IMAGENETTE_CLASSES[i] if i < len(IMAGENETTE_CLASSES) else folder}")

        # 收集所有图像路径
        from PIL import Image
        image_paths = []
        for folder in class_folders:
            folder_path = os.path.join(train_dir, folder)
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith(('.jpeg', '.jpg', '.png', '.webp')):
                    image_paths.append(
                        (os.path.join(folder_path, fname),
                         folder_to_label[folder]))

        # Shuffle to mix classes
        rng = np.random.default_rng(42)
        rng.shuffle(image_paths)

        # Limit
        if len(image_paths) > n_images:
            image_paths = image_paths[:n_images]

        print(f"  Loading {len(image_paths)} images...")
        images_list = []
        labels_list = []
        for pi, (path, label) in enumerate(image_paths):
            if (pi + 1) % 2000 == 0:
                print(f"    Loaded {pi + 1}/{len(image_paths)}...")
            try:
                img = Image.open(path).convert('RGB')
                images_list.append(np.array(img, dtype=np.uint8))
                labels_list.append(label)
            except Exception:
                pass  # skip corrupted images

        labels = np.array(labels_list, dtype=np.int64)
        print(f"  ImageNette: {len(images_list)} images loaded")

        for i in range(len(class_folders)):
            count = int(np.sum(labels == i))
            name = IMAGENETTE_CLASSES[i] if i < len(IMAGENETTE_CLASSES) else class_folders[i]
            print(f"    [{i}] {name}: {count}")

        return images_list, labels, IMAGENETTE_CLASSES[:len(class_folders)]

    # ================================================================
    # 编码 + PCA
    # ================================================================

    def _encode_with_pca(self, cache_path: str, pca_path: str
                         ) -> tuple[np.ndarray, object]:
        """编码全部图像 (V1 only) + 拟合 PCA → 缓存."""
        from sklearn.decomposition import PCA

        n = self.n_images
        raw_dim = self._gabor.raw_dim  # 1024

        print(f"  Encoding {n} images (Gabor V1 {raw_dim}d "
              f"+ PCA -> {self.pca_components}d)...")
        raw = np.zeros((n, raw_dim), dtype=np.float32)
        for i in range(n):
            if (i + 1) % 1000 == 0:
                print(f"    Gabor V1 encode {i + 1}/{n}...")
            raw[i] = self._gabor.encode(self.images[i], learn=True)

        profile = self._gabor.get_gain_profile()
        print(f"  Hebb gain: mean={profile['mean_gain']:.3f}, "
              f"std={profile['std_gain']:.3f}")

        print(f"  Fitting PCA ({raw_dim}d -> {self.pca_components}d)...")
        pca = PCA(n_components=self.pca_components, random_state=42)
        encodings = pca.fit_transform(raw).astype(np.float32)

        norms = np.linalg.norm(encodings, axis=1, keepdims=True)
        encodings /= (norms + 1e-8)

        print(f"  PCA explained variance: "
              f"{pca.explained_variance_ratio_.sum():.1%}")

        np.save(cache_path, encodings)
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"  Cached: {cache_path}")

        return encodings, pca

    def _encode_with_pca_v2(self, cache_path: str, v2_cache_path: str,
                             v4_cache_path: str, color_cache_path: str,
                             gestalt_cache_path: str, pca_path: str
                             ) -> tuple:
        """编码 V1 + V2 + V4 + Color + Gestalt + PCA → 缓存。

        V1: 1024d raw → PCA → pca_components
        V2: ~276d raw → PCA → v2_components
        V4: ~72d raw → PCA or raw
        Gestalt: ~19d raw (no PCA, already compact)
        Color: ~512d raw → PCA → color_components
        """
        from sklearn.decomposition import PCA

        n = self.n_images
        v1_raw_dim = self._gabor.raw_dim  # 1024

        # 先计算 V2/V4 raw dims
        sample_v2 = self._gabor.encode_v2(self.images[0])
        v2_raw_dim = len(sample_v2)
        sample_v4 = self._gabor.encode_v4(self.images[0])
        v4_raw_dim = len(sample_v4)
        do_color = self.use_color
        if do_color:
            sample_color = self._gabor.encode_color(self.images[0])
            color_raw_dim = len(sample_color)
        else:
            color_raw_dim = 0
        print(f"  V1: {v1_raw_dim}d, V2: {v2_raw_dim}d, V4: {v4_raw_dim}d"
              + (f", Color: {color_raw_dim}d" if do_color else ""))

        do_v2 = self.use_v2
        do_v4 = self.use_v4

        # Gestalt dim (Module A)
        from cerebrum.occipital_lobe.gestalt import GestaltGrouping
        gestalt_dim = GestaltGrouping(
            image_size=self.image_size,
            n_scales=self._gabor.n_scales,
            n_orientations=self._gabor.n_orientations).feature_dim  # 19

        print(f"  Encoding {n} images (V1 {v1_raw_dim}d -> "
              f"{self.pca_components}d"
              + (f" + V2 {v2_raw_dim}d -> {self.v2_components}d" if do_v2 else "")
              + (f" + V4 {v4_raw_dim}d" if do_v4 else "")
              + f" + Gestalt {gestalt_dim}d"
              + (f" + Color {color_raw_dim}d -> {self.color_components}d" if do_color else "")
              + ")...")

        v1_raw = np.zeros((n, v1_raw_dim), dtype=np.float32)
        v2_raw = np.zeros((n, v2_raw_dim), dtype=np.float32) if do_v2 else None
        v4_raw = np.zeros((n, v4_raw_dim), dtype=np.float32) if do_v4 else None
        gestalt_raw = np.zeros((n, gestalt_dim), dtype=np.float32)
        color_raw = np.zeros((n, color_raw_dim), dtype=np.float32) if do_color else None

        for i in range(n):
            if (i + 1) % 1000 == 0:
                print(f"    Encode {i + 1}/{n}...")
            v1_raw[i] = self._gabor.encode(self.images[i], learn=True)
            if do_v2:
                v2_raw[i] = self._gabor.encode_v2(self.images[i])
            if do_v4:
                v4_raw[i] = self._gabor.encode_v4(self.images[i])
            if do_color:
                color_raw[i] = self._gabor.encode_color(self.images[i])
            # Module A: gestalt features
            gestalt_raw[i] = compute_gestalt_from_image(
                self.images[i], self._gabor)

        profile = self._gabor.get_gain_profile()
        print(f"  Hebb gain: mean={profile['mean_gain']:.3f}, "
              f"std={profile['std_gain']:.3f}")

        # V1 PCA
        print(f"  Fitting V1 PCA ({v1_raw_dim}d -> {self.pca_components}d)...")
        pca_v1 = PCA(n_components=self.pca_components, random_state=42)
        v1_enc = pca_v1.fit_transform(v1_raw).astype(np.float32)
        norms = np.linalg.norm(v1_enc, axis=1, keepdims=True)
        v1_enc /= (norms + 1e-8)
        print(f"  V1 PCA explained: {pca_v1.explained_variance_ratio_.sum():.1%}")

        pca_data = {'v1': pca_v1}

        # V2 PCA
        if do_v2:
            v2_nc = min(self.v2_components, v2_raw_dim, v2_raw.shape[0])
            print(f"  Fitting V2 PCA ({v2_raw_dim}d -> {v2_nc}d)...")
            pca_v2 = PCA(n_components=v2_nc, random_state=42)
            v2_enc = pca_v2.fit_transform(v2_raw).astype(np.float32)
            norms = np.linalg.norm(v2_enc, axis=1, keepdims=True)
            v2_enc /= (norms + 1e-8)
            print(f"  V2 PCA explained: {pca_v2.explained_variance_ratio_.sum():.1%}")
            pca_data['v2'] = pca_v2
            pca_data['v2_nc'] = v2_nc
        else:
            v2_enc = None

        # V4: optional PCA (default keep raw ~72d) but L2-normalize
        if do_v4:
            if self.v4_components > 0 and self.v4_components < v4_raw_dim:
                v4_nc = min(self.v4_components, v4_raw_dim, v4_raw.shape[0])
                print(f"  Fitting V4 PCA ({v4_raw_dim}d -> {v4_nc}d)...")
                pca_v4 = PCA(n_components=v4_nc, random_state=42)
                v4_enc = pca_v4.fit_transform(v4_raw).astype(np.float32)
                norms = np.linalg.norm(v4_enc, axis=1, keepdims=True)
                v4_enc /= (norms + 1e-8)
                print(f"  V4 PCA explained: {pca_v4.explained_variance_ratio_.sum():.1%}")
                pca_data['v4'] = pca_v4
                pca_data['v4_dim'] = v4_nc
            else:
                v4_enc = v4_raw.astype(np.float32)
                norms = np.linalg.norm(v4_enc, axis=1, keepdims=True)
                v4_enc /= (norms + 1e-8)
                pca_data['v4_dim'] = v4_raw_dim
        else:
            v4_enc = None

        # Color: PCA to color_components
        if do_color and color_raw is not None:
            col_nc = min(self.color_components, color_raw_dim, color_raw.shape[0])
            print(f"  Fitting Color PCA ({color_raw_dim}d -> {col_nc}d)...")
            pca_color = PCA(n_components=col_nc, random_state=42)
            color_enc = pca_color.fit_transform(color_raw).astype(np.float32)
            norms = np.linalg.norm(color_enc, axis=1, keepdims=True)
            color_enc /= (norms + 1e-8)
            print(f"  Color PCA explained: {pca_color.explained_variance_ratio_.sum():.1%}")
            pca_data['color'] = pca_color
            pca_data['color_dim'] = col_nc
        else:
            color_enc = None

        # Gestalt: L2 normalize (no PCA, already compact 19d)
        gestalt_norms = np.linalg.norm(gestalt_raw, axis=1, keepdims=True)
        gestalt_enc = gestalt_raw / (gestalt_norms + 1e-8)
        pca_data['gestalt_dim'] = gestalt_dim

        # 缓存
        np.save(cache_path, v1_enc)
        if do_v2 and v2_enc is not None:
            np.save(v2_cache_path, v2_enc)
        if do_v4 and v4_enc is not None:
            np.save(v4_cache_path, v4_enc)
        np.save(gestalt_cache_path, gestalt_enc)
        if do_color and color_enc is not None:
            np.save(color_cache_path, color_enc)
        with open(pca_path, 'wb') as f:
            pickle.dump(pca_data, f)
        print(f"  Cached: {cache_path}"
              + (f", {v2_cache_path}" if do_v2 else "")
              + (f", {v4_cache_path}" if do_v4 else "")
              + f", {gestalt_cache_path}"
              + (f", {color_cache_path}" if do_color else ""))

        return v1_enc, v2_enc, v4_enc, gestalt_enc, color_enc, pca_data

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """单张图像 → V1 视觉向量 (Gabor V1 → PCA)。

        Args:
            image: (H, W) 灰度 或 (H, W, 3) RGB, uint8 或 float

        Returns:
            (pca_components,) float32
        """
        raw = self._gabor.encode(image, learn=False)
        if hasattr(self.pca, 'transform'):  # single PCA object
            enc = self.pca.transform(raw.reshape(1, -1))[0].astype(np.float32)
        else:  # dict with 'v1' key
            enc = self.pca['v1'].transform(raw.reshape(1, -1))[0].astype(np.float32)
        norm = np.linalg.norm(enc)
        if norm > 1e-8:
            enc /= norm
        return enc

    def encode_image_v2(self, image: np.ndarray) -> np.ndarray | None:
        """单张图像 → V2 视觉向量 (Gabor V2 → PCA)。"""
        if not self.use_v2 or self.encodings_v2 is None:
            return None
        raw = self._gabor.encode_v2(image)
        enc = self.pca['v2'].transform(raw.reshape(1, -1))[0].astype(np.float32)
        norm = np.linalg.norm(enc)
        if norm > 1e-8:
            enc /= norm
        return enc

    def encode_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """批量编码 → (N, pca_components)"""
        n = len(images)
        raw = np.zeros((n, self._gabor.raw_dim), dtype=np.float32)
        for i, img in enumerate(images):
            raw[i] = self._gabor.encode(img, learn=False)
        pca_obj = self.pca if hasattr(self.pca, 'transform') else self.pca['v1']
        enc = pca_obj.transform(raw).astype(np.float32)
        norms = np.linalg.norm(enc, axis=1, keepdims=True)
        enc /= (norms + 1e-8)
        return enc

    # ================================================================
    # 感知接口
    # ================================================================

    def get_sensory(self, index: int = None,
                    include_v2: bool = True,
                    include_v4: bool = True,
                    include_color: bool = False) -> np.ndarray:
        """获取 V1 (或 V1+V2+V4+Color) 视觉向量。

        Returns:
            拼接后的视觉特征向量
        """
        if index is None:
            index = self.cursor
        index = int(index) % self.n_images
        parts = [self.encodings[index].copy()]
        if include_v2 and self.use_v2 and self.encodings_v2 is not None:
            parts.append(self.encodings_v2[index].copy())
        if include_v4 and self.use_v4 and self.encodings_v4 is not None:
            parts.append(self.encodings_v4[index].copy())
        # Module A: Gestalt features
        if getattr(self, 'encodings_gestalt', None) is not None:
            parts.append(self.encodings_gestalt[index].copy())
        if include_color and self.use_color and getattr(self, 'encodings_color', None) is not None:
            parts.append(self.encodings_color[index].copy())
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts)

    def step(self, action_idx: int = 0):
        """移动视觉注意 (简化版)。

        A₀: 下一张
        A₁: 随机跳转
        A₂: 跳转到语义最远的图像
        A₃: 跳转到同类别的下一张
        """
        if action_idx == 0:
            self.cursor = (self.cursor + 1) % self.n_images
        elif action_idx == 1:
            self.cursor = int(np.random.randint(0, self.n_images))
        elif action_idx == 2:
            # 跳转到与当前编码余弦相似度最低的图像
            cur = self.encodings[self.cursor]
            sims = np.dot(self.encodings, cur) / (
                np.linalg.norm(self.encodings, axis=1)
                * np.linalg.norm(cur) + 1e-8)
            sims[self.cursor] = 2.0
            self.cursor = int(np.argmin(sims))
        elif action_idx == 3:
            # 跳转到同类别下一张
            cur_label = self.labels[self.cursor]
            same_class = np.where(self.labels == cur_label)[0]
            if len(same_class) > 1:
                cur_pos = np.where(same_class == self.cursor)[0][0]
                self.cursor = int(same_class[(cur_pos + 1) % len(same_class)])

    def get_image(self, index: int = None) -> np.ndarray:
        """获取原始图像 (用于可视化)。

        Args:
            index: 图像索引。None → 当前 cursor。

        Returns:
            (32, 32, 3) uint8 RGB 图像
        """
        if index is None:
            index = self.cursor
        return self.images[int(index) % self.n_images].copy()

    def get_label(self, index: int = None) -> str:
        """获取图像类别名称"""
        if index is None:
            index = self.cursor
        label_id = self.labels[int(index) % self.n_images]
        return self.label_names[label_id]

    # ================================================================
    # 统计与诊断
    # ================================================================

    def get_class_centroids(self, use_v2: bool = True) -> dict[str, np.ndarray]:
        """计算每个类别的视觉编码质心。

        Returns:
            {class_name: (N,) centroid vector}
        """
        encs = self.encodings
        if use_v2 and self.use_v2 and self.encodings_v2 is not None:
            encs = np.concatenate([self.encodings, self.encodings_v2], axis=1)
        centroids = {}
        for i, name in enumerate(self.label_names):
            mask = self.labels == i
            if mask.sum() > 0:
                centroids[name] = encs[mask].mean(axis=0)
        return centroids

    def evaluate_class_separation(self, use_v2: bool = True) -> dict:
        """评估 Gabor + PCA 编码的类间分离度。

        Returns:
            {class_name: {avg_intra_sim, avg_inter_sim, margin}}
        """
        centroids = self.get_class_centroids(use_v2=use_v2)
        encs = self.encodings
        if use_v2 and self.use_v2 and self.encodings_v2 is not None:
            encs = np.concatenate([self.encodings, self.encodings_v2], axis=1)
        results = {}

        for name, centroid in centroids.items():
            mask = self.labels == self.label_names.index(name)
            class_encs = encs[mask]
            intra_sims = np.dot(class_encs, centroid) / (
                np.linalg.norm(class_encs, axis=1)
                * np.linalg.norm(centroid) + 1e-8)
            avg_intra = float(np.mean(intra_sims))

            inter_sims = []
            for other_name, other_centroid in centroids.items():
                if other_name != name:
                    sim = float(np.dot(centroid, other_centroid) / (
                        np.linalg.norm(centroid)
                        * np.linalg.norm(other_centroid) + 1e-8))
                    inter_sims.append(sim)
            avg_inter = float(np.mean(inter_sims))

            margin = avg_intra - avg_inter  # >0 = separable

            results[name] = {
                'avg_intra_sim': avg_intra,
                'avg_inter_sim': avg_inter,
                'margin': margin,
            }

        return results

    def get_gabor_profile(self) -> dict:
        """返回 Gabor 滤波器增益概况"""
        return self._gabor.get_gain_profile()

    @property
    def n_classes(self) -> int:
        return len(self.label_names)


# ================================================================
# 自测
# ================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  VisualEnvironment Test")
    print("=" * 60)

    # 小规模测试: 2000 张图像
    venv = VisualEnvironment(dataset='cifar10', n_images=2000)
    print(f"\n  Loaded: {venv.n_images} images, {venv.n_classes} classes")
    print(f"  Encoding shape: {venv.encodings.shape}")

    # 测试: get_sensory()
    vec = venv.get_sensory(0)
    print(f"  First image encoding: shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")

    # 测试: 类间分离度
    print("\n  --- Class Separation ---")
    sep = venv.evaluate_class_separation()
    avg_margin = 0.0
    for name, stats in sep.items():
        m = stats['margin']
        avg_margin += m
        bar = '+' * max(0, min(20, int(m * 40))) + (
            '-' * max(0, 20 - max(0, min(20, int(m * 40)))))
        print(f"    {name:12s}: intra={stats['avg_intra_sim']:+.3f}  "
              f"inter={stats['avg_inter_sim']:+.3f}  "
              f"margin={m:+.3f} {bar}")
    avg_margin /= len(sep)
    print(f"\n  Average margin: {avg_margin:+.3f} "
          f"(>0 = classes are separable)")

    # 测试: Gabor 增益分布
    profile = venv.get_gabor_profile()
    print(f"\n  Hebb gain profile:")
    print(f"    mean={profile['mean_gain']:.3f}, std={profile['std_gain']:.3f}")
    print(f"    top filters: {profile['top_filters'][:5]}")
    print(f"    bottom filters: {profile['bottom_filters'][:5]}")

    # 测试: 同一类别 vs 不同类别的余弦相似度
    print("\n  --- Intra vs Inter class similarity ---")
    cat_mask = venv.labels == 3  # cat
    car_mask = venv.labels == 1  # automobile

    if cat_mask.sum() >= 2:
        cat_vecs = venv.encodings[cat_mask][:50]
        intra_cos = []
        for i in range(len(cat_vecs)):
            for j in range(i + 1, len(cat_vecs)):
                sim = np.dot(cat_vecs[i], cat_vecs[j]) / (
                    np.linalg.norm(cat_vecs[i]) * np.linalg.norm(cat_vecs[j]) + 1e-8)
                intra_cos.append(sim)
        print(f"    Intra-class (cat-cat): {np.mean(intra_cos):.3f} ± {np.std(intra_cos):.3f}")

    if cat_mask.sum() >= 1 and car_mask.sum() >= 1:
        cat_sample = venv.encodings[cat_mask][:20]
        car_sample = venv.encodings[car_mask][:20]
        inter_cos = []
        for ci in range(len(cat_sample)):
            for cj in range(len(car_sample)):
                sim = np.dot(cat_sample[ci], car_sample[cj]) / (
                    np.linalg.norm(cat_sample[ci]) * np.linalg.norm(car_sample[cj]) + 1e-8)
                inter_cos.append(sim)
        print(f"    Inter-class (cat-car): {np.mean(inter_cos):.3f} ± {np.std(inter_cos):.3f}")

    print("\n  [PASS] VisualEnvironment test complete")
