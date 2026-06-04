"""
record_multimodal.py —— 录制多模态数据 (Stage 3A)
自由能原理智能体

从 images/ 目录加载图片 → CLIP 编码 + 文本编码 → multimodal_log.npy
每 10 句配一张图，模拟"文本↔视觉"共现。
"""

import os, glob
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from text_interface import TextEnvironment


def main():
    # ---- 加载图片 ----
    img_files = sorted(glob.glob('images/*.jpg') + glob.glob('images/*.png'))
    if not img_files:
        print("ERROR: No images in images/. Run generator first.")
        return
    print(f"Images: {len(img_files)}")

    # ---- 加载语料 ----
    print("Loading corpus...")
    text_env = TextEnvironment()
    sentences = text_env.chunks[:100]
    print(f"  {len(sentences)} sentences, each with image[{len(img_files)}]")

    # ---- 文本编码 ----
    print("Loading sentence-transformer...")
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_full = text_model.encode(sentences, show_progress_bar=False)
    pca_t = PCA(n_components=min(64, len(sentences)))
    text_emb = pca_t.fit_transform(text_full).astype(np.float32)
    if text_emb.shape[1] < 64:
        p = np.zeros((len(sentences), 64), dtype=np.float32)
        p[:, :text_emb.shape[1]] = text_emb
        text_emb = p
    print(f"  Text: {text_full.shape[1]}d -> {text_emb.shape[1]}d PCA")

    # ---- CLIP 编码图片 ----
    print("Loading CLIP ViT-B/32...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    images = [Image.open(f).convert('RGB').resize((224,224)) for f in img_files]
    print(f"Encoding {len(images)} images...")
    inputs = clip_proc(images=images, return_tensors="pt")
    with torch.no_grad():
        vision_full = clip_model.get_image_features(**inputs).pooler_output.numpy()  # (N_img, 512)
    print(f"  Vision: {vision_full.shape}")

    # PCA 到 32 (保留区分度，留有零空间降低全局余弦相似度)
    n_v_pca = min(32, len(images))
    pca_v = PCA(n_components=n_v_pca)
    v_tmp = pca_v.fit_transform(vision_full).astype(np.float32)
    vision_emb = np.zeros((len(images), 64), dtype=np.float32)
    vision_emb[:, :n_v_pca] = v_tmp
    print(f"  Vision: 512d -> {n_v_pca}d PCA + {64-n_v_pca}d zeros")

    # ---- 组装帧: 每句配一张图 (循环使用) ----
    n = len(sentences)
    log = np.zeros((n, 256), dtype=np.float32)  # text|vision|audio|body
    for i in range(n):
        img_idx = i % len(images)  # 1 句 1 图
        log[i, 0:64] = text_emb[i]
        log[i, 64:128] = vision_emb[img_idx]
        # audio+body: 零

    np.save('multimodal_log.npy', log)

    # 视觉方差检查
    vis_variance = float(np.var(log[:, 64:128]))
    print(f"\nSaved multimodal_log.npy: ({n}, 256)")
    print(f"  Vision variance: {vis_variance:.4f} {'OK' if vis_variance > 0.01 else 'LOW - need diverse images'}")
    print(f"  Scenes: {len(images)} images x {(n // len(images) // 10)} blocks")
    print(f"\nRun: python main_multimodal.py {n}")


if __name__ == '__main__':
    main()
