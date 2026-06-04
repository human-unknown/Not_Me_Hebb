"""
multimodal_interface.py —— 多模态环境 (Stage 3A/3B)
自由能原理智能体

模式:
1. realtime:  摄像头实时 CLIP 编码 + 文本语料循环
2. full_log:  从 multimodal_log.npy 离线回放
3. clip_emb:  CLIP vision .npy + 文本语料
4. pseudo:    随机伪数据
"""

import numpy as np
from data_types import D


class MultimodalReplay:
    """多模态环境 — 离线回放或实时采集"""

    def __init__(self, log_path: str = 'multimodal_log.npy',
                 text_embeddings: np.ndarray = None,
                 body=None, n_frames: int = 500,
                 clip_vision_path: str = 'clip_vision.npy',
                 realtime: bool = False):
        self.realtime = realtime
        self.use_pseudo = False
        self.use_clip = False
        self.use_full_log = False

        if realtime:
            self._init_realtime(text_embeddings, n_frames)
        elif log_path and self._exists(log_path):
            self.log = np.load(log_path)
            self.n_frames = min(len(self.log), n_frames)
            self.use_full_log = True
            print(f"  Full multimodal log: {len(self.log)} frames from {log_path}")
        elif text_embeddings is not None:
            self.text_emb = text_embeddings
            self.n_frames = min(len(text_embeddings), n_frames)
            if self._exists(clip_vision_path):
                self.clip_vision = np.load(clip_vision_path)
                self.n_frames = min(len(self.clip_vision), self.n_frames)
                self.use_clip = True
                print(f"  CLIP vision: {len(self.clip_vision)} frames")
            else:
                self._generate_pseudo()
                self.use_pseudo = True
                print(f"  Pseudo mode: {self.n_frames} frames")
        else:
            self.n_frames = n_frames
            self._generate_pseudo()
            self.use_pseudo = True
            print(f"  Pseudo mode: {self.n_frames} frames")

        self.cursor = 0
        self.steps = 0
        self.last_output_embedding = None
        self.last_output_text = ''

    # ==========================================================
    # 实时模式 (Stage 3B)
    # ==========================================================
    def _init_realtime(self, text_embeddings, n_frames):
        print("  Mode: REALTIME (camera + CLIP)")
        import cv2
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        import torch

        self.text_emb = text_embeddings
        self.n_frames = n_frames
        self.use_realtime = True

        # 摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("  WARNING: Camera not available, falling back to pseudo")
            self.use_realtime = False
            self._generate_pseudo()
            self.use_pseudo = True
            return
        # 预热
        for _ in range(5):
            self.cap.read()
        print("  Camera ready")

        # CLIP
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.torch = torch
        self.Image = Image
        self.cv2 = cv2
        self.clip_frames = 0
        print("  CLIP ViT-B/32 ready")

    def _capture_frame(self) -> np.ndarray:
        """捕获并 CLIP 编码当前摄像头帧 → 64-dim"""
        ret, frame = self.cap.read()
        if not ret:
            return np.zeros(64, dtype=np.float32)
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        pil_img = self.Image.fromarray(rgb)
        inp = self.clip_proc(images=[pil_img], return_tensors="pt")
        with self.torch.no_grad():
            vis = self.clip_model.get_image_features(
                **inp).pooler_output.numpy()[0]
        self.clip_frames += 1
        vis64 = vis[:64].astype(np.float32)
        return vis64 / (np.linalg.norm(vis64) + 1e-8)

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    # ==========================================================
    # 伪数据
    # ==========================================================
    @staticmethod
    def _exists(path):
        import os
        return os.path.exists(path)

    def _generate_pseudo(self):
        rng = np.random.default_rng(42)
        self.pseudo_vision = np.zeros((self.n_frames, 64))
        for i in range(0, self.n_frames, 50):
            scene = rng.normal(0, 0.5, 64)
            end = min(i + 50, self.n_frames)
            for j in range(i, end):
                self.pseudo_vision[j] = scene + rng.normal(0, 0.05, 64)
        self.pseudo_audio = np.zeros((self.n_frames, 64))
        self.pseudo_audio[0] = rng.normal(0, 0.3, 64)
        for i in range(1, self.n_frames):
            self.pseudo_audio[i] = (0.95 * self.pseudo_audio[i-1]
                                    + 0.05 * rng.normal(0, 0.3, 64))
        self.pseudo_text = np.zeros((self.n_frames, 64))
        for i in range(self.n_frames):
            self.pseudo_text[i] = self.text_emb[i % len(self.text_emb)]

    # ==========================================================
    # 感知
    # ==========================================================
    def get_sensory(self, body=None) -> np.ndarray:
        s = np.zeros(D)

        if hasattr(self, 'use_realtime') and self.use_realtime:
            s[0:64] = self.text_emb[self.cursor % len(self.text_emb)]
            s[64:128] = self._capture_frame()
            s[128:192] = np.zeros(64)
        elif self.use_full_log:
            log_cols = self.log.shape[1]
            idx = self.cursor % self.n_frames
            if log_cols >= 320:
                s[0:320] = self.log[idx, :320]
            elif log_cols == 256:
                s[0:256] = self.log[idx]
            else:
                s[:log_cols] = self.log[idx]
        elif self.use_clip:
            idx = self.cursor % self.n_frames
            s[0:64] = self.text_emb[idx % len(self.text_emb)]
            s[64:128] = self.clip_vision[idx]
            s[128:192] = np.zeros(64)
        elif self.use_pseudo:
            idx = self.cursor % self.n_frames
            s[0:64] = self.pseudo_text[idx]
            s[64:128] = self.pseudo_vision[idx]
            s[128:192] = self.pseudo_audio[idx]

        if body is not None:
            b = body.b if hasattr(body, 'b') else body
            s[192:192 + min(len(b), 8)] = b[:8]

        s[256] = np.sin(2 * np.pi * self.steps / 1000.0)
        s[257] = np.cos(2 * np.pi * self.steps / 1000.0)
        s[258] = self.steps / max(self.steps, 1)

        if self.last_output_embedding is not None:
            s[320:328] = self.last_output_embedding[:8]

        return s

    # ==========================================================
    # 行动
    # ==========================================================
    def step(self, action_idx: int) -> float:
        if action_idx == 0:
            self.cursor = (self.cursor + 1) % self.n_frames
        elif action_idx == 1:
            self.cursor = max(0, self.cursor - 5)
        elif action_idx == 2:
            self.cursor = (self.cursor + 20) % self.n_frames
        elif action_idx == 3:
            pass
        elif action_idx == 4:
            pass
        self.steps += 1
        return 0.0

    def get_state_summary(self) -> dict:
        return {'cursor': self.cursor, 'n_frames': self.n_frames,
                'steps': self.steps, 'realtime': getattr(self, 'use_realtime', False)}
