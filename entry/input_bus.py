"""
input_bus.py — NotMe v5.7 多模态同步输入总线

核心设计:
  每帧同时采集所有活跃输入通道，构建完整感知向量 s ∈ R^516。
  文本+视觉+听觉+痛觉+触觉同时激活 → Hebb 跨模态绑定从共现统计中涌现。

与 agent.step() 的分工:
  - InputBus: 采集 + 编码原始输入 → 构建 s[0:64](text) + s[64:372](vision) + ...
  - agent.step(): 层级处理 (视觉/听觉/痛觉管线前馈+反馈+PE) + 自由能 + 行动选择

用法:
  bus = InputBus()
  bus.set_text("你好")
  bus.set_image(frame)        # 摄像头帧
  bus.set_audio(audio_data)   # 麦克风频谱
  s, info = bus.build_sensory(text_env, img_enc)
  agent.step(s, t)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple


@dataclass
class InputBus:
    """统一多模态输入采集器 — 每帧构建完整感知向量.

    所有活跃通道同时存在于同一个 s 向量中，
    确保 Hebb 网络一次性学习跨模态共现模式。

    Attributes:
        text: 当前文本输入 (可为 None)
        image: 当前图像帧 (H,W,3) RGB uint8, 可为 None
        audio_data: 当前音频数据 (AudioInput dict 格式), 可为 None
        pain_intensity: 伤害性刺激强度 [0, 1]
        touch_intensity: Aβ触觉强度 [0, 1] (闸门控制)
        speaker_name: 说话人身份 (→ TPJ 意图推断)
    """

    text: Optional[str] = None
    image: Optional[np.ndarray] = None
    audio_data: Optional[dict] = None
    pain_intensity: float = 0.0
    touch_intensity: float = 0.0
    speaker_name: str = "human"

    def set_text(self, text: str):
        """设置文本输入."""
        self.text = text

    def set_image(self, image: np.ndarray):
        """设置图像帧 (RGB uint8, H×W×3)."""
        self.image = image

    def set_audio(self, audio_data: dict):
        """设置音频数据 (AudioInput.from_file/from_mic 格式)."""
        self.audio_data = audio_data

    def set_pain(self, intensity: float):
        """设置伤害性刺激强度 [0, 1]."""
        self.pain_intensity = float(np.clip(intensity, 0.0, 1.0))

    def set_touch(self, intensity: float):
        """设置 Aβ触觉强度 [0, 1] (激活闸门控制)."""
        self.touch_intensity = float(np.clip(intensity, 0.0, 1.0))

    def set_speaker(self, name: str):
        """设置说话人身份 (→ TPJ 心理理论)."""
        self.speaker_name = name

    # ---- 通道标记 ----
    @property
    def has_text(self) -> bool:
        return self.text is not None and len(self.text.strip()) > 0

    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def has_audio(self) -> bool:
        return self.audio_data is not None

    @property
    def has_pain(self) -> bool:
        return self.pain_intensity > 0.001

    @property
    def active_channels(self) -> list:
        """返回当前活跃的通道列表."""
        ch = []
        if self.has_text:    ch.append('text')
        if self.has_image:   ch.append('vision')
        if self.has_audio:   ch.append('audio')
        if self.has_pain:    ch.append('pain')
        if self.touch_intensity > 0.001: ch.append('touch')
        return ch

    # ================================================================
    # 感知向量构建
    # ================================================================

    def build_sensory(
        self,
        text_env,              # TextEnvironment instance
        img_enc=None,          # ImageEncoder instance (optional)
        vis_brain: dict = None,  # cross-modal visual brain (optional)
    ) -> Tuple[np.ndarray, dict]:
        """构建完整感知向量 s ∈ R^516.

        同时填入所有非零通道:
          s[0:64]    ← encode_text(text)       if text
          s[64:372]  ← Gabor(image)            if image
          s[80:88]   ← sentiment(text)         if text
          s[372:468] ← (由 agent.step() 听觉管线填充)
          s[468:516] ← (由 agent.step() 痛觉管线填充)

        Args:
            text_env: TextEnvironment — 提供文本编码
            img_enc: ImageEncoder — 提供视觉 Gabor 编码 (可选)
            vis_brain: 跨模态视觉脑 (可选, 用于 Scene B 补全)

        Returns:
            (s, info) where:
              s ∈ R^516 完整感知向量
              info: dict with channel metadata
        """
        from cns.data_types import D, M_V1_START, BINDING_END, D_VISUAL_V5

        VIS_START = M_V1_START      # 64
        VIS_END = BINDING_END       # 372

        s = np.zeros(D, dtype=np.float32)
        info = {
            'channels': [],
            'has_text': False,
            'has_vision': False,
            'has_audio': False,
            'has_pain': False,
            'display_text': '',
            'speaker_name': self.speaker_name,
        }

        # ---- Channel 1: Text [0:64] ----
        if self.has_text:
            try:
                s[0:64] = text_env.encode_text(self.text)
                info['has_text'] = True
                info['display_text'] = self.text
            except Exception:
                # Fallback: use random embedding from corpus
                s[0:64] = text_env.embeddings[0]
        else:
            # 无文本时给一个中性嵌入 (避免全零导致 Hebb 不学习)
            s[0:64] = np.zeros(64, dtype=np.float32)

        info['text_vec'] = s[0:64].copy()

        # ---- Channel 2: Vision [64:372] ----
        if self.has_image and img_enc is not None:
            try:
                from cerebrum.occipital_lobe.retina_lgn import build_visual_sensory
                vis_feat = img_enc.encode_from_array(self.image)
                s_vis = build_visual_sensory(vis_feat)
                s[VIS_START:VIS_END] = s_vis[VIS_START:VIS_END]
                info['has_vision'] = True
                info['channels'].append('vision')
            except Exception:
                pass
        elif self.has_text and vis_brain is not None:
            # Scene B: 文本→视觉跨模态补全 (无真实图像时)
            try:
                from entry.main_dialogue import _cross_modal_complete
                cross = _cross_modal_complete(
                    vis_brain, self.text, min_similarity=0.25)
                if cross:
                    s[VIS_START:VIS_END] = cross['visual_vec']
                    info['cross_modal_vision'] = True
                    info['channels'].append('crossmodal_vision')
            except Exception:
                pass

        # ---- Channel 3: Audio [372:468] ----
        # 音频数据交给 agent (agent.step() 内部调用 auditory_hierarchy)
        # InputBus 仅标记音频存在, 实际频谱由 agent.set_audio_input() 传入
        if self.has_audio:
            info['has_audio'] = True
            info['channels'].append('audio')
            info['audio_duration'] = self.audio_data.get('duration', 0.0)
            info['audio_is_speech'] = self.audio_data.get('is_speech', False)

        # ---- Channel 4: Sentiment [80:88] ----
        if self.has_text:
            try:
                from cerebrum.limbic_system.amygdala import (
                    analyze_sentiment, sentiment_to_social_signal,
                    get_emotional_lexicon,
                )
                emo_lexicon = get_emotional_lexicon()
                sentiment = analyze_sentiment(self.text, lexicon=emo_lexicon)
                social_signal = sentiment_to_social_signal(sentiment)
                s[80:88] = social_signal
                info['sentiment'] = sentiment
                info['social_signal'] = social_signal
            except Exception:
                s[80:88] = np.zeros(8, dtype=np.float32)
        else:
            s[80:88] = np.zeros(8, dtype=np.float32)

        # ---- Channel 5: Pain [468:516] ----
        # 痛觉数据交给 agent (agent.step() 内部调用 nociception_hierarchy)
        if self.has_pain:
            info['has_pain'] = True
            info['channels'].append('pain')
            info['pain_intensity'] = self.pain_intensity
            info['touch_intensity'] = self.touch_intensity

        # ---- 通道摘要 ----
        if info['has_text']:
            info['channels'].insert(0, 'text')
        info['n_active_channels'] = len(info['channels'])

        return s, info

    # ================================================================
    # 多模态命令解析
    # ================================================================

    @staticmethod
    def parse(raw_text: str) -> 'InputBus':
        """从原始输入文本解析多模态命令语法.

        支持的前缀:
          img:<path> [text]     → 图像 + 可选文本
          audio:<path> [text]   → 音频文件 + 可选文本
          mic:<duration> [text] → 麦克风录制 + 可选文本
          speaker:<name> <text> → 指定说话人 + 文本
          pain:<0-1> [text]     → 痛觉刺激 + 可选文本
          touch:<0-1> [text]    → 触觉刺激 + 可选文本

        纯文本 (无前缀) 直接作为 text 通道.

        Returns:
            InputBus 实例 (仅解析元数据, 不加载实际文件)
        """
        bus = InputBus()
        text = raw_text.strip()

        if not text:
            return bus

        # ---- img: prefix ----
        if text.startswith('img:'):
            parts = text[4:].strip().split(' ', 1)
            img_path = parts[0].strip('"\'')
            extra_text = parts[1] if len(parts) > 1 else None
            bus.text = extra_text or f"看到了 {img_path}"
            bus._pending_image_path = img_path
            bus._input_type = 'image'

        # ---- audio: prefix ----
        elif text.startswith('audio:'):
            audio_part = text[6:].strip()
            if ' ' in audio_part:
                parts = audio_part.split(' ', 1)
                audio_path = parts[0].strip('"\'')
                extra_text = parts[1] if len(parts) > 1 else None
            else:
                audio_path = audio_part.strip('"\'')
                extra_text = None
            bus.text = extra_text or "听到了声音"
            bus._pending_audio_path = audio_path
            bus._input_type = 'audio_file'

        # ---- mic: prefix ----
        elif text.startswith('mic:'):
            mic_part = text[4:].strip()
            duration = 3.0
            extra_text = None
            parts = mic_part.split(' ', 1)
            try:
                duration = float(parts[0])
                extra_text = parts[1] if len(parts) > 1 else None
            except ValueError:
                # No duration given, treat whole as text
                extra_text = mic_part
            duration = max(0.5, min(duration, 10.0))
            bus.text = extra_text or "听到了声音"
            bus._pending_mic_duration = duration
            bus._input_type = 'mic'

        # ---- speaker: prefix ----
        elif text.startswith('speaker:'):
            parts = text[8:].strip().split(' ', 1)
            bus.speaker_name = parts[0]
            bus.text = parts[1] if len(parts) > 1 else ""
            bus._input_type = 'text'

        # ---- pain: prefix ----
        elif text.startswith('pain:'):
            parts = text[5:].strip().split(' ', 1)
            try:
                bus.pain_intensity = float(parts[0])
            except ValueError:
                bus.pain_intensity = 0.5
            bus.text = parts[1] if len(parts) > 1 else None
            bus._input_type = 'pain'

        # ---- touch: prefix ----
        elif text.startswith('touch:'):
            parts = text[6:].strip().split(' ', 1)
            try:
                bus.touch_intensity = float(parts[0])
            except ValueError:
                bus.touch_intensity = 0.5
            bus.text = parts[1] if len(parts) > 1 else None
            bus._input_type = 'touch'

        # ---- plain text ----
        else:
            bus.text = text
            bus._input_type = 'text'

        return bus

    def resolve_pending(self, img_enc=None) -> Dict[str, Any]:
        """解析 parse() 阶段标记的待加载资源 (图像/音频).

        在 build_sensory() 之前调用, 完成实际 I/O.

        Args:
            img_enc: ImageEncoder (加载图像时必需)

        Returns:
            dict with resolved resources and any errors
        """
        result = {'errors': [], 'warnings': []}

        input_type = getattr(self, '_input_type', 'text')

        # ---- 加载图像 ----
        if input_type == 'image':
            img_path = getattr(self, '_pending_image_path', None)
            if img_path:
                try:
                    from PIL import Image
                    img = Image.open(img_path).convert('RGB')
                    self.image = np.array(img)
                    result['image_loaded'] = True
                    result['image_path'] = img_path
                    result['image_shape'] = self.image.shape
                except Exception as e:
                    result['errors'].append(f"Image load failed: {e}")
                    self.image = None
            else:
                result['errors'].append("No image path specified")

        # ---- 加载音频文件 ----
        elif input_type == 'audio_file':
            audio_path = getattr(self, '_pending_audio_path', None)
            if audio_path:
                try:
                    from tools.audio_io import AudioInput
                    self.audio_data = AudioInput.from_file(audio_path)
                    result['audio_loaded'] = True
                    result['audio_path'] = audio_path
                    result['audio_duration'] = self.audio_data.get('duration', 0.0)
                except Exception as e:
                    result['errors'].append(f"Audio load failed: {e}")
                    self.audio_data = None
            else:
                result['errors'].append("No audio path specified")

        # ---- 麦克风录制 ----
        elif input_type == 'mic':
            duration = getattr(self, '_pending_mic_duration', 3.0)
            try:
                from tools.audio_io import AudioInput
                self.audio_data = AudioInput.from_mic(duration_sec=duration)
                result['audio_loaded'] = True
                result['audio_duration'] = duration
            except Exception as e:
                result['errors'].append(f"Mic recording failed: {e}")
                self.audio_data = None

        return result

    def reset(self):
        """清空所有通道 (用于下一帧)."""
        self.text = None
        self.image = None
        self.audio_data = None
        self.pain_intensity = 0.0
        self.touch_intensity = 0.0
        # speaker_name persists across frames
        for attr in ('_pending_image_path', '_pending_audio_path',
                     '_pending_mic_duration', '_input_type'):
            if hasattr(self, attr):
                delattr(self, attr)

    def describe(self) -> str:
        """人类可读的输入描述 (用于终端显示)."""
        parts = []
        input_type = getattr(self, '_input_type', 'text')

        if input_type == 'image':
            path = getattr(self, '_pending_image_path', '?')
            parts.append(f"[img]: {path}")
        elif input_type == 'audio_file':
            path = getattr(self, '_pending_audio_path', '?')
            parts.append(f"[audio]: {path}")
        elif input_type == 'mic':
            dur = getattr(self, '_pending_mic_duration', 3.0)
            parts.append(f"[mic]: {dur:.1f}s")
        elif input_type == 'pain':
            parts.append(f"[pain]: {self.pain_intensity:.2f}")
        elif input_type == 'touch':
            parts.append(f"[touch]: {self.touch_intensity:.2f}")

        if self.speaker_name != 'human':
            parts.append(f"speaker={self.speaker_name}")

        if self.has_text:
            parts.append(self.text)

        return ' '.join(parts) if parts else '(empty)'
