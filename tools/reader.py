"""
reader.py — 通用文本阅读器 (v6.4)

Agent "读书"能力: 逐句消化任意 UTF-8 文本文件。

设计原则:
  - 不是"预训练"——是 Agent 以正常速度逐句阅读
  - 每句经过完整的 comprehend → learn → 情感反应 → 内部回应 管线
  - 有阅读疲劳模型——累了就"合上书"
  - 内容无关——不关心文本主题，所有处理由现有 Hebb 网络完成
  - corpus.txt 不使用——用户自行提供书籍文本

用法:
  reader = Reader()
  reader.load("book.txt")
  while (sentence := reader.next_sentence()):
      if not reader.should_read(agent_body, tpn_act):
          break  # Agent 累了，休息
      # ... 将 sentence 喂入 Agent 的 comprehend/step 管线
"""

import re
import os
import time
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 句子切分
# ============================================================

# 中文句子结束标点
_SENTENCE_END = re.compile(r'[。！？…！？\n]+')

# 最小句子长度 (过滤纯标点/空白行)
_MIN_SENTENCE_LEN = 2


def _split_sentences(text: str) -> list[str]:
    """将文本切分为句子列表。

    按中文标点切分，保留有意义的句子。
    """
    # 先按换行预分割
    paragraphs = text.split('\n')
    sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # 按标点切分
        parts = _SENTENCE_END.split(para)
        for part in parts:
            part = part.strip()
            if len(part) >= _MIN_SENTENCE_LEN:
                # 过滤纯符号/数字行
                char_ratio = sum(1 for c in part if '一' <= c <= '鿿'
                                or '぀' <= c <= 'ヿ'
                                or c.isalpha()) / max(len(part), 1)
                if char_ratio > 0.3:
                    sentences.append(part)
    return sentences


# ============================================================
# 阅读状态
# ============================================================

@dataclass
class ReadingState:
    """阅读器状态。"""
    file_path: str = ""
    file_name: str = ""           # 文件名 (显示用)
    total_sentences: int = 0      # 总句子数
    current_index: int = 0        # 当前句子索引
    sentences_read: int = 0       # 已读句子数
    started_at: float = 0.0       # 开始阅读时间 (time.time())
    paused_at: float = 0.0        # 暂停时间
    comprehension_history: list[float] = field(default_factory=list)  # 最近N句理解深度
    is_active: bool = False       # 是否有活跃的阅读任务
    is_paused: bool = False       # 是否暂停 (疲劳)


# ============================================================
# Reader
# ============================================================

class Reader:
    """通用文本阅读器。

    模拟 Agent "读书"——逐句提取文本，跟踪进度。

    疲劳模型:
      - 连续阅读提升 body.b[7] (认知负荷)
      - 认知负荷过高 → should_read() 返回 False → Agent "合上书"
      - 休息后自动恢复

    Attributes:
        state: 当前阅读状态
        fatigue_per_sentence: 每句阅读增加的认知负荷 [0, 1]
        recovery_rate: 休息时认知负荷衰减率
        max_fatigue: 触发暂停的认知负荷阈值
    """

    def __init__(self,
                 fatigue_per_sentence: float = 0.03,
                 recovery_rate: float = 0.01,
                 max_fatigue: float = 0.70):
        self.state = ReadingState()
        self._sentences: list[str] = []
        self._cognitive_load: float = 0.0   # 当前累积认知负荷

        self.fatigue_per_sentence = fatigue_per_sentence
        self.recovery_rate = recovery_rate
        self.max_fatigue = max_fatigue

    # ---- 文件操作 ----

    def load(self, file_path: str) -> ReadingState:
        """加载文本文件。

        Args:
            file_path: UTF-8 文本文件路径

        Returns:
            ReadingState — 阅读状态

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件无有效内容
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self._sentences = _split_sentences(text)
        if not self._sentences:
            raise ValueError(f"No valid sentences found in: {file_path}")

        self.state = ReadingState(
            file_path=os.path.abspath(file_path),
            file_name=os.path.basename(file_path),
            total_sentences=len(self._sentences),
            current_index=0,
            sentences_read=0,
            started_at=time.time(),
            is_active=True,
            is_paused=False,
        )
        self._cognitive_load = 0.0
        return self.state

    def load_from_text(self, text: str, name: str = "<string>") -> ReadingState:
        """从字符串加载 (用于测试)。

        Args:
            text: 文本内容
            name: 标识名
        """
        self._sentences = _split_sentences(text)
        if not self._sentences:
            raise ValueError("No valid sentences found in text")

        self.state = ReadingState(
            file_path=name,
            file_name=name,
            total_sentences=len(self._sentences),
            current_index=0,
            sentences_read=0,
            started_at=time.time(),
            is_active=True,
            is_paused=False,
        )
        self._cognitive_load = 0.0
        return self.state

    def close(self):
        """关闭阅读任务，重置状态。"""
        self.state = ReadingState()
        self._sentences = []
        self._cognitive_load = 0.0

    # ---- 阅读循环 ----

    def next_sentence(self) -> Optional[str]:
        """获取下一个句子。

        Returns:
            句子文本，或 None (已读完/无活跃任务)
        """
        if not self.state.is_active or self.state.is_paused:
            return None
        if self.state.current_index >= len(self._sentences):
            self.state.is_active = False
            return None

        sentence = self._sentences[self.state.current_index]
        self.state.current_index += 1
        self.state.sentences_read += 1

        # 累积认知负荷
        self._cognitive_load = min(1.0,
            self._cognitive_load + self.fatigue_per_sentence)

        return sentence

    def record_comprehension(self, depth: float):
        """记录当前句的理解深度 (用于追踪阅读质量)。

        Args:
            depth: 理解深度 [0, 1] (来自 understanding['n_triggered_memories'] 等)
        """
        self.state.comprehension_history.append(depth)
        # 只保留最近 50 句
        if len(self.state.comprehension_history) > 50:
            self.state.comprehension_history = \
                self.state.comprehension_history[-50:]

    # ---- 疲劳模型 ----

    def should_read(self, body_b, tpn_activation: float = 0.3) -> bool:
        """判断 Agent 是否应该继续阅读。

        基于:
          - body.b[7] (认知负荷) + 阅读累积负荷
          - body.b[4] (专注度)
          - TPN 激活度 (需要一定任务参与)

        Args:
            body_b: BodyVector.b numpy array
            tpn_activation: TPN 激活度 [0, 1]

        Returns:
            True = 应该继续阅读, False = "合上书"
        """
        if not self.state.is_active:
            return False
        if self.state.is_paused:
            return False
        if self.state.current_index >= len(self._sentences):
            return False

        # 综合认知负荷 = 身体固有 + 阅读累积
        body_cognitive = float(body_b[7]) if len(body_b) > 7 else 0.0
        total_load = 0.6 * body_cognitive + 0.4 * self._cognitive_load

        # 专注度
        focus = float(body_b[4]) if len(body_b) > 4 else 0.3

        # 太累 → 暂停
        if total_load > self.max_fatigue:
            self.state.is_paused = True
            self.state.paused_at = time.time()
            return False

        # 不专注 → 暂停
        if focus < 0.15:
            self.state.is_paused = True
            self.state.paused_at = time.time()
            return False

        # TPN 太低 → 不想读 (DMN 主导 = 走神)
        if tpn_activation < 0.15:
            self.state.is_paused = True
            self.state.paused_at = time.time()
            return False

        return True

    def try_resume(self, body_b, tpn_activation: float = 0.3) -> bool:
        """尝试从暂停中恢复。

        当认知负荷回落且专注度恢复时自动恢复阅读。

        Returns:
            True = 已恢复
        """
        if not self.state.is_paused:
            return False
        if not self.state.is_active:
            return False

        # 认知负荷恢复 (衰减)
        self._cognitive_load = max(0.0,
            self._cognitive_load - self.recovery_rate)

        body_cognitive = float(body_b[7]) if len(body_b) > 7 else 0.0
        total_load = 0.6 * body_cognitive + 0.4 * self._cognitive_load
        focus = float(body_b[4]) if len(body_b) > 4 else 0.3

        # 恢复条件: 认知负荷 < 阈值 × 0.7 (滞后) 且 专注度 > 0.25
        if total_load < self.max_fatigue * 0.7 and focus > 0.25 and tpn_activation > 0.2:
            self.state.is_paused = False
            return True
        return False

    # ---- 状态查询 ----

    @property
    def is_active(self) -> bool:
        return self.state.is_active and not self.state.is_paused

    @property
    def is_finished(self) -> bool:
        return self.state.current_index >= self.state.total_sentences and self.state.total_sentences > 0

    @property
    def progress(self) -> float:
        """阅读进度 [0, 1]."""
        if self.state.total_sentences == 0:
            return 0.0
        return min(1.0, self.state.current_index / self.state.total_sentences)

    @property
    def cognitive_load(self) -> float:
        return self._cognitive_load

    def get_progress(self) -> dict:
        """获取阅读进度摘要。"""
        return {
            'file_name': self.state.file_name,
            'file_path': self.state.file_path,
            'total_sentences': self.state.total_sentences,
            'current_index': self.state.current_index,
            'sentences_read': self.state.sentences_read,
            'progress': self.progress,
            'is_active': self.state.is_active,
            'is_paused': self.state.is_paused,
            'is_finished': self.is_finished,
            'cognitive_load': self._cognitive_load,
            'avg_comprehension': (sum(self.state.comprehension_history)
                                 / max(len(self.state.comprehension_history), 1))
                                 if self.state.comprehension_history else 0.0,
            'elapsed_seconds': time.time() - self.state.started_at
                               if self.state.started_at > 0 else 0.0,
        }

    def get_state_for_save(self) -> dict:
        """获取可序列化的状态 (用于持久化)。"""
        return {
            'file_path': self.state.file_path,
            'file_name': self.state.file_name,
            'total_sentences': self.state.total_sentences,
            'current_index': self.state.current_index,
            'sentences_read': self.state.sentences_read,
            'started_at': self.state.started_at,
            'paused_at': self.state.paused_at,
            'is_active': self.state.is_active,
            'is_paused': self.state.is_paused,
            'cognitive_load': self._cognitive_load,
            'comprehension_history': self.state.comprehension_history[-20:],  # 只保留最近
        }

    def restore_from_save(self, data: dict):
        """从持久化数据恢复。"""
        if not data or not data.get('is_active'):
            return
        self.state = ReadingState(
            file_path=data.get('file_path', ''),
            file_name=data.get('file_name', ''),
            total_sentences=data.get('total_sentences', 0),
            current_index=data.get('current_index', 0),
            sentences_read=data.get('sentences_read', 0),
            started_at=data.get('started_at', 0.0),
            paused_at=data.get('paused_at', 0.0),
            is_active=data.get('is_active', False),
            is_paused=data.get('is_paused', False),
            comprehension_history=data.get('comprehension_history', []),
        )
        self._cognitive_load = data.get('cognitive_load', 0.0)
        # 重新加载文件以获取句子列表
        if self.state.file_path and os.path.exists(self.state.file_path):
            try:
                with open(self.state.file_path, 'r', encoding='utf-8') as f:
                    self._sentences = _split_sentences(f.read())
            except Exception:
                self._sentences = []
