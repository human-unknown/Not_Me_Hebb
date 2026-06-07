"""
telemetry.py — 长期遥测记录器 (v6.4)

结构化日志记录 Agent 生命周期内所有关键指标。
支持时间序列分析、可视化、和长期运行诊断。

设计原则:
  - 缓冲写入 CSV (避免高频 I/O)
  - 每行 = 一个 step 的所有关键指标
  - 独立的事件日志 (睡眠/对话/阅读)
  - 可查询的时间窗口摘要

CSV 输出路径: .notme/telemetry/
  - steps.csv      每步快照
  - sleep.csv      睡眠事件
  - dialogue.csv   对话轮次
  - reading.csv    阅读活动

用法:
  tel = Telemetry()
  tel.record_step(agent, activity='reading', extra_stats={...})
  tel.record_sleep(stats)
  tel.record_dialogue(turn_data)
  tel.flush()  # 每 N 步调用一次
"""

import os
import csv
import time
import numpy as np
from typing import Optional


# ============================================================
# 常量
# ============================================================

TELEMETRY_DIR = ".notme/telemetry"
FLUSH_INTERVAL = 60   # 每 60 步 flush 一次
BUFFER_MAX = 500      # 缓冲区上限 (超过强制 flush)


# ============================================================
# Telemetry
# ============================================================

class Telemetry:
    """长期遥测记录器。

    Attributes:
        session_id: 会话标识 (启动时间戳)
        step_count: 已记录步数
        buffer: 步数据缓冲区
        _last_flush: 上次 flush 的步数
    """

    # CSV 列定义
    STEP_COLUMNS = [
        'timestamp', 'step', 'session_step',
        'circadian_phase', 'is_asleep', 'sleep_state',
        'F_total', 'F_body', 'F_social', 'F_cognitive', 'F_accuracy',
        'valence', 'arousal',
        'tpn_act', 'dmn_act',
        'tonic_ne', 'rpe',
        'n_clusters', 'total_activation',
        'activity_mode',
        'body_b0', 'body_b1', 'body_b2', 'body_b3', 'body_b4',
        'body_b5', 'body_b6', 'body_b7', 'body_b8',
    ]

    SLEEP_COLUMNS = [
        'timestamp', 'step',
        'clusters_before', 'clusters_after',
        'total_replayed', 'total_downscaled', 'total_cleared',
        'total_emotional', 'total_cross_linked',
        'nrem_steps', 'rem_steps',
    ]

    DIALOGUE_COLUMNS = [
        'timestamp', 'step', 'turn',
        'human_text_len', 'agent_response_len',
        'comprehension_depth', 'n_triggered_memories',
        'valence_before', 'valence_after',
        'arousal_before', 'arousal_after',
        'eval_score',
    ]

    READING_COLUMNS = [
        'timestamp', 'step',
        'sentence_len', 'comprehension',
        'cognitive_load', 'progress',
    ]

    def __init__(self, session_id: str = None):
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.step_count: int = 0
        self._buffer: list[dict] = []
        self._last_flush_step: int = 0

        # 事件缓冲区
        self._sleep_events: list[dict] = []
        self._dialogue_events: list[dict] = []
        self._reading_events: list[dict] = []

        # 确保目录存在
        os.makedirs(TELEMETRY_DIR, exist_ok=True)

    # ---- 记录方法 ----

    def record_step(self, agent, activity: str = 'idle',
                    extra_stats: dict = None) -> dict:
        """记录单步快照。

        Args:
            agent: Agent 实例
            activity: 活动模式 ('idle'|'reading'|'wandering'|'monologue'|'streaming'|'dialogue')
            extra_stats: 额外统计

        Returns:
            记录的数据 dict
        """
        # 提取核心指标
        F_total = agent.F_history[-1] if agent.F_history else 0.0
        F_body = agent.F_body_history[-1] if agent.F_body_history else 0.0
        F_social = agent.F_social_history[-1] if agent.F_social_history else 0.0
        F_cognitive = agent.F_cognitive_history[-1] if agent.F_cognitive_history else 0.0
        F_accuracy = agent.F_accuracy_history[-1] if agent.F_accuracy_history else 0.0

        valence = agent.valence_history[-1] if agent.valence_history else 0.0
        arousal = agent.arousal_history[-1] if agent.arousal_history else 0.0

        tpn_act = agent.tpn.tpn_activation if hasattr(agent, 'tpn') else 0.0
        dmn_act = 1.0 - tpn_act  # TPN↔DMN 跷跷板

        tonic_ne = agent._lc_result.get('tonic_ne', 0.2) \
            if agent._lc_result else 0.2
        rpe = agent._vta_result.get('rpe', 0.0) \
            if agent._vta_result else 0.0

        n_clusters = agent.net.n_clusters
        total_act = agent.net.total_activation

        # SCN/VLPO
        circadian_phase = agent._circadian_state.circadian_phase \
            if hasattr(agent, '_circadian_state') else 0.0
        is_asleep = 1 if agent.vlpo.is_asleep else 0
        sleep_state = agent._sleep_state.state \
            if hasattr(agent, '_sleep_state') else 'awake'

        # 身体
        body_b = agent.body.b if agent.body else np.zeros(9)

        row = {
            'timestamp': time.time(),
            'step': self.step_count,
            'session_step': self.step_count,
            'circadian_phase': float(circadian_phase),
            'is_asleep': is_asleep,
            'sleep_state': sleep_state,
            'F_total': float(F_total),
            'F_body': float(F_body),
            'F_social': float(F_social),
            'F_cognitive': float(F_cognitive),
            'F_accuracy': float(F_accuracy),
            'valence': float(valence),
            'arousal': float(arousal),
            'tpn_act': float(tpn_act),
            'dmn_act': float(dmn_act),
            'tonic_ne': float(tonic_ne),
            'rpe': float(rpe),
            'n_clusters': int(n_clusters),
            'total_activation': float(total_act),
            'activity_mode': activity,
            'body_b0': float(body_b[0]) if len(body_b) > 0 else 0.0,
            'body_b1': float(body_b[1]) if len(body_b) > 1 else 0.0,
            'body_b2': float(body_b[2]) if len(body_b) > 2 else 0.0,
            'body_b3': float(body_b[3]) if len(body_b) > 3 else 0.0,
            'body_b4': float(body_b[4]) if len(body_b) > 4 else 0.0,
            'body_b5': float(body_b[5]) if len(body_b) > 5 else 0.0,
            'body_b6': float(body_b[6]) if len(body_b) > 6 else 0.0,
            'body_b7': float(body_b[7]) if len(body_b) > 7 else 0.0,
            'body_b8': float(body_b[8]) if len(body_b) > 8 else 0.0,
        }

        if extra_stats:
            row.update(extra_stats)

        self._buffer.append(row)
        self.step_count += 1

        # 自动 flush
        if len(self._buffer) >= BUFFER_MAX:
            self.flush()
        elif self.step_count - self._last_flush_step >= FLUSH_INTERVAL:
            self.flush()

        return row

    def record_sleep(self, sleep_stats: dict):
        """记录睡眠事件。

        Args:
            sleep_stats: dual_phase_sleep() 返回的 combined 统计
        """
        row = {
            'timestamp': time.time(),
            'step': self.step_count,
            'clusters_before': sleep_stats.get('clusters_before', 0),
            'clusters_after': sleep_stats.get('clusters_after', 0),
            'total_replayed': sleep_stats.get('total_replayed', 0),
            'total_downscaled': sleep_stats.get('total_downscaled', 0),
            'total_cleared': sleep_stats.get('total_cleared', 0),
            'total_emotional': sleep_stats.get('total_emotional', 0),
            'total_cross_linked': sleep_stats.get('total_cross_linked', 0),
            'nrem_steps': sleep_stats.get('nrem_steps', 0),
            'rem_steps': sleep_stats.get('rem_steps', 0),
        }
        self._sleep_events.append(row)

    def record_dialogue(self, turn_data: dict):
        """记录对话轮次。

        Args:
            turn_data: {
                'human_text_len': int,
                'agent_response_len': int,
                'comprehension_depth': float,
                'n_triggered_memories': int,
                'valence_before': float,
                'valence_after': float,
                'arousal_before': float,
                'arousal_after': float,
                'eval_score': float | None,
            }
        """
        row = {
            'timestamp': time.time(),
            'step': self.step_count,
            'turn': len(self._dialogue_events) + 1,
            **turn_data,
        }
        row['eval_score'] = row.get('eval_score') or 0.0
        self._dialogue_events.append(row)

    def record_reading(self, sentence: str, comprehension: float,
                       cognitive_load: float, progress: float):
        """记录阅读活动。

        Args:
            sentence: 阅读的句子文本
            comprehension: 理解深度 [0, 1]
            cognitive_load: 当前认知负荷
            progress: 阅读进度 [0, 1]
        """
        row = {
            'timestamp': time.time(),
            'step': self.step_count,
            'sentence_len': len(sentence),
            'comprehension': float(comprehension),
            'cognitive_load': float(cognitive_load),
            'progress': float(progress),
        }
        self._reading_events.append(row)

    # ---- I/O ----

    def flush(self):
        """将缓冲区写入磁盘。"""
        if not self._buffer and not self._sleep_events \
                and not self._dialogue_events and not self._reading_events:
            return

        # Steps CSV (追加)
        if self._buffer:
            steps_path = os.path.join(TELEMETRY_DIR,
                                      f"steps_{self.session_id}.csv")
            write_header = not os.path.exists(steps_path)
            with open(steps_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.STEP_COLUMNS,
                                        extrasaction='ignore')
                if write_header:
                    writer.writeheader()
                for row in self._buffer:
                    writer.writerow(row)
            self._buffer = []

        # Sleep CSV (追加)
        if self._sleep_events:
            sleep_path = os.path.join(TELEMETRY_DIR,
                                      f"sleep_{self.session_id}.csv")
            write_header = not os.path.exists(sleep_path)
            with open(sleep_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.SLEEP_COLUMNS)
                if write_header:
                    writer.writeheader()
                for row in self._sleep_events:
                    writer.writerow(row)
            self._sleep_events = []

        # Dialogue CSV (追加)
        if self._dialogue_events:
            dial_path = os.path.join(TELEMETRY_DIR,
                                     f"dialogue_{self.session_id}.csv")
            write_header = not os.path.exists(dial_path)
            with open(dial_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.DIALOGUE_COLUMNS)
                if write_header:
                    writer.writeheader()
                for row in self._dialogue_events:
                    writer.writerow(row)
            self._dialogue_events = []

        # Reading CSV (追加)
        if self._reading_events:
            read_path = os.path.join(TELEMETRY_DIR,
                                     f"reading_{self.session_id}.csv")
            write_header = not os.path.exists(read_path)
            with open(read_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.READING_COLUMNS)
                if write_header:
                    writer.writeheader()
                for row in self._reading_events:
                    writer.writerow(row)
            self._reading_events = []

        self._last_flush_step = self.step_count

    def get_summary(self, window_steps: int = 100) -> dict:
        """获取最近 N 步的统计摘要。

        Args:
            window_steps: 窗口大小 (步数)

        Returns:
            dict with mean/max/min for key metrics
        """
        recent = self._buffer[-window_steps:] if self._buffer else []
        if not recent:
            return {'n_steps': 0}

        def _safe_mean(key):
            vals = [r[key] for r in recent if key in r]
            return float(np.mean(vals)) if vals else 0.0

        def _safe_std(key):
            vals = [r[key] for r in recent if key in r]
            return float(np.std(vals)) if vals else 0.0

        return {
            'n_steps': len(recent),
            'window_step_start': recent[0]['step'] if recent else 0,
            'window_step_end': recent[-1]['step'] if recent else 0,
            'F_total': {'mean': _safe_mean('F_total'), 'std': _safe_std('F_total')},
            'F_body': {'mean': _safe_mean('F_body'), 'std': _safe_std('F_body')},
            'valence': {'mean': _safe_mean('valence'), 'std': _safe_std('valence')},
            'arousal': {'mean': _safe_mean('arousal'), 'std': _safe_std('arousal')},
            'sleep_ratio': _safe_mean('is_asleep'),
            'tpn_act': _safe_mean('tpn_act'),
            'tonic_ne': {'mean': _safe_mean('tonic_ne'), 'std': _safe_std('tonic_ne')},
            'n_clusters': {'mean': _safe_mean('n_clusters'), 'max': max(
                (r.get('n_clusters', 0) for r in recent), default=0)},
            'total_activation': {'mean': _safe_mean('total_activation')},
            'body_b0': _safe_mean('body_b0'),  # 社交需求
            'body_b2': _safe_mean('body_b2'),  # 压力
            'body_b7': _safe_mean('body_b7'),  # 认知负荷
            'activity_modes': {
                mode: sum(1 for r in recent if r.get('activity_mode') == mode)
                for mode in set(r.get('activity_mode', 'idle') for r in recent)
            },
        }

    def get_recent_steps(self, n: int = 200) -> list[dict]:
        """获取最近 N 步数据 (用于 Web API)。"""
        return self._buffer[-n:] if self._buffer else []

    def get_session_info(self) -> dict:
        """获取会话元信息。"""
        return {
            'session_id': self.session_id,
            'total_steps': self.step_count,
            'buffer_size': len(self._buffer),
            'sleep_events': len(self._sleep_events),
            'dialogue_events': len(self._dialogue_events),
            'reading_events': len(self._reading_events),
            'last_flush_step': self._last_flush_step,
        }

    def clear(self):
        """清空所有数据 (慎用)。"""
        self._buffer = []
        self._sleep_events = []
        self._dialogue_events = []
        self._reading_events = []
        self.step_count = 0
        self._last_flush_step = 0
