"""
internal_life.py — 内部生命系统 (v6.4)

DMN 主导的自主心理活动——在无外部输入时维持 Agent 的"内心世界"。

核心功能:
  1. mind_wander:      DMN 主导的随机回忆 + 联想链 — 模拟走神
  2. internal_monologue: 对自己说话 (亚发声) — 完整语言闭环
  3. emotional_rumination: 高唤醒负效价记忆的重复回访 — 情感巩固

设计原则:
  - 全部复用现有模块 (Hippocampus/DMN/Broca/Wernicke/AF/PhonologicalLoop)
  - 不创造新记忆系统, 不引入新网络
  - 走神/独白/反刍的产出经过正常的 Hebb 学习管线 → 强化已有记忆
  - 模拟真实大脑在静息状态下的自发活动

相关脑区:
  - DMN (Default Mode Network):    走神/自我参照/自传体记忆
  - 海马:                          记忆回放/模式补全
  - Broca + 运动皮层 (亚发声):      内部言语
  - 杏仁核:                        情感记忆标记
  - 前额叶:                        反刍抑制 (低时 → 反刍失控, 类比抑郁症)

参考:
  - Buckner, R. L., Andrews-Hanna, J. R., & Schacter, D. L. (2008).
    The brain's default network. Annals of the New York Academy of Sciences.
  - Andrews-Hanna, J. R. (2012). The brain's default network and its
    adaptive role in internal mentation. The Neuroscientist.
"""

import numpy as np
import random as _random
from typing import Optional


# ============================================================
# 常量
# ============================================================

# 联想链参数
WANDER_CHAIN_LENGTH = 3           # 每次走神联想的步数
WANDER_RECALL_TEMP = 0.6          # 回忆温度 (控制联想广度)
WANDER_EMOTIONAL_ALPHA = 0.08     # 走神情感波动 → self_valence_ema

# 独白参数
MONOLOGUE_MAX_WORDS = 10          # 内部独白最大词数
MONOLOGUE_TEMPERATURE = 0.85      # 独白生成温度 (略高 = 更有创造性)

# 反刍参数
RUMINATION_MIN_VALENCE = -0.2     # 触发反刍的效价阈值
RUMINATION_MIN_AROUSAL = 0.5      # 触发反刍的唤醒阈值
RUMINATION_MAX_CYCLES = 4         # 每次反刍最多重复回访次数

# 自听情感传染
SELF_HEARING_ALPHA = 0.12         # 自听 → self_valence_ema 的 EMA 系数


# ============================================================
# InternalLife
# ============================================================

class InternalLife:
    """DMN 主导的自主内部活动引擎。

    用法:
      il = InternalLife()
      agent.internal_life = il

      # 在自主循环中调用:
      result = il.trigger(agent, 'wander')
      result = il.trigger(agent, 'monologue', broca=broca)
      result = il.trigger(agent, 'rumination')
    """

    def __init__(self):
        # 追踪
        self._wander_count: int = 0
        self._monologue_count: int = 0
        self._rumination_count: int = 0
        self._last_wander_cluster_id: int = -1  # 避免连续走神到同一记忆

    # ---- 主入口 ----

    def trigger(self, agent, thought_type: str = 'wander',
                broca=None) -> dict:
        """触发一次内部思维活动。

        Args:
            agent: Agent 实例
            thought_type: 'wander' | 'monologue' | 'rumination'
            broca: Broca 实例 (monologue 必需)

        Returns:
            dict with activity summary
        """
        if thought_type == 'wander':
            return self.mind_wander(agent)
        elif thought_type == 'monologue':
            return self.internal_monologue(agent, broca=broca)
        elif thought_type == 'rumination':
            return self.emotional_rumination(agent)
        else:
            return {'thought_type': thought_type, 'activity': 'unknown',
                    'error': f'Unknown thought_type: {thought_type}'}

    # ---- 走神 (Mind Wandering) ----

    def mind_wander(self, agent) -> dict:
        """走神模式: 随机回忆 → 联想链 → 情感波动。

        模拟 DMN 主导的自发思维流:
          1. 从海马随机选一个高激活集群作为起点
          2. recall() 该集群 → 将质心作为下一个查询
          3. 重复 2-3 步 → 形成联想链
          4. 每次 recall 的情感信号 → 轻量更新 self_valence_ema

        这同时测试了:
          - Hebb 模式补全能力 (部分激活 → 完整回忆)
          - 记忆联想结构 (链的连贯性)
          - 情感记忆持久性

        Returns:
            dict with chain_length, recalled_clusters, emotional_shift
        """
        net = agent.net
        if net.n_clusters < 3:
            return {'thought_type': 'wander', 'activity': 'wander',
                    'chain_length': 0, 'n_recalled': 0,
                    'reason': 'insufficient_clusters'}

        result = {
            'thought_type': 'wander',
            'activity': 'wander',
            'chain_length': 0,
            'n_recalled': 0,
            'emotional_shift': 0.0,
        }

        # 选择起点: 高激活集群中随机选 (模拟海马自发活动)
        top_n = min(15, net.n_clusters)
        top_clusters = sorted(net.clusters,
                             key=lambda c: c.activation, reverse=True)[:top_n]

        # 避免连续选同一个
        candidates = [c for i, c in enumerate(top_clusters)
                     if i != self._last_wander_cluster_id]
        if not candidates:
            candidates = top_clusters

        start_cluster = _random.choice(candidates)
        # Find index safely (Cluster.__eq__ returns array, don't use `in`)
        try:
            self._last_wander_cluster_id = top_clusters.index(start_cluster)
        except ValueError:
            self._last_wander_cluster_id = -1

        # 联想链: 使用完整 D-dim 质心进行 recall
        from cns.data_types import D as D_DIM
        current_vec = start_cluster.centroid.copy()
        chain_emotional = 0.0
        recalled_count = 0

        for step in range(WANDER_CHAIN_LENGTH):
            # recall: 用完整 D-dim 质心 + 文本段掩码
            mask = np.zeros(D_DIM, dtype=bool)
            mask[:64] = True  # 只用文本段匹配
            recalled = net.recall(current_vec, mask=mask)
            if recalled is None:
                break

            recalled_count += 1

            # 提取情感分量 (质心 [64:72] = emotion snapshot)
            if len(recalled.centroid) > 65:
                recalled_valence = float(recalled.centroid[64])
                recalled_arousal = float(recalled.centroid[65])
                chain_emotional += recalled_valence * 0.3 + recalled_arousal * 0.1

            # 下一个查询 = 已回忆质心的混合 (70% 旧链 + 30% 新回忆)
            current_vec = (0.7 * current_vec + 0.3 * recalled.centroid).astype(
                np.float32)

            # 温和的学习: 加强回忆路径 (用完整向量)
            old_mod = net.learn_rate_modifier
            net.learn_rate_modifier = 0.06  # 低学习率
            net.learn(current_vec)
            net.learn_rate_modifier = old_mod

        # 情感传染: 走神的情感色彩轻微影响 self_valence_ema
        if recalled_count > 0:
            avg_emotional = chain_emotional / recalled_count
            agent.self_valence_ema += WANDER_EMOTIONAL_ALPHA * avg_emotional
            agent.self_valence_ema = float(np.clip(
                agent.self_valence_ema, -1.0, 1.0))
            result['emotional_shift'] = float(
                WANDER_EMOTIONAL_ALPHA * avg_emotional)

        result['chain_length'] = recalled_count
        result['n_recalled'] = recalled_count
        self._wander_count += 1

        return result

    # ---- 内部独白 (Internal Monologue) ----

    def internal_monologue(self, agent, broca=None) -> dict:
        """内部独白: 对自己说话 (亚发声模式)。

        完整语言闭环 —— Agent "在心里说话" 而不发出声音:
          1. 从 DMN self_anchor 构建查询向量
          2. 调用 agent.speak() 的完整管线 (AF→Broca→Motor Cortex)
          3. MotorCortex.subvocal_plan() 替代 TTS (无声)
          4. 自听回路: 生成词 → 语音回路 → 自听情感传染
          5. AF 背侧: 运动副本 → 预期听觉 → 自我监控 PE

        这测试了:
          - 语言产出全管线 (在没有真实听众的情况下)
          - 语音工作记忆 (内部言语的语音编码)
          - 自我一致性 (我说的话是否符合我的自我锚点)

        Returns:
            dict with words_generated, speech_diag
        """
        result = {
            'thought_type': 'monologue',
            'activity': 'monologue',
            'words_generated': 0,
            'text': '',
            'self_monitoring_pe': 0.0,
        }

        if broca is None:
            result['error'] = 'broca_required'
            return result

        if agent.net.n_clusters < 5:
            result['error'] = 'insufficient_clusters'
            return result

        try:
            # 构建查询: 自我锚点 + 最近理解 + 最活跃信念
            self_anchor = agent.self_model.get_self_anchor()
            comp_vec = agent._last_comprehension

            top = max(agent.net.clusters, key=lambda c: c.activation)
            belief_vec = top.centroid.copy()

            # 查询混合 (让独白既反映自我又回应最近经历)
            query = (0.5 * self_anchor[:64]
                    + 0.3 * comp_vec[:64]
                    + 0.2 * belief_vec[:64]).astype(np.float32)

            v = agent.valence_history[-1] if agent.valence_history else 0.0
            a = agent.arousal_history[-1] if agent.arousal_history else 0.5

            # 调用完整 speak 管线 (内部使用 MotorCortex.subvocal_plan)
            words, audio, diag = agent.speak(
                broca=broca,
                query_vec=query,
                belief_vec=belief_vec,
                valence=v,
                arousal=a,
                max_words=MONOLOGUE_MAX_WORDS,
                temperature=MONOLOGUE_TEMPERATURE,
                use_phrase_structure=(agent.phrase_structure._trained
                                     if hasattr(agent, 'phrase_structure')
                                     and agent.phrase_structure._trained
                                     else False),
            )

            result['words_generated'] = len(words)
            if words:
                text = "".join(words)
                result['text'] = text
                agent._last_thought = text

                # 自听 → 语音回路 → 情感传染
                try:
                    from environments.text_interface import TextEnvironment
                    te = TextEnvironment()
                    thought_vec = te.encode_text(text).astype(np.float32)
                    agent.phonological_loop.hear(thought_vec[:64])

                    # 自听情感分析 (简化)
                    agent.self_valence_ema += SELF_HEARING_ALPHA * v
                    agent.self_valence_ema = float(np.clip(
                        agent.self_valence_ema, -1.0, 1.0))
                except Exception:
                    pass

            result['self_monitoring_pe'] = float(
                diag.get('self_monitoring_pe', 0.0))

            self._monologue_count += 1

        except Exception as e:
            result['error'] = str(e)

        return result

    # ---- 情绪反刍 (Emotional Rumination) ----

    def emotional_rumination(self, agent) -> dict:
        """情绪反刍: 高唤醒负效价记忆的重复回访。

        仅在 valence < RUMINATION_MIN_VALENCE 且 a > RUMINATION_MIN_AROUSAL 时触发。

        模拟抑郁症 DMN 过度活跃的机制:
          - 负效价记忆被反复 recall
          - 每次 recall 加强该记忆 (恶性循环)
          - 但适度反刍有助于情感记忆巩固 → 情感学习

        如果效价极低 (< -0.5) → 减少反刍次数 (保护机制, 避免"崩溃")

        Returns:
            dict with n_ruminated, emotional_intensity
        """
        result = {
            'thought_type': 'rumination',
            'activity': 'rumination',
            'n_ruminated': 0,
            'emotional_intensity': 0.0,
        }

        v = agent.valence_history[-1] if agent.valence_history else 0.0
        a = agent.arousal_history[-1] if agent.arousal_history else 0.0

        # 检查触发条件
        if v > RUMINATION_MIN_VALENCE or a < RUMINATION_MIN_AROUSAL:
            result['reason'] = 'below_threshold'
            return result

        net = agent.net
        if net.n_clusters < 5:
            result['reason'] = 'insufficient_clusters'
            return result

        # 情感强度 = |valence| * arousal → 决定反刍深度
        emotional_intensity = abs(v) * a
        n_cycles = max(1, min(RUMINATION_MAX_CYCLES,
                             int(emotional_intensity * 5)))

        # 极低效价 → 减少反刍 (保护)
        if v < -0.5:
            n_cycles = max(1, n_cycles - 2)

        # 找最接近当前情感状态的记忆集群
        # 构建完整 D-dim 查询 (文本+情感+身体快照)
        from cns.data_types import D as D_DIM
        query = np.zeros(D_DIM, dtype=np.float32)
        query[:64] = agent._last_comprehension[:64]
        query[64] = v
        query[65] = a
        # body snapshot
        if agent.body is not None and len(agent.body.b) >= 8:
            query[72:80] = agent.body.b[:8].astype(np.float32)

        mask = np.zeros(D_DIM, dtype=bool)
        mask[:72] = True  # 文本 + 情感

        ruminated = 0
        for _ in range(n_cycles):
            c = net.recall(query, mask=mask)
            if c is None:
                break

            # 提取情感分量
            if len(c.centroid) > 65:
                recalled_v = float(c.centroid[64])
                recalled_a = float(c.centroid[65])

                # 情感传染
                agent.self_valence_ema += 0.05 * recalled_v
                agent.self_arousal_ema += 0.03 * recalled_a
                agent.self_valence_ema = float(np.clip(
                    agent.self_valence_ema, -1.0, 1.0))
                agent.self_arousal_ema = float(np.clip(
                    agent.self_arousal_ema, 0.0, 1.0))

            # 加强该记忆 (反刍的核心: 重复激活)
            old_mod = net.learn_rate_modifier
            net.learn_rate_modifier = 0.10  # 稍高学习率
            net.learn(c.centroid)
            net.learn_rate_modifier = old_mod

            # 下一个查询用新回忆的质心 (可能进入不同情感区域)
            query = (0.8 * query + 0.2 * c.centroid).astype(np.float32)
            ruminated += 1

        result['n_ruminated'] = ruminated
        result['emotional_intensity'] = float(emotional_intensity)
        self._rumination_count += 1

        return result

    # ---- 状态查询 ----

    def get_state(self) -> dict:
        """获取内部生命系统状态摘要。"""
        return {
            'wander_count': self._wander_count,
            'monologue_count': self._monologue_count,
            'rumination_count': self._rumination_count,
            'total_internal_events': (self._wander_count
                                     + self._monologue_count
                                     + self._rumination_count),
        }

    def get_state_for_save(self) -> dict:
        """可序列化状态 (用于持久化)。"""
        return {
            'wander_count': self._wander_count,
            'monologue_count': self._monologue_count,
            'rumination_count': self._rumination_count,
            'last_wander_cluster_id': self._last_wander_cluster_id,
        }

    def restore_from_save(self, data: dict):
        """从持久化数据恢复。"""
        if not data:
            return
        self._wander_count = data.get('wander_count', 0)
        self._monologue_count = data.get('monologue_count', 0)
        self._rumination_count = data.get('rumination_count', 0)
        self._last_wander_cluster_id = data.get('last_wander_cluster_id', -1)
