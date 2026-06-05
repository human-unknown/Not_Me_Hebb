"""
dialogue_memory.py —— 对话工作记忆 + Wernicke 区理解回路

脑区映射:
- DialogueContext:    海马体 (情景记忆) + DLPFC (工作记忆)
- comprehend():       Wernicke 区 (语言理解)
- ResponseMonitor:    ACC (冲突监控) + OFC (社会适当性)

核心原则:
- 对话是时间序列，不是孤立回合
- 理解 = 激活相关记忆 + 提取意图 + 评估情感
- 回应质量 = 相关性 × 新颖性 × 连贯性
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DialogueTurn:
    """单轮对话记录"""
    human_text: str = ""
    human_vec: np.ndarray = None        # 人类输入的语义编码 (64,)
    human_sentiment: np.ndarray = None   # 情感信号 (8,)
    agent_response: str = ""
    agent_vec: np.ndarray = None         # Agent 回应的语义编码 (64,)
    agent_valence: float = 0.0
    agent_arousal: float = 0.0
    comprehension_vec: np.ndarray = None  # Agent 理解的内容 (64,)
    turn_id: int = 0

    def __post_init__(self):
        if self.human_vec is None:
            self.human_vec = np.zeros(64, dtype=np.float32)
        if self.human_sentiment is None:
            self.human_sentiment = np.zeros(8, dtype=np.float32)
        if self.agent_vec is None:
            self.agent_vec = np.zeros(64, dtype=np.float32)
        if self.comprehension_vec is None:
            self.comprehension_vec = np.zeros(64, dtype=np.float32)


class DialogueContext:
    """对话情景记忆 —— 海马体 + DLPFC 等价物。

    存储最近 N 轮对话，提供上下文向量用于:
    1. 理解: "刚才在说什么？"
    2. 生成: "我应该接着说什么？"
    3. 自评: "我是不是在重复自己？"

    记忆衰减: 越老的轮次权重越低 (时间折扣 gamma=0.85)
    """

    def __init__(self, max_turns: int = 10, gamma: float = 0.85):
        self.turns: list[DialogueTurn] = []
        self.max_turns = max_turns
        self.gamma = gamma          # 时间折扣
        self.turn_counter: int = 0
        # EMA 追踪对话质量
        self.coherence_ema: float = 0.5   # 连贯性
        self.engagement_ema: float = 0.5  # 参与度

    def add_turn(self, human_text: str, human_vec: np.ndarray,
                 human_sentiment: np.ndarray,
                 agent_response: str, agent_vec: np.ndarray,
                 agent_valence: float, agent_arousal: float,
                 comprehension_vec: np.ndarray):
        """添加一轮对话"""
        turn = DialogueTurn(
            human_text=human_text,
            human_vec=human_vec.astype(np.float32),
            human_sentiment=human_sentiment.astype(np.float32),
            agent_response=agent_response,
            agent_vec=agent_vec.astype(np.float32),
            agent_valence=agent_valence,
            agent_arousal=agent_arousal,
            comprehension_vec=comprehension_vec.astype(np.float32),
            turn_id=self.turn_counter,
        )
        self.turns.append(turn)
        self.turn_counter += 1

        # 维持窗口
        while len(self.turns) > self.max_turns:
            self.turns.pop(0)

        # 更新连贯性 EMA
        if len(self.turns) >= 2:
            prev_h = self.turns[-2].human_vec
            curr_r = agent_vec
            sim = float(np.dot(prev_h, curr_r) / (
                np.linalg.norm(prev_h) * np.linalg.norm(curr_r) + 1e-8))
            self.coherence_ema += 0.2 * (sim - self.coherence_ema)

    def get_context_vector(self) -> np.ndarray:
        """获取对话上下文的加权语义向量。

        最近轮次权重高，旧轮次指数衰减。
        Returns: (64,) float32 vector
        """
        if not self.turns:
            return np.zeros(64, dtype=np.float32)

        weights = np.array([self.gamma ** (len(self.turns) - 1 - i)
                           for i in range(len(self.turns))])
        weights /= weights.sum()

        ctx = np.zeros(64, dtype=np.float32)
        for i, turn in enumerate(self.turns):
            # v3: 精度加权 — comprehension 有内容时精度更高
            has_comp = float(np.sum(np.abs(turn.comprehension_vec)) > 0.01)
            h_p = 1.0; a_p = 1.0; c_p = 0.5 + 0.5 * has_comp
            total_p = h_p + a_p + c_p
            turn_vec = (turn.human_vec * (h_p / total_p)
                        + turn.agent_vec * (a_p / total_p)
                        + turn.comprehension_vec * (c_p / total_p))
            ctx += weights[i] * turn_vec
        return ctx.astype(np.float32)

    def get_recent_topic(self) -> np.ndarray:
        """最近一轮的话题向量"""
        if not self.turns:
            return np.zeros(64, dtype=np.float32)
        return self.turns[-1].human_vec.copy()

    def get_emotion_context(self) -> dict:
        """最近的对话情感状态"""
        if not self.turns:
            return {'human_valence': 0.0, 'agent_valence': 0.0,
                    'coherence': 0.5}
        recent = self.turns[-3:] if len(self.turns) >= 3 else self.turns
        return {
            'human_valence': float(np.mean([t.human_sentiment[6]
                                  if len(t.human_sentiment) > 6 else 0
                                  for t in recent])),
            'agent_valence': float(np.mean([t.agent_valence for t in recent])),
            'coherence': float(self.coherence_ema),
        }

    def is_repeating(self, candidate_response: str, threshold: float = 0.7
                     ) -> bool:
        """检查是否在重复最近说过的话"""
        if not self.turns or not candidate_response:
            return False
        recent_responses = [t.agent_response for t in self.turns[-3:]]
        for prev in recent_responses:
            if prev and len(prev) > 1:
                # 简单子串匹配: 长句包含短句
                if len(candidate_response) >= len(prev) * 0.7:
                    if prev in candidate_response or candidate_response in prev:
                        return True
        return False

    def n_turns(self) -> int:
        return len(self.turns)


# ============================================================
# Wernicke 区: 语言理解
# ============================================================

def comprehend(human_vec: np.ndarray,              # 人类输入语义 (64,)
               human_sentiment: np.ndarray,        # 情感信号 (8,)
               agent_net,                          # L0 ClusterNetwork
               dialogue_ctx: DialogueContext,      # 对话记忆
               body_state,                         # BodyVector
               valence: float,                     # 当前效价
               arousal: float,                     # 当前唤醒
               ) -> tuple[np.ndarray, dict]:
    """Wernicke 区等价物 —— 语言理解。

    不是做 NLP 解析，而是:
    1. 在记忆中激活相关集群 (这让我想起什么？)
    2. 提取对话上下文 (刚才在说什么？)
    3. 评估情感影响 (这让我感觉如何？)
    4. 形成"理解向量" — 用于驱动 Broca 区回应

    Returns:
        (comprehension_vec, understanding)
        - comprehension_vec: (64,) 理解内容 — 传入 Broca
        - understanding: dict — 可解释的理解摘要
    """
    # ---- 1. 记忆激活: 人类输入触发了哪些记忆？ ----
    from cns.data_types import D
    s_full = np.zeros(D, dtype=np.float32)
    s_full[:64] = human_vec[:64].astype(np.float32)

    triggered_memories = []
    triggered_sims = []
    if agent_net.n_clusters > 0:
        h = agent_net.hash_features(s_full)
        mask = np.zeros(D, dtype=bool)
        mask[:64] = True
        from cerebrum.limbic_system.hippocampus import _masked_cosine

        # 跨桶检索 top-5 触发记忆
        for c in agent_net.clusters:
            sim = _masked_cosine(h, c.centroid, mask)
            triggered_memories.append((c, float(sim)))
        triggered_memories.sort(key=lambda x: x[1], reverse=True)
        triggered_memories = triggered_memories[:5]
        triggered_sims = [s for _, s in triggered_memories]

    # ---- 2. 从触发记忆中提取语义 ----
    memory_context = np.zeros(64, dtype=np.float32)
    if triggered_memories:
        total_sim = sum(max(s, 0.01) for s in triggered_sims)
        for (c, sim), s in zip(triggered_memories, triggered_sims):
            weight = max(0.0, sim) / total_sim
            memory_context += weight * c.centroid[:64]

    # ---- 3. 对话上下文 ----
    dialogue_context = dialogue_ctx.get_context_vector()

    # ---- 4. 情感评估: 这个输入让我感觉如何？ ----
    # 人类情感信号影响理解 (s[80:88])
    human_v = float(human_sentiment[6]) if len(human_sentiment) > 6 else 0.0
    human_a = float(human_sentiment[1]) if len(human_sentiment) > 1 else 0.0

    # 情感一致性: 人类效价 vs Agent 效价
    emotional_congruence = 1.0 - abs(human_v - valence) * 0.5

    # 社交需求: 身体需求 vs 互动质量
    social_need = max(0.0, float(body_state.setpoints[0] - body_state.b[0]))

    # ---- 5. 合成理解向量 (v3: 精度加权 — 高精度通道自然获得更高权重) ----
    # 精度 = 通道信号的信噪比代理
    # human_vec[:64]:       输入总是精确的 (precision=1.0)
    # memory_context:       记忆匹配度 = 精度 (top_sim 越高 → 越精确)
    # dialogue_context:     上下文窗口存在 = 精度 (有对话历史 → higher)
    # emotional_congruence: 情感一致性 = 调节器, 非独立通道

    mem_precision = max(0.1, triggered_sims[0] if triggered_sims else 0.1)
    ctx_precision = min(1.0, dialogue_ctx.n_turns() / 5.0)  # 5+ 轮 → 满精度
    input_precision = 1.0

    total_precision = input_precision + mem_precision + ctx_precision

    comprehension_vec = (
        human_vec[:64] * (input_precision / total_precision)
        + memory_context * (mem_precision / total_precision)
        + dialogue_context * (ctx_precision / total_precision)
    ).astype(np.float32)
    # 情感一致性作为调节器 (缩放, 不是混合)
    comprehension_vec *= (0.7 + 0.3 * emotional_congruence)

    # 归一化
    norm = np.linalg.norm(comprehension_vec)
    if norm > 1e-8:
        comprehension_vec /= norm
        comprehension_vec *= np.linalg.norm(human_vec[:64])  # 保持原始规模

    # ---- 6. 构建可解释的理解摘要 ----
    top_memory_sim = triggered_sims[0] if triggered_sims else 0.0
    understanding = {
        'comprehension_vec': comprehension_vec,
        'top_memory_similarity': top_memory_sim,
        'n_triggered_memories': len([s for s in triggered_sims if s > 0.3]),
        'emotional_congruence': emotional_congruence,
        'social_need': social_need,
        'dialogue_coherence': dialogue_ctx.coherence_ema,
        'human_valence': human_v,
        'human_arousal': human_a,
    }
    return comprehension_vec, understanding


# ============================================================
# ACC + OFC: 响应监控
# ============================================================

def evaluate_response(response_vec: np.ndarray,       # 回应的语义编码 (64,)
                       comprehension_vec: np.ndarray,  # 理解内容 (64,)
                       dialogue_ctx: DialogueContext,
                       agent_net,
                       ) -> dict:
    """ACC + OFC 等价物 —— 评估生成的回应。

    检查:
    1. 相关性: 回应是否与理解相关？
    2. 新颖性: 是否在重复自己？
    3. 连贯性: 是否延续了对话？

    Returns:
        {'relevance': float, 'novelty': float, 'coherence': float,
         'overall_score': float, 'acceptable': bool}
    """
    # 相关性 (v2: FEP 驱动 — 集群历史经验)
    # 回应在 L0 中触发什么记忆？该记忆的历史 F_signal 低 = 好模式
    from cns.data_types import D
    from cerebrum.limbic_system.hippocampus import _masked_cosine
    query = np.zeros(D, dtype=np.float32)
    query[:64] = response_vec[:64].astype(np.float32)
    mask = np.zeros(D, dtype=bool); mask[:64] = True
    h = agent_net.hash_features(query)
    best_F_signal = 0.5; best_sim = 0.0
    if agent_net.n_clusters > 0:
        for c in agent_net.clusters:
            sim = _masked_cosine(h, c.centroid, mask)
            if sim > best_sim:
                best_sim = sim; best_F_signal = float(c.F_signal)
    # 高匹配 + 低历史 F = 好回应
    relevance = best_sim * max(0.0, 1.0 - best_F_signal * 0.15)

    # 新颖性: 是否与最近回应高度相似？
    novelty = 1.0
    if dialogue_ctx.n_turns() > 0:
        recent_responses = [t.agent_vec for t in dialogue_ctx.turns[-3:]
                          if t.agent_vec is not None]
        if recent_responses:
            sims = []
            for prev in recent_responses:
                d = (np.linalg.norm(response_vec[:64]) * np.linalg.norm(prev[:64]) + 1e-8)
                sims.append(float(np.dot(response_vec[:64], prev[:64]) / d))
            max_sim_to_past = max(sims)
            novelty = max(0.0, 1.0 - max_sim_to_past)

    # 连贯性: 与对话上下文的契合度
    ctx_vec = dialogue_ctx.get_context_vector()
    denom_c = (np.linalg.norm(response_vec[:64]) * np.linalg.norm(ctx_vec[:64]) + 1e-8)
    coherence = float(np.dot(response_vec[:64], ctx_vec[:64]) / denom_c)

    # 综合: 精度加权 (v3: 各维度精度驱动权重, 无手设比例)
    # relevance精度 ∝ cluster_match (高匹配 = 高置信度)
    # novelty精度  ∝ 1.0 (客观计算)
    # coherence精度 ∝ n_turns/5 (更多上下文 = 更高精度)
    ctx_precision = min(1.0, dialogue_ctx.n_turns() / 5.0)
    rel_precision = min(1.0, best_sim * 2.0)
    nov_precision = 1.0
    total_p = rel_precision + nov_precision + ctx_precision
    overall = (relevance * rel_precision + novelty * nov_precision
              + coherence * ctx_precision) / total_p
    acceptable = overall > 0.12 or (best_sim > 0.3 and best_F_signal < 1.0)

    return {
        'relevance': relevance,
        'novelty': novelty,
        'coherence': coherence,
        'overall_score': overall,
        'acceptable': acceptable,
        'F_signal': best_F_signal,     # v2: 集群历史自由能
        'cluster_match': best_sim,      # v2: 集群匹配度
    }


# ============================================================
# 睡眠巩固: 海马 → 皮层记忆转移
# ============================================================

def consolidate_dialogue_memory(
    dialogue_ctx: 'DialogueContext',
    agent_net,                          # L0 ClusterNetwork
    self_model,                         # SelfModel (DMN/vmPFC)
    body_state,                         # BodyVector
    valence: float,
    arousal: float,
    broca=None,                         # optional: Broca for word-order consolidation
) -> dict:
    """睡眠巩固：将对话经验从海马体整合到皮层长期记忆。

    模拟:
    1. **海马重放** (Hippocampal replay): 对话轮次以压缩时间重放至 L0
       — 情感显著性决定重放次数 (1-5x)
       — 高唤醒 → 更多重放 (杏仁核调制)
    2. **系统巩固** (Systems consolidation): 海马记忆 → 皮层长期记忆
       — 临时 learn_rate 提升至 2-3x
       — 创建综合记忆痕迹: [输入|回应|理解|情感快照]
    3. **交叉关联** (Cross-association): 语义相关的轮次建立 Hebb 连接
       — 轮次对的余弦相似度 > 0.4 → 创建桥接模式
    4. **自我模型整合**: 会话摘要 → DMN 锚点更新
       — 加权平均所有体验 → add_experience()
    5. **突触稳态** (Synaptic homeostasis): 弱集群修剪
       — 移除激活度最低的 20% 对话集群
    6. **词序巩固**: 对话中高频 trigram 强化

    神经学等价检查:
    - 无巩固 → 顺行性遗忘 (无法形成长期记忆)
    - 仅海马存储 → 记忆随时间衰减 (艾宾浩斯遗忘曲线)
    - 巩固后 → 记忆抵抗干扰，可被多线索检索

    Returns:
        dict: 巩固统计信息
    """
    turns = dialogue_ctx.turns
    n_turns = len(turns)
    if n_turns == 0:
        return {
            'consolidated': 0, 'replays': 0, 'cross_links': 0,
            'pruned': 0, 'self_updated': False,
            'phase': 'skip', 'n_turns': 0,
        }

    from cns.data_types import D
    import numpy as np

    # ---- 备份原始参数 ----
    orig_lr = agent_net.theta.learn_rate_l0
    n_replays_total = 0
    n_consolidated = 0
    n_cross_links = 0

    # ---- Phase 1: 情感门控的海马重放 ----
    for turn in turns:
        # 情感显著性: 高唤醒 + 极端效价 → 更多重放
        emotional_salience = abs(turn.agent_valence) * 0.4 + turn.agent_arousal * 0.6
        n_replays = int(1 + emotional_salience * 4)  # 1-5 次重放
        n_replays = max(1, min(5, n_replays))

        # 临时提升学习率: 2x-3x (睡眠中可塑性增强)
        boosted_lr = orig_lr * (2.0 + emotional_salience)
        agent_net.theta.learn_rate_l0 = min(0.30, boosted_lr)

        for replay_i in range(n_replays):
            # 构建综合记忆痕迹
            # 布局: [human(64)|agent_resp(64)|comprehension(64)|emotion(8)|body(8)|padding]
            trace = np.zeros(D, dtype=np.float32)
            trace[:64] = turn.human_vec[:64].astype(np.float32)
            trace[64:128] = turn.agent_vec[:64].astype(np.float32)
            if turn.comprehension_vec is not None:
                trace[128:192] = turn.comprehension_vec[:64].astype(np.float32)

            # 情感快照 [192:200]
            trace[192] = turn.agent_valence
            trace[193] = turn.agent_arousal
            trace[194] = float(turn.human_sentiment[6]) if len(turn.human_sentiment) > 6 else 0.0
            trace[195] = float(turn.human_sentiment[1]) if len(turn.human_sentiment) > 1 else 0.0

            # 身体快照 [200:208]
            b = body_state.b
            for i in range(min(8, len(b))):
                trace[200 + i] = b[i]

            # 首轮重放: 高学习率 → 强记忆痕迹
            # 后续重放: 递减学习率 → 巩固但不过度
            decayed_lr = boosted_lr * (0.85 ** replay_i)
            agent_net.theta.learn_rate_l0 = max(orig_lr * 1.5, decayed_lr)

            agent_net.learn(trace)
            n_replays_total += 1

        n_consolidated += 1

    # ---- Phase 2: 交叉关联 (轮次间的 Hebb 桥接) ----
    # 两种关联:
    #   (a) 时序相邻: 连续轮次天然有上下文关联 (对话连贯性)
    #   (b) 语义相似: 非相邻但话题相关的轮次 (长程语义关联)
    if n_turns >= 2:
        for i in range(n_turns):
            for j in range(i + 1, n_turns):
                # 计算轮次对的人类输入语义相似度
                sim = float(np.dot(
                    turns[i].human_vec[:64], turns[j].human_vec[:64]) / (
                    np.linalg.norm(turns[i].human_vec[:64])
                    * np.linalg.norm(turns[j].human_vec[:64]) + 1e-8))

                # 时序相邻 (j == i+1) 或语义相似 (sim > 0.25)
                is_adjacent = (j == i + 1)
                should_link = is_adjacent or sim > 0.25

                if should_link:
                    # 桥接权重: 相邻轮次基线 0.6, 语义相似度加成
                    link_weight = 0.6 if is_adjacent else 0.0
                    link_weight = max(link_weight, sim)
                    # 桥接模式: [turn_i 的输入 | turn_j 的回应]
                    bridge = np.zeros(D, dtype=np.float32)
                    bridge[:64] = turns[i].human_vec[:64].astype(np.float32)
                    bridge[64:128] = turns[j].agent_vec[:64].astype(np.float32)
                    agent_net.theta.learn_rate_l0 = orig_lr * (0.8 + link_weight)
                    agent_net.learn(bridge)
                    n_cross_links += 1

    # ---- Phase 3: 自我模型整合 ----
    # 构建会话摘要: 加权平均所有体验
    if n_turns > 0:
        weights = np.array([
            abs(t.agent_valence) * 0.4 + t.agent_arousal * 0.3 + 0.3
            for t in turns
        ])
        weights /= weights.sum()

        session_response = np.zeros(64, dtype=np.float32)
        session_comprehension = np.zeros(64, dtype=np.float32)
        session_valence = 0.0
        session_arousal = 0.0

        for i, turn in enumerate(turns):
            w = weights[i]
            session_response += w * turn.agent_vec[:64].astype(np.float32)
            if turn.comprehension_vec is not None:
                session_comprehension += w * turn.comprehension_vec[:64].astype(np.float32)
            session_valence += w * turn.agent_valence
            session_arousal += w * turn.agent_arousal

        ctx_vec = dialogue_ctx.get_context_vector()

        # 获取 self_model 的当前情感 EMA (如果有)
        sv_ema = getattr(self_model, 'anchor_emotion', np.zeros(8, dtype=np.float32))
        sv = float(sv_ema[0]) if len(sv_ema) > 0 else session_valence
        sa = float(sv_ema[1]) if len(sv_ema) > 1 else session_arousal
        coh = float(sv_ema[4]) if len(sv_ema) > 4 else 0.5

        self_model.add_experience(
            response_vec=session_response,
            valence=session_valence,
            arousal=session_arousal,
            self_valence_ema=sv,
            self_arousal_ema=sa,
            self_coherence=coh,
            body_state=body_state,
            comprehension_vec=session_comprehension,
            dialogue_ctx_vec=ctx_vec,
        )
        self_updated = True
    else:
        self_updated = False

    # ---- Phase 4: 词序巩固 (可选: Broca trigram 强化) ----
    if broca is not None and hasattr(broca, '_learn_from_sentence'):
        for turn in turns:
            if turn.agent_response and len(turn.agent_response) > 3:
                try:
                    broca._learn_from_sentence(turn.agent_response)
                except Exception:
                    pass

    # ---- Phase 5: 突触稳态 (弱集群修剪) ----
    n_pruned = 0
    if agent_net.n_clusters > 20:
        # 衰减所有集群
        agent_net.decay()
        # 移除最弱的 15% (但保留至少 20 个集群)
        min_keep = 20
        threshold_idx = max(min_keep, int(agent_net.n_clusters * 0.85))
        if agent_net.n_clusters > threshold_idx:
            sorted_clusters = sorted(agent_net.clusters,
                                     key=lambda c: c.activation, reverse=True)
            keep_set = set(id(c) for c in sorted_clusters[:threshold_idx])
            to_remove = [c for c in agent_net.clusters if id(c) not in keep_set]
            for c in to_remove:
                key = agent_net._hash_to_bucket(c.centroid)
                if key in agent_net.buckets:
                    agent_net.buckets[key] = [
                        x for x in agent_net.buckets[key] if x is not c]
            agent_net.clusters = sorted_clusters[:threshold_idx]
            n_pruned = len(to_remove)

    # ---- 恢复原始学习率 ----
    agent_net.theta.learn_rate_l0 = orig_lr

    # ---- Post-consolidation: 保留最后 2 轮维持上下文连续性 ----
    while len(dialogue_ctx.turns) > 2:
        dialogue_ctx.turns.pop(0)

    # 重置连贯性追踪 (新会话段)
    dialogue_ctx.coherence_ema = 0.5

    return {
        'consolidated': n_consolidated,
        'replays': n_replays_total,
        'cross_links': n_cross_links,
        'pruned': n_pruned,
        'self_updated': self_updated,
        'phase': 'full',
        'n_turns': n_turns,
        'session_valence': float(session_valence) if n_turns > 0 else 0.0,
        'session_arousal': float(session_arousal) if n_turns > 0 else 0.0,
        'n_self_clusters': self_model.net.n_clusters,
        'n_l0_clusters': agent_net.n_clusters,
    }


def micro_consolidation(
    dialogue_ctx: 'DialogueContext',
    agent_net,                          # L0 ClusterNetwork
    body_state,                         # BodyVector
) -> dict:
    """微量巩固: 单轮后的快速记忆强化 (不涉及完整的睡眠周期)。

    在每次对话轮次后自动触发——低成本的即时巩固:
    1. 最近一轮的 2x 重放
    2. 相邻轮次的快速关联
    3. 不做修剪和词序巩固 (保留给完整睡眠周期)

    Returns:
        dict: 微量巩固统计
    """
    turns = dialogue_ctx.turns
    if len(turns) < 1:
        return {'consolidated': 0, 'phase': 'skip'}

    from cns.data_types import D
    import numpy as np

    orig_lr = agent_net.theta.learn_rate_l0
    last = turns[-1]

    # 最近轮次的双重重放
    for _ in range(2):
        trace = np.zeros(D, dtype=np.float32)
        trace[:64] = last.human_vec[:64].astype(np.float32)
        trace[64:128] = last.agent_vec[:64].astype(np.float32)
        if last.comprehension_vec is not None:
            trace[128:192] = last.comprehension_vec[:64].astype(np.float32)
        trace[192] = last.agent_valence
        trace[193] = last.agent_arousal

        b = body_state.b
        for i in range(min(8, len(b))):
            trace[200 + i] = b[i]

        agent_net.theta.learn_rate_l0 = orig_lr * 2.0
        agent_net.learn(trace)

    # 与前一轮关联 (如果存在且语义相关)
    cross_links = 0
    if len(turns) >= 2:
        prev = turns[-2]
        sim = float(np.dot(
            prev.human_vec[:64], last.human_vec[:64]) / (
            np.linalg.norm(prev.human_vec[:64])
            * np.linalg.norm(last.human_vec[:64]) + 1e-8))
        if sim > 0.3:
            bridge = np.zeros(D, dtype=np.float32)
            bridge[:64] = prev.human_vec[:64].astype(np.float32)
            bridge[64:128] = last.agent_vec[:64].astype(np.float32)
            agent_net.theta.learn_rate_l0 = orig_lr * (1.0 + sim)
            agent_net.learn(bridge)
            cross_links = 1

    agent_net.theta.learn_rate_l0 = orig_lr

    return {
        'consolidated': 1,
        'replays': 2,
        'cross_links': cross_links,
        'phase': 'micro',
    }
