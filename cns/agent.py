"""
agent.py —— Agent 主类
自由能原理智能体 — M1 单智能体生存

组装 L0-L3 所有组件，提供统一的 step() 接口。

全链路:
s → L0.learn(s) → L1.compute_F(z,s) → L2.select_action() → a
  → 状态更新: z' = predict_next_state(z,a) → L3.meta.update(F)
"""

import numpy as np
from cns.data_types import Theta, Action, FreeEnergy, AgentBelief, BodyVector
from cerebrum.limbic_system.hippocampus import (
    predict_sensations, ClusterNetwork, sleep_cycle,
)
from cerebrum.limbic_system.cingulate import (
    compute_free_energy, HabituationTracker,
)
from cerebrum.frontal_lobe.prefrontal import (
    compute_G, select_action, predict_next_state,
    update_social_beliefs,
)
from cerebrum.basal_ganglia.action_gating import MoEGate
from brainstem_cerebellum.neuromodulatory.meta_learning import (
    create_default_theta, MetaLearner,
)
from cerebrum.temporal_lobe.wernicke import (
    DialogueContext, comprehend, evaluate_response,
    consolidate_dialogue_memory, micro_consolidation,
)
from cerebrum.association.dmn import SelfModel


class Agent:
    """自由能原理智能体

    组件:
    - L0: ClusterNetwork (簇记忆) + predict_sensations (生成模型)
    - L1: HabituationTracker (习惯化) + FreeEnergy 计算
    - L2: Cluster.G_ema (集群经验) + MoEGate (门控)
    - L3: MetaLearner (元学习) + Theta (参数)

    生命周期:
    - 每个时间步: observe → think → act → learn
    - 每 100 步: 一次睡眠巩固
    """

    def __init__(self, rng: np.random.Generator = None,
                 agent_id: int = 0, n_agents: int = 1):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.agent_id = agent_id
        self.n_agents = n_agents

        # L3: 参数
        self.theta = create_default_theta()

        # Body: 身体稳态 (v2)
        self.body = BodyVector()

        # L0: 生成模型 (信念 = 集群激活状态, 非显式向量)
        self.net = ClusterNetwork(self.theta)

        # L1: 习惯化
        self.hab = HabituationTracker(self.theta.habituation_tau)

        # L2: MoE 门控 (v2: 经验在 Cluster.G_ema 中)
        self.moe = MoEGate()

        # L2 社会信念 (M3+: 预填充其他 agent ID)
        self.beliefs = AgentBelief()
        for i in range(n_agents):
            if i != agent_id:
                self.beliefs.other_positions[i] = np.zeros(2)
                self.beliefs.trust_levels[i] = 0.5
                self.beliefs.second_order[i] = np.zeros(2)

        # 自听情感追踪: 自己说的话/想的事对自己的情绪传染
        self.self_valence_ema: float = 0.0   # 自听效价 EMA ("我最近对自己说了什么感觉的话")
        self.self_arousal_ema: float = 0.0   # 自听唤醒 EMA
        self.self_coherence: float = 1.0     # 自我一致性 [0,1] (自听与当前效价是否一致)

        # L3: 元学习
        self.meta = MetaLearner(self.theta)

        # 对话工作记忆 (海马体 + DLPFC)
        self.dialogue_ctx: DialogueContext = DialogueContext(max_turns=8)

        # 自我模型 (Default Mode Network / vmPFC)
        self.self_model: SelfModel = SelfModel(max_clusters=256)

        # 睡眠巩固追踪
        self.consolidation_counter: int = 0      # 距离上次完整巩固的步数
        self.consolidation_interval: int = 100   # 完整巩固间隔 (步)
        self.dialogue_since_consolidation: int = 0  # 距离上次巩固的对话轮数
        self.consolidation_history: list[dict] = []  # 巩固历史记录

        # 追踪（核心）
        self.F_history: list[float] = []
        self.F_body_history: list[float] = []
        self.F_social_history: list[float] = []
        self.F_cognitive_history: list[float] = []
        self.F_accuracy_history: list[float] = []
        self.valence_history: list[float] = []
        self.arousal_history: list[float] = []
        self.attention_history: list[float] = []
        self.action_history: list[int] = []
        self.reward_history: list[float] = []
        self.theta_snapshots: list[dict] = []  # 参数快照 (面板4)
        self.last_action: Action = None

    def step(self, sensory: np.ndarray, step_count: int,
             my_pos: np.ndarray = None,
             social_ctx = None) -> Action:
        """单步决策：串联 L0-L3 全链路

        Args:
            sensory: 感知向量 (D,) 来自环境（含社会感知 M3+）
            step_count: 当前全局步数
            my_pos: 我的实际位置 (2,) — M3+ 社会信念更新需要
            social_ctx: SocialContext — Stage 6 人类对话社会信号

        Returns:
            选定的 Action
        """
        # ---- L0: 学习感知 + 周期性睡眠 ----
        self.net.learn(sensory)

        if step_count > 0 and step_count % 100 == 0:
            # Phase 1: 对话记忆巩固 (海马 → 皮层, 在衰减前)
            # 把最近的对话经验从工作记忆转移到长期记忆
            cb_result = self.maybe_consolidate(broca=None)

            # Phase 2: L0 集群衰减 + 弱簇清理
            n_removed = sleep_cycle(self.net, self.theta)

            # Phase 3: 睡眠后自我模型修剪 (v3: F_cognitive 驱动)
            # 自我模型过大 → F_cognitive ↑ → 需要修剪
            # 保留比例 ∝ 1/F_cognitive: F_cognitive 越高 → 剪掉越多
            max_self = max(20, int(150 / max(1.0, self.F_cognitive_history[-1]
                                             if self.F_cognitive_history else 1.0)))
            if self.self_model.net.n_clusters > max_self:
                self.self_model.net.decay()
                sorted_sc = sorted(self.self_model.net.clusters,
                                   key=lambda c: c.activation, reverse=True)
                if len(sorted_sc) > max_self:
                    self.self_model.net.clusters = sorted_sc[:max_self]

            self.consolidation_counter += 100
        else:
            self.consolidation_counter += 1

        # ---- L1: 自由能计算 + 注意力 ----
        belief = self._belief_vector()
        F = compute_free_energy(belief, sensory, self.net, self.theta,
                                self.hab, self.beliefs, self.body, social_ctx)
        self.hab.update(F.total)
        self.F_history.append(F.total)
        self.F_body_history.append(F.body)
        self.F_social_history.append(F.social)
        self.F_cognitive_history.append(F.cognitive)
        self.F_accuracy_history.append(F.accuracy)
        self.valence_history.append(F.valence)
        self.arousal_history.append(F.arousal)
        self.attention_history.append(F.attention_precision)

        # ---- 社会信念更新 (M3) ----
        if my_pos is not None and self.n_agents > 1:
            update_social_beliefs(sensory, self.beliefs, my_pos, self.theta)

        # ---- 信念已在 recall() 侧抑制中更新 (Phase 3) ----
        # z 仅作为 legacy fallback 保留，不执行显式更新

        # ---- 构建 F_context (Phase 2) ----
        F_context = np.array([
            F.body, F.social, F.cognitive, F.valence, F.arousal
        ])

        # ---- L2: 行动选择 (v2: 集群驱动经验) ----
        # 对话模式下启用 A₃ 表达耦合 (有人在 → 鼓励回应; 没人 → 抑制)
        is_dialogue = (social_ctx is not None)
        human_active = True
        if is_dialogue:
            human_active = (social_ctx.steps_since_input == 0)
        action = select_action(
            z=belief,
            net=self.net,
            theta=self.theta,
            moe_gate=self.moe,
            beliefs=self.beliefs,
            step=step_count,
            recent_F=self.F_history[-1] if self.F_history else 0.0,
            F_context=F_context,
            rng=self.rng,
            human_active=human_active,
            dialogue_mode=is_dialogue,
        )

        # ---- 状态更新: z → z' (Phase 2: 集群模式补全) ----
        self._last_belief = predict_next_state(belief, action.index, self.theta,
                                                self.rng, F_context, self.net)

        # ---- Body ODE: 身体动力学 (v2) ----
        self.body.step(action.index)
        # 多模态刺激累积 (b₅:视觉, b₆:音频)
        if self.body.mode == 'text' and len(self.body.b) >= 8:
            vis_norm = float(np.linalg.norm(sensory[64:128]))
            aud_norm = float(np.linalg.norm(sensory[128:192]))
            self.body.b[5] = np.clip(self.body.b[5] - 0.003 + 0.01 * vis_norm, 0, 1)
            self.body.b[6] = np.clip(self.body.b[6] - 0.003 + 0.01 * aud_norm, 0, 1)
            # A₄ 恢复 b₅,b₆,b₇
            if action.index == 4:
                self.body.b[5] = max(0.0, self.body.b[5] - 0.02)
                self.body.b[6] = max(0.0, self.body.b[6] - 0.02)
                self.body.b[7] = max(0.0, self.body.b[7] - 0.02)

                # ---- 自听情感传染 (v2: 强化反馈) ----
            # s[96:104] 携带上一轮自己回应/内部言语的情感编码
            self_sent = sensory[96:104]
            has_self = np.sum(np.abs(self_sent)) > 0.01

            if has_self:
                self_valence = float(self_sent[6])   # raw valence [-1, 1]
                self_arousal = float(self_sent[1])    # arousal [0, 1]
                # v3: 注意力精度调制 EMA alpha (高唤醒 → 更易被影响)
                attn = F.attention_precision if hasattr(F, 'attention_precision') else 0.5
                effective_alpha = 0.15 + attn * 0.30  # [0.15, 0.45]
                self.self_valence_ema += effective_alpha * (self_valence - self.self_valence_ema)
                self.self_arousal_ema += effective_alpha * (self_arousal - self.self_arousal_ema)

                # 强化身体耦合: 系数从 0.008 → 0.025 (3x 强度)
                self.body.b[0] = np.clip(
                    self.body.b[0] + 0.025 * self_valence, 0.0, 1.0)
            else:
                # 无自听信号时，自听情感 EMA 缓慢回归中性
                self.self_valence_ema *= 0.95
                self.self_arousal_ema *= 0.95
                # 情绪记忆: 自听 EMA 对 b[0] 有残余影响
                self.body.b[0] = np.clip(
                    self.body.b[0] + 0.005 * self.self_valence_ema, 0.0, 1.0)

            # ---- 自我一致性检测 (认知失调) ----
            # 当前效价 vs 自听效价: 不一致 → 唤醒升高
            current_v = F.valence
            self.self_coherence = float(
                np.exp(-abs(current_v - self.self_valence_ema) * 3.0))
            # 低一致性 → 修正 arousal_history (事后注入, 影响情感追踪)
            dissonance = (1.0 - self.self_coherence) * 0.2
            if self.arousal_history:
                boosted = min(1.0, self.arousal_history[-1] + dissonance)
                self.arousal_history[-1] = boosted

            # ---- DMN 耦合: F_accuracy 突发增加 → 自动写入自我模型 ----
            # v3: 连续函数, 不用任意阈值
            # 存储概率 ∝ F_accuracy_spike × arousal (杏仁核门控)
            # F_accuracy 突增 = 预期与实际不符 = "意外事件" → 值得记住
            arousal_now = self.arousal_history[-1] if self.arousal_history else 0.0
            valence_now = self.valence_history[-1] if self.valence_history else 0.0
            if len(self.F_accuracy_history) >= 2:
                F_acc_spike = max(0.0, self.F_accuracy_history[-1]
                                 - self.F_accuracy_history[-2])
                # 耦合概率: F_acc spike 大 + 至少中等唤醒 → 高概率存储
                coupling_prob = min(1.0, (F_acc_spike * 3.0 + arousal_now * 0.5))
            else:
                coupling_prob = 0.5 * arousal_now  # 冷启动: 只用唤醒
            if coupling_prob > 0.5:
                self._auto_update_self_model(sensory, F, arousal_now, valence_now)

        # ---- L3: 元学习 (M5 真实有限差分) ----
        self.meta.update(F.total, self._last_belief, sensory, self.net,
                         self.hab, self.beliefs)

        # ---- Theta 快照 (每 10 步记录，节省内存) ----
        if step_count % 10 == 0:
            self.theta_snapshots.append({
                'step': step_count,
                **self.theta.to_dict(),
            })

        # ---- 追踪 ----
        self.last_action = action
        self.action_history.append(action.index)

        return action

    def _belief_vector(self) -> np.ndarray:
        """从集群激活状态推导隐状态。
        激活度最高的集群的 centroid → 信念向量。
        无集群 → 零向量。
        """
        from cns.data_types import H
        if self.net.n_clusters == 0:
            return np.zeros(H)
        top = max(self.net.clusters, key=lambda c: c.activation)
        if top.activation > 0:
            v = np.zeros(H)
            v[:min(H, len(top.centroid))] = top.centroid[:min(H, len(top.centroid))]
            return v
        return np.zeros(H)

    def add_reward(self, reward: float):
        """记录环境返回的奖励"""
        self.reward_history.append(reward)

    def record_action_consequence(self, s_next: np.ndarray):
        """Phase 2: 创建行动-后果集群

        存储 [s_next(48) | F_context(5) | action_onehot(5)] = 58 dims
        供 predict_next_state → recall() 模式补全。
        """
        if self.last_action is None:
            return
        Fb = self.F_body_history[-1] if self.F_body_history else 0.0
        Fs = self.F_social_history[-1] if self.F_social_history else 0.0
        Fc = self.F_cognitive_history[-1] if self.F_cognitive_history else 0.0
        v = self.valence_history[-1] if self.valence_history else 0.0
        ar = self.arousal_history[-1] if self.arousal_history else 0.0
        F_context = np.array([Fb, Fs, Fc, v, ar])

        onehot = np.zeros(5)
        onehot[self.last_action.index] = 1.0

        from cns.data_types import D, S_CORE
        pattern = np.zeros(D)
        # 行动-后果集群: 只存 F_context(5) + action(5), 不存 s_next
        # s_next 会导致下一帧 raw sensory 命中此集群 → 全合并
        pattern[S_CORE:S_CORE+5] = F_context
        pattern[S_CORE+5:S_CORE+10] = onehot
        self.net.learn(pattern)

    def get_state_summary(self) -> dict:
        """返回智能体状态摘要"""
        return {
            'top_activation': float(max((c.activation for c in self.net.clusters), default=0.0)),
            'n_clusters': self.net.n_clusters,
            'total_activation': self.net.total_activation,
            'F_latest': self.F_history[-1] if self.F_history else 0.0,
            'valence': 0.0,  # 由 step 中的 F 提供
            'arousal': 0.0,
            'meta_step': self.meta.step_count,
            'is_critical': self.meta.is_critical,
        }

    # ================================================================
    # Wernicke 区: 语言理解 (v5.1)
    # ================================================================

    def comprehend(self, human_vec: np.ndarray,
                   human_sentiment: np.ndarray = None
                   ) -> tuple[np.ndarray, dict]:
        """Wernicke 区等价物 —— 理解人类输入。

        调用 dialogue_memory.comprehend()，传入当前身体/情感状态。
        理解向量驱动 Broca 区生成回应，而不是直接用原始输入检索。

        Args:
            human_vec: 人类输入的语义编码 (64,)
            human_sentiment: 情感信号 (8,), 可选

        Returns:
            (comprehension_vec, understanding_dict)
        """
        if human_sentiment is None:
            human_sentiment = np.zeros(8, dtype=np.float32)

        v = self.valence_history[-1] if self.valence_history else 0.0
        a = self.arousal_history[-1] if self.arousal_history else 0.0

        comp_vec, understanding = comprehend(
            human_vec=human_vec,
            human_sentiment=human_sentiment,
            agent_net=self.net,
            dialogue_ctx=self.dialogue_ctx,
            body_state=self.body,
            valence=v,
            arousal=a,
        )
        # 存储最近的理解向量 (供自听回路使用)
        self._last_comprehension = comp_vec
        return comp_vec, understanding

    def warmup_l0(self, sentences: list[str], text_env, n: int = 30):
        """L0 预热: 喂入种子句子建立初始记忆集群。

        没有初始记忆 → 理解时找不到触发记忆 → Wernicke 区无输出。
        预热后的集群提供"知识基础"，让 comprehend() 能激活相关记忆。

        Args:
            sentences: 种子句子列表
            text_env: TextEnvironment (用于编码)
            n: 创建的集群数量上限
        """
        from cns.data_types import D
        for i, sent in enumerate(sentences[:n]):
            try:
                vec = text_env.encode_text(sent)
                s = np.zeros(D, dtype=np.float32)
                s[:64] = vec.astype(np.float32)
                self.net.learn(s)
            except Exception:
                pass

    def evaluate_own_response(self, response_vec: np.ndarray) -> dict:
        """ACC+OFC —— 评估自己刚生成的回应"""
        comp = getattr(self, '_last_comprehension', None)
        if comp is None:
            comp = np.zeros(64, dtype=np.float32)
        return evaluate_response(response_vec, comp,
                                 self.dialogue_ctx, self.net)

    # ================================================================
    # 睡眠巩固: 海马 → 皮层记忆转移
    # ================================================================

    def consolidate_dialogue(self, broca=None) -> dict:
        """完整睡眠巩固周期 —— 将对话经验整合到长期记忆。

        触发时机:
        - 每 100 步 L0 睡眠周期
        - 累积 5+ 轮对话后自动触发
        - 手动调用 (会话结束)

        Returns:
            dict: 巩固统计
        """
        v = self.valence_history[-1] if self.valence_history else 0.0
        a = self.arousal_history[-1] if self.arousal_history else 0.0

        result = consolidate_dialogue_memory(
            dialogue_ctx=self.dialogue_ctx,
            agent_net=self.net,
            self_model=self.self_model,
            body_state=self.body,
            valence=v,
            arousal=a,
            broca=broca,
        )

        # 追踪
        self.consolidation_counter = 0
        self.dialogue_since_consolidation = 0
        self.consolidation_history.append(result)

        return result

    def micro_consolidate(self) -> dict:
        """微量巩固: 单轮对话后的即时记忆强化。

        每轮对话后自动调用 — 低成本:
        - 最近轮次双重重放
        - 相邻轮次快速关联
        - 不做修剪 (留给完整睡眠周期)
        """
        result = micro_consolidation(
            dialogue_ctx=self.dialogue_ctx,
            agent_net=self.net,
            body_state=self.body,
        )
        self.dialogue_since_consolidation += 1
        return result

    def maybe_consolidate(self, broca=None, force: bool = False) -> dict | None:
        """按需触发巩固: 对话轮数 ≥5 或强制。

        Returns:
            dict or None: 如果未触发则返回 None
        """
        n_turns = self.dialogue_ctx.n_turns()
        if force or (n_turns >= 5 and self.dialogue_since_consolidation >= 5):
            return self.consolidate_dialogue(broca=broca)
        return None

    def _auto_update_self_model(self, sensory: np.ndarray, F,
                                arousal: float, valence: float):
        """DMN 自动耦合: 高唤醒/高情感时刻 → 自传体记忆。

        由 step() 自动调用——不依赖外部代码手动写入。
        模拟: 杏仁核标记情感事件 → 海马编码 → DMN 整合。
        """
        if self.dialogue_ctx.n_turns() == 0:
            return  # 无对话上下文时不存储

        # 用当前最激活的信念集群作为"我在想什么"
        if self.net.n_clusters == 0:
            return
        top = max(self.net.clusters, key=lambda c: c.activation)
        if top.activation < 0.05:
            return

        # 构建体验向量
        resp_vec = top.centroid[:64].copy().astype(np.float32)
        ctx = self.dialogue_ctx.get_context_vector()
        comp = getattr(self, '_last_comprehension', np.zeros(64, dtype=np.float32))

        self.self_model.add_experience(
            response_vec=resp_vec,
            valence=valence,
            arousal=arousal,
            self_valence_ema=self.self_valence_ema,
            self_arousal_ema=self.self_arousal_ema,
            self_coherence=self.self_coherence,
            body_state=self.body,
            comprehension_vec=comp,
            dialogue_ctx_vec=ctx,
        )
