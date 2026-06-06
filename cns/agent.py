"""
agent.py —— Agent 主类
自由能原理智能体 — M1 单智能体生存

组装 L0-L3 所有组件，提供统一的 step() 接口。

全链路 (v5.6):
s → visual_hierarchy → auditory_hierarchy → nociception →
  hypothalamus → VTA → LC → L0.learn(s) →
  L1.compute_F(z,s) + F_language(N400/P600) → L2.select_action() → a
  → 状态更新: z' = predict_next_state(z,a) → L3.meta.update(F)

v5.6 语言全管线 (comprehend → speak):
  human_input → Wernicke.comprehend(+N400/P600) → TPJ.pragmatic_enrich →
  ArcuateFasciculus.ventral → Broca.speak(+PhraseStructure) →
  MotorCortex.plan_sequence → ArcuateFasciculus.dorsal(efference_copy) →
  PhonologicalLoop(self-hearing) → self-monitoring
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
    compute_language_PE,  # v5.6: 语言预测误差
)
from cerebrum.association.dmn import SelfModel
from cerebrum.association.fpn import FrontoparietalNetwork
from cerebrum.association.tpn import TaskPositiveNetwork

# v5.6: 语言系统新模块
from cerebrum.association.arcuate_fasciculus import ArcuateFasciculus
from cerebrum.frontal_lobe.phonological_loop import PhonologicalLoop
from cerebrum.frontal_lobe.phrase_structure import PhraseStructureNetwork
from cerebrum.parietal_lobe.angular_gyrus import AngularGyrus
from cerebrum.frontal_lobe.motor_cortex import MotorCortex
from cerebrum.parietal_lobe.tpj import TPJ


def _estimate_azimuth_from_stereo(left_spectrum, right_spectrum) -> float:
    """v5.3: 从立体声频谱估算方位角.

    通过比较左右耳 mel 频谱的能量差 (ILD) 粗略估计声源方位。
    正值=右侧, 负值=左侧。

    Args:
        left_spectrum: 左耳 mel 频谱 (32,) 或 None
        right_spectrum: 右耳 mel 频谱 (32,) 或 None

    Returns:
        azimuth: 方位角估计 (度, -90=左, +90=右, 0=正前方)
    """
    if left_spectrum is None or right_spectrum is None:
        return 0.0
    left = np.asarray(left_spectrum, dtype=np.float32).ravel()
    right = np.asarray(right_spectrum, dtype=np.float32).ravel()
    if len(left) == 0 or len(right) == 0:
        return 0.0

    left_energy = float(np.sum(left))
    right_energy = float(np.sum(right))
    total = left_energy + right_energy + 1e-8

    # ILD ratio → azimuth
    ild_ratio = (right_energy - left_energy) / total  # [-1, +1]
    azimuth = float(np.degrees(np.arcsin(np.clip(ild_ratio, -1.0, 1.0))))
    return azimuth


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

        # FPN — 额顶网络: 选择性注意"探照灯" (v4.4 集成)
        self.fpn: FrontoparietalNetwork = FrontoparietalNetwork()

        # TPN — 任务正网络: TPN↔DMN 跷跷板动态 (v4.4 集成)
        self.tpn: TaskPositiveNetwork = TaskPositiveNetwork()

        # v5.1: 视觉层级管线 (full visual hierarchy, 替代旧 ImageEncoder)
        from cerebrum.occipital_lobe.visual_hierarchy import VisualHierarchy
        self.visual_hierarchy: VisualHierarchy = VisualHierarchy(image_size=64)
        self._current_visual_result: dict = {}  # 存本次 step 的视觉处理结果
        self._current_image: np.ndarray = None  # 当前帧图像 (由外部设置)

        # v5.2: 听觉层级管线 (耳蜗核→SOC→IC→MGB→听皮层)
        from cerebrum.temporal_lobe.auditory_hierarchy import AuditoryHierarchy
        self.auditory_hierarchy: AuditoryHierarchy = AuditoryHierarchy()
        self._current_auditory_result: dict = {}  # 存本次 step 的听觉处理结果
        self._current_audio_data: dict = None     # v5.3: 真实音频输入数据 (由外部设置)

        # v5.4: 痛觉层级管线 (背角闸门→双通路→丘脑→皮层→PAG→RVM下行)
        from brainstem_cerebellum.nociception_hierarchy import NociceptionHierarchy
        self.nociception_hierarchy: NociceptionHierarchy = NociceptionHierarchy()
        self._current_pain_result: dict = {}      # 存本次 step 的痛觉处理结果
        self._current_pain_input: float = 0.0     # 当前痛觉输入强度 (由外部/环境设置)
        self._current_abeta_input: float = 0.0    # 当前Aβ触觉输入 (按摩/触摸)

        # v5.5: 下丘脑稳态调节 (SetpointModel + DriveSystem → BodyVector homeostasis)
        from cerebrum.limbic_system.hypothalamus import Hypothalamus
        self.hypothalamus: Hypothalamus = Hypothalamus()
        self._hypo_result: dict = {}              # 存本次 step 的下丘脑调节结果

        # v5.5: VTA 奖赏预测误差 → 事件驱动学习率
        from brainstem_cerebellum.midbrain.vta import VTA
        self.vta: VTA = VTA()
        self._vta_result: dict = {}               # 存本次 step 的 VTA 结果

        # v5.5: 蓝斑核 NE 唤醒度调制 (phasic/tonic NE + SNR + explore/exploit)
        from brainstem_cerebellum.pons.locus_coeruleus import LocusCoeruleus
        self.locus_coeruleus: LocusCoeruleus = LocusCoeruleus()
        self._lc_result: dict = {}                # 存本次 step 的 LC 结果

        # v5.6: 弓状束 — 连接 Wernicke ↔ Broca (腹侧+背侧双通路)
        self.arcuate_fasciculus: ArcuateFasciculus = ArcuateFasciculus()

        # v5.6: 语音回路 — 言语工作记忆 (~7组块, ~2秒消退)
        self.phonological_loop: PhonologicalLoop = PhonologicalLoop()

        # v5.6: 短语结构网络 — 层级句法 (从语料统计中涌现)
        self.phrase_structure: PhraseStructureNetwork = PhraseStructureNetwork()

        # v5.6: 角回 — 阅读通路 (视觉字形→语音)
        self.angular_gyrus: AngularGyrus = AngularGyrus()

        # v5.6: 运动皮层 — 言语发音规划 (M1+SMA, 16维发音特征)
        self.motor_cortex: MotorCortex = MotorCortex()

        # v5.6: 颞顶联合区 — 心理理论与语用语言
        self.tpj: TPJ = TPJ()

        # v6.0: 语义记忆 — 皮层知识存储 (慢学慢衰大容量)
        from cerebrum.temporal_lobe.semantic_memory import SemanticMemory
        self.semantic_memory: SemanticMemory = SemanticMemory(max_clusters=1024)

        # v5.6: 语言预测误差追踪
        self.F_language_history: list[float] = []
        self.language_pe_history: list[dict] = []
        self._last_comprehension: np.ndarray = np.zeros(64, dtype=np.float32)

        # v5.1: 自听回路状态 (从 sensory 向量中移出, 变为 Agent 内部状态)
        self._audio_semantic: np.ndarray = np.zeros(64, dtype=np.float32)
        self._self_sentiment: np.ndarray = np.zeros(8, dtype=np.float32)

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

        # v4.4: FPN/TPN 状态追踪
        self.tpn_state_history: list[dict] = []       # TPN 每次 update_seesaw() 后快照
        self.fpn_gain_mean_history: list[float] = []   # FPN 注意力模板平均增益
        self.attended_sensory: np.ndarray = None       # 最新 FPN 门控后的感知

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
        # ---- v5.1 Phase 0: 视觉层级处理 (全管线前馈+反馈+PE+绑定) ----
        from cns.data_types import (M_V1_START, BINDING_END, D_VISUAL_V5)
        VIS_START, VIS_END = M_V1_START, BINDING_END  # s[64:372]
        vis_active = bool(np.any(np.abs(sensory[VIS_START:VIS_END]) > 0.01))

        # v5.7: 视觉层级常开 — 有真实图像时用全管线, 无图像时用语义代理
        # 这确保 _current_visual_result 始终被填充 (perception面板不再显示inactive)
        if hasattr(self, 'visual_hierarchy') and hasattr(self, 'fpn'):
            try:
                brainstem_arousal = float(np.clip(
                    self.arousal_history[-1] if self.arousal_history else 0.5,
                    0.1, 1.0))
                if self._current_image is not None:
                    # 模式A: 真实图像 → 全视觉层级管线
                    vis_result = self.visual_hierarchy.process(
                        self._current_image,
                        brainstem_arousal=brainstem_arousal,
                        fpn=self.fpn,
                        learn=True,
                    )
                    sensory[VIS_START:VIS_END] = vis_result['sensory'][VIS_START:VIS_END]
                    self._current_visual_result = vis_result
                elif vis_active:
                    # 模式B: cross-modal视觉补全已填充sensory → 保持现有值
                    self._current_visual_result = {
                        'F_accuracy': 0.05, 'PE_total': 0.0,
                        'diagnostics': {
                            'V1_mean_norm': float(np.mean(np.abs(
                                sensory[VIS_START:VIS_START+64]))),
                            'V2_mean_norm': float(np.mean(np.abs(
                                sensory[VIS_START+64:VIS_START+128]))),
                            'V4_mean_norm': float(np.mean(np.abs(
                                sensory[VIS_START+128:VIS_START+192]))),
                            'IT_mean_norm': float(np.mean(np.abs(
                                sensory[VIS_START+240:VIS_END]))),
                            'mode': 'crossmodal_fill',
                        },
                    }
                else:
                    # 模式C: 语义代理 — 从文本生成低分辨率视觉特征
                    text_vec = sensory[0:64].copy()
                    if np.linalg.norm(text_vec) > 0.01:
                        # 用 Hebb 网络跨模态补全: text → vision
                        # 简单实现: text_vec → 网络检索 → 最佳匹配质心的视觉段
                        vis_proxy = np.zeros(D_VISUAL_V5, dtype=np.float32)
                        if self.net.n_clusters > 0:
                            c = self.net.recall(text_vec[:48])
                            if c is not None:
                                vis_proxy = c.centroid[VIS_START:VIS_END].copy()
                        # 即使质心全零也没关系 — 至少结果已填充
                        sensory[VIS_START:VIS_END] = vis_proxy
                        self._current_visual_result = {
                            'F_accuracy': 0.05, 'PE_total': 0.0,
                            'diagnostics': {
                                'V1_mean_norm': float(np.mean(np.abs(vis_proxy[:64]))),
                                'V2_mean_norm': float(np.mean(np.abs(vis_proxy[64:128]))),
                                'V4_mean_norm': float(np.mean(np.abs(vis_proxy[128:192]))),
                                'IT_mean_norm': float(np.mean(np.abs(vis_proxy[240:]))),
                                'mode': 'semantic_proxy',
                            },
                        }
                    else:
                        self._current_visual_result = {
                            'F_accuracy': 0.0, 'PE_total': 0.0,
                            'diagnostics': {
                                'V1_mean_norm': 0.0, 'V2_mean_norm': 0.0,
                                'V4_mean_norm': 0.0, 'IT_mean_norm': 0.0,
                                'mode': 'no_input',
                            },
                        }
            except Exception:
                self._current_visual_result = {
                    'F_accuracy': 0.0, 'PE_total': 0.0,
                    'diagnostics': {
                        'V1_mean_norm': 0.0, 'V2_mean_norm': 0.0,
                        'V4_mean_norm': 0.0, 'IT_mean_norm': 0.0,
                        'mode': 'error',
                    },
                }

        # ---- v5.2 Phase 0b: 听觉层级处理 (耳蜗核→SOC→IC→MGB→听皮层) ----
        # v5.3: 真实音频优先 — 有真实频谱时用它, 否则回退语义代理模式
        from cns.data_types import CN_START, AC_END, CN_WIDTH, SOC_WIDTH, IC_WIDTH, AC_WIDTH
        AUD_START, AUD_END = CN_START, AC_END  # s[372:468]
        aud_active = True  # 听觉管线常开

        if aud_active and hasattr(self, 'auditory_hierarchy'):
            try:
                brainstem_arousal = float(np.clip(
                    self.arousal_history[-1] if self.arousal_history else 0.5,
                    0.1, 1.0))
                # 视觉空间信息 (用于ICx多感官整合)
                vis_spatial = None
                if 'sensory' in self._current_visual_result:
                    vis_vec = self._current_visual_result.get('sensory', None)
                    if vis_vec is not None:
                        vis_spatial = vis_vec[VIS_START:VIS_START+16]

                # v5.3: 真实音频输入 → 替换语义代理
                if self._current_audio_data is not None:
                    audio_data = self._current_audio_data
                    spectrum = audio_data.get('spectrum', None)
                    left_spec = audio_data.get('left_spectrum', None)
                    right_spec = audio_data.get('right_spectrum', None)
                    # 从立体声数据估算方位角 (用于双耳处理)
                    azimuth_hint = _estimate_azimuth_from_stereo(
                        left_spec, right_spec)

                    aud_result = self.auditory_hierarchy.process(
                        spectrum=spectrum,
                        left_spectrum=left_spec,
                        right_spectrum=right_spec,
                        azimuth_hint=azimuth_hint,
                        arousal=brainstem_arousal,
                        fpn=self.fpn if hasattr(self, 'fpn') else None,
                        visual_spatial=vis_spatial,
                        learn=True,
                    )
                else:
                    # 语义代理: text[0:64] → 伪频谱 → 全听觉通路
                    text_vec = sensory[0:64].copy()
                    aud_result = self.auditory_hierarchy.process(
                        semantic_vec=text_vec,
                        arousal=brainstem_arousal,
                        fpn=self.fpn if hasattr(self, 'fpn') else None,
                        visual_spatial=vis_spatial,
                        learn=True,
                    )

                # 将听觉管线输出写入感知向量的听觉段
                sensory[AUD_START:AUD_END] = aud_result['sensory'][:AUD_END-AUD_START]
                self._current_auditory_result = aud_result
            except Exception:
                self._current_auditory_result = {
                    'F_accuracy': 0.0, 'PE_total': 0.0,
                    'diagnostics': {
                        'CN_mean_norm': 0.0, 'SOC_ITD_std': 0.0,
                        'IC_mean_norm': 0.0, 'AC_mean_norm': 0.0,
                        'n_asa_streams': 0, 'mode': 'error',
                    },
                }

        # ---- v5.4 Phase 0c: 痛觉层级处理 (背角闸门→双通路→丘脑→皮层→PAG→RVM下行) ----
        from cns.data_types import PAIN_DH_START, PAIN_THALAMIC_END, D_PAIN
        PAIN_START, PAIN_END = PAIN_DH_START, PAIN_THALAMIC_END  # s[468:516]

        # 从身体状态推导伤害性输入
        if self.body is not None and hasattr(self, 'nociception_hierarchy'):
            try:
                body_b = self.body.b
                # b[-1] for pain if available (text mode has 9, grid has 5)
                if len(body_b) >= 9:
                    tissue_integrity = float(body_b[8])
                    tissue_damage = float(np.clip(tissue_integrity, 0.0, 1.0))
                else:
                    # Grid mode: use overall body deviation as proxy
                    tissue_damage = float(np.clip(
                        self.body.compute_deviation() * 0.5, 0.0, 1.0))
                # v5.4: 伤害性输入 = max(身体组织损伤, 外部输入)
                # 外部输入主导 (环境伤害), 身体组织损伤提供持续背景
                nociceptive = float(np.clip(
                    max(tissue_damage * 0.6, self._current_pain_input), 0.0, 1.0))

                # Aβ触觉输入 (来自外部, 如按摩/触摸 = 关闭闸门)
                abeta_input = float(self._current_abeta_input)

                # 从当前状态获取情感/认知参数
                v = self.valence_history[-1] if self.valence_history else 0.0
                a = self.arousal_history[-1] if self.arousal_history else 0.5
                # ACC情感信号 = F_body归一化
                f_body_now = self.F_body_history[-1] if self.F_body_history else 0.0
                acc_affect = float(np.clip(f_body_now * 0.5, 0.0, 1.0))
                # 杏仁核恐惧 = 负效价 × 唤醒
                amygdala_fear = float(np.clip(max(0.0, -v) * a, 0.0, 1.0))
                # PFC认知调控 = TPN激活度 (任务模式 → 更强的下行抑制)
                pfc_cognitive = float(self.tpn.tpn_activation if hasattr(self, 'tpn') else 0.3)
                # 应激水平 = 高唤醒 + 高F_body
                stress = float(np.clip(a * 0.6 + f_body_now * 0.4, 0.0, 1.0))
                # 安慰剂预期 = 正效价 + 低F_body (预期好转)
                placebo = float(np.clip(max(0.0, v) * (1.0 - f_body_now), 0.0, 1.0))

                pain_result = self.nociception_hierarchy.process(
                    nociceptive_input=nociceptive,
                    tissue_damage=tissue_damage,
                    abeta_input=abeta_input,
                    valence=v,
                    arousal=a,
                    body_vector=body_b,
                    acc_affect=acc_affect,
                    insula_intero=0.0,       # 首次运行无岛叶反馈
                    amygdala_fear=amygdala_fear,
                    pfc_cognitive=pfc_cognitive,
                    stress_level=stress,
                    placebo_expectation=placebo,
                    fpn=self.fpn if hasattr(self, 'fpn') else None,
                    learn=True,
                )

                # 将痛觉管线输出写入感知向量的痛觉段
                sensory[PAIN_START:PAIN_END] = pain_result['sensory'][:D_PAIN]
                self._current_pain_result = pain_result

                # 痛觉反馈 → 身体状态
                pain_intensity = pain_result['pain_intensity']
                # 高疼痛 → b[2] 压力↑ (所有模式通用)
                if len(self.body.b) >= 3:
                    self.body.b[2] = min(1.0,
                        self.body.b[2] + 0.003 * pain_intensity)
                # 疼痛 → b[0] 社交需求(安慰)↑ (所有模式通用)
                if len(self.body.b) >= 1 and pain_intensity > 0.3:
                    self.body.b[0] = max(0.0,
                        self.body.b[0] - 0.002 * pain_intensity)
                # b[8] 组织完整性: text模式专有
                if len(body_b) >= 9:
                    if pain_intensity > 0.5:
                        self.body.b[8] = min(1.0,
                            self.body.b[8] + 0.002 * (pain_intensity - 0.5))
                    # 下行调控: analgesia → 恢复加速
                    if pain_result.get('descending_signal', 0) > 0.3:
                        self.body.b[8] = max(0.0,
                            self.body.b[8] - 0.001 * pain_result['descending_signal'])

            except Exception:
                self._current_pain_result = {
                    'F_accuracy': 0.0, 'PE_total': 0.0,
                    'pain_intensity': 0.0, 'diagnostics': {
                        'DH_gate_output': 0.0, 'mode': 'error',
                    },
                }

        # ---- v5.5 Phase 0d: 下丘脑稳态调节 (动态调定点 + 驱力 + HPA轴) ----
        if hasattr(self, 'hypothalamus'):
            try:
                latest_arousal = self.arousal_history[-1] if self.arousal_history else 0.5
                stress_level = float(self.body.b[2]) if self.body is not None else 0.0
                hypo_result = self.hypothalamus.process(
                    body_vector=self.body,
                    time_of_day=(step_count % 1440) / 1440.0,
                    stress_level=stress_level,
                    arousal=latest_arousal,
                )
                # HPA激活 → 压力/疲劳维度调制 (b[2])
                if hypo_result['hpa_activation'] > 0.5 and self.body is not None:
                    self.body.b[2] = min(1.0,
                        self.body.b[2] + 0.002 * hypo_result['hpa_activation'])
                # 自主神经平衡 → 能量调制 (b[1])
                if hypo_result['sympathetic_dominant'] and self.body is not None:
                    self.body.b[1] = max(0.1,
                        self.body.b[1] - 0.001 * abs(hypo_result['autonomic_balance']))
                elif hypo_result['parasympathetic_dominant'] and self.body is not None:
                    self.body.b[1] = min(1.0,
                        self.body.b[1] + 0.001 * abs(hypo_result['autonomic_balance']))
                self._hypo_result = hypo_result
            except Exception:
                self._hypo_result = {'total_drive': 0.0, 'hpa_activation': 0.0,
                                    'autonomic_balance': 0.0, 'regulatory_urgency': 0.0}

        # ---- v5.5 Phase 0e: VTA 奖赏预测误差 → 事件驱动学习率调制 ----
        if hasattr(self, 'vta'):
            try:
                v = self.valence_history[-1] if self.valence_history else 0.0
                f_body = self.F_body_history[-1] if self.F_body_history else 0.0
                # Δvalence (改善=正), ΔF_body (下降=改善, 所以取负)
                delta_v = v - (self.valence_history[-2]
                               if len(self.valence_history) >= 2 else v)
                delta_f = ((self.F_body_history[-2] - f_body)
                           if len(self.F_body_history) >= 2 else 0.0)
                # 社会奖赏: 互动质量 (信任度高 → 社会奖赏高)
                social_r = 0.0
                if social_ctx is not None:
                    social_r = float(np.clip(social_ctx.trust_level, 0.0, 1.0))
                # 新颖性: 来自 TPN salience (在下面TPN段计算, 此处预估值)
                novelty_est = 0.0
                if len(self.F_accuracy_history) >= 2:
                    acc_change = abs(self.F_accuracy_history[-1]
                                    - self.F_accuracy_history[-2])
                    novelty_est = float(np.tanh(acc_change * 5.0))

                vta_result = self.vta.process(
                    valence=v, delta_valence=delta_v,
                    F_body=f_body, delta_F_body=delta_f,
                    social_reward=social_r,
                    novelty=novelty_est,
                    arousal=latest_arousal if 'latest_arousal' in dir() else 0.5,
                    base_learn_rate=self.theta.learn_rate_l0,
                )
                # VTA RPE → 海马学习率调制
                self.net.learn_rate_modifier = vta_result['learn_rate_multiplier']
                self._vta_result = vta_result
            except Exception:
                self.net.learn_rate_modifier = 1.0
                self._vta_result = {'rpe': 0.0, 'learn_rate_multiplier': 1.0,
                                   'total_da': 0.3, 'motivation': 0.5}

        # ---- v5.5 Phase 0f: 蓝斑核 NE 调制 (唤醒度 + SNR + 探索/利用) ----
        if hasattr(self, 'locus_coeruleus'):
            try:
                latest_arousal_lc = self.arousal_history[-1] if self.arousal_history else 0.5
                f_body_now_lc = self.F_body_history[-1] if self.F_body_history else 0.0
                stress_lc = stress_level if 'stress_level' in dir() else 0.0
                novelty_lc = novelty_est if 'novelty_est' in dir() else 0.0
                task_eng = self.tpn.tpn_activation if hasattr(self, 'tpn') else 0.3

                lc_result = self.locus_coeruleus.process(
                    arousal=latest_arousal_lc,
                    novelty=novelty_lc,
                    stress=stress_lc,
                    F_body=f_body_now_lc,
                    task_engagement=task_eng,
                    # sensory SNR will be applied after FPN attention gate
                )
                # Wire LC → RVM (v5.4 reserved NE interface)
                if hasattr(self, 'nociception_hierarchy'):
                    self.nociception_hierarchy.rvm._norepinephrine_tone = float(
                        lc_result['tonic_ne'])
                self._lc_result = lc_result
            except Exception:
                self._lc_result = {'tonic_ne': 0.2, 'phasic_ne': 0.0,
                                  'total_ne': 0.2, 'snr_gain': 1.0,
                                  'yd_performance': 0.5, 'exploration_bias': 0.0}

        # ---- L0: 学习感知 + 周期性睡眠 ----
        self.net.learn(sensory)
        # v5.5: 学习后重置 VTA 学习率调制 (避免跨步累积)
        self.net.learn_rate_modifier = 1.0

        # v6.0: 语义记忆并行学习 (慢速, 主旨提取)
        # 每 5 步学习一次语义记忆 (减少开销)
        if hasattr(self, 'semantic_memory') and step_count % 5 == 0:
            from cns.data_types import D
            gist_vec = np.zeros(D, dtype=np.float32)
            gist_vec[:64] = sensory[:64]  # 文本段
            # 身体+情感快照
            if self.body is not None and len(self.body.b) >= 8:
                gist_vec[64:72] = self.body.b[:8].astype(np.float32)
            v = self.valence_history[-1] if self.valence_history else 0.0
            a = self.arousal_history[-1] if self.arousal_history else 0.0
            gist_vec[72] = v
            gist_vec[73] = a
            self.semantic_memory.learn_fact(gist_vec, weight=0.3)

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

        # v5.4: 痛觉预测误差汇入 F_total
        pain_pe = 0.0
        if hasattr(self, '_current_pain_result') and self._current_pain_result:
            pain_pe = float(self._current_pain_result.get('PE_total', 0.0))
            # 将痛觉PE加入F (权重 0.3 = 痛觉预测误差的影响)
            F.total += 0.3 * pain_pe
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

        # ---- TPN ↔ DMN 跷跷板: 突显信号驱动注意力切换 (v4.4 图3 规则4) ----
        # 1. 突显网络 (dACC+AI): 从 ACC 自由能信号提取冲突/新颖/紧迫
        conflict = float(np.tanh(abs(F.total - self.hab.running_F) * 2.0))
        novelty = 0.0
        if len(self.F_accuracy_history) >= 2:
            # 新颖性: F_accuracy 的突增 — "世界不再符合预期"
            acc_change = abs(self.F_accuracy_history[-1] - self.F_accuracy_history[-2])
            novelty = float(np.tanh(acc_change * 5.0))
        urgency = float(np.tanh(self.body.compute_deviation() * 3.0)) if self.body is not None else 0.0
        # v5.4: 痛觉增强紧迫感
        if hasattr(self, '_current_pain_result') and self._current_pain_result:
            pain_intensity = float(self._current_pain_result.get('pain_intensity', 0.0))
            urgency = float(np.clip(urgency + pain_intensity * 0.5, 0.0, 1.0))
            # 疼痛新颖性: allodynia/hyperalgesia → 异常状态 → 新颖
            if self._current_pain_result.get('allodynia', False):
                novelty += 0.2
            if self._current_pain_result.get('hyperalgesia', False):
                novelty += 0.1

        self.tpn.receive_salience(conflict_signal=conflict, novelty_signal=novelty, urgency=urgency)

        # 2. 跷跷板动态: 任务需求 vs 走神基线
        task_demand = conflict + 0.3 * novelty  # 冲突/新颖 → 需要 TPN 介入
        mind_wandering = max(0.05, 1.0 - F.arousal)  # 低唤醒 → 走神倾向
        tpn_act, dmn_act = self.tpn.update_seesaw(
            task_demand=float(np.clip(task_demand, 0.0, 1.0)),
            mind_wandering_baseline=mind_wandering,
            salience=self.tpn.salience_signal,
        )
        # TPN 激活度反馈给 DMN (跷跷板对面)
        self.self_model.tpn_suppression = max(0.0, 1.0 - dmn_act)

        # ---- FPN 探照灯: 注意力增益调制感觉输入 (v4.4 图3 规则4) ----
        # FPN 模板朝向当前任务目标 — 由 TPN 激活度和 F 结构共同定义
        # 目标特征 = 感觉通道中与当前自由能梯度相关的维度
        if self.tpn.tpn_activation > 0.3:
            # 基于信念向量构建注意力目标掩码
            belief_norm = belief / (np.linalg.norm(belief) + 1e-8)
            # 将信念(H-dim)映射到感知空间(D-dim): 视觉通道 [VIS_START:VIS_END]
            goal_features = np.ones(self.fpn.input_dim, dtype=np.float32)
            # 前额叶信念主要投射到视觉区 → 增强视觉注意力
            vis_gain = 1.0 + 0.3 * float(np.tanh(np.mean(np.abs(belief_norm))))
            goal_features[VIS_START:VIS_END] *= vis_gain
            # 注意力精度调制探照灯亮度
            goal_features *= (0.5 + F.attention_precision * 0.5)
            self.fpn.update_template(goal_features, lr=0.05)
        else:
            # TPN 低时 → 探照灯缩回，均匀注意力
            self.fpn.attention_template = np.ones_like(self.fpn.attention_template)

        # 应用探照灯: 增益调制感觉输入 (自下而上信号的增益调制)
        attended_sensory = self.fpn.gate_attention(sensory)

        # v5.5: LC NE → SNR增强 (Yerkes-Dodson倒U曲线调制)
        if hasattr(self, '_lc_result') and self._lc_result:
            lc_snr = float(self._lc_result.get('snr_gain', 1.0))
            if lc_snr != 1.0:
                # 增益 >1 → 增强信号, <1 → 噪声放大
                attended_sensory = attended_sensory * lc_snr

        self.attended_sensory = attended_sensory  # v4.4: 存储供dashboard/debug

        # v4.4: FPN/TPN 状态追踪
        self.tpn_state_history.append(self.tpn.get_state())
        self.fpn_gain_mean_history.append(float(np.mean(self.fpn.attention_template)))

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
        # 多模态刺激累积 (b₅:视觉, b₆:音频) [v5.1: 使用 V5 布局 + agent 状态]
        if self.body.mode == 'text' and len(self.body.b) >= 8:
            vis_norm = float(np.linalg.norm(sensory[VIS_START:VIS_END]))
            aud_norm = float(np.linalg.norm(self._audio_semantic))
            self.body.b[5] = np.clip(self.body.b[5] - 0.003 + 0.01 * vis_norm, 0, 1)
            self.body.b[6] = np.clip(self.body.b[6] - 0.003 + 0.01 * aud_norm, 0, 1)
            # A₄ 恢复 b₅,b₆,b₇
            if action.index == 4:
                self.body.b[5] = max(0.0, self.body.b[5] - 0.02)
                self.body.b[6] = max(0.0, self.body.b[6] - 0.02)
                self.body.b[7] = max(0.0, self.body.b[7] - 0.02)

            # ---- 自听情感传染 (v5.1: 从 agent 内部状态读取, 不再从 sensory 向量) ----
            # self._self_sentiment 由外部 (main_dialogue) 通过 set_self_audio() 写入
            self_sent = self._self_sentiment
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
                self._auto_update_self_model(attended_sensory, F, arousal_now, valence_now)

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
            # v4.4: FPN/TPN 状态
            'tpn_activation': self.tpn.tpn_activation,
            'dmn_activation': self.tpn.dmn_activation,
            'cognitive_effort': self.tpn.cognitive_effort,
            'task_fatigue': self.tpn.task_fatigue,
            'fpn_gain_mean': self.fpn_gain_mean_history[-1] if self.fpn_gain_mean_history else 1.0,
        }

    # ================================================================
    # Wernicke 区: 语言理解 (v5.1)
    # ================================================================

    def comprehend(self, human_vec: np.ndarray,
                   human_sentiment: np.ndarray = None,
                   speaker_name: str = "human",
                   human_text: str = None
                   ) -> tuple[np.ndarray, dict]:
        """Wernicke 区等价物 —— 理解人类输入 (v5.7: TPJ常驻+角回双路径).

        调用 dialogue_memory.comprehend()，传入当前身体/情感状态。
        理解向量驱动 Broca 区生成回应，而不是直接用原始输入检索。

        v5.6 增强:
          - 语音回路: 将输入词写入语音工作记忆
          - TPJ: 推断说话人意图, 语用丰富化理解
          - 语言PE: N400/P600 分量追踪
        v5.7 增强:
          - TPJ: 默认 speaker_name="human", 始终活跃
          - AngularGyrus: 双路径阅读 (脑路径+快速路径混合)

        Args:
            human_vec: 人类输入的语义编码 (64,)
            human_sentiment: 情感信号 (8,), 可选
            speaker_name: 说话人标识 (用于TPJ意图推断), 默认 "human"
            human_text: 原始文本 (用于AngularGyrus阅读通路), 可选

        Returns:
            (comprehension_vec, understanding_dict)
        """
        if human_sentiment is None:
            human_sentiment = np.zeros(8, dtype=np.float32)

        v = self.valence_history[-1] if self.valence_history else 0.0
        a = self.arousal_history[-1] if self.arousal_history else 0.0

        # v5.6: 写入语音回路 (模拟"听到"的过程)
        self.phonological_loop.hear(human_vec[:64])

        comp_vec, understanding = comprehend(
            human_vec=human_vec,
            human_sentiment=human_sentiment,
            agent_net=self.net,
            dialogue_ctx=self.dialogue_ctx,
            body_state=self.body,
            valence=v,
            arousal=a,
        )

        # v5.7: TPJ 语用丰富化 (默认活跃, speaker_name="human")
        if hasattr(self, 'tpj'):
            try:
                ctx_vec = self.dialogue_ctx.get_context_vector()
                intent_vec, tpj_inference = self.tpj.infer_speaker_intent(
                    utterance_vec=human_vec,
                    speaker_name=speaker_name,
                    context_vec=ctx_vec,
                )
                # 语用丰富化: 字面理解 + 意图推断
                social_ctx = getattr(self, '_social_ctx', None)
                enriched = self.tpj.pragmatic_enrichment(
                    literal_comprehension=comp_vec,
                    speaker_intent=intent_vec,
                    social_context=social_ctx,
                )
                comp_vec = enriched
                understanding['tpj_inference'] = tpj_inference
                understanding['pragmatic_enriched'] = True
            except Exception:
                understanding['pragmatic_enriched'] = False

        # v5.7: Angular Gyrus 双路径阅读 (脑路径 + 快速路径混合)
        if human_text is not None and hasattr(self, 'angular_gyrus'):
            try:
                ag_phon, ag_conf = self.angular_gyrus.read(
                    human_text, context_vec=human_vec)
                understanding['ag_phon_norm'] = float(np.linalg.norm(ag_phon))
                understanding['ag_confidence'] = float(ag_conf)
                # 双路径混合: 脑路径(AG)权重随训练量增长
                ag_weight = min(0.4, 0.1 + ag_conf * 0.3)
                if ag_conf > 0.15:
                    comp_vec = ((1.0 - ag_weight) * comp_vec
                                + ag_weight * ag_phon).astype(np.float32)
                understanding['ag_weight'] = float(ag_weight)
            except Exception:
                understanding['ag_confidence'] = 0.0

        # v5.6: 追踪语言PE
        if 'F_language' in understanding:
            self.F_language_history.append(understanding['F_language'])
            self.language_pe_history.append({
                'semantic_pe': understanding.get('semantic_pe', 0.0),
                'syntactic_pe': understanding.get('syntactic_pe', 0.0),
                'phonological_pe': understanding.get('phonological_pe', 0.0),
                'F_language': understanding['F_language'],
            })

        # v6.0: 语义记忆查询 — 用已学知识丰富理解
        if hasattr(self, 'semantic_memory'):
            try:
                semantic_hits = self.semantic_memory.query(human_vec, top_k=3)
                understanding['semantic_knowledge'] = [
                    {'sim': float(sim), 'activation': float(c.activation)}
                    for c, sim in semantic_hits
                ]
                familiarity = self.semantic_memory.knows_about(human_vec)
                understanding['familiarity'] = familiarity
                # 熟悉度高 → 理解更"确信" (类似语义启动效应)
                if familiarity > 0.3:
                    boost = min(0.2, familiarity * 0.2)
                    comp_vec = comp_vec + boost * np.sign(comp_vec).astype(np.float32)
            except Exception:
                understanding['semantic_knowledge'] = []
                understanding['familiarity'] = 0.0

        # 存储最近的理解向量 (供自听回路和AF使用)
        self._last_comprehension = comp_vec
        return comp_vec, understanding

    def speak(self, broca, query_vec: np.ndarray = None,
              belief_vec: np.ndarray = None,
              valence: float = 0.0, arousal: float = 0.0,
              max_words: int = 18, temperature: float = 0.7,
              use_phrase_structure: bool = True,
              human_text: str = None,
              ) -> tuple[list[str], np.ndarray | None, dict]:
        """v5.6: 全语言产出管线 — AF → Broca → Motor Cortex.

        流程 (对应 Wernicke-Geschwind 模型 + 双流模型):
          1. AF腹侧: 理解 → 言语种子 (弓状束复述通路)
          2. Broca: 种子词 → Hebb词序链生成 (Broca区)
          3. 短语结构: 调制词候选 (BA44层级句法)
          4. Motor Cortex: 发音计划 + 运动指令副本 (M1/SMA)
          5. AF背侧: 运动副本 → 预期听觉 (自我监控)

        Args:
            broca: Broca实例
            query_vec: 查询向量 (64,) — 用于Broca生成
            belief_vec: 信念向量 (64,) — Agent内部状态
            valence: 当前效价
            arousal: 当前唤醒
            max_words: 最大词数
            temperature: 生成温度
            use_phrase_structure: 是否启用短语结构约束
            human_text: 原始人类输入文本 (v5.7: 用于婴儿模仿模式)

        Returns:
            (words, audio, speech_diagnostics)
        """
        diagnostics = {
            'af_seed_used': False,
            'phrase_structure_used': False,
            'motor_plan_executed': False,
            'self_monitoring_pe': 0.0,
        }

        from cns.data_types import D

        if query_vec is None:
            query_vec = self._last_comprehension.copy()

        if belief_vec is None:
            # Use top cluster's full centroid (D-dim) — broca expects >=64 dims
            if self.net.n_clusters > 0:
                top = max(self.net.clusters, key=lambda c: c.activation)
                if top.activation > 0:
                    belief_vec = top.centroid.copy()
                else:
                    belief_vec = np.zeros(D, dtype=np.float32)
            else:
                belief_vec = np.zeros(D, dtype=np.float32)

        # ---- Step 1: AF 腹侧通路 → 言语种子 ----
        comprehension_vec = self._last_comprehension
        af_seed_vec, af_confidence = self.arcuate_fasciculus.repeat(
            comprehension_vec, temperature=0.4)

        diagnostics['af_seed_used'] = af_confidence > 0.1
        diagnostics['af_confidence'] = float(af_confidence)

        # ---- Step 2: AF 种子调制查询向量 ----
        # AF种子与查询混合 → 言语产出同时受理解和当前状态驱动
        if af_confidence > 0.15:
            query_vec = (0.5 * query_vec[:64] + 0.5 * af_seed_vec[:64]
                        ).astype(np.float32)

        # ---- Step 3: Broca 词序 Hebb 链生成 (v5.7: 短语结构约束) ----
        phrase_net = self.phrase_structure if (
            use_phrase_structure and hasattr(self, 'phrase_structure')
            and self.phrase_structure._trained) else None
        words, audio = broca.speak_from_state(
            belief_vec=belief_vec,
            body_state=self.body,
            query_vec=query_vec,
            valence=valence,
            arousal=arousal,
            max_words=max_words,
            temperature=temperature,
            phrase_network=phrase_net,
            phrase_strength=0.25,
            human_text=human_text,
        )

        # ---- Step 4: 短语结构约束 (v5.6) ----
        if use_phrase_structure and hasattr(self, 'phrase_structure'):
            if self.phrase_structure._trained and len(words) >= 3:
                diagnostics['phrase_structure_used'] = True
                # 获取短语结构诊断 (不重新生成, 只记录)
                phrase_info = self.phrase_structure.get_boundary_examples(
                    ''.join(words))
                diagnostics['phrase_boundaries'] = len(phrase_info)

        # ---- Step 5: 运动皮层发音计划 ----
        if len(words) >= 1:
            try:
                word_vecs = [broca._word_to_vec(w) for w in words]
                word_vecs = [wv for wv in word_vecs if wv is not None]
                if word_vecs:
                    motor_plans = self.motor_cortex.plan_sequence(
                        word_vecs, words)
                    diagnostics['motor_plan_executed'] = True
                    diagnostics['motor_plan_length'] = len(motor_plans)

                    # ---- Step 6: AF 背侧通路 (运动副本 → 预期听觉) ----
                    if motor_plans:
                        # 从运动计划生成预期听觉
                        expected_audio = self.motor_cortex.efference_copy(
                            motor_plans[-1])

                        # AF 背侧: 言语计划 → 预期听觉
                        speech_plan_vec = np.mean(
                            [wv[:64] for wv in word_vecs], axis=0)
                        af_expected, _ = self.arcuate_fasciculus.efference_copy(
                            speech_plan_vec, motor_plans[-1])

                        # 学习背侧关联
                        self.arcuate_fasciculus.learn_dorsal(
                            speech_plan=speech_plan_vec,
                            actual_auditory=expected_audio,
                            motor_plan=motor_plans[-1],
                            weight=0.5,
                        )
            except Exception:
                pass

        # ---- Step 7: 自听 → 语音回路 ----
        if len(words) >= 1:
            try:
                word_vecs_audio = []
                for w in words:
                    wv = broca._word_to_vec(w)
                    if wv is not None:
                        word_vecs_audio.append(wv[:64])
                if word_vecs_audio:
                    self.phonological_loop.hear_sequence(word_vecs_audio)
            except Exception:
                pass

        # ---- Step 8: AF 腹侧学习 (comprehension → speech) ----
        if len(words) >= 2 and af_confidence > 0.1:
            try:
                # 提取回应中的关键种子词
                seed_vecs = []
                for w in words[:3]:
                    wv = broca._word_to_vec(w)
                    if wv is not None:
                        seed_vecs.append(wv[:64])
                if seed_vecs:
                    speech_seed = np.mean(seed_vecs, axis=0)
                    self.arcuate_fasciculus.learn_ventral(
                        comprehension_vec=comprehension_vec,
                        speech_seed_vec=speech_seed,
                        weight=0.3,
                    )
            except Exception:
                pass

        return words, audio, diagnostics

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

    def set_self_audio(self, audio_semantic: np.ndarray,
                       self_sentiment: np.ndarray):
        """v5.1: 设置自听回路状态 (替代旧 s[128:192] + s[96:104] 方案).

        由 main_dialogue 在 agent.step() 之前调用,
        将上一轮自己说话的语义和情感写入 agent 内部状态.

        Args:
            audio_semantic: 自听语义向量 (64,)
            self_sentiment: 自听情感信号 (8,)
        """
        if audio_semantic is not None:
            flen = min(len(audio_semantic), len(self._audio_semantic))
            self._audio_semantic[:flen] = audio_semantic[:flen]
        if self_sentiment is not None:
            flen = min(len(self_sentiment), len(self._self_sentiment))
            self._self_sentiment[:flen] = self_sentiment[:flen]

    def set_current_image(self, image: np.ndarray = None):
        """v5.1: 设置当前帧图像 (供 Phase 0 视觉层级管线使用).

        Args:
            image: (H, W, 3) uint8 图像, 或 None 清除
        """
        self._current_image = image

    def set_audio_input(self, audio_data: dict = None):
        """v5.3: 设置真实音频输入 (供 Phase 0b 听觉层级管线使用).

        当设置真实音频数据时, Phase 0b 将使用真实的 mel 频谱
        替代语义代理模式。设置 None 回到语义代理模式。

        Args:
            audio_data: AudioInput.from_file()/from_mic() 返回的 dict,
                       或 None 清除 (回到语义代理模式).
                       dict 应包含:
                         'spectrum': mono mel 频谱 (32,)
                         'left_spectrum': 左耳频谱 (可选, 立体声)
                         'right_spectrum': 右耳频谱 (可选, 立体声)
                         'is_stereo': 是否立体声
        """
        self._current_audio_data = audio_data

    def set_pain_input(self, nociceptive: float = 0.0,
                       abeta_input: float = 0.0):
        """v5.4: 设置外部痛觉输入 (供环境/场景驱动疼痛).

        Args:
            nociceptive: 伤害性信号强度 [0, 1]
                         >0.5 = 组织损伤/有害刺激
                         >0.8 = 严重损伤
            abeta_input: Aβ触觉输入 [0, 1]
                         按摩/触摸/摩擦 → 关闭闸门 → 缓解疼痛
        """
        self._current_pain_input = float(np.clip(nociceptive, 0.0, 1.0))
        self._current_abeta_input = float(np.clip(abeta_input, 0.0, 1.0))

    # ================================================================
    # 会话持久化 (v5.7)
    # ================================================================

    def save(self, path: str = None, name: str = None,
             n_sessions: int = 1, n_turns: int = 0) -> str:
        """保存 Agent 完整状态到磁盘.

        Args:
            path: 完整路径 (优先)
            name: 存档名 (不含路径)
            n_sessions: 累计会话数
            n_turns: 当前对话轮数

        Returns:
            保存路径
        """
        from cns.persistence import save_agent
        return save_agent(self, path=path, name=name,
                         n_sessions=n_sessions,
                         extra={'total_turns': n_turns,
                                'total_steps': self.meta.step_count
                                if hasattr(self, 'meta') else 0})

    @classmethod
    def load(cls, path: str, rng=None, agent_id: int = 0,
             n_agents: int = 1, verbose: bool = True):
        """从存档加载 Agent.

        创建全新 Agent 实例并用存档数据恢复所有状态.
        跳过热身和预训练 — 网络已有知识.

        Args:
            path: 存档路径
            rng: 随机数生成器 (可选)
            agent_id: Agent ID
            n_agents: Agent 数量
            verbose: 是否打印恢复进度

        Returns:
            (agent, metadata) — agent 实例和存档元数据
        """
        from cns.persistence import load_agent_state, restore_agent

        if rng is None:
            rng = np.random.default_rng()

        data = load_agent_state(path)
        agent = cls(rng=rng, agent_id=agent_id, n_agents=n_agents)

        if verbose:
            print(f"  [Persistence] Loading agent from: {path}")
            print(f"    Version: {data.get('version', 'unknown')}")
            print(f"    Sessions: {data.get('n_sessions', 1)}")
            print(f"    Turns: {data.get('total_turns', 0)}")

        restore_agent(agent, data, verbose=verbose)

        metadata = {
            'version': data.get('version', 'unknown'),
            'n_sessions': data.get('n_sessions', 1),
            'total_turns': data.get('total_turns', 0),
            'total_steps': data.get('total_steps', 0),
            'timestamp': data.get('timestamp', 0),
        }

        return agent, metadata

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
