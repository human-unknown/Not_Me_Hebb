"""
persistence.py — Agent 全状态持久化 (v5.7)

提供 save/load 完整 Agent 状态:
  - 所有 Hebb 网络 (L0, SelfModel, AF ventral/dorsal, AngularGyrus, TPJ)
  - Body state + 稳态调节器 (Hypothalamus)
  - 神经调节状态 (VTA, Locus Coeruleus)
  - 语言系统状态 (PhonologicalLoop, PhraseStructure, MotorCortex)
  - 元学习参数 (Theta, MetaLearner)
  - 追踪历史 (F/valence/arousal/attention/action)
  - 对话上下文 (DialogueContext)

自动保存策略:
  - 每 10 轮对话自动保存
  - Ctrl+C / exit 时保存
  - 启动时自动加载最新存档

存档路径: .notme/sessions/agent_YYYYMMDD_HHMMSS.pkl
"""

import os
import pickle
import time
import glob
import numpy as np
from typing import Optional


# ================================================================
# 版本管理
# ================================================================

SAVE_VERSION = "5.7"  # v6.2: schema updated with tag/persistence/consolidation fields
SAVE_DIR = ".notme/sessions"


def _get_save_dir() -> str:
    """获取存档目录 (相对于当前工作目录)."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    return SAVE_DIR


def _make_path(name: str = None) -> str:
    """生成存档路径."""
    save_dir = _get_save_dir()
    if name:
        return os.path.join(save_dir, f"{name}.pkl")
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(save_dir, f"agent_{ts}.pkl")


def list_saves() -> list[str]:
    """列出所有存档文件, 按时间排序 (最新的在前)."""
    save_dir = _get_save_dir()
    files = glob.glob(os.path.join(save_dir, "*.pkl"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def latest_save() -> Optional[str]:
    """返回最新存档路径, 无存档则 None."""
    files = list_saves()
    return files[0] if files else None


# ================================================================
# State extraction helpers
# ================================================================

def _save_cluster_network(net) -> dict:
    """序列化 ClusterNetwork 状态 (v6.1: +保护/PNN/STDP/候选)."""
    if net is None:
        return {'n_clusters': 0, 'clusters': [], 'buckets': {},
                'total_activation': 0.0, 'candidate_clusters': []}
    clusters_data = []
    for c in net.clusters:
        clusters_data.append({
            'centroid': c.centroid.copy(),
            'activation': float(c.activation),
            'G_ema': float(c.G_ema),
            'count': int(c.count),
            'age': int(getattr(c, 'age', 0)),
            # v6.1: 新字段
            'protection_score': float(getattr(c, 'protection_score', 0.0)),
            'pnn_level': float(getattr(c, 'pnn_level', 0.0)),
            'stdp_links': {str(k): float(v) for k, v
                          in getattr(c, 'stdp_links', {}).items()},
            # v6.2: 记忆巩固优化字段
            'tag': float(getattr(c, 'tag', 0.0)),
            'tag_age': int(getattr(c, 'tag_age', 0)),
            'activation_persistence': float(getattr(c, 'activation_persistence', 0.0)),
            'consolidation_count': int(getattr(c, 'consolidation_count', 0)),
        })
    # v6.1: 候选集群
    candidates_data = []
    for cc in getattr(net, '_candidate_clusters', []):
        candidates_data.append({
            'centroid': cc.centroid.copy(),
            'exposure_count': int(cc.exposure_count),
            'max_similarity': float(cc.max_similarity),
            'age': int(getattr(cc, 'age', 0)),
        })
    return {
        'n_clusters': net.n_clusters,
        'clusters': clusters_data,
        'buckets': {k: list(v) for k, v in net.buckets.items()},
        'total_activation': float(net.total_activation),
        'theta': {
            'cluster_threshold': float(net.theta.cluster_threshold),
            'learn_rate_l0': float(net.theta.learn_rate_l0),
            'decay_rate': float(net.theta.decay_rate),
        },
        # v6.1
        'candidate_clusters': candidates_data,
        'n_candidates': len(candidates_data),
    }


def _restore_cluster_network(net, data: dict):
    """从序列化数据恢复 ClusterNetwork (v6.1: +保护/PNN/STDP/候选)."""
    if net is None or not data:
        return
    from cns.data_types import Cluster, CandidateCluster, D as D_DIM
    net.clusters = []
    for cd in data.get('clusters', []):
        centroid = np.array(cd['centroid'], dtype=np.float32)
        if len(centroid) < D_DIM:
            padded = np.zeros(D_DIM, dtype=np.float32)
            padded[:len(centroid)] = centroid
            centroid = padded
        c = Cluster(centroid=centroid)
        c.activation = float(cd['activation'])
        c.G_ema = float(cd['G_ema'])
        c.count = int(cd.get('count', 0))
        c.age = int(cd.get('age', 0))
        # v6.1: 恢复新字段
        c.protection_score = float(cd.get('protection_score', 0.0))
        c.pnn_level = float(cd.get('pnn_level', 0.0))
        c.stdp_links = {int(k): float(v) for k, v
                       in cd.get('stdp_links', {}).items()}
        # v6.2: 恢复记忆巩固优化字段
        c.tag = float(cd.get('tag', 0.0))
        c.tag_age = int(cd.get('tag_age', 0))
        c.activation_persistence = float(cd.get('activation_persistence', 0.0))
        c.consolidation_count = int(cd.get('consolidation_count', 0))
        net.clusters.append(c)
    net._n_clusters = len(net.clusters)
    # v6.1: 从恢复的簇重建桶 (旧桶引用指向序列化的旧对象)
    net.buckets = {}
    for c in net.clusters:
        key = net._hash_to_bucket(c.centroid)
        net.buckets.setdefault(key, []).append(c)
    if 'theta' in data:
        net.theta.cluster_threshold = float(data['theta']['cluster_threshold'])
        net.theta.learn_rate_l0 = float(data['theta']['learn_rate_l0'])
        net.theta.decay_rate = float(data['theta']['decay_rate'])

    # v6.1: 恢复候选集群
    net._candidate_clusters = []
    for ccd in data.get('candidate_clusters', []):
        centroid = np.array(ccd['centroid'], dtype=np.float32)
        if len(centroid) < D_DIM:
            padded = np.zeros(D_DIM, dtype=np.float32)
            padded[:len(centroid)] = centroid
            centroid = padded
        cc = CandidateCluster(centroid=centroid)
        cc.exposure_count = int(ccd.get('exposure_count', 1))
        cc.max_similarity = float(ccd.get('max_similarity', 0.0))
        cc.age = int(ccd.get('age', 0))
        net._candidate_clusters.append(cc)


def _default_setpoints(mode: str) -> np.ndarray:
    """返回 BodyVector 默认设定点."""
    if mode == 'text':
        return np.array([0.7, 0.7, 0.0, 0.0, 0.3, 0.3, 0.3, 0.5, 0.0], dtype=np.float32)
    else:
        return np.array([0.7, 0.7, 0.0, 0.0, 0.3], dtype=np.float32)


def _default_decays(mode: str) -> np.ndarray:
    """返回 BodyVector 默认衰减率."""
    if mode == 'text':
        return np.array([-0.003, 0.0, 0.002, 0.0, 0.0,
                         -0.003, -0.003, 0.001, -0.001], dtype=np.float32)
    else:
        return np.array([-0.003, -0.002, 0.004, 0.0, 0.001], dtype=np.float32)


def _save_body(body) -> dict:
    """序列化 BodyVector."""
    if body is None:
        return {}
    return {
        'b': body.b.copy().tolist(),
        'mode': body.mode,
        'M': body.M,
        'setpoints': body.setpoints.copy().tolist() if body.setpoints is not None and len(body.setpoints) > 0 else [],
        'decays': body.decays.copy().tolist() if body.decays is not None else [],
    }


def _restore_body(body, data: dict):
    """恢复 BodyVector."""
    if body is None or not data:
        return
    body.b = np.array(data['b'], dtype=np.float32)
    body.mode = data.get('mode', 'text')
    body.M = data.get('M', len(body.b))

    # 恢复 setpoints: 优先使用存档数据, 否则按 body.mode 默认
    if 'setpoints' in data and data['setpoints']:
        body.setpoints = np.array(data['setpoints'], dtype=np.float32)
    elif body.setpoints is None or len(body.setpoints) != body.M:
        body.setpoints = _default_setpoints(body.mode)

    # 恢复 decays: 优先使用存档数据, 否则按 body.mode 默认
    if 'decays' in data and data['decays']:
        body.decays = np.array(data['decays'], dtype=np.float32)
    elif body.decays is None or len(body.decays) != body.M:
        body.decays = _default_decays(body.mode)


# ================================================================
# 主接口
# ================================================================

def save_agent(agent, path: str = None, name: str = None,
               n_sessions: int = 1, extra: dict = None) -> str:
    """保存 Agent 完整状态.

    Args:
        agent: Agent 实例
        path: 完整路径 (优先于 name)
        name: 存档名 (不含路径和扩展名)
        n_sessions: 累计会话数
        extra: 额外元数据

    Returns:
        实际保存路径
    """
    if path is None:
        path = _make_path(name)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # ---- 收集所有状态 ----
    data = {
        'version': SAVE_VERSION,
        'timestamp': time.time(),
        'n_sessions': n_sessions,
        'total_turns': 0,
        'total_steps': 0,
    }

    if extra:
        data.update(extra)

    # L0: ClusterNetwork
    data['net'] = _save_cluster_network(agent.net)

    # L1: HabituationTracker
    data['hab'] = {
        'running_F': float(agent.hab.running_F),
        'n': int(agent.hab.n),
        'tau': float(agent.hab.tau),
    }

    # L2: MoEGate
    data['moe'] = {
        'n_experts': int(agent.moe.n_experts),
        'budgets': agent.moe.budgets.copy().tolist(),
        'performance': agent.moe.performance.copy().tolist(),
    }

    # Body
    data['body'] = _save_body(agent.body)

    # SelfModel
    if hasattr(agent, 'self_model') and agent.self_model is not None:
        data['self_model'] = _save_cluster_network(agent.self_model.net)
        data['self_model_meta'] = {
            'n_experiences': agent.self_model.n_experiences,
            'tpn_suppression': float(getattr(agent.self_model, 'tpn_suppression', 0.5)),
        }

    # DialogueContext
    if hasattr(agent, 'dialogue_ctx') and agent.dialogue_ctx is not None:
        ctx = agent.dialogue_ctx
        data['dialogue_ctx'] = {
            'turns': list(ctx.turns) if hasattr(ctx, 'turns') else [],
            'max_turns': ctx.max_turns,
        }

    # Theta
    if hasattr(agent, 'theta'):
        data['theta'] = agent.theta.to_dict()

    # MetaLearner
    if hasattr(agent, 'meta'):
        data['meta_learner'] = {
            'step_count': agent.meta.step_count,
            'is_critical': agent.meta.is_critical,
            'gradient_history': list(getattr(agent.meta, 'gradient_history', [])),
        }

    # Tracking histories (keep last 500 to bound size)
    max_hist = 500
    for key in ['F_history', 'F_body_history', 'F_social_history',
                'F_cognitive_history', 'F_accuracy_history',
                'valence_history', 'arousal_history', 'attention_history',
                'action_history', 'reward_history', 'F_language_history']:
        if hasattr(agent, key):
            val = getattr(agent, key)
            data[key] = list(val[-max_hist:]) if val else []

    data['language_pe_history'] = list(
        getattr(agent, 'language_pe_history', [])[-max_hist:])

    # Self audio tracking
    data['self_audio'] = {
        'self_valence_ema': float(agent.self_valence_ema),
        'self_arousal_ema': float(agent.self_arousal_ema),
        'self_coherence': float(agent.self_coherence),
    }

    # TPN/FPN state
    if hasattr(agent, 'tpn'):
        data['tpn_state'] = agent.tpn.get_state()
    if hasattr(agent, 'fpn'):
        data['fpn_state'] = {
            'attention_template': agent.fpn.attention_template.copy().tolist(),
        }

    # ---- v5.5: Neuromodulatory ----
    # Hypothalamus
    if hasattr(agent, 'hypothalamus') and agent.hypothalamus is not None:
        hypo = agent.hypothalamus
        data['hypothalamus'] = {
            'setpoints': {k: float(v) for k, v in hypo.setpoints.items()}
            if hasattr(hypo, 'setpoints') else {},
        }

    # VTA
    if hasattr(agent, 'vta') and agent.vta is not None:
        vta = agent.vta
        data['vta'] = {
            'da_level': float(getattr(vta, 'da_level', 0.3)),
            'rpe_ema': float(getattr(vta, 'rpe_ema', 0.0)),
        }

    # Locus Coeruleus
    if hasattr(agent, 'locus_coeruleus') and agent.locus_coeruleus is not None:
        lc = agent.locus_coeruleus
        data['locus_coeruleus'] = {
            'tonic_ne': float(getattr(lc, 'tonic_ne', 0.2)),
        }

    # ---- v5.6: Language system ----
    # Arcuate Fasciculus
    if hasattr(agent, 'arcuate_fasciculus') and agent.arcuate_fasciculus is not None:
        af = agent.arcuate_fasciculus
        data['arcuate_fasciculus'] = {
            'ventral': _save_cluster_network(af.ventral_net),
            'dorsal': _save_cluster_network(af.dorsal_net),
            'n_ventral_clusters': af.n_ventral_clusters,
            'n_dorsal_clusters': af.n_dorsal_clusters,
        }

    # Phonological Loop
    if hasattr(agent, 'phonological_loop') and agent.phonological_loop is not None:
        pl = agent.phonological_loop
        data['phonological_loop'] = {
            'store': [s.copy().tolist() if isinstance(s, np.ndarray)
                      else s for s in pl.store],
            'max_chunks': pl.max_chunks,
        }

    # Phrase Structure
    if hasattr(agent, 'phrase_structure') and agent.phrase_structure is not None:
        ps = agent.phrase_structure
        data['phrase_structure'] = {
            'trained': ps._trained,
            'n_phrase_types': ps.n_phrase_types,
            # bigram_probs and unigram_counts are too large;
            # they will be re-computed from corpus on next startup
            # but we keep the phrase boundary threshold
            'boundary_threshold': float(getattr(ps, 'boundary_threshold', 0.1)),
        }

    # Angular Gyrus
    if hasattr(agent, 'angular_gyrus') and agent.angular_gyrus is not None:
        ag = agent.angular_gyrus
        data['angular_gyrus'] = {
            'grapheme_to_phoneme': _save_cluster_network(ag.grapheme_to_phoneme),
            'trained': ag._trained,
            'n_associations': ag._n_associations,
            'word_cache': {k: v.tolist() for k, v in ag._word_to_phoneme.items()},
        }

    # Motor Cortex
    if hasattr(agent, 'motor_cortex') and agent.motor_cortex is not None:
        mc = agent.motor_cortex
        data['motor_cortex'] = {
            'state': mc.get_state() if hasattr(mc, 'get_state') else {},
        }

    # TPJ
    if hasattr(agent, 'tpj') and agent.tpj is not None:
        tpj = agent.tpj
        data['tpj'] = {
            'speaker_models': _save_cluster_network(tpj.speaker_net)
            if hasattr(tpj, 'speaker_net') else None,
            'intent_clusters': _save_cluster_network(tpj.intent_net)
            if hasattr(tpj, 'intent_net') else None,
            '_n_intents': getattr(tpj, '_n_intents', 0),
        }

    # Consolidation state
    data['consolidation'] = {
        'counter': agent.consolidation_counter,
        'dialogue_since': agent.dialogue_since_consolidation,
    }

    # ---- v6.0: Semantic Memory ----
    if hasattr(agent, 'semantic_memory') and agent.semantic_memory is not None:
        sm = agent.semantic_memory
        data['semantic_memory'] = _save_cluster_network(sm.net)
        data['semantic_memory_meta'] = {
            'n_facts': sm.n_facts,
            'consolidation_count': sm.consolidation_count,
        }

    # ---- v6.0: Striatum ----
    if hasattr(agent, 'striatum') and agent.striatum is not None:
        st = agent.striatum
        data['striatum'] = {
            'd1_weights': {str(k): v.tolist() for k, v in st.d1_weights.items()},
            'd2_weights': {str(k): v.tolist() for k, v in st.d2_weights.items()},
            'habit_strength': {str(k): float(v) for k, v in st.habit_strength.items()},
            'cooccurrence': {f"{k[0]}_{k[1]}": v
                           for k, v in st.cooccurrence.items()},
            'n_learn_events': st.n_learn_events,
            '_habit_ema': float(st._habit_ema),
        }

    # ---- 写入磁盘 ----
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return path


def load_agent_state(path: str) -> dict:
    """加载 Agent 状态 (不恢复对象, 返回原始数据).

    返回完整的状态 dict, 供 Agent.load_state() 使用.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    version = data.get('version', 'unknown')
    if version != SAVE_VERSION:
        print(f"  [Persistence] Warning: save version {version} != "
              f"current {SAVE_VERSION}, attempting migration...")

    return data


def restore_agent(agent, data: dict, verbose: bool = True):
    """从状态 dict 恢复 Agent.

    将持久化数据写回 Agent 实例的所有子模块.

    Args:
        agent: Agent 实例 (已初始化但可能未预热)
        data: load_agent_state() 返回的状态 dict
        verbose: 是否打印恢复进度
    """
    # L0: ClusterNetwork
    if 'net' in data:
        _restore_cluster_network(agent.net, data['net'])
        if verbose:
            print(f"  L0: {agent.net.n_clusters} clusters restored")

    # L1: HabituationTracker
    if 'hab' in data:
        agent.hab.running_F = float(data['hab']['running_F'])
        agent.hab.n = int(data['hab'].get('n', 0))
        agent.hab.tau = float(data['hab'].get('tau', 10.0))

    # L2: MoEGate
    if 'moe' in data:
        agent.moe.n_experts = int(data['moe'].get('n_experts', 3))
        budgets = data['moe'].get('budgets', None)
        if budgets is not None:
            agent.moe.budgets = np.array(budgets, dtype=np.float64)
        perf = data['moe'].get('performance', None)
        if perf is not None:
            agent.moe.performance = np.array(perf, dtype=np.float64)

    # Body
    if 'body' in data:
        _restore_body(agent.body, data['body'])

    # SelfModel
    if 'self_model' in data and hasattr(agent, 'self_model'):
        _restore_cluster_network(agent.self_model.net, data['self_model'])
        if 'self_model_meta' in data:
            agent.self_model.n_experiences = data['self_model_meta'].get('n_experiences', 0)
            agent.self_model.tpn_suppression = float(
                data['self_model_meta'].get('tpn_suppression', 0.5))
        if verbose:
            print(f"  SelfModel: {agent.self_model.net.n_clusters} clusters restored")

    # DialogueContext
    if 'dialogue_ctx' in data and hasattr(agent, 'dialogue_ctx'):
        ctx_data = data['dialogue_ctx']
        agent.dialogue_ctx.turns = list(ctx_data.get('turns', []))
        agent.dialogue_ctx.max_turns = ctx_data.get('max_turns', 8)

    # Theta
    if 'theta' in data and hasattr(agent, 'theta'):
        for k, v in data['theta'].items():
            if hasattr(agent.theta, k):
                setattr(agent.theta, k, v)
        if verbose:
            print(f"  Theta: {len(data['theta'])} params restored")

    # MetaLearner
    if 'meta_learner' in data and hasattr(agent, 'meta'):
        agent.meta.step_count = data['meta_learner'].get('step_count', 0)
        agent.meta.is_critical = data['meta_learner'].get('is_critical', False)
        if hasattr(agent.meta, 'gradient_history'):
            agent.meta.gradient_history = list(
                data['meta_learner'].get('gradient_history', []))

    # Tracking histories
    for key in ['F_history', 'F_body_history', 'F_social_history',
                'F_cognitive_history', 'F_accuracy_history',
                'valence_history', 'arousal_history', 'attention_history',
                'action_history', 'reward_history', 'F_language_history',
                'language_pe_history']:
        if key in data and hasattr(agent, key):
            setattr(agent, key, list(data[key]))

    # Self audio
    if 'self_audio' in data:
        agent.self_valence_ema = float(data['self_audio'].get('self_valence_ema', 0.0))
        agent.self_arousal_ema = float(data['self_audio'].get('self_arousal_ema', 0.0))
        agent.self_coherence = float(data['self_audio'].get('self_coherence', 1.0))

    # TPN/FPN
    if 'tpn_state' in data and hasattr(agent, 'tpn'):
        ts = data['tpn_state']
        agent.tpn.tpn_activation = float(ts.get('tpn_activation', 0.3))
        agent.tpn.dmn_activation = float(ts.get('dmn_activation', 0.7))
        agent.tpn.cognitive_effort = float(ts.get('cognitive_effort', 0.0))
        agent.tpn.task_fatigue = float(ts.get('task_fatigue', 0.0))
        agent.tpn.salience_signal = float(ts.get('salience_signal', 0.0))
    if 'fpn_state' in data and hasattr(agent, 'fpn'):
        template = data['fpn_state'].get('attention_template', None)
        if template is not None:
            agent.fpn.attention_template = np.array(template, dtype=np.float32)

    # ---- v5.5: Neuromodulatory ----
    if 'hypothalamus' in data and hasattr(agent, 'hypothalamus'):
        if agent.hypothalamus is not None and hasattr(agent.hypothalamus, 'setpoints'):
            for k, v in data['hypothalamus'].get('setpoints', {}).items():
                agent.hypothalamus.setpoints[k] = float(v)

    if 'vta' in data and hasattr(agent, 'vta'):
        if agent.vta is not None:
            agent.vta.da_level = float(data['vta'].get('da_level', 0.3))
            agent.vta.rpe_ema = float(data['vta'].get('rpe_ema', 0.0))

    if 'locus_coeruleus' in data and hasattr(agent, 'locus_coeruleus'):
        if agent.locus_coeruleus is not None:
            agent.locus_coeruleus.tonic_ne = float(
                data['locus_coeruleus'].get('tonic_ne', 0.2))

    # ---- v5.6: Language system ----
    if 'arcuate_fasciculus' in data and hasattr(agent, 'arcuate_fasciculus'):
        af = agent.arcuate_fasciculus
        if af is not None:
            af_data = data['arcuate_fasciculus']
            _restore_cluster_network(af.ventral_net, af_data.get('ventral', {}))
            _restore_cluster_network(af.dorsal_net, af_data.get('dorsal', {}))
            if verbose:
                print(f"  AF: {af.n_ventral_clusters}v/{af.n_dorsal_clusters}d "
                      f"clusters restored")

    if 'phonological_loop' in data and hasattr(agent, 'phonological_loop'):
        pl = agent.phonological_loop
        if pl is not None:
            store = data['phonological_loop'].get('store', [])
            restored_store = []
            for s in store:
                try:
                    restored_store.append(np.array(s, dtype=np.float32))
                except (ValueError, TypeError):
                    # Skip items that can't be restored
                    pass
            pl.store = restored_store

    if 'phrase_structure' in data and hasattr(agent, 'phrase_structure'):
        ps = agent.phrase_structure
        if ps is not None:
            ps._trained = data['phrase_structure'].get('trained', False)
            ps.boundary_threshold = float(
                data['phrase_structure'].get('boundary_threshold', 0.1))

    if 'angular_gyrus' in data and hasattr(agent, 'angular_gyrus'):
        ag = agent.angular_gyrus
        if ag is not None:
            ag_data = data['angular_gyrus']
            _restore_cluster_network(ag.grapheme_to_phoneme,
                                     ag_data.get('grapheme_to_phoneme', {}))
            ag._trained = ag_data.get('trained', False)
            ag._n_associations = ag_data.get('n_associations', 0)
            ag._word_to_phoneme = {
                k: np.array(v, dtype=np.float32)
                for k, v in ag_data.get('word_cache', {}).items()}
            if verbose:
                print(f"  AngularGyrus: {ag.n_grapheme_clusters} glyphs, "
                      f"{len(ag._word_to_phoneme)} cached words restored")

    if 'motor_cortex' in data and hasattr(agent, 'motor_cortex'):
        # MotorCortex state is mostly stateless (Hebb mappings)
        pass

    if 'tpj' in data and hasattr(agent, 'tpj'):
        tpj = agent.tpj
        if tpj is not None:
            tpj_data = data['tpj']
            if tpj_data.get('speaker_models'):
                _restore_cluster_network(tpj.speaker_net,
                                         tpj_data['speaker_models'])
            if tpj_data.get('intent_clusters'):
                _restore_cluster_network(tpj.intent_net,
                                         tpj_data['intent_clusters'])
            tpj._n_intents = tpj_data.get('_n_intents', 0)
            if verbose:
                print(f"  TPJ: {tpj._n_intents} intents restored")

    # ---- v6.0: Semantic Memory ----
    if 'semantic_memory' in data and hasattr(agent, 'semantic_memory'):
        sm = agent.semantic_memory
        if sm is not None:
            _restore_cluster_network(sm.net, data['semantic_memory'])
            meta = data.get('semantic_memory_meta', {})
            sm.n_facts = meta.get('n_facts', 0)
            sm.consolidation_count = meta.get('consolidation_count', 0)
            if verbose:
                print(f"  Semantic: {sm.n_clusters} clusters, "
                      f"{sm.n_facts} facts restored")

    # ---- v6.0: Striatum ----
    if 'striatum' in data and hasattr(agent, 'striatum'):
        st = agent.striatum
        if st is not None:
            st_data = data['striatum']
            st.d1_weights = {int(k): np.array(v, dtype=np.float32)
                           for k, v in st_data.get('d1_weights', {}).items()}
            st.d2_weights = {int(k): np.array(v, dtype=np.float32)
                           for k, v in st_data.get('d2_weights', {}).items()}
            st.habit_strength = {int(k): float(v)
                               for k, v in st_data.get('habit_strength', {}).items()}
            cooc_data = st_data.get('cooccurrence', {})
            st.cooccurrence = {}
            for k, v in cooc_data.items():
                parts = k.split('_')
                if len(parts) == 2:
                    st.cooccurrence[(int(parts[0]), int(parts[1]))] = v
            st.n_learn_events = st_data.get('n_learn_events', 0)
            st._habit_ema = float(st_data.get('_habit_ema', 0.0))
            if verbose:
                print(f"  Striatum: {st.get_state()['n_states_known']} states, "
                      f"habit={st.global_habit_strength:.3f}")

    # Consolidation
    if 'consolidation' in data:
        agent.consolidation_counter = data['consolidation'].get('counter', 0)
        agent.dialogue_since_consolidation = data['consolidation'].get(
            'dialogue_since', 0)

    # Total steps/turns from meta
    total_steps = data.get('total_steps', 0)
    total_turns = data.get('total_turns', 0)
    n_sessions = data.get('n_sessions', 1)

    if verbose:
        print(f"  Meta: {total_turns} turns, {total_steps} steps, "
              f"session #{n_sessions}")


def auto_save(agent, n_turns: int, n_sessions: int = 1,
              save_every: int = 10, verbose: bool = False) -> Optional[str]:
    """如果达到保存间隔, 自动保存.

    Args:
        agent: Agent 实例
        n_turns: 当前对话轮数
        n_sessions: 累计会话数
        save_every: 每 N 轮自动保存
        verbose: 是否打印

    Returns:
        保存路径, 或 None (未达间隔)
    """
    if n_turns > 0 and n_turns % save_every == 0:
        path = _make_path()
        save_agent(agent, path, n_sessions=n_sessions,
                   extra={'total_turns': n_turns})
        if verbose:
            print(f"  [auto-saved] {path}")
        return path
    return None
