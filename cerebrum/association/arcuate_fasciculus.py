"""
arcuate_fasciculus.py — 弓状束 (Arcuate Fasciculus) [v5.6 新增]

对应脑区: 弓状束白质通路 (连接 Wernicke区 ↔ Broca区)
所属层级: 大脑 → 联合皮层 → 弓状束 (白质通路)

脑区标记: arcuate fasciculus (BA22 ↔ BA44/45 之间的长距离纤维束)

功能职责 (参考: 语言与大脑 §2.1, §3.4):
  - 腹侧通路 (Ventral AF): Wernicke→Broca — 理解到的内容传递给言语产出
  - 背侧通路 (Dorsal AF): Broca→Wernicke — 运动指令副本 (efference copy) → 自我监控
  - 复述 (Repetition): 听到词 → Wernicke 理解 → AF 传递 → Broca 编码 → 运动皮层 → 发声

临床对应:
  传导性失语症 (Conduction Aphasia): 弓状束损伤
    - 理解正常, 言语流畅
    - 核心缺陷: 无法复述 (听到但无法说出来)
    - 音位性错语 (phonemic paraphasia) + 自我纠正 (conduite d'approche)
    - 为 "Broca区和Wernicke区之间的白质连接独立于两个语言中枢" 提供临床证据

解剖不对称性:
  - 人类左侧弓状束显著粗壮于右侧 (尤其颞叶段)
  - 部分个体右侧缺少完整的直接弓状束连接
  - 这种不对称性可能是人类语言功能的重要结构基础

关键机制:
  1. Hebb 桥接: comprehension_vec ↔ speech_seed_vec 通过共现学习
     — 不是手编路径, 而是从"听到+说出"的统计共现中涌现
  2. 掩码召回: 腹侧 mask[:64]=True → 从理解检索言语种子
               背侧 mask[:64]=True → 从言语计划检索预期听觉
  3. 预测编码: 背侧通路产生预期听觉, 与实际自听比较 → PE → F_language
  4. 在线学习: 每次对话回合强化 AF 连接 (fire together, wire together)

在 NotMe 中的集成:
  - comprehend() 输出 → AF.repeat() → 种子词 (直接复述通路)
  - broca.speak_from_state() 前 → AF 提供 comprehension-driven 种子词偏好
  - broca 产出言语后 → AF.efference_copy() → 预期听觉 → 自我监控
  - AF 预测误差 → F_language (分解为语义PE + 语音PE)

参考:
  - Geschwind, N. (1970). The organization of language and the brain.
  - Catani, M., Jones, D. K., & ffytche, D. H. (2005). Perisylvian language
    networks of the human brain. Annals of Neurology.
  - Hickok, G., & Poeppel, D. (2007). The cortical organization of speech
    processing. Nature Reviews Neuroscience. (双流模型)
"""

import numpy as np
import os
from typing import Optional

from cns.data_types import D, Theta, Cluster
from cerebrum.limbic_system.hippocampus import ClusterNetwork, _masked_cosine


class ArcuateFasciculus:
    """弓状束 — Broca区与Wernicke区之间的Hebb桥接通路.

    双网络架构 (对应双流模型):
      ventral_net: Wernicke → Broca (理解→言语产出)
        - centroid[:64]  = comprehension_vec (理解了什么)
        - centroid[64:128] = speech_seed_vec   (该说什么词)
        - centroid[128:192] = emotional_context (情感上下文)

      dorsal_net: Broca → Wernicke (言语计划→预期听觉, 运动指令副本)
        - centroid[:64]   = speech_plan_vec    (计划说什么)
        - centroid[64:128] = expected_auditory (预期听到什么)
        - centroid[128:192] = motor_copy        (运动指令副本)

    两个网络都是从共现中学习的 Hebb 网络 —
    不是手编的符号规则, 而是统计学习的结果.
    """

    def __init__(self, cache_dir: str = None):
        """初始化弓状束.

        Args:
            cache_dir: 缓存目录 (用于预训练模型的加载/保存).
                      默认在 cerebrum/association/.cache/ 下.
        """
        # ---- 腹侧通路: Wernicke → Broca (理解→言语) ----
        ventral_theta = Theta()
        ventral_theta.cluster_threshold = 0.12   # 低阈值: 广泛关联
        ventral_theta.learn_rate_l0 = 0.06
        self.ventral_net = ClusterNetwork(ventral_theta)
        self._ventral_n_associations: int = 0

        # ---- 背侧通路: Broca → Wernicke (言语计划→预期听觉) ----
        dorsal_theta = Theta()
        dorsal_theta.cluster_threshold = 0.15
        dorsal_theta.learn_rate_l0 = 0.06
        self.dorsal_net = ClusterNetwork(dorsal_theta)
        self._dorsal_n_associations: int = 0

        # ---- 预测误差追踪 ----
        self.ventral_pe_ema: float = 0.0    # 腹侧通路预测误差 EMA
        self.dorsal_pe_ema: float = 0.0     # 背侧通路预测误差 EMA
        self.n_calls_ventral: int = 0
        self.n_calls_dorsal: int = 0

        # ---- 缓存 ----
        if cache_dir is None:
            base = os.path.dirname(__file__)
            cache_dir = os.path.join(base, '.cache')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # ---- 解剖不对称性: 左>右 (模拟人类优势半球) ----
        # 左侧通路权重 1.0, 右侧可设为较低值表示非优势半球
        self.left_dominance: float = 1.0

        # ---- 传导性: 通路效率 [0,1] ----
        # 模拟传导性失语症: conduction < 0.3 → 复述困难
        self.conduction: float = 1.0

    # ================================================================
    # 腹侧通路: Wernicke → Broca (理解→言语种子)
    # ================================================================

    def repeat(self, comprehension_vec: np.ndarray,
               emotional_context: np.ndarray = None,
               top_k: int = 5,
               temperature: float = 0.5) -> tuple[np.ndarray, float]:
        """腹侧通路召回: 从理解检索言语种子词.

        这是"复述"的神经基础: 听到的话 → Wernicke理解 → AF腹侧 → Broca种子词.
        传导性失语症中此通路受损 → 复述不能.

        Args:
            comprehension_vec: Wernicke区理解向量 (64,)
            emotional_context: 情感上下文 (8,) 可选
            top_k: 返回 top-k 候选
            temperature: softmax 温度

        Returns:
            (speech_seed_vec, confidence)
            - speech_seed_vec: 言语种子词向量 (64,), 可传入 Broca
            - confidence: Hebb 召回置信度 [0,1]
        """
        self.n_calls_ventral += 1

        if self.ventral_net.n_clusters == 0:
            # 空网络 → 回退: 理解向量本身就是种子
            return comprehension_vec[:64].copy().astype(np.float32), 0.0

        # 构建查询: 仅理解通道有效
        q = np.zeros(D, dtype=np.float32)
        q[:64] = comprehension_vec[:64].astype(np.float32)

        # 情感上下文 (如果提供)
        if emotional_context is not None:
            q[128:128 + min(8, len(emotional_context))] = \
                emotional_context[:8].astype(np.float32)

        h = self.ventral_net.hash_features(q)
        mask = np.zeros(D, dtype=bool)
        mask[:64] = True  # 只在理解空间比较

        # ---- Hebb 检索: hash → bucket → 竞争 ----
        hash_key = self.ventral_net._hash_to_bucket(h)
        bucket = self.ventral_net.buckets.get(hash_key, [])

        if not bucket:
            bucket = self.ventral_net.clusters[:min(500,
                len(self.ventral_net.clusters))]

        scored = []
        for c in bucket:
            sim = _masked_cosine(h, c.centroid, mask)
            scored.append((c, float(sim)))

        if not scored:
            return comprehension_vec[:64].copy().astype(np.float32), 0.0

        scored.sort(key=lambda x: x[1], reverse=True)
        k_effective = min(top_k, len(scored))
        top = scored[:k_effective]

        # 温度 softmax 采样
        top_sims = np.array([s for _, s in top], dtype=np.float32)
        eff_temp = max(temperature, 0.05)
        probs = np.exp((top_sims - top_sims.max()) / eff_temp)
        probs /= probs.sum()

        chosen_idx = int(np.random.choice(len(top), p=probs))
        chosen_cluster, confidence = top[chosen_idx]

        # centroid[64:128] = 关联的言语种子词向量
        speech_seed = chosen_cluster.centroid[64:128].copy().astype(np.float32)

        # 更新腹侧 PE
        best_sim = top[0][1]
        self.ventral_pe_ema += 0.1 * ((1.0 - best_sim) - self.ventral_pe_ema)

        return speech_seed, confidence

    def repeat_topk(self, comprehension_vec: np.ndarray,
                    emotional_context: np.ndarray = None,
                    k: int = 8) -> list[tuple[np.ndarray, float]]:
        """腹侧通路: top-k 言语种子候选 (不采样).

        Args:
            comprehension_vec: 理解向量 (64,)
            emotional_context: 情感上下文 (8,) 可选
            k: 返回的候选数

        Returns:
            [(speech_seed_vec, similarity), ...] 按相似度降序排列
        """
        if self.ventral_net.n_clusters == 0:
            return [(comprehension_vec[:64].copy().astype(np.float32), 0.0)]

        q = np.zeros(D, dtype=np.float32)
        q[:64] = comprehension_vec[:64].astype(np.float32)
        if emotional_context is not None:
            q[128:128 + min(8, len(emotional_context))] = \
                emotional_context[:8].astype(np.float32)

        h = self.ventral_net.hash_features(q)
        mask = np.zeros(D, dtype=bool)
        mask[:64] = True

        hash_key = self.ventral_net._hash_to_bucket(h)
        bucket = self.ventral_net.buckets.get(hash_key, [])
        if not bucket:
            bucket = self.ventral_net.clusters[:min(500,
                len(self.ventral_net.clusters))]

        scored = []
        seen = set()
        for c in bucket:
            sim = _masked_cosine(h, c.centroid, mask)
            seed_vec = c.centroid[64:128].copy()
            seed_key = tuple(seed_vec[:8].astype(np.float32).tobytes())
            if seed_key not in seen:
                seen.add(seed_key)
                scored.append((seed_vec, float(sim)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ================================================================
    # 背侧通路: Broca → Wernicke (言语计划→预期听觉, 运动指令副本)
    # ================================================================

    def efference_copy(self, speech_plan: np.ndarray,
                       motor_plan: np.ndarray = None
                       ) -> tuple[np.ndarray, float]:
        """背侧通路: 言语计划 → 预期听觉 (运动指令副本).

        大脑在发出运动指令的同时发送一份"副本"到感觉皮层,
        用于预测行动的感觉后果. 这是自我监控的基础 —
        "我即将说的话听起来应该是什么样?"

        Args:
            speech_plan: Broca产出的言语计划向量 (64,)
            motor_plan: 运动皮层计划 (16,) 可选, 用于更精确的预测

        Returns:
            (expected_auditory, confidence)
            - expected_auditory: 预期听觉向量 (64,)
            - confidence: 预测置信度 [0,1]
        """
        self.n_calls_dorsal += 1

        if self.dorsal_net.n_clusters == 0:
            # 空网络 → 回退: 言语计划 ≈ 预期听觉 (零延迟反馈)
            return speech_plan[:64].copy().astype(np.float32), 0.0

        q = np.zeros(D, dtype=np.float32)
        q[:64] = speech_plan[:64].astype(np.float32)
        if motor_plan is not None:
            q[128:128 + min(16, len(motor_plan))] = \
                motor_plan[:16].astype(np.float32)

        h = self.dorsal_net.hash_features(q)
        mask = np.zeros(D, dtype=bool)
        mask[:64] = True  # 在言语计划空间比较

        hash_key = self.dorsal_net._hash_to_bucket(h)
        bucket = self.dorsal_net.buckets.get(hash_key, [])

        if not bucket:
            bucket = self.dorsal_net.clusters[:min(500,
                len(self.dorsal_net.clusters))]

        best_sim = -1.0
        best_c = None
        for c in bucket:
            sim = _masked_cosine(h, c.centroid, mask)
            if sim > best_sim:
                best_sim = sim
                best_c = c

        if best_c is not None and best_sim >= self.dorsal_net.theta.cluster_threshold:
            expected_auditory = best_c.centroid[64:128].copy().astype(np.float32)
            confidence = float(best_sim)
        else:
            # 回退: 言语计划本身
            expected_auditory = speech_plan[:64].copy().astype(np.float32)
            confidence = 0.0

        # 更新背侧 PE
        self.dorsal_pe_ema += 0.1 * ((1.0 - confidence) - self.dorsal_pe_ema)

        return expected_auditory, confidence

    # ================================================================
    # Hebb 学习
    # ================================================================

    def learn_ventral(self, comprehension_vec: np.ndarray,
                      speech_seed_vec: np.ndarray,
                      emotional_context: np.ndarray = None,
                      weight: float = 1.0):
        """腹侧通路 Hebb 学习: 强化 (理解, 言语种子) 关联.

        每次 Agent 听到话并成功回应后调用,
        建立 "理解到这个 → 该说这些词" 的映射.

        Args:
            comprehension_vec: 理解向量 (64,)
            speech_seed_vec: 言语种子向量 (64,) — Agent回应中的关键词
            emotional_context: 情感上下文 (8,) 可选
            weight: 学习权重 (显著回合 > 1.0, 普通回合 = 1.0)
        """
        # 构建 Hebb 模式
        pattern = np.zeros(D, dtype=np.float32)
        pattern[:64] = comprehension_vec[:64].astype(np.float32)
        pattern[64:128] = speech_seed_vec[:64].astype(np.float32)
        if emotional_context is not None:
            pattern[128:128 + min(8, len(emotional_context))] = \
                emotional_context[:8].astype(np.float32)

        # 临时提升学习率 (权重调制)
        orig_lr = self.ventral_net.theta.learn_rate_l0
        self.ventral_net.theta.learn_rate_l0 = min(0.30,
            orig_lr * weight)
        self.ventral_net.learn(pattern)
        self.ventral_net.theta.learn_rate_l0 = orig_lr

        self._ventral_n_associations += 1

    def learn_dorsal(self, speech_plan: np.ndarray,
                     actual_auditory: np.ndarray,
                     motor_plan: np.ndarray = None,
                     weight: float = 1.0):
        """背侧通路 Hebb 学习: 强化 (言语计划, 实际听到) 关联.

        每次 Agent 说话后, 用实际自听反馈更新背侧通路,
        使未来的 efference_copy() 预测更准确.

        Args:
            speech_plan: 言语计划向量 (64,)
            actual_auditory: 实际自听反馈 (64,)
            motor_plan: 运动皮层计划 (16,) 可选
            weight: 学习权重
        """
        pattern = np.zeros(D, dtype=np.float32)
        pattern[:64] = speech_plan[:64].astype(np.float32)
        pattern[64:128] = actual_auditory[:64].astype(np.float32)
        if motor_plan is not None:
            pattern[128:128 + min(16, len(motor_plan))] = \
                motor_plan[:16].astype(np.float32)

        orig_lr = self.dorsal_net.theta.learn_rate_l0
        self.dorsal_net.theta.learn_rate_l0 = min(0.30,
            orig_lr * weight)
        self.dorsal_net.learn(pattern)
        self.dorsal_net.theta.learn_rate_l0 = orig_lr

        self._dorsal_n_associations += 1

    # ================================================================
    # 复述 (Repetition) — 完整传导通路
    # ================================================================

    def conduction_repeat(self, heard_comprehension: np.ndarray,
                         broca=None, max_words: int = 6
                         ) -> tuple[list[str], float]:
        """完整传导复述通路: 听到 → 理解 → AF → Broca → 复述.

        这是测试弓状束完整性的经典诊断任务.
        传导性失语症的核心缺陷就是无法完成这个任务.

        流程:
          1. Wernicke comprehend → comprehension_vec
          2. AF ventral repeat() → speech_seed_vec
          3. AF conduction 调制 → Broca speak_from_state()

        Args:
            heard_comprehension: Wernicke理解后的向量 (64,)
            broca: Broca实例 (可选, 用于完整生成)
            max_words: 最大复述词数

        Returns:
            (words, conduction_quality)
            - words: 词列表
            - conduction_quality: 传导质量 [0,1] (受 conduction 参数影响)
        """
        # Step 1: AF腹侧 → 种子词
        seed_vec, af_confidence = self.repeat(
            heard_comprehension, temperature=0.3)  # 低温度 = 忠实复述

        # 传导效率调制
        conduction_quality = af_confidence * self.conduction

        # Step 2: 如果 Broca 可用, 用种子词生成简短回应
        if broca is not None and conduction_quality > 0.1:
            try:
                # 用种子词向量找到词汇表中最接近的词
                seed_word = broca._vec_to_word(seed_vec)
                if seed_word:
                    # 简短复述: 只用种子词 + 少量 Hebb 链
                    words = [seed_word]
                    prev_vec = seed_vec
                    for _ in range(max_words - 1):
                        nv = broca.next_word(prev_vec)
                        if nv is None:
                            break
                        nw = broca._vec_to_word(nv)
                        if nw in words[-2:]:  # 防重复
                            break
                        words.append(nw)
                        prev_vec = nv
                    return words, conduction_quality
            except Exception:
                pass

        # 回退: 空网络时返回空
        return [], conduction_quality

    # ================================================================
    # 预测误差计算 (汇入 F_language)
    # ================================================================

    def compute_af_pe(self, expected_auditory: np.ndarray,
                      actual_auditory: np.ndarray) -> dict:
        """计算弓状束预测误差 (自听监控PE).

        背侧通路预测了自己说话的声音 → 与实际自听比较 → PE.
        高PE → 言语产出与预期不符 → 可能提示言语错误或外部干扰.

        Args:
            expected_auditory: efference_copy() 预测的听觉 (64,)
            actual_auditory: 实际自听的听觉反馈 (64,)

        Returns:
            {'af_phonological_pe': float,   # 语音层面PE
             'af_semantic_pe': float,       # 语义层面PE
             'af_total_pe': float,          # 总PE
             'self_monitoring_alert': bool} # 需要自我纠正?
        """
        # 语音PE: 前32维 (频谱相关)
        phono_pe = float(1.0 - np.dot(
            expected_auditory[:32], actual_auditory[:32]) / (
            np.linalg.norm(expected_auditory[:32])
            * np.linalg.norm(actual_auditory[:32]) + 1e-8))

        # 语义PE: 整体余弦距离
        semantic_pe = float(1.0 - np.dot(
            expected_auditory, actual_auditory) / (
            np.linalg.norm(expected_auditory)
            * np.linalg.norm(actual_auditory) + 1e-8))

        # 总PE (精度加权)
        total_pe = 0.5 * phono_pe + 0.5 * semantic_pe

        # 自我监控: PE > 阈值 → 可能需要自我纠正
        alert = total_pe > 0.5

        return {
            'af_phonological_pe': phono_pe,
            'af_semantic_pe': semantic_pe,
            'af_total_pe': total_pe,
            'self_monitoring_alert': alert,
        }

    # ================================================================
    # 预训练: 从语料暖机 AF 连接
    # ================================================================

    def pretrain_from_corpus(self, broca, text_env=None,
                            n_samples: int = 5000, verbose: bool = True):
        """从语料预训练弓状束 Hebb 连接.

        对语料中每个句子:
          1. 编码句子 → 模拟"理解向量" (comprehension)
          2. 提取前 2-3 个实词 → 模拟"言语种子" (speech_seed)
          3. Hebb学习: 建立 (理解, 种子) 关联

        这模拟了发育过程中 "听别人说话 → 自己学说话" 的统计学习.

        Args:
            broca: Broca 实例 (需要 word_list/word_vecs 用于词汇映射)
            text_env: TextEnvironment (用于编码句子)
            n_samples: 采样句子数
            verbose: 是否打印进度
        """
        if broca is None:
            return

        import jieba

        sentences = broca.sentences
        if len(sentences) == 0:
            return

        # 均匀采样
        n_available = len(sentences)
        n_samples = min(n_samples, n_available)
        if n_samples < n_available:
            stride = max(1, n_available // n_samples)
            sample_indices = list(range(0, n_available, stride))[:n_samples]
        else:
            sample_indices = list(range(n_available))

        if verbose:
            print(f"  AF pretrain: {n_samples} sentences → "
                  f"ventral+dorsal Hebb associations...")

        n_ventral = 0
        n_dorsal = 0

        for idx in sample_indices:
            sent = sentences[idx]
            cleaned = broca._clean_sentence(sent)

            # ---- 腹侧通路: (理解向量, 种子词) ----
            words = [w for w in jieba.lcut(cleaned) if len(w.strip()) >= 1]
            if len(words) < 3:
                continue

            # 编码句子作为理解向量
            if text_env is not None:
                try:
                    comp_vec = text_env.encode_text(cleaned)
                except Exception:
                    continue
            else:
                # 用词向量的加权平均作为理解代理
                word_vecs = []
                for w in words:
                    wv = broca._word_to_vec(w)
                    if wv is not None:
                        word_vecs.append(wv[:64])
                if not word_vecs:
                    continue
                comp_vec = np.mean(word_vecs, axis=0)
                comp_vec = comp_vec / (np.linalg.norm(comp_vec) + 1e-8)

            # 提取前几个词作为言语种子
            seed_words = words[:min(3, len(words))]
            seed_vecs = []
            for sw in seed_words:
                wv = broca._word_to_vec(sw)
                if wv is not None:
                    seed_vecs.append(wv[:64])

            if not seed_vecs:
                continue
            seed_vec = np.mean(seed_vecs, axis=0)
            seed_vec = seed_vec / (np.linalg.norm(seed_vec) + 1e-8)

            # 学习腹侧
            self.learn_ventral(comp_vec, seed_vec, weight=0.5)
            n_ventral += 1

            # ---- 背侧通路: (言语种子, 预期听觉=种子词频谱) ----
            # 用种子词的音频频谱作为预期听觉
            if len(seed_vecs) >= 1:
                expected_aud = seed_vec  # 简化: 种子向量 ≈ 预期听觉
                self.learn_dorsal(seed_vec, expected_aud, weight=0.3)
                n_dorsal += 1

        if verbose:
            print(f"  AF pretrain done: {n_ventral} ventral + "
                  f"{n_dorsal} dorsal associations "
                  f"({self.ventral_net.n_clusters} + "
                  f"{self.dorsal_net.n_clusters} clusters)")

    # ================================================================
    # 持久化
    # ================================================================

    def save(self, filepath: str):
        """保存 AF 状态到 .npz 文件 (仅保存网络数据)."""
        data = {
            'ventral_centroids': np.stack(
                [c.centroid for c in self.ventral_net.clusters]),
            'ventral_activations': np.array(
                [c.activation for c in self.ventral_net.clusters],
                dtype=np.float32),
            'dorsal_centroids': np.stack(
                [c.centroid for c in self.dorsal_net.clusters]),
            'dorsal_activations': np.array(
                [c.activation for c in self.dorsal_net.clusters],
                dtype=np.float32),
            'ventral_pe_ema': self.ventral_pe_ema,
            'dorsal_pe_ema': self.dorsal_pe_ema,
        }
        np.savez_compressed(filepath, **data)

    def load(self, filepath: str) -> bool:
        """从 .npz 文件加载 AF 状态."""
        if not os.path.exists(filepath):
            return False

        try:
            data = np.load(filepath, allow_pickle=True)

            # 重建腹侧网络
            centroids = data['ventral_centroids']
            activations = data['ventral_activations']
            self.ventral_net.clusters = []
            self.ventral_net.buckets = {}
            for i in range(len(centroids)):
                c = Cluster(centroid=centroids[i].astype(np.float32))
                c.activation = float(activations[i])
                c.count = 1
                self.ventral_net.clusters.append(c)
                hk = self.ventral_net._hash_to_bucket(centroids[i])
                self.ventral_net.buckets.setdefault(hk, []).append(c)
            self._ventral_n_associations = len(centroids)

            # 重建背侧网络
            centroids = data['dorsal_centroids']
            activations = data['dorsal_activations']
            self.dorsal_net.clusters = []
            self.dorsal_net.buckets = {}
            for i in range(len(centroids)):
                c = Cluster(centroid=centroids[i].astype(np.float32))
                c.activation = float(activations[i])
                c.count = 1
                self.dorsal_net.clusters.append(c)
                hk = self.dorsal_net._hash_to_bucket(centroids[i])
                self.dorsal_net.buckets.setdefault(hk, []).append(c)
            self._dorsal_n_associations = len(centroids)

            self.ventral_pe_ema = float(data.get('ventral_pe_ema', 0.0))
            self.dorsal_pe_ema = float(data.get('dorsal_pe_ema', 0.0))

            return True
        except Exception:
            return False

    # ================================================================
    # 诊断
    # ================================================================

    @property
    def n_ventral_clusters(self) -> int:
        return self.ventral_net.n_clusters

    @property
    def n_dorsal_clusters(self) -> int:
        return self.dorsal_net.n_clusters

    def get_state(self) -> dict:
        """返回当前 AF 状态 (供 dashboard 使用)."""
        return {
            'ventral_n': self.n_ventral_clusters,
            'dorsal_n': self.n_dorsal_clusters,
            'ventral_pe_ema': float(self.ventral_pe_ema),
            'dorsal_pe_ema': float(self.dorsal_pe_ema),
            'conduction': float(self.conduction),
            'n_calls_ventral': self.n_calls_ventral,
            'n_calls_dorsal': self.n_calls_dorsal,
        }

    def __repr__(self) -> str:
        return (f"ArcuateFasciculus(ventral={self.n_ventral_clusters}, "
                f"dorsal={self.n_dorsal_clusters}, "
                f"conduction={self.conduction:.2f})")
