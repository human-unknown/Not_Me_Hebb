"""
main_dialogue.py —— Stage 6: 人机对话闭环 (v2: 社会情感回路)
自由能原理智能体

持续流模式: Agent 不等人类，10 FPS 持续循环。
人类输入 → 语义编码 s[0:64] + 情感编码 s[80:88] → Agent 被对话改变。
"""
import sys, time, queue, numpy as np
from data_types import D, BodyVector, Action, ACTION_DIRECTIONS
from agent import Agent
from text_interface import TextEnvironment
from word_speech import get_speaker
from broca import Broca
from stdin_reader import StdinReader
from sentiment import analyze_sentiment, sentiment_to_social_signal, get_emotional_lexicon
from layer1_free_energy import SocialContext


def _safe_str(s: str) -> str:
    """GBK-safe: replace non-GBK chars for Windows console"""
    try:
        s.encode('gbk'); return s
    except UnicodeEncodeError:
        return s.encode('gbk', errors='replace').decode('gbk')


def run_dialogue():
    rng = np.random.default_rng(42)
    agent = Agent(rng=rng, agent_id=0, n_agents=1)
    agent.body = BodyVector(mode='text')
    agent.theta.cluster_threshold = 0.55
    agent.theta.w_social = 2.0  # 社会域权重加倍 — 对话中社会信号很重要
    agent.record_action_consequence = lambda s: None
    ACTION_DIRECTIONS[3] = [0.0, 0.0]
    ACTION_DIRECTIONS[4] = [0.0, 0.0]

    text_env = TextEnvironment()
    speaker = get_speaker()
    broca = Broca(text_env=text_env)  # 共享 PCA 空间
    reader = StdinReader(); reader.start()

    # ---- L0 预热: 从语料均匀采样 (v2: 无手选, 无人类偏好) ----
    # 从语料中等距采样 15 句 — 语义多样, 无人类偏好注入
    corpus_sents = text_env.chunks
    n_corpus = len(corpus_sents)
    warmup_indices = [int(i * n_corpus / 15) for i in range(15)]
    warmup_sents = [corpus_sents[idx] for idx in warmup_indices]
    agent.warmup_l0(warmup_sents, text_env, n=15)
    print(f"  L0 warmup: {agent.net.n_clusters} initial clusters "
          f"(sampled from corpus)")

    # ---- 社会情感上下文 ----
    social_ctx = SocialContext(tau=15.0)  # tau 越小，情感越容易被改变

    last_human_vec = np.zeros(64)
    last_self_semantic = np.zeros(64, dtype=np.float32)   # 上一轮自己说的话 (语义)
    last_self_sentiment = np.zeros(8, dtype=np.float32)   # 上一轮自己说的话 (情感)
    comprehension_vec = np.zeros(64, dtype=np.float32)     # Wernicke 理解向量
    emo_lexicon = get_emotional_lexicon()                  # Hebb 情感词汇网络
    t = 0; expr_cooldown = 0; inner_cooldown = 0

    print("=" * 60)
    print("  自由能原理智能体 — Stage 6: 人机对话")
    print("  v5.1: Wernicke 理解回路 + Hebb 词序链生成 + 响应评估")
    print("=" * 60)
    print("  你说的话会改变 Agent 的情感状态")
    print("  温暖的话 → F_social ↓ → valence ↑ → Agent 感到好")
    print("  攻击的话 → F_social ↑ → valence ↓ → Agent 感到不好")
    print("  Agent 先理解，再回忆，然后用自己的话说出来")
    print("  沉默时 Agent 会自己「想」→ 内部言语链")
    print("-" * 60)
    print("  输入文字后回车 · 输入 'exit' 退出 · Ctrl+C 退出")
    print("-" * 60)

    try:
        while True:
            # ---- 1. 非阻塞读人类输入 ----
            human_text = None
            try:
                human_text = reader.queue.get_nowait()
            except queue.Empty:
                pass

            if human_text and human_text.strip().lower() == 'exit':
                print("\n[System]: 退出")
                break

            # ---- 2. 构建感觉向量 ----
            s = np.zeros(D)

            if human_text and human_text.strip():
                # 语义编码
                try:
                    s[0:64] = text_env.encode_text(human_text.strip())
                    last_human_vec = s[0:64].copy()
                except Exception:
                    s[0:64] = text_env.embeddings[t % len(text_env.embeddings)]

                # 情感编码 → s[80:88] (v2: Hebb 学习, 无手标词典)
                sentiment = analyze_sentiment(human_text.strip(),
                                             lexicon=emo_lexicon)
                social_signal = sentiment_to_social_signal(sentiment)
                s[80:88] = social_signal

                # 更新社会上下文
                social_ctx.update(sentiment['valence'], sentiment['arousal'])

                # ---- Wernicke 区: 理解人类输入 ----
                human_sent = social_signal[:8].astype(np.float32)
                comprehension_vec, understanding = agent.comprehend(
                    last_human_vec, human_sent)

                # 显示
                valence_icon = ':)' if sentiment['valence'] > 0.3 else (
                    ':(' if sentiment['valence'] < -0.3 else ':|')
                print(f"\n[You] {valence_icon}: {human_text.strip()}")
                print(f"       sentiment: v={sentiment['valence']:+.2f} "
                      f"a={sentiment['arousal']:.2f} "
                      f"intensity={sentiment['intensity']:.2f} "
                      f"learned={sentiment.get('learned', False)} "
                      f"trust={social_ctx.trust_level:.2f} "
                      f"understood={understanding['n_triggered_memories']}mem "
                      f"emo_words={emo_lexicon.get_stats()['n_words']}")

            else:
                # 沉默: 语料背景 + 10% 上一句人类残差
                s[0:64] = (text_env.embeddings[t % len(text_env.embeddings)] * 0.9
                          + last_human_vec * 0.1)
                # 沉默时 s[80:88] 保持零（无人类社会信号）
                social_ctx.tick()

            # ---- 2b. 自听回路: 把自己上一轮说的话填入听觉通道 ----
            # s[128:192] = 听觉通道 (自己说的话的语义)
            # s[96:104]  = 输出反馈 (自己说的话的情感)
            # 这形成了自感知闭环: 说话 → 听到 → 影响下一步状态
            s[128:192] = last_self_semantic
            s[96:104] = last_self_sentiment

            # ---- 3. Agent 处理 ----
            F_before = agent.F_body_history[-1] if agent.F_body_history else 0.0
            action = agent.step(s, t, social_ctx=social_ctx)
            F_after = agent.F_body_history[-1] if agent.F_body_history else 0.0

            # ---- Hebb 情感学习: F_body 变化 → 词汇情感关联 ----
            if human_text and human_text.strip():
                delta_F = F_before - F_after  # 正值=变好, 负值=变差
                import jieba
                input_words = [w for w in jieba.lcut(human_text.strip())
                              if len(w.strip()) >= 1]
                arousal_now = agent.arousal_history[-1] if agent.arousal_history else 0.5
                emo_lexicon.learn_from_feedback(input_words, delta_F, arousal_now)

            # ---- 3b. 表达耦合: A₃ 与人类输入绑定 ----
            # 沉默时 Agent 不应该对外说话——把 A₃ 重定向为内部言语
            # G 偏置 (layer2) 已大幅抑制沉默时的 A₃，这里是安全网
            is_silence = not (human_text and human_text.strip())
            if action.index == 3 and is_silence:
                # 强制触发内部言语: 覆盖行动为 REST，清空 cooldown
                inner_cooldown = 0
                action = Action(index=4, expected_F=action.expected_F,
                                expected_G=action.expected_G, confidence=action.confidence)

            # ---- 4. A₃ 表达: 信念锚定整句检索 ----
            # 不从 raw sensory 做 recall，而是用 Agent 当前的信念状态
            # （最激活的集群 = Hebb 竞争胜出者 = "我在想什么"）
            # 只在有人类输入时触发 (沉默时 A₃ 已被上面重定向)
            if action.index == 3 and expr_cooldown <= 0:
                if agent.net.n_clusters > 0:
                    # 信念状态: 激活度最高的集群 (已受 body/历史/竞争调制)
                    top = max(agent.net.clusters, key=lambda c: c.activation)
                    if top.activation > 0.01:
                        # ---- 构建查询: 精度加权 (v3: 无硬编码权重) ----
                        # 每个通道的精度 = 信号可靠性代理:
                        #   comprehension: 触发记忆数 → 理解深度
                        #   belief:        集群激活度 → 信念强度
                        #   ctx:           对话轮数 → 上下文可用性
                        #   self_anchor:   自我体验数 → 人格锚定度
                        sv = agent.self_valence_ema
                        belief_sem = top.centroid[:64].copy().astype(np.float32)
                        ctx = agent.dialogue_ctx.get_context_vector()
                        self_anchor = agent.self_model.get_self_anchor()

                        n_triggered = understanding.get('n_triggered_memories', 0)
                        comp_precision = 0.15 + min(1.0, n_triggered / 5.0)
                        belief_precision = 0.15 + min(1.0, top.activation * 2.0)
                        ctx_precision = 0.15 + min(1.0, agent.dialogue_ctx.n_turns() / 5.0)
                        n_self = agent.self_model.n_experiences
                        self_precision = 0.05 + min(0.20, n_self * 0.005)

                        total_p = (comp_precision + belief_precision
                                   + ctx_precision + self_precision)
                        query = (comprehension_vec * (comp_precision / total_p)
                                 + belief_sem * (belief_precision / total_p)
                                 + ctx * (ctx_precision / total_p)
                                 + self_anchor * (self_precision / total_p))
                        query = query.astype(np.float32)

                        # Valence 调制温度: 极端情绪 → 更确定 (低温)
                        v = agent.valence_history[-1] if agent.valence_history else 0
                        a = agent.arousal_history[-1] if agent.arousal_history else 0
                        sa = agent.self_arousal_ema
                        temp_base = (0.5 + abs(v) * 0.8 + sa * 0.2)

                        # Body 需求调制
                        body = agent.body
                        social_need = max(0.0, body.setpoints[0] - body.b[0])
                        novelty_need = max(0.0, 0.5 - body.b[3])
                        temp = temp_base * (1.0 + social_need * 0.6 + novelty_need * 0.4)

                        # ---- 生成回应 ----
                        words, audio = broca.speak_from_state(
                            belief_vec=top.centroid,
                            body_state=agent.body,
                            query_vec=query,
                            valence=v,
                            arousal=a,
                            temperature=temp,
                            max_words=20,
                        )

                        response = "".join(words) if words else ""

                        # ---- ACC+OFC: 响应评估 ----
                        eval_score = None
                        if response:
                            try:
                                resp_vec = text_env.encode_text(response).astype(np.float32)
                                eval_result = agent.evaluate_own_response(resp_vec)
                                eval_score = eval_result['overall_score']
                                # 低于阈值 → 重试一次 (更高温度)
                                if not eval_result['acceptable'] and len(words) < 8:
                                    words2, audio2 = broca.speak_from_state(
                                        belief_vec=top.centroid,
                                        body_state=agent.body,
                                        query_vec=query,
                                        valence=v,
                                        arousal=a,
                                        temperature=temp * 1.5,
                                        max_words=20,
                                    )
                                    response2 = "".join(words2) if words2 else ""
                                    if response2 and len("".join(words2)) > len(response):
                                        words, audio, response = words2, audio2, response2
                            except Exception:
                                pass

                        if audio is not None and len(audio) > 0:
                            import soundfile as sf
                            import os as _os
                            out = f'audio_output/dialogue_t{t:04d}.wav'
                            _os.makedirs('audio_output', exist_ok=True)
                            sf.write(out, audio, 22050)
                            try:
                                import sounddevice as sd
                                sd.play(audio, 22050)
                                sd.wait()
                            except Exception:
                                pass

                        if not response:
                            response = "(silence)"

                        v_now = agent.valence_history[-1] if agent.valence_history else 0
                        feel = ':)' if v_now > 0.1 else (':(' if v_now < -0.1 else ':|')
                        a_now = agent.arousal_history[-1] if agent.arousal_history else 0
                        b0 = agent.body.b[0]
                        eval_str = f' score={eval_score:.2f}' if eval_score is not None else ''
                        print(f'  [Agent] {feel}: {_safe_str(response)}')
                        print(f'         V={v_now:+.2f} A={a_now:.2f} '
                              f'b0={b0:.2f}{eval_str} temp={temp:.2f}')
                        expr_cooldown = 8

                        # ---- 反重复: 最近说过的话不重复 ----
                        if agent.dialogue_ctx.is_repeating(response):
                            words2, audio2 = broca.speak_from_state(
                                belief_vec=top.centroid,
                                body_state=agent.body,
                                query_vec=query,
                                valence=v, arousal=a,
                                temperature=temp * 1.5,
                                max_words=20,
                            )
                            response2 = "".join(words2) if words2 else ""
                            if response2 and not agent.dialogue_ctx.is_repeating(response2):
                                words, audio, response = words2, audio2, response2

                        # ---- 存入对话记忆 ----
                        if response != "(silence)":
                            try:
                                resp_vec = text_env.encode_text(response).astype(np.float32)
                            except Exception:
                                resp_vec = np.zeros(64, dtype=np.float32)
                            agent.dialogue_ctx.add_turn(
                                human_text=human_text.strip() if human_text else "",
                                human_vec=last_human_vec,
                                human_sentiment=s[80:88].astype(np.float32),
                                agent_response=response,
                                agent_vec=resp_vec,
                                agent_valence=v_now,
                                agent_arousal=a_now,
                                comprehension_vec=comprehension_vec,
                            )

                            # ---- 自我模型: 存储"我是谁"的体验 ----
                            agent.self_model.add_experience(
                                response_vec=resp_vec,
                                valence=v_now,
                                arousal=a_now,
                                self_valence_ema=agent.self_valence_ema,
                                self_arousal_ema=agent.self_arousal_ema,
                                self_coherence=agent.self_coherence,
                                body_state=agent.body,
                                comprehension_vec=comprehension_vec,
                                dialogue_ctx_vec=ctx,
                            )

                            # ---- 微量巩固: 即时记忆强化 ----
                            mc_result = agent.micro_consolidate()

                            # ---- 检查是否需要完整睡眠巩固 ----
                            cb_result = agent.maybe_consolidate(broca=broca)
                            if cb_result and cb_result.get('phase') == 'full':
                                print(f'  [sleep] consolidated {cb_result["n_turns"]} turns → '
                                      f'{cb_result["n_l0_clusters"]} L0 clusters, '
                                      f'{cb_result["n_self_clusters"]} self clusters, '
                                      f'{cb_result["replays"]} replays, '
                                      f'{cb_result["cross_links"]} cross-links, '
                                      f'pruned {cb_result["pruned"]} weak clusters')

                        # ---- 自听编码: 把回应写入下一帧的听觉通道 ----
                        if response and response != "(silence)":
                            try:
                                last_self_semantic = text_env.encode_text(response).astype(np.float32)
                            except Exception:
                                last_self_semantic = np.zeros(64, dtype=np.float32)
                            self_sent = analyze_sentiment(response)
                            last_self_sentiment = sentiment_to_social_signal(self_sent)[:8].astype(np.float32)

            expr_cooldown -= 1
            inner_cooldown -= 1

            # ---- 4b. 内部言语: 沉默时信念→检索→自听闭环 (v2: 情感传染) ----
            # 只在无人类输入、未选择表达、cooldown 归零时触发
            # 不输出音频，编码后直接写入自听通道 → 下一帧影响自身
            # 人类说话后短暂抑制内心独白 → 让 Agent "听"人类说完
            if not is_silence:
                inner_cooldown = max(inner_cooldown, 3)
            if is_silence and action.index != 3 and inner_cooldown <= 0:
                if agent.net.n_clusters > 0:
                    top = max(agent.net.clusters, key=lambda c: c.activation)
                    if top.activation > 0.01:
                        # v3: 身体状态驱动信念权重 (FEP-derived, 不用硬编码)
                        # F_body 偏离大 → 更自我聚焦 (rumination)
                        # 自我效价 EMA 作为二次调制 (subtle: negative → +rumination)
                        body_dev = float(agent.body.compute_deviation())
                        belief_weight = 0.5 + min(0.4, body_dev * 0.7)  # [0.5, 0.9]
                        sv_mod = -agent.self_valence_ema * 0.05
                        belief_weight = float(np.clip(
                            belief_weight + sv_mod, 0.45, 0.92))
                        sensory_weight = 1.0 - belief_weight

                        belief_sem = top.centroid[:64].copy().astype(np.float32)
                        sensory_ctx = s[:64].astype(np.float32)
                        inner_query = (belief_sem * belief_weight
                                      + sensory_ctx * sensory_weight)

                        # 内部言语参数: 自听情感 EMA 增强调制
                        v = agent.valence_history[-1] if agent.valence_history else 0
                        a = agent.arousal_history[-1] if agent.arousal_history else 0
                        sa = agent.self_arousal_ema
                        body = agent.body
                        social_need = max(0.0, body.setpoints[0] - body.b[0])
                        # 自听唤醒 EMA 增加思维温度 (情绪化思维更发散)
                        inner_temp = ((0.6 + abs(v) * 0.4 + sa * 0.3)
                                     * (1.0 + a * 0.5 + social_need * 0.3))
                        inner_k = max(5, min(20, int(8 + a * 10 + sa * 6)))

                        words, _ = broca.speak_from_state(
                            belief_vec=top.centroid,
                            body_state=agent.body,
                            query_vec=inner_query,
                            valence=v,
                            arousal=a,
                            temperature=inner_temp,
                            max_words=16,
                        )
                        thought = "".join(words) if words else ""

                        if thought:
                            # 编码为自听信号 — 不输出音频，纯内部
                            try:
                                last_self_semantic = text_env.encode_text(thought).astype(np.float32)
                            except Exception:
                                last_self_semantic = np.zeros(64, dtype=np.float32)
                            self_sent = analyze_sentiment(thought)
                            last_self_sentiment = sentiment_to_social_signal(self_sent)[:8].astype(np.float32)

                            # 轻量显示 (含情感传染信息)
                            coh = agent.self_coherence
                            coh_mark = '=' if coh > 0.8 else ('~' if coh > 0.5 else '!')
                            print(f'  [thinks{coh_mark}]: {_safe_str(thought[:90])}')

                        # 重置内部言语 cooldown: 唤醒越高越想, 低一致性加快
                        inner_cooldown = max(3, int(14 * (1.0 - a * 0.5)
                                                    * (0.7 + 0.3 * agent.self_coherence)))

            # 表达后短暂抑制内部言语
            if action.index == 3:
                inner_cooldown = max(inner_cooldown, 4)

            # ---- 5. 定期状态显示 ----
            if t % 30 == 0:
                Fa = agent.F_history[-1] if agent.F_history else 0
                Fs = agent.F_social_history[-1] if agent.F_social_history else 0
                Fb = agent.F_body_history[-1] if agent.F_body_history else 0
                v = agent.valence_history[-1] if agent.valence_history else 0
                a = agent.arousal_history[-1] if agent.arousal_history else 0
                sv = agent.self_valence_ema
                coh = agent.self_coherence
                feel = ':)' if v > 0.1 else (':(' if v < -0.1 else ':|')
                dc = agent.dialogue_ctx
                sm = agent.self_model
                last_cb = (agent.consolidation_history[-1]
                          if agent.consolidation_history else None)
                cb_str = (f" cb=t{last_cb['n_turns']}r{last_cb['replays']}"
                         if last_cb and last_cb.get('phase') == 'full' else "")
                print(f"  [T={t:04d}] {feel} "
                      f"F={Fa:.3f} Fb={Fb:.3f} Fs={Fs:.3f} "
                      f"V={v:+.2f} A={a:.2f} "
                      f"self-V={sv:+.2f} coh={coh:.2f} "
                      f"b0={agent.body.b[0]:.2f} C={agent.net.n_clusters} "
                      f"mem={dc.n_turns()} self={sm.n_experiences}{cb_str}")

            t += 1
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[System]: Ctrl+C 退出")

    reader.stop()

    # 最终状态
    v_final = agent.valence_history[-1] if agent.valence_history else 0
    trust_final = social_ctx.trust_level
    n_cb = len(agent.consolidation_history)
    cb_info = ""
    if n_cb > 0:
        last_cb = agent.consolidation_history[-1]
        cb_info = (f"\n  Consolidations: {n_cb}"
                   f"\n  Last: {last_cb.get('n_turns', 0)} turns → "
                   f"{last_cb.get('replays', 0)} replays, "
                   f"{last_cb.get('cross_links', 0)} cross-links, "
                   f"pruned {last_cb.get('pruned', 0)}")
    print(f"\n{'='*60}")
    print(f"  Session ended")
    print(f"  Steps: {t}  |  Clusters: {agent.net.n_clusters}")
    print(f"  Final valence: {v_final:+.2f}")
    print(f"  Final trust:   {trust_final:.2f}")
    print(f"  Body b0:       {agent.body.b[0]:.2f}")
    print(f"  Interactions:  {social_ctx.n_interactions}")
    print(f"  Self-model:    {agent.self_model.n_experiences} exp, "
          f"{agent.self_model.net.n_clusters} clusters{cb_info}")
    print(f"{'='*60}")


if __name__ == '__main__':
    run_dialogue()
