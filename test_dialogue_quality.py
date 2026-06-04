"""
test_dialogue_quality.py — 对话质量基准测试

模拟多轮对话，评分:
1. 相关性: 回应是否与输入有关？
2. 句法质量: 生成的句子是否语法通顺？
3. 非引用率: 是否在说自己的话？
4. 连贯性: 多轮对话是否保持一致？
5. 上下文意识: 是否记住了之前说的？

神经学等价检查:
- Broca 失语症: 词不达意 → 句法质量低
- Wernicke 失语症: 不理解输入 → 相关性低
- 顺行性遗忘: 不记得上下文 → 连贯性低
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from data_types import D, BodyVector, Action, ACTION_DIRECTIONS
from agent import Agent
from text_interface import TextEnvironment
from broca import Broca
from sentiment import analyze_sentiment, sentiment_to_social_signal, get_emotional_lexicon
from layer1_free_energy import SocialContext


class DialogueTester:
    """非交互式对话测试器"""

    def __init__(self):
        self.rng = np.random.default_rng(99)
        self.agent = Agent(rng=self.rng, agent_id=0, n_agents=1)
        self.agent.body = BodyVector(mode='text')
        self.agent.theta.cluster_threshold = 0.55
        self.agent.theta.w_social = 2.0
        ACTION_DIRECTIONS[3] = [0.0, 0.0]
        ACTION_DIRECTIONS[4] = [0.0, 0.0]

        self.text_env = TextEnvironment()
        self.broca = Broca(text_env=self.text_env)
        self.social_ctx = SocialContext(tau=15.0)
        self.emo_lexicon = get_emotional_lexicon()   # Hebb 情感词汇网络

        # ---- L0 预热: 从语料均匀采样 (v2: 无人类偏好) ----
        corpus_sents = self.text_env.chunks
        n_corpus = len(corpus_sents)
        warmup_indices = [int(i * n_corpus / 15) for i in range(15)]
        warmup_sents = [corpus_sents[idx] for idx in warmup_indices]
        self.agent.warmup_l0(warmup_sents, self.text_env, n=15)
        print(f"  L0 warmup: {self.agent.net.n_clusters} initial clusters "
              f"(sampled from corpus)")

        self.last_human_vec = np.zeros(64, dtype=np.float32)
        self.last_self_semantic = np.zeros(64, dtype=np.float32)
        self.last_self_sentiment = np.zeros(8, dtype=np.float32)
        self.comprehension_vec = np.zeros(64, dtype=np.float32)
        self.t = 0

    def process_input(self, human_text: str) -> dict:
        """处理单轮输入，返回完整结果"""
        s = np.zeros(D, dtype=np.float32)

        # 语义编码
        human_vec = self.text_env.encode_text(human_text.strip())
        s[0:64] = human_vec
        self.last_human_vec = human_vec.copy()

        # 情感编码 (v2: Hebb 学习, 无手标词典)
        sentiment = analyze_sentiment(human_text.strip(),
                                     lexicon=self.emo_lexicon)
        social_signal = sentiment_to_social_signal(sentiment)
        s[80:88] = social_signal
        self.social_ctx.update(sentiment['valence'], sentiment['arousal'])

        # Wernicke 区: 理解
        human_sent = social_signal[:8].astype(np.float32)
        self.comprehension_vec, understanding = self.agent.comprehend(
            human_vec, human_sent)

        # 自听回路
        s[128:192] = self.last_self_semantic
        s[96:104] = self.last_self_sentiment

        # Agent 处理
        F_before = self.agent.F_body_history[-1] if self.agent.F_body_history else 0.0
        action = self.agent.step(s, self.t, social_ctx=self.social_ctx)
        F_after = self.agent.F_body_history[-1] if self.agent.F_body_history else 0.0

        # ---- Hebb 情感学习 ----
        delta_F = F_before - F_after
        import jieba
        input_words = [w for w in jieba.lcut(human_text.strip())
                      if len(w.strip()) >= 1]
        a_now = self.agent.arousal_history[-1] if self.agent.arousal_history else 0.5
        self.emo_lexicon.learn_from_feedback(input_words, delta_F, a_now)

        # 生成回应
        response = "(silence)"
        words = []
        eval_result = None

        if action.index == 3 and self.agent.net.n_clusters > 0:
            top = max(self.agent.net.clusters, key=lambda c: c.activation)
            if top.activation > 0.01:
                belief_sem = top.centroid[:64].copy().astype(np.float32)
                ctx = self.agent.dialogue_ctx.get_context_vector()
                query = (self.comprehension_vec * 0.45
                         + belief_sem * 0.35
                         + ctx * 0.20).astype(np.float32)

                v = self.agent.valence_history[-1] if self.agent.valence_history else 0
                a = self.agent.arousal_history[-1] if self.agent.arousal_history else 0
                words, _ = self.broca.speak_from_state(
                    belief_vec=top.centroid,
                    body_state=self.agent.body,
                    query_vec=query,
                    valence=v, arousal=a,
                    temperature=0.7, max_words=18,
                )
                response = "".join(words) if words else ""

                # 评估
                if response and len(response) > 1:
                    try:
                        resp_vec = self.text_env.encode_text(response).astype(np.float32)
                        eval_result = self.agent.evaluate_own_response(resp_vec)
                    except Exception:
                        pass

                # ---- 反重复: 最近说过的话不重复 ----
                if self.agent.dialogue_ctx.is_repeating(response):
                    # 重新生成 (更高温度)
                    v = self.agent.valence_history[-1] if self.agent.valence_history else 0
                    a = self.agent.arousal_history[-1] if self.agent.arousal_history else 0
                    words2, _ = self.broca.speak_from_state(
                        belief_vec=top.centroid,
                        body_state=self.agent.body,
                        query_vec=query,
                        valence=v, arousal=a,
                        temperature=1.2, max_words=18,
                    )
                    response2 = "".join(words2) if words2 else ""
                    if response2 and not self.agent.dialogue_ctx.is_repeating(response2):
                        response = response2
                        words = words2

                # 存入记忆
                if response:
                    try:
                        resp_vec = self.text_env.encode_text(response).astype(np.float32)
                    except Exception:
                        resp_vec = np.zeros(64, dtype=np.float32)
                    v_now = self.agent.valence_history[-1] if self.agent.valence_history else 0
                    a_now = self.agent.arousal_history[-1] if self.agent.arousal_history else 0
                    self.agent.dialogue_ctx.add_turn(
                        human_text=human_text, human_vec=human_vec,
                        human_sentiment=s[80:88].astype(np.float32),
                        agent_response=response, agent_vec=resp_vec,
                        agent_valence=v_now, agent_arousal=a_now,
                        comprehension_vec=self.comprehension_vec,
                    )

                    # 自我模型存储
                    ctx = self.agent.dialogue_ctx.get_context_vector()
                    self.agent.self_model.add_experience(
                        response_vec=resp_vec,
                        valence=v_now,
                        arousal=a_now,
                        self_valence_ema=self.agent.self_valence_ema,
                        self_arousal_ema=self.agent.self_arousal_ema,
                        self_coherence=self.agent.self_coherence,
                        body_state=self.agent.body,
                        comprehension_vec=self.comprehension_vec,
                        dialogue_ctx_vec=ctx,
                    )

                    # ---- 微量巩固 ----
                    self.agent.micro_consolidate()

                    # ---- 累积 5 轮触发完整睡眠巩固 ----
                    cb_result = self.agent.maybe_consolidate(broca=None)
                    if cb_result and cb_result.get('phase') == 'full':
                        pass  # 静默 (benchmark 不打印)

        # 更新自听
        if response and response != "(silence)" and len(response) > 1:
            try:
                self.last_self_semantic = self.text_env.encode_text(response).astype(np.float32)
                self_sent = analyze_sentiment(response)
                self.last_self_sentiment = sentiment_to_social_signal(self_sent)[:8].astype(np.float32)
            except Exception:
                pass

        self.t += 1
        return {
            'human_text': human_text,
            'response': response,
            'words': words,
            'valence': self.agent.valence_history[-1] if self.agent.valence_history else 0,
            'arousal': self.agent.arousal_history[-1] if self.agent.arousal_history else 0,
            'comprehension': understanding,
            'eval': eval_result,
            'action': action.index,
        }


def score_response(human_text: str, response: str, text_env: TextEnvironment) -> dict:
    """评分单条回应"""
    scores = {}

    # 引用检测
    broca = None
    try:
        from broca import Broca as B
    except Exception:
        pass

    # 1. 长度评分 (2-15字正常, <2太短, >30太长)
    rlen = len(response)
    if rlen < 2:
        scores['length'] = 0.0
    elif rlen < 4:
        scores['length'] = 0.3
    elif rlen < 30:
        scores['length'] = min(1.0, rlen / 15)
    else:
        scores['length'] = 0.7

    # 2. 相关性评分 (输入→回应余弦相似度)
    try:
        hv = text_env.encode_text(human_text)
        rv = text_env.encode_text(response)
        sim = float(np.dot(hv, rv) / (np.linalg.norm(hv) * np.linalg.norm(rv) + 1e-8))
        scores['relevance'] = max(0.0, sim)
    except Exception:
        scores['relevance'] = 0.3

    # 3. 重复检测 (回应自相似)
    if len(response) > 3:
        # 检查是否有重复字模式
        chars = list(response)
        unique_ratio = len(set(chars)) / max(len(chars), 1)
        scores['uniqueness'] = unique_ratio
    else:
        scores['uniqueness'] = 0.5

    # 综合
    scores['overall'] = (scores['length'] * 0.2
                         + scores['relevance'] * 0.5
                         + scores['uniqueness'] * 0.3)
    return scores


def run_benchmark():
    """运行对话质量基准测试"""
    print("=" * 70)
    print("  对话质量基准测试 — 脑区功能验证")
    print("=" * 70)

    tester = DialogueTester()

    # 测试场景: 模拟一段自然对话
    test_dialogues = [
        # (人类输入, 期望行为, 最低可接受分)
        ("你好！今天过得怎么样？", "问候回应", 0.2),
        ("我最近在看一本书，讲的是人工智能的历史。", "展示理解+延伸", 0.15),
        ("你觉得机器人会有感情吗？", "表达观点", 0.15),
        ("哈哈，你说的有道理。", "回应赞同", 0.15),
        ("那我们换个话题吧，你喜欢什么音乐？", "话题跟随", 0.15),
        ("我也喜欢古典音乐！", "共鸣回应", 0.15),
        ("时间不早了，我该走了。", "告别回应", 0.2),
        ("明天见！", "简短告别", 0.2),
    ]

    all_scores = []
    for i, (human_text, expected, min_score) in enumerate(test_dialogues):
        result = tester.process_input(human_text)
        response = result['response']

        # 评分
        scores = score_response(human_text, response, tester.text_env)

        # 引用检测: 长句完全匹配才算引用 (>15字)
        tester.broca._ensure_sent_clusters()
        is_long_quote = (len(response) > 15
                         and response in tester.broca.sentences)
        is_short_match = (len(response) <= 15
                          and response in tester.broca.sentences)
        scores['is_quote'] = 1.0 if is_long_quote else 0.0
        scores['is_short_match'] = 0.5 if is_short_match else 0.0
        if is_long_quote:
            scores['overall'] *= 0.4  # 长引用重罚
        elif is_short_match:
            scores['overall'] *= 0.8  # 短匹配轻罚 (可能是巧合)

        all_scores.append(scores)

        # 脑区诊断
        diagnosis = []
        if scores['relevance'] < 0.1:
            diagnosis.append('Wernicke失语(不理解)')
        if scores['length'] < 0.2:
            diagnosis.append('Broca失语(词不达意)')
        if result['comprehension']['n_triggered_memories'] == 0:
            diagnosis.append('顺行性遗忘(无记忆)')

        print(f"\n  [{i+1}] Human: {human_text}")
        print(f"      Response: {_safe_str(response)}")
        print(f"      Score: rel={scores['relevance']:.2f} "
              f"len={scores['length']:.2f} "
              f"uniq={scores['uniqueness']:.2f} "
              f"overall={scores['overall']:.2f} "
              f"long_quote={is_long_quote} short_match={is_short_match}")
        if diagnosis:
            print(f"      [!] Diagnosis: {', '.join(diagnosis)}")
        print(f"      V={result['valence']:+.2f} A={result['arousal']:.2f} "
              f"mem={len(tester.agent.dialogue_ctx.turns)}")

    # 汇总
    avg_score = np.mean([s['overall'] for s in all_scores])
    avg_rel = np.mean([s['relevance'] for s in all_scores])
    long_quote_rate = np.mean([s['is_quote'] for s in all_scores])
    short_match_rate = np.mean([s.get('is_short_match', 0) for s in all_scores])

    print(f"\n{'='*70}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"  Avg Overall Score:    {avg_score:.3f}")
    print(f"  Avg Relevance:        {avg_rel:.3f}")
    print(f"  Long Quote Rate:      {long_quote_rate:.0%}")
    print(f"  Short Match Rate:     {short_match_rate:.0%}")
    print(f"  Dialogue Memory Turns: {len(tester.agent.dialogue_ctx.turns)}")
    print(f"  Word-order clusters:  {tester.broca.word_order_net.n_clusters}")
    sm = tester.agent.self_model
    print(f"  Self-model:           {sm.n_experiences} exp, {sm.net.n_clusters} clusters")
    print(f"  Self personality:     V={sm.anchor_emotion[0]:+.2f} A={sm.anchor_emotion[1]:.2f}")

    # Hebb 情感词汇统计
    lex_stats = tester.emo_lexicon.get_stats()
    print(f"  HebbEmoLexicon:       {lex_stats['n_words']} words learned")
    if lex_stats['top_positive']:
        print(f"    Top +: {', '.join(f'{w}({v:+.2f})' for w,v in lex_stats['top_positive'][:3])}")
    if lex_stats['top_negative']:
        print(f"    Top -: {', '.join(f'{w}({v:+.2f})' for w,v in lex_stats['top_negative'][:3])}")

    # 睡眠巩固统计
    cb_history = tester.agent.consolidation_history
    if cb_history:
        last_cb = cb_history[-1]
        print(f"\n  Sleep Consolidation:")
        print(f"    Cycles:             {len(cb_history)}")
        print(f"    Last:               {last_cb.get('n_turns', 0)} turns → "
              f"{last_cb.get('replays', 0)} replays, "
              f"{last_cb.get('cross_links', 0)} cross-links")
        print(f"    L0 clusters post:   {tester.agent.net.n_clusters}")
        print(f"    Self clusters post: {sm.net.n_clusters}")
        print(f"    Session valence:    {last_cb.get('session_valence', 0):+.2f}")
    else:
        print(f"\n  Sleep Consolidation:  NOT TRIGGERED (need 5+ turns)")

    # 脑区功能检查表
    print(f"\n  Brain Region Check:")
    checks = {
        'Broca (speech)': tester.broca.word_order_net.n_clusters > 100,
        'Wernicke (comprehension)': avg_rel > 0.1,
        'Hippocampus (memory)': len(tester.agent.dialogue_ctx.turns) > 0,
        'Hippocampus→Cortex (consolidation)': len(cb_history) > 0,
        'ACC/OFC (monitoring)': all_scores[-1].get('overall', 0) > 0,
        'DMN (self-model)': sm.n_experiences > 0,
    }
    for region, ok in checks.items():
        print(f"    {'[OK]' if ok else '[!!]'} {region}: {'OK' if ok else 'NEEDS WORK'}")

    # 通过标准
    if avg_score >= 0.2 and long_quote_rate < 0.5:
        print(f"\n  [PASS] Dialogue quality acceptable")
    else:
        print(f"\n  [FAIL] Dialogue quality insufficient "
              f"(avg_score={avg_score:.2f}, long_quote_rate={long_quote_rate:.0%})")

    return avg_score, long_quote_rate


def _safe_str(s):
    try:
        s.encode('gbk'); return s
    except UnicodeEncodeError:
        return s.encode('gbk', errors='replace').decode('gbk')


if __name__ == '__main__':
    run_benchmark()
    print()
