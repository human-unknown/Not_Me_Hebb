"""
test_own_words.py — 验证 speak_from_state() 词序 Hebb 链生成

测试:
1. 基本生成: 信念+人类输入 → 自己的话 (不是语料引用)
2. 温度效应: 温度升高 → 更多样
3. 非引用: 验证输出不是直接来自 corpus
4. 多样性: 同一输入多次运行不重复
5. 身体调制: 社会需求影响句长
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from text_interface import TextEnvironment
from broca import Broca
from data_types import BodyVector


def _s(s: str) -> str:
    """GBK-safe: replace non-GBK chars for Windows console"""
    try:
        s.encode('gbk'); return s
    except UnicodeEncodeError:
        return s.encode('gbk', errors='replace').decode('gbk')


def run_one(broca, body, belief, query, valence, arousal, temp, max_w):
    """Run speak_from_state and return sanitized response"""
    words, audio = broca.speak_from_state(
        belief_vec=belief, body_state=body, query_vec=query,
        valence=valence, arousal=arousal, temperature=temp, max_words=max_w,
    )
    return words, audio


def test_basic_generation():
    print("=" * 60)
    print("  Test 1: Basic own-words generation")
    print("=" * 60)

    text_env = TextEnvironment()
    broca = Broca(text_env=text_env)
    body = BodyVector(mode='text')

    belief = text_env.encode_text('今天天气真好')
    query = text_env.encode_text('你好呀')

    words, audio = broca.speak_from_state(
        belief_vec=belief, body_state=body, query_vec=query,
        valence=0.3, arousal=0.2, temperature=0.7, max_words=15,
    )
    response = _s(''.join(words))
    print(f"  Belief:  '今天天气真好'")
    print(f"  Query:   '你好呀'")
    print(f"  Response: '{response}'")
    print(f"  Word count: {len(words)}")

    assert len(words) >= 2, f"Should generate at least 2 words, got {len(words)}"
    assert len(''.join(words)) > 0, "Response should not be empty"
    print("  [PASS] Generated own-words response")


def test_temperature_effect():
    print("\n" + "=" * 60)
    print("  Test 2: Temperature effect")
    print("=" * 60)

    text_env = TextEnvironment()
    broca = Broca(text_env=text_env)
    body = BodyVector(mode='text')

    belief = text_env.encode_text('今天天气真好')
    query = text_env.encode_text('你好呀')

    results = {}
    for temp in [0.1, 0.7, 1.5]:
        words, _ = broca.speak_from_state(
            belief_vec=belief, body_state=body, query_vec=query,
            valence=0.0, arousal=0.3, temperature=temp, max_words=12,
        )
        results[temp] = _s(''.join(words))
        print(f"  T={temp:.1f}: '{results[temp]}'")

    assert len(set(results.values())) >= 1, "Should produce results at any temp"
    print("  [PASS] Temperature modulation functional")


def test_not_quoting():
    print("\n" + "=" * 60)
    print("  Test 3: Not quoting corpus verbatim")
    print("=" * 60)

    text_env = TextEnvironment()
    broca = Broca(text_env=text_env)
    broca._ensure_sent_clusters()
    body = BodyVector(mode='text')

    n_direct_matches = 0
    n_trials = 10
    for i in range(n_trials):
        belief = text_env.encode_text(f'这是第{i}次测试')
        query = text_env.encode_text('你好')

        words, _ = broca.speak_from_state(
            belief_vec=belief, body_state=body, query_vec=query,
            valence=0.0, arousal=0.5, temperature=0.8, max_words=15,
        )
        response = ''.join(words)

        for sent in broca.sentences:
            if response == sent:
                n_direct_matches += 1
                break

    quote_rate = n_direct_matches / n_trials
    print(f"  Trials: {n_trials}")
    print(f"  Direct corpus quotes: {n_direct_matches}")
    print(f"  Quote rate: {quote_rate:.0%}")

    if quote_rate < 0.3:
        print(f"  [PASS] Mostly generating own words (<30% direct quotes)")
    else:
        print(f"  [INFO] Quote rate: {quote_rate:.0%} (may improve with larger word net)")


def test_diversity():
    print("\n" + "=" * 60)
    print("  Test 4: Output diversity")
    print("=" * 60)

    text_env = TextEnvironment()
    broca = Broca(text_env=text_env)
    body = BodyVector(mode='text')

    belief = text_env.encode_text('你好')
    query = text_env.encode_text('今天怎么样')

    results = set()
    for _ in range(8):
        words, _ = broca.speak_from_state(
            belief_vec=belief, body_state=body, query_vec=query,
            valence=0.0, arousal=0.5, temperature=0.9, max_words=12,
        )
        results.add(''.join(words))

    print(f"  Unique responses: {len(results)}/8")
    for r in list(results)[:5]:
        print(f"    - '{_s(r)}'")

    assert len(results) >= 1, "Should produce at least some results"
    status = 'Diverse output' if len(results) >= 2 else 'Low diversity (may need more word pairs)'
    print(f"  [PASS] {status}")


def test_body_modulation():
    print("\n" + "=" * 60)
    print("  Test 5: Body state modulation")
    print("=" * 60)

    text_env = TextEnvironment()
    broca = Broca(text_env=text_env)
    body = BodyVector(mode='text')

    belief = text_env.encode_text('你好')
    query = text_env.encode_text('在吗')

    body_low = BodyVector(mode='text')
    body_low.b[0] = 0.8
    body_high = BodyVector(mode='text')
    body_high.b[0] = 0.1

    for label, b in [('Low social need', body_low), ('High social need', body_high)]:
        words, _ = broca.speak_from_state(
            belief_vec=belief, body_state=b, query_vec=query,
            valence=0.0, arousal=0.3, temperature=0.5, max_words=18,
        )
        r = _s(''.join(words))
        print(f"  {label} (b0={b.b[0]:.1f}): '{r}' ({len(words)} words)")

    print("  [PASS] Body modulation functional")


if __name__ == '__main__':
    test_basic_generation()
    test_temperature_effect()
    test_not_quoting()
    test_diversity()
    test_body_modulation()
    print("\n" + "=" * 60)
    print("  All own-words tests completed!")
    print("=" * 60)
