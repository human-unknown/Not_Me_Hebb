"""
main_dialogue.py —— Stage 6: 人机对话闭环 (v3: 视觉通道)
自由能原理智能体

持续流模式: Agent 不等人类，10 FPS 持续循环。
人类输入 → 语义编码 s[0:64] + 情感编码 s[80:88] → Agent 被对话改变。

v3: 视觉通道 — 输入 img:<path> 让 Agent ""看到"" 图片并描述。
     管线: 图像 → Gabor 编码 → 跨模态模型 V→T recall → 中文标签 → 对话。
"""
import sys, time, queue, pickle, os as _os, numpy as np
from data_types import D, BodyVector, Action, ACTION_DIRECTIONS
from agent import Agent
from text_interface import TextEnvironment
from word_speech import get_speaker
from broca import Broca
from stdin_reader import StdinReader
from sentiment import analyze_sentiment, sentiment_to_social_signal, get_emotional_lexicon
from layer1_free_energy import SocialContext
from image_encoder import (ImageEncoder, build_visual_sensory,
                           make_visual_mask, make_text_mask)


def _safe_str(s: str) -> str:
    """GBK-safe: replace non-GBK chars for Windows console"""
    try:
        s.encode('gbk'); return s
    except UnicodeEncodeError:
        return s.encode('gbk', errors='replace').decode('gbk')


# ================================================================
# COCO 80 类别中英文映射 (用于视觉→文本→中文对话桥接)
# ================================================================

COCO_CATEGORIES_ZH = {
    'person': '人', 'bicycle': '自行车', 'car': '汽车',
    'motorcycle': '摩托车', 'airplane': '飞机', 'bus': '公共汽车',
    'train': '火车', 'truck': '卡车', 'boat': '船',
    'traffic light': '红绿灯', 'fire hydrant': '消防栓',
    'stop sign': '停车标志', 'parking meter': '停车计时器', 'bench': '长椅',
    'bird': '鸟', 'cat': '猫', 'dog': '狗', 'horse': '马',
    'sheep': '羊', 'cow': '牛', 'elephant': '大象', 'bear': '熊',
    'zebra': '斑马', 'giraffe': '长颈鹿',
    'backpack': '背包', 'umbrella': '雨伞', 'handbag': '手提包', 'tie': '领带',
    'suitcase': '行李箱', 'frisbee': '飞盘', 'skis': '滑雪板',
    'snowboard': '滑雪板', 'sports ball': '球', 'kite': '风筝',
    'baseball bat': '棒球棒', 'baseball glove': '棒球手套',
    'skateboard': '滑板', 'surfboard': '冲浪板', 'tennis racket': '网球拍',
    'bottle': '瓶子', 'wine glass': '酒杯', 'cup': '杯子',
    'fork': '叉子', 'knife': '刀', 'spoon': '勺子', 'bowl': '碗',
    'banana': '香蕉', 'apple': '苹果', 'sandwich': '三明治',
    'orange': '橙子', 'broccoli': '西兰花', 'carrot': '胡萝卜',
    'hot dog': '热狗', 'pizza': '披萨', 'donut': '甜甜圈', 'cake': '蛋糕',
    'chair': '椅子', 'couch': '沙发', 'potted plant': '盆栽',
    'bed': '床', 'dining table': '餐桌', 'toilet': '马桶',
    'tv': '电视', 'laptop': '笔记本电脑', 'mouse': '鼠标',
    'remote': '遥控器', 'keyboard': '键盘', 'cell phone': '手机',
    'microwave': '微波炉', 'oven': '烤箱', 'toaster': '烤面包机',
    'sink': '水槽', 'refrigerator': '冰箱',
    'book': '书', 'clock': '时钟', 'vase': '花瓶',
    'scissors': '剪刀', 'teddy bear': '泰迪熊', 'hair drier': '吹风机',
    'toothbrush': '牙刷',
}
"""COCO 80 类别 → 中文 (用于 Agent 对话)"""


def _channel_normalize_visual(vis_part: np.ndarray) -> np.ndarray:
    """逐通道 L2 归一化视觉特征 [V1|V2|V4|Color]。

    防止 V4 (norm ~1.0) 主导 V1 (norm ~0.01)。
    布局: V1[0:96] | V2[96:160] | V4[160:224] | Color[224:266]
    """
    result = vis_part.copy().astype(np.float32)
    for start, end in [(0, 96), (96, 160), (160, 224), (224, 266)]:
        n = np.linalg.norm(result[start:end])
        if n > 1e-8:
            result[start:end] /= n
    # L2 normalize the whole thing too
    total_n = np.linalg.norm(result)
    if total_n > 1e-8:
        result /= total_n
    return result


def _setup_visual_brain(model_path: str = '.cache/stage2_crossmodal_coco_5000_s42.pkl'):
    """加载跨模态视觉模型 + 构建 COCO 文本编码器。

    Returns:
        (vis_net, coco_embs, coco_captions_zh, coco_captions_en)
        或 None (若模型未找到)
    """
    if not _os.path.exists(model_path):
        print(f"  [Vision] 未找到跨模态模型: {model_path}")
        print(f"  [Vision] 请先运行: python stage2_crossmodal.py --dataset coco --n 5000 --mode all")
        return None

    print(f"  [Vision] 加载跨模态模型: {model_path}")
    with open(model_path, 'rb') as f:
        vis_model = pickle.load(f)
    vis_net = vis_model['net']
    print(f"  [Vision] {vis_net.n_clusters} 跨模态集群已加载")

    # ---- 构建 COCO 文本编码器 (与 stage2 相同的 pipeline) ----
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    # 英文模板 (与训练时一致，用于 PCA 匹配)
    EN_TEMPLATES = ['a photo of a {}', 'a {} in the scene',
                    'an image containing a {}']
    # 中文自然描述模板 (用于对话 Agent)
    ZH_TEMPLATES = ['一张{}的照片', '场景中的{}', '包含{}的图像']

    coco_captions_en = []
    coco_captions_zh = []  # 用于显示 + 对话输入
    coco_categories_zh = []  # 纯类别名 (用于去重)
    for en_name, zh_name in COCO_CATEGORIES_ZH.items():
        for en_tmpl, zh_tmpl in zip(EN_TEMPLATES, ZH_TEMPLATES):
            coco_captions_en.append(en_tmpl.format(en_name))
            coco_captions_zh.append(zh_tmpl.format(zh_name))
            coco_categories_zh.append(zh_name)  # 存纯类别名用于去重

    print(f"  [Vision] 拟合文本编码器 ({len(coco_captions_en)} 条)...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    full_embs = st_model.encode(coco_captions_en, show_progress_bar=False,
                                batch_size=64)
    pca = PCA(n_components=64, random_state=42)
    pca.fit(full_embs)
    coco_embs = pca.transform(full_embs).astype(np.float32)
    # L2 归一化
    norms = np.linalg.norm(coco_embs, axis=1, keepdims=True)
    coco_embs /= (norms + 1e-8)
    print(f"  [Vision] 文本编码器就绪: {full_embs.shape[1]}d → 64d, "
          f"explained={pca.explained_variance_ratio_.sum():.1%}")

    # 预计算所有簇的逐通道归一化视觉特征 (用于快速视觉检索)
    coco_vis_normed = np.zeros((vis_net.n_clusters, 266), dtype=np.float32)
    for i, cl in enumerate(vis_net.clusters):
        coco_vis_normed[i] = _channel_normalize_visual(cl.centroid[64:330])

    vis_brain = {
        'net': vis_net,
        'coco_embs': coco_embs,
        'coco_captions_zh': coco_captions_zh,
        'coco_captions_en': coco_captions_en,
        'coco_categories_zh': coco_categories_zh,
        'coco_vis_normed': coco_vis_normed,
        'st_model': st_model,
        'pca': pca,
    }
    return vis_brain


def _visual_recall(vis_brain: dict, image_path: str,
                   img_enc: ImageEncoder,
                   top_k: int = 3) -> list[dict]:
    """视觉→文本 recall: 图像 → Gabor 编码 → 跨模态检索 → 中文描述。

    使用逐通道 L2 归一化: V1/V2/V4/Color 各自归一化，
    防止 V4 (norm ~1.0) 主导 V1/V2/Color (norm ~0.01)。

    Returns:
        [{zh_caption, en_caption, similarity, rank}, ...]
    """
    # 编码图像 (build_visual_sensory 默认 normalize_channels=True)
    try:
        vis_feat = img_enc.encode_from_path(image_path)
    except Exception as e:
        print(f"  [Vision] 无法加载图像: {e}")
        return []

    s_vis = build_visual_sensory(vis_feat)  # 通道归一化的 query
    query_vis = s_vis[64:330].astype(np.float32)
    # 与 centroids 使用相同的归一化: 逐通道 + 整体 L2
    query_vis = _channel_normalize_visual(query_vis)

    net = vis_brain['net']
    coco_embs = vis_brain['coco_embs']

    # 与所有簇的预归一化视觉特征比较
    if 'coco_vis_normed' in vis_brain:
        # 快速路径: 使用预归一化的 centroid 视觉特征
        centroid_vis_normed = vis_brain['coco_vis_normed']  # (N, 266)
        sims = np.dot(centroid_vis_normed, query_vis)
        best_idx = int(np.argmax(sims))
        best_c = net.clusters[best_idx]
    else:
        # 慢速路径: 逐簇归一化比较
        best_sim = -1.0
        best_c = None
        for cl in net.clusters:
            c_vis = _channel_normalize_visual(cl.centroid[64:330])
            sim = float(np.dot(c_vis, query_vis))
            if sim > best_sim:
                best_sim = sim
                best_c = cl

    if best_c is None:
        return []

    # 从 centroid 提取文本部分 → 匹配 COCO 标签
    text_vec = best_c.centroid[:64].astype(np.float32)

    # 归一化
    text_vec = text_vec / (np.linalg.norm(text_vec) + 1e-8)

    # 计算与所有 COCO caption 的余弦相似度
    sims = np.dot(coco_embs, text_vec)

    results = []
    seen_cats = set()  # 按类别名去重
    for rank, idx in enumerate(np.argsort(sims)[::-1]):
        zh_cat = vis_brain['coco_categories_zh'][idx]
        if zh_cat in seen_cats:
            continue
        seen_cats.add(zh_cat)

        results.append({
            'zh_caption': vis_brain['coco_captions_zh'][idx],
            'zh_category': zh_cat,
            'en_caption': vis_brain['coco_captions_en'][idx],
            'similarity': float(sims[idx]),
            'rank': rank,
        })
        if len(results) >= top_k:
            break

    return results


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

    # ---- 视觉大脑: 加载跨模态模型 (可选) ----
    vis_brain = _setup_visual_brain()
    img_enc = ImageEncoder(image_size=128) if vis_brain else None
    if vis_brain:
        print(f"  [Vision] 视觉通道就绪 — 输入 'img:path/to/image.jpg' 让 Agent ""看到"" 图片")
    else:
        print(f"  [Vision] 视觉通道未启用 — 运行 stage2_crossmodal.py 训练跨模态模型")

    # ---- 社会情感上下文 ----
    social_ctx = SocialContext(tau=15.0)  # tau 越小，情感越容易被改变

    last_human_vec = np.zeros(64)
    last_self_semantic = np.zeros(64, dtype=np.float32)   # 上一轮自己说的话 (语义)
    last_self_sentiment = np.zeros(8, dtype=np.float32)   # 上一轮自己说的话 (情感)
    comprehension_vec = np.zeros(64, dtype=np.float32)     # Wernicke 理解向量
    emo_lexicon = get_emotional_lexicon()                  # Hebb 情感词汇网络
    t = 0; expr_cooldown = 0; inner_cooldown = 0

    print("=" * 60)
    print("  自由能原理智能体 — Stage 6: 人机对话 (v6: 视觉)")
    print("  v6: Wernicke 理解 + Hebb 词序链 + 响应评估 + 视觉通道")
    print("=" * 60)
    print("  你说的话会改变 Agent 的情感状态")
    print("  温暖的话 → F_social ↓ → valence ↑ → Agent 感到好")
    print("  攻击的话 → F_social ↑ → valence ↓ → Agent 感到不好")
    print("  Agent 先理解，再回忆，然后用自己的话说出来")
    print("  沉默时 Agent 会自己「想」→ 内部言语链")
    if vis_brain:
        print("  🖼️ 输入 'img:path/to/image.jpg' 让 Agent \"看到\" 图片")
    print("-" * 60)
    print("  输入文字后回车 · 输入 'exit' 退出 · Ctrl+C 退出")
    print("-" * 60)

    try:
        while True:
            img_path = None; extra_text = None  # 视觉输入变量
            # ---- 1. 非阻塞读人类输入 ----
            human_text = None
            try:
                human_text = reader.queue.get_nowait()
            except queue.Empty:
                pass

            if human_text and human_text.strip().lower() == 'exit':
                print("\n[System]: 退出")
                break

            # ---- 视觉输入处理: img:<path> ----
            visual_context = None  # 视觉理解结果 (用于注入对话)
            if (human_text and human_text.strip().startswith('img:')
                    and vis_brain is not None):
                img_path = human_text.strip()[4:].strip()
                # 支持 img:path 和 img:path 额外文本
                extra_text = None
                if ' ' in img_path:
                    parts = img_path.split(' ', 1)
                    img_path = parts[0]
                    extra_text = parts[1] if len(parts) > 1 else None

                # 去除引号
                img_path = img_path.strip('"\'')

                if not _os.path.exists(img_path):
                    print(f"\n[Vision] 文件未找到: {img_path}")
                    continue

                print(f"\n[Vision] 正在看: {img_path} ...")
                results = _visual_recall(vis_brain, img_path, img_enc, top_k=3)

                if results:
                    top = results[0]
                    print(f"  [Vision] 看到: {top['zh_caption']} "
                          f"(sim={top['similarity']:.3f})")
                    if len(results) > 1:
                        others = ', '.join(r['zh_caption'] for r in results[1:])
                        print(f"  [Vision]   其他可能: {others}")

                    # 构建视觉上下文: 用最匹配的中文描述作为"看到的内容"
                    visual_context = top['zh_caption']
                else:
                    print(f"  [Vision] 未找到匹配 — 视觉识别失败")
                    continue

            # ---- 2. 构建感觉向量 ----
            s = np.zeros(D)

            if human_text and human_text.strip():
                # 检查是否是视觉输入
                is_visual_input = (human_text.strip().startswith('img:')
                                   and visual_context is not None)

                if is_visual_input:
                    # ---- 视觉输入: 用识别到的中文描述替代 img: 命令 ----
                    display_text = f"[看到 {visual_context}]"
                    # 编码视觉描述为语义向量
                    try:
                        s[0:64] = text_env.encode_text(visual_context)
                    except Exception:
                        s[0:64] = np.zeros(64)
                    last_human_vec = s[0:64].copy()

                    # 视觉通道: 填入 Gabor 特征
                    try:
                        vis_feat = img_enc.encode_from_path(img_path)
                        s_vis = build_visual_sensory(vis_feat)
                        s[64:330] = s_vis[64:330]
                    except Exception:
                        pass

                    # 中性偏正情感编码 (看到东西通常是好奇/中性)
                    s[80:88] = np.array(
                        [0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.1, 0.0],
                        dtype=np.float32)

                    # 更新社会上下文
                    social_ctx.update(0.2, 0.3)  # mildly positive, curious

                    # Wernicke 理解
                    human_sent = s[80:88].astype(np.float32)
                    comprehension_vec, understanding = agent.comprehend(
                        last_human_vec, human_sent)

                    # 显示
                    print(f"\n[You] 🖼️: {human_text.strip()}")
                    print(f"       → Agent 看到: {visual_context}")
                    if extra_text:
                        print(f"       → 附带问题: {extra_text}")
                    print(f"       vision_input=True "
                          f"understood={understanding['n_triggered_memories']}mem")
                else:
                    # ---- 正常文本输入 ----
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
