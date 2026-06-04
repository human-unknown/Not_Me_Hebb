"""
clean_corpus.py — 从 b-corpus 中提取并清洗二次元对话语料

流程:
1. 扫描 b_corpus_raw 所有 .txt / .txt.gz 文件
2. 白名单 + 内容过滤 双重机制去除 R18 内容
3. 清洗格式、去重、生成干净的 corpus.txt
"""

import os, re, gzip, hashlib
from pathlib import Path

RAW_DIR = Path(__file__).parent / 'b_corpus_raw'
OUTPUT_FILE = Path(__file__).parent / 'corpus.txt'

# ============================================================
# 1. 白名单 — 只从这些目录提取
# ============================================================
SAFE_DIRS = {
    # v-corpus-zh: 视觉小说 (全年龄 / 主流作品)
    '米哈游',       # 原神, 崩坏:星穹铁道
    'Key',          # CLANNAD, AIR, Summer Pockets, Rewrite, Little Busters
    '5pb',          # Steins;Gate, Chaos;Head
    'TYPE-MOON',    # Fate系列 (过滤后)
    '猫猫社',       # 全年龄向
    '橘子班',       # 国产AVG
    'CIRCUS',       # 初音岛系列
    'AUGUST',       # 八月社
    'BugSystem',
    'Campus',
    'FAVORITE',
    'FLAT',
    'Feng',
    'GIGA',
    'InnocentGrey',
    'KID',
    'Leaf',         # White Album 2
    'Liar-soft',
    'Lump of Sugar',
    'MOONSTONE',
    'Minori',
    'NEXTON',
    'NanaWind',
    'Navel',
    'OVERDRIVE',
    'Palette',
    'Purple',
    'SAGA PLANETS',
    'SEGA',
    'SMEE',
    'Sprite',
    'TOPCAT',
    '柚子社',
    '方糖社',
    '枕社',
    '演绘',
    '猫之日',
    '白玉社',
    '零创',
    '绘恋',
    '河豚屋',
    '东方Project',
    '大宇',
    '库洛',
    'ANIPLEX.EXE',
    'Acacia',
    'BugSystem',
    'Citrus',
    'DreaMory',
    '07thExpansion', # 海猫鸣泣之时 / 寒蝉
    'YUZUSOFT',      # 柚子社 (英文目录名)
    'TYPEMOON',      # TYPE-MOON (英文目录名)
    'Nitro+',        # Fate/Zero, Steins;Gate
    'FrontWing',     # Grisaia系列
    'PULLTOP',       # 大空翼等
    'SAGAPLANETS',   # SAGA PLANETS (英文目录名)
    'NEKOWORKs',     # Nekopara (全年龄内容)
    'Madosoft',      # 任性HighSpec等
    'Lose',          # 茂伸等
    'Laplacian',     # 牛顿与苹果树等
    'Sphere',        # 缘之空等 (仅全年龄部分)
    'Novectacle',    # 海市蜃楼之馆
    'KeroQ',         # 美好的每一天
    'LIFE0',         # 7days等
    'Noesis',        # 免费游戏
    'SP-time',       # 国产AVG
    'WonderFool',    # 国产AVG
    'Recette',       # 国产AVG
    'U0U',           # 国产AVG
    'WaterPhoenix',  # 国产AVG
    'NEXON',         # 蔚蓝档案等

    # 轻小说
    '轻小说',

    # 互动小说
    '互动小说',

    # ChatHaruhi
    'ChatHaruhi',

    # 畅销书 / 散文集
    '畅销书',
    '散文集',
}

# ============================================================
# 2. R18 内容过滤关键词
# ============================================================
R18_KEYWORDS = [
    # 显式性描写
    '乳房', '乳头', '乳首', '巨乳', '爆乳', '贫乳',
    '阴道', '阴部', '阴唇', '阴蒂', '阴核',
    '肉棒', '阴茎', '阳具', '龟头', '勃起',
    '插入', '抽插', '性交', '做爱', '交合',
    '淫', '强奸', '凌辱', '调教',
    '精液', '射精', '高潮', '潮吹',
    '发情', '春药', '媚药',
    '露出', '痴汉', '轮奸',
    '性器', '性欲', '性奴',
    '口交', '肛交', '中出',
    '下体', '私处',
    '肏', '屌', '操你', '操我', '艹',
    '妓女', '卖春', '娼',
    # 极端暴力/猎奇
    '肢解', '虐杀', '凌迟',
    # 乱伦关键词
    '乱伦',
    # 边缘暗示 (不过滤日常用语)
    '爱液', '蜜汁', '花蜜',
    '小穴', '蜜穴', '肉穴',
    '自慰', '手淫',
    '情妇', '情欲',
    '处女膜', '破处',
    '发骚', '欠操',
    '爆乳', '贫乳', '巨乳',
]

# 编译为正则，忽略大小写
R18_PATTERN = re.compile('|'.join(re.escape(kw) for kw in R18_KEYWORDS))


# ============================================================
# 3. 辅助函数
# ============================================================

def is_safe_directory(path: Path) -> bool:
    """检查路径是否在白名单内"""
    parts = set(path.parts)
    return bool(parts & SAFE_DIRS)


def is_safe_line(line: str) -> bool:
    """单行内容过滤"""
    line = line.strip()
    if len(line) < 4:           # 太短
        return False
    if len(line) > 300:          # 太长（可能是旁白/描述）
        return False
    if R18_PATTERN.search(line):  # 命中 R18 关键词
        return False
    # 纯标点/数字/英文行
    chinese_chars = sum(1 for c in line if '一' <= c <= '鿿')
    if chinese_chars < 3:
        return False
    return True


def clean_line(line: str) -> str | None:
    """清洗单行文本"""
    line = line.strip()
    if not line:
        return None

    # 去掉过长的旁白描述（保留对话为主的短句）
    # 大多数对话 < 150 字
    if len(line) > 200:
        return None

    # 统一全角半角标点
    line = line.replace('：', '：')  # keep full-width colon as separator
    line = line.replace('…', '...')
    line = line.replace('——', '——')

    # 去掉开头多余标点
    line = line.lstrip('，。！？、；：""''（）…—')

    # 去掉纯符号行
    if all(c in '，。！？、；：""''（）…—～~ \t' for c in line):
        return None

    # 去掉表情符号标记 (如 【笑】 【哭】)
    # 保留文本内容

    return line


def extract_dialogue_parts(line: str) -> list[str]:
    """
    从一行中提取可用的中文文本。

    格式通常是: "角色名：对话内容"
    也可能有多轮对话: "A：xxx B：yyy"
    只提取对话部分（冒号后的内容），忽略角色名。
    """
    parts = []
    # 按全角冒号分割
    segments = line.split('：')
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # 跳过纯角色名（通常很短，不含标点）
        if len(seg) <= 3 and all('一' <= c <= '鿿' or c.isascii() for c in seg):
            continue
        if len(seg) >= 3:
            parts.append(seg)
    return parts


def read_file_safe(filepath: Path) -> list[str]:
    """安全读取文件，尝试多种编码"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.readlines()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='gb18030') as f:
                return f.readlines()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='shift-jis') as f:
                    return f.readlines()
            except:
                return []


def read_gz_file(filepath: Path) -> list[str]:
    """读取 .txt.gz 文件"""
    for enc in ['utf-8', 'gb18030', 'shift-jis']:
        try:
            with gzip.open(filepath, 'rt', encoding=enc) as f:
                return f.readlines()
        except (UnicodeDecodeError, gzip.BadGzipFile):
            continue
    return []


def should_skip_file(filepath: Path) -> bool:
    """判断是否应该跳过某个文件"""
    fname = filepath.name.lower()
    # 跳过非文本文件
    if fname.endswith('.py') or fname.endswith('.json'):
        return True
    if fname.endswith('.md') or fname.endswith('.txt'):
        return False
    if fname.endswith('.gz'):
        return False
    return True


# ============================================================
# 4. 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  b-corpus 清洗工具")
    print("=" * 60)

    # 收集所有文件
    all_files = []
    for root, dirs, files in os.walk(RAW_DIR):
        root_path = Path(root)
        if not is_safe_directory(root_path):
            continue
        for fname in files:
            fpath = root_path / fname
            if should_skip_file(fpath):
                continue
            all_files.append(fpath)

    print(f"\n  白名单内文件: {len(all_files)} 个")

    # 读取并清洗
    seen_hashes = set()
    clean_lines = []
    stats = {
        'total_lines': 0,
        'too_short': 0,
        'too_long': 0,
        'r18_filtered': 0,
        'few_chinese': 0,
        'duplicate': 0,
        'kept': 0,
    }

    for i, fpath in enumerate(all_files):
        if i % 500 == 0:
            print(f"  处理中... {i}/{len(all_files)}")

        # 读取
        if fpath.suffix == '.gz' or '.gz' in fpath.name:
            lines = read_gz_file(fpath)
        else:
            lines = read_file_safe(fpath)

        for line in lines:
            stats['total_lines'] += 1
            line = line.strip()
            if not line:
                continue

            # 清洗
            cleaned = clean_line(line)
            if cleaned is None:
                stats['too_short'] += 1
                continue

            # R18 过滤
            if R18_PATTERN.search(cleaned):
                stats['r18_filtered'] += 1
                continue

            # 中文占比
            chinese = sum(1 for c in cleaned if '一' <= c <= '鿿')
            if chinese < 4:
                stats['few_chinese'] += 1
                continue

            # 长度检查
            if len(cleaned) > 250:
                stats['too_long'] += 1
                continue

            # 去重（用 hash）
            h = hashlib.md5(cleaned.encode('utf-8')).hexdigest()
            if h in seen_hashes:
                stats['duplicate'] += 1
                continue
            seen_hashes.add(h)

            clean_lines.append(cleaned)
            stats['kept'] += 1

    print(f"\n  清洗完成!")
    print(f"  原始行数:   {stats['total_lines']}")
    print(f"  太短跳过:   {stats['too_short']}")
    print(f"  太长跳过:   {stats['too_long']}")
    print(f"  R18 过滤:   {stats['r18_filtered']}")
    print(f"  中文不足:   {stats['few_chinese']}")
    print(f"  重复去重:   {stats['duplicate']}")
    print(f"  保留:       {stats['kept']}")

    # 打乱并限制数量
    import random
    random.seed(42)
    random.shuffle(clean_lines)

    # 限制到 N 行（平衡质量和处理时间）
    max_lines = 200000
    if len(clean_lines) > max_lines:
        clean_lines = clean_lines[:max_lines]
        print(f"  截断至:     {max_lines} 行")

    # 写入
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in clean_lines:
            f.write(line + '\n')

    print(f"\n  输出文件: {OUTPUT_FILE}")
    print(f"  最终行数: {len(clean_lines)}")

    # 打印样本
    print(f"\n  样本 (前 20 行):")
    print("  " + "-" * 56)
    for line in clean_lines[:20]:
        # 截断长行显示
        display = line[:80] + ('...' if len(line) > 80 else '')
        try:
            print(f"  {display}")
        except UnicodeEncodeError:
            print(f"  [encoding issue, len={len(line)}]")

    return clean_lines


if __name__ == '__main__':
    main()
