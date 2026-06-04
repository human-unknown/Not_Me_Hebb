"""
viz.py —— 四面板仪表板可视化
自由能原理智能体 — M1 单智能体生存

面板:
1. 自由能时序 — F_total + F_body/F_social/F_cognitive 曲线
2. 效价-唤醒  — Valence × Arousal 散点图 + 注意力精度
3. 行为空间  — 行动分布 + 行动序列 + 位置轨迹
4. 参数空间  — Theta 参数随时间变化的热力图
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，兼容无 GUI 环境
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
import os
import time as _time

# 中文字体设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 颜色常量 (与架构文档一致)
C_L0 = '#3b82f6'   # blue
C_L1 = '#f59e0b'   # amber
C_L2 = '#10b981'   # green
C_L3 = '#8b5cf6'   # purple
C_MOE = '#06b6d4'  # cyan
C_BG = '#0a0e14'
C_CARD = '#131820'
C_BORDER = '#1e2a3a'
C_TEXT = '#c8ccd4'
C_DIM = '#7a8494'

ACTION_NAMES = {0: 'N', 1: 'S', 2: 'W', 3: 'E'}
ACTION_COLORS = {0: C_L0, 1: C_L2, 2: C_L1, 3: C_L3}


def build_dashboard(result: dict, output_path: str = None,
                    dpi: int = 150, tag: str = None) -> str:
    """生成四面板仪表板并保存为 PNG

    Args:
        result: run_episode() 的返回值
        output_path: 完整输出路径。若为 None，自动生成带时间戳的文件名
        dpi: 图片分辨率
        tag: 自定义标签，追加到文件名（如 'seed42_steps500'）

    Returns:
        output_path: 生成的图片绝对路径
    """
    # 自动生成带时间戳的文件名，放入 dashboards/ 目录
    if output_path is None:
        os.makedirs('dashboards', exist_ok=True)
        ts = _time.strftime('%Y%m%d_%H%M%S')
        if tag:
            output_path = os.path.join('dashboards', f'{tag}_{ts}.png')
        else:
            output_path = os.path.join('dashboards', f'run_{ts}.png')
    elif not os.path.dirname(output_path):
        # 只给了文件名，放入 dashboards/
        os.makedirs('dashboards', exist_ok=True)
        output_path = os.path.join('dashboards', output_path)
    # 深色主题
    plt.rcParams.update({
        'figure.facecolor': C_BG,
        'axes.facecolor': C_CARD,
        'axes.edgecolor': C_BORDER,
        'axes.labelcolor': C_TEXT,
        'text.color': C_TEXT,
        'xtick.color': C_DIM,
        'ytick.color': C_DIM,
        'grid.color': C_BORDER,
        'legend.facecolor': C_CARD,
        'legend.edgecolor': C_BORDER,
        'legend.labelcolor': C_TEXT,
    })

    fig = plt.figure(figsize=(20, 12), dpi=dpi)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.06, right=0.98, top=0.93, bottom=0.07)

    # ---- 标题 ----
    fig.suptitle('Free Energy Principle Agent — M1 Dashboard',
                 fontsize=18, fontweight='bold', color='#e8ecf2', y=0.97)

    # ---- 面板 1: 自由能时序 ----
    ax1 = fig.add_subplot(gs[0, 0])
    _panel_free_energy(ax1, result)

    # ---- 面板 2: 效价-唤醒 ----
    ax2 = fig.add_subplot(gs[0, 1])
    _panel_valence_arousal(ax2, result)

    # ---- 面板 3: 行为空间 ----
    ax3 = fig.add_subplot(gs[1, 0])
    _panel_behavior(ax3, result)

    # ---- 面板 4: 参数空间 ----
    ax4 = fig.add_subplot(gs[1, 1])
    _panel_parameters(ax4, result)

    # 保存
    fig.savefig(output_path, dpi=dpi, facecolor=C_BG, edgecolor='none',
                bbox_inches='tight')
    plt.close(fig)

    return output_path


# ============================================================
# 面板 1: 自由能时序
# ============================================================

def _panel_free_energy(ax, result):
    """F_total + 三分量随时间变化"""
    steps = np.arange(len(result['F_history']))

    # F_total 主曲线 (填充)
    ax.fill_between(steps, 0, result['F_history'],
                    alpha=0.15, color=C_L1)
    ax.plot(steps, result['F_history'],
            color=C_L1, linewidth=1.2, label='F total', alpha=0.9)

    # 三分量虚线
    if result.get('F_body_history'):
        ax.plot(steps, result['F_body_history'],
                color=C_L0, linewidth=0.6, alpha=0.6, linestyle='--', label='F body')
    if result.get('F_social_history'):
        ax.plot(steps, result['F_social_history'],
                color=C_L3, linewidth=0.6, alpha=0.6, linestyle='--', label='F social')
    if result.get('F_cognitive_history'):
        ax.plot(steps, result['F_cognitive_history'],
                color=C_MOE, linewidth=0.6, alpha=0.6, linestyle='--', label='F cognitive')

    # 移动平均 (平滑趋势)
    window = max(10, len(steps) // 20)
    if len(steps) > window:
        F_smooth = np.convolve(result['F_history'],
                               np.ones(window) / window, mode='valid')
        ax.plot(steps[window - 1:], F_smooth,
                color='white', linewidth=2.0, alpha=0.8, label=f'MA({window})')

    ax.set_title('Panel 1: Free Energy', fontsize=13, fontweight='bold',
                 color='#e8ecf2', loc='left')
    ax.set_xlabel('Step')
    ax.set_ylabel('Free Energy')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, linestyle=':')

    # 统计标注
    F_arr = np.array(result['F_history'])
    stats_text = (f"mean={np.mean(F_arr):.2f}  "
                  f"std={np.std(F_arr):.2f}\n"
                  f"min={np.min(F_arr):.2f}  "
                  f"final={F_arr[-1]:.2f}")
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            fontsize=7, color=C_DIM, va='top', ha='right',
            fontfamily='monospace')


# ============================================================
# 面板 2: 效价-唤醒散点图
# ============================================================

def _panel_valence_arousal(ax, result):
    """Valence × Arousal 散点图，按时间和注意力着色"""
    valence = np.array(result.get('valence_history', []))
    arousal = np.array(result.get('arousal_history', []))
    attention = np.array(result.get('attention_history', []))
    steps = np.arange(len(valence))

    if len(valence) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center',
                color=C_DIM, fontsize=12)
        ax.set_title('Panel 2: Valence–Arousal', fontsize=13, fontweight='bold',
                     color='#e8ecf2', loc='left')
        return

    # 散点：颜色=时间步，大小=注意力精度
    colors = steps
    sizes = 10 + attention * 60  # 注意力越高点越大
    sc = ax.scatter(valence, arousal, c=colors, s=sizes,
                    cmap='plasma', alpha=0.7, edgecolors='none',
                    norm=Normalize(vmin=0, vmax=max(1, len(steps))))

    # 颜色条
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('Step', fontsize=7, color=C_DIM)
    cbar.ax.tick_params(colors=C_DIM, labelsize=6)

    # 象限线
    ax.axhline(y=0.5, color=C_BORDER, linewidth=0.8, linestyle=':')
    ax.axvline(x=0.0, color=C_BORDER, linewidth=0.8, linestyle=':')

    # 象限标签
    ax.text(0.5, 0.95, 'High Arousal\nPositive', fontsize=6, color=C_L0,
            ha='center', va='top', alpha=0.7)
    ax.text(-0.5, 0.95, 'High Arousal\nNegative', fontsize=6, color=C_L1,
            ha='center', va='top', alpha=0.7)
    ax.text(0.5, 0.05, 'Calm\nPositive', fontsize=6, color=C_L2,
            ha='center', va='bottom', alpha=0.7)
    ax.text(-0.5, 0.05, 'Calm\nNegative', fontsize=6, color=C_L3,
            ha='center', va='bottom', alpha=0.7)

    ax.set_title('Panel 2: Valence–Arousal Space', fontsize=13, fontweight='bold',
                 color='#e8ecf2', loc='left')
    ax.set_xlabel('Valence [−1, +1]')
    ax.set_ylabel('Arousal [0, 1]')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3, linestyle=':')

    # 统计
    ax.text(0.98, 0.05,
            f"mean V={np.mean(valence):+.2f}  A={np.mean(arousal):.2f}\n"
            f"attn={np.mean(attention):.2f}",
            transform=ax.transAxes, fontsize=7, color=C_DIM,
            va='bottom', ha='right', fontfamily='monospace')


# ============================================================
# 面板 3: 行为空间 (三合一: 直方图 + 序列 + 轨迹)
# ============================================================

def _panel_behavior(ax, result):
    """行动分布 + 行动序列片段 + 位置轨迹 (小图)"""
    actions = np.array(result['actions'])
    pos_history = result.get('pos_history', [])

    # --- 3a: 行动分布直方图 (左半) ---
    ax_hist = ax.inset_axes([0.05, 0.55, 0.42, 0.40])
    counts = np.bincount(actions, minlength=4)
    bars = ax_hist.bar(range(4), counts,
                       color=[ACTION_COLORS[i] for i in range(4)],
                       alpha=0.8, edgecolor=C_BORDER, linewidth=0.5)
    ax_hist.set_xticks(range(4))
    ax_hist.set_xticklabels(['N', 'S', 'W', 'E'], fontsize=7)
    ax_hist.set_ylabel('Count', fontsize=7, color=C_DIM)
    ax_hist.set_title('Action Distribution', fontsize=8, color=C_DIM, loc='left')
    ax_hist.tick_params(colors=C_DIM, labelsize=6)
    ax_hist.set_facecolor(C_BG)
    for bar, count in zip(bars, counts):
        ax_hist.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(count), ha='center', fontsize=7, color=C_TEXT)

    # --- 3b: 行动序列 (右下) ---
    ax_seq = ax.inset_axes([0.52, 0.55, 0.45, 0.40])
    seq_len = min(200, len(actions))
    seq = actions[-seq_len:]
    ax_seq.plot(range(seq_len), seq, color=C_DIM, linewidth=0.4, alpha=0.5)
    # 用散点标记行动变化
    for i in range(seq_len):
        ax_seq.scatter(i, seq[i], color=ACTION_COLORS.get(seq[i], C_DIM),
                       s=3, alpha=0.6)
    ax_seq.set_ylim(-0.5, 3.5)
    ax_seq.set_yticks(range(4))
    ax_seq.set_yticklabels(['N', 'S', 'W', 'E'], fontsize=6)
    ax_seq.set_xlabel(f'Last {seq_len} steps', fontsize=7, color=C_DIM)
    ax_seq.set_title('Action Sequence', fontsize=8, color=C_DIM, loc='left')
    ax_seq.tick_params(colors=C_DIM, labelsize=6)
    ax_seq.set_facecolor(C_BG)
    ax_seq.grid(True, alpha=0.2, linestyle=':')

    # --- 3c: 位置轨迹 (下半全宽) ---
    if pos_history:
        ax_pos = ax.inset_axes([0.05, 0.05, 0.92, 0.43])
        pos = np.array(pos_history)
        # 颜色映射时间
        t_norm = Normalize(vmin=0, vmax=len(pos) - 1)
        for i in range(len(pos) - 1):
            ax_pos.plot(pos[i:i + 2, 0], pos[i:i + 2, 1],
                        color=plt.cm.viridis(t_norm(i)), linewidth=0.5, alpha=0.7)
        ax_pos.scatter(pos[0, 0], pos[0, 1], color=C_L2, s=30, marker='o',
                       label='Start', zorder=5)
        ax_pos.scatter(pos[-1, 0], pos[-1, 1], color=C_L1, s=30, marker='X',
                       label='End', zorder=5)
        ax_pos.set_xlim(-0.5, 10.5)
        ax_pos.set_ylim(-0.5, 10.5)
        ax_pos.set_xlabel('X', fontsize=7, color=C_DIM)
        ax_pos.set_ylabel('Y', fontsize=7, color=C_DIM)
        ax_pos.set_title('Position Trajectory', fontsize=8, color=C_DIM, loc='left')
        ax_pos.legend(fontsize=6, loc='upper right')
        ax_pos.tick_params(colors=C_DIM, labelsize=6)
        ax_pos.set_aspect('equal')
        ax_pos.set_facecolor(C_BG)
        ax_pos.grid(True, alpha=0.3, linestyle=':')

    ax.set_title('Panel 3: Behavior Space', fontsize=13, fontweight='bold',
                 color='#e8ecf2', loc='left')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(C_BG)

    # 收尾统计
    total_r = sum(result['rewards'])
    ax.text(0.98, 0.02, f"Total Reward: {total_r:+.1f}  |  "
            f"Clusters: {result.get('n_clusters', '?')}  |  "
            f"Entropy: {_action_entropy(actions):.2f}",
            transform=ax.transAxes, fontsize=7, color=C_DIM,
            va='bottom', ha='right', fontfamily='monospace')


def _action_entropy(actions: np.ndarray) -> float:
    """行动分布的香农熵"""
    counts = np.bincount(actions, minlength=4)
    probs = counts / (counts.sum() + 1e-8)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ============================================================
# 面板 4: 参数空间
# ============================================================

def _panel_parameters(ax, result):
    """Theta 参数热力图 / 发育轨迹"""
    snapshots = result.get('theta_snapshots', [])

    # 可展示的参数（数值型）
    DISPLAY_PARAMS = [
        'sigma_z', 'sigma_x', 'decay_rate', 'cluster_threshold', 'learn_rate_l0',
        'w_body', 'w_social', 'w_cognitive',
        'eta_valence', 'eta_arousal', 'habituation_tau',
        'gamma', 'exploration_bonus', 'temperature', 'urgency_weight',
        'meta_lr', 'plasticity_decay',
    ]
    N_PARAMS = len(DISPLAY_PARAMS)

    if len(snapshots) <= 1:
        # M1: 参数不变 → 用柱状图展示当前值
        _panel_params_static(ax, result, DISPLAY_PARAMS)
    else:
        # M5: 参数随时间变化 → 用热力图
        _panel_params_heatmap(ax, snapshots, DISPLAY_PARAMS)

    ax.set_title('Panel 4: Parameter Space (Theta)', fontsize=13, fontweight='bold',
                 color='#e8ecf2', loc='left')


def _panel_params_static(ax, result, params):
    """静态参数：横向柱状图展示初始化值"""
    theta_dict = result.get('theta_snapshots', [{}])
    if theta_dict:
        values = [theta_dict[0].get(p, 0) for p in params]
    else:
        # Fallback: 直接从 Theta 默认值
        from layer3_meta import create_default_theta
        t = create_default_theta().to_dict()
        values = [t.get(p, 0) for p in params]

    y_pos = range(len(params))

    # 归一化到 [0, 1] 以颜色编码
    vals_arr = np.array(values)
    v_norm = (vals_arr - vals_arr.min()) / (vals_arr.max() - vals_arr.min() + 1e-8)

    colors = plt.cm.Spectral(v_norm)
    bars = ax.barh(y_pos, vals_arr, color=colors, alpha=0.85,
                   edgecolor=C_BORDER, linewidth=0.3, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(params, fontsize=6.5, fontfamily='monospace')
    ax.set_xlabel('Value', fontsize=7, color=C_DIM)
    ax.invert_yaxis()
    ax.tick_params(colors=C_DIM, labelsize=6)
    ax.grid(True, alpha=0.2, linestyle=':', axis='x')

    # 每条柱子末尾标数值
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_width() + 0.01 * max(vals_arr), bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}'.rstrip('0').rstrip('.'),
                fontsize=5.5, color=C_DIM, va='center', fontfamily='monospace')

    ax.text(0.98, 0.02, 'M1: static (no meta-learning)',
            transform=ax.transAxes, fontsize=7, color=C_DIM,
            va='bottom', ha='right')


def _panel_params_heatmap(ax, snapshots, params):
    """动态参数：热力图展示参数随时间的演化"""
    steps = [s['step'] for s in snapshots]
    data = np.array([[s.get(p, np.nan) for p in params] for s in snapshots])

    # Z-score 归一化 (按列)
    data_norm = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        if np.std(col) > 1e-10:
            data_norm[:, j] = (col - np.mean(col)) / np.std(col)
        else:
            data_norm[:, j] = 0.0

    im = ax.imshow(data_norm.T, aspect='auto', cmap='RdBu_r',
                   origin='upper', interpolation='nearest')

    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params, fontsize=6, fontfamily='monospace')
    ax.set_xlabel('Snapshot index', fontsize=7, color=C_DIM)

    # x 轴标签（抽样展示步数）
    n_xticks = min(10, len(steps))
    if len(steps) > 1:
        tick_indices = np.linspace(0, len(steps) - 1, n_xticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([str(steps[i]) for i in tick_indices],
                           fontsize=6, rotation=45)
    ax.tick_params(colors=C_DIM, labelsize=6)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('Z-score', fontsize=7, color=C_DIM)
    cbar.ax.tick_params(colors=C_DIM, labelsize=6)


# ============================================================
# 便捷入口：从 Agent 和 World 直接生成
# ============================================================

def quick_dashboard(agent, world, output_path: str = None,
                    dpi=150, tag: str = None) -> str:
    """从 Agent 和 World 对象直接生成仪表板"""
    result = {
        'F_history': agent.F_history,
        'F_body_history': agent.F_body_history,
        'F_social_history': agent.F_social_history,
        'F_cognitive_history': agent.F_cognitive_history,
        'valence_history': agent.valence_history,
        'arousal_history': agent.arousal_history,
        'attention_history': agent.attention_history,
        'actions': agent.action_history,
        'rewards': agent.reward_history,
        'pos_history': [],  # 需要外部传入
        'n_clusters': agent.net.n_clusters,
        'theta_snapshots': agent.theta_snapshots,
    }
    return build_dashboard(result, output_path, dpi, tag)


def dashboard_from_episode(result: dict, output_path: str = None,
                           dpi=150, tag: str = None) -> str:
    """从 run_episode 返回值生成仪表板（推荐）"""
    return build_dashboard(result, output_path, dpi, tag)


def dashboard_multi(result: dict, output_path: str = None,
                    dpi=150, tag: str = None) -> str:
    """从 run_episode_multi 返回值生成多智能体仪表板 (M3)

    面板:
    1. 各 agent F_total 时序对比
    2. 各 agent F_social 时序
    3. 各 agent 累计奖励 + 信任度
    4. 各 agent 簇数 + 最终信任矩阵
    """
    if output_path is None:
        os.makedirs('dashboards', exist_ok=True)
        ts = _time.strftime('%Y%m%d_%H%M%S')
        prefix = f'{tag}_' if tag else 'm3_'
        output_path = os.path.join('dashboards', f'{prefix}{ts}.png')

    plt.rcParams.update({
        'figure.facecolor': C_BG, 'axes.facecolor': C_CARD,
        'axes.edgecolor': C_BORDER, 'axes.labelcolor': C_TEXT,
        'text.color': C_TEXT, 'xtick.color': C_DIM, 'ytick.color': C_DIM,
        'grid.color': C_BORDER, 'legend.facecolor': C_CARD,
        'legend.edgecolor': C_BORDER, 'legend.labelcolor': C_TEXT,
    })

    fig = plt.figure(figsize=(20, 12), dpi=dpi)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.06, right=0.98, top=0.93, bottom=0.07)

    n_agents = result.get('n_agents', 1)
    w_social = result.get('w_social', 1.0)
    agent_colors = [C_L0, C_L2, C_L3]  # blue, green, purple

    fig.suptitle(f'FEP Multi-Agent Dashboard (n={n_agents}, w_social={w_social})',
                 fontsize=16, fontweight='bold', color='#e8ecf2', y=0.97)

    # ---- Panel 1: F_total per agent ----
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(n_agents):
        F_hist = result['per_agent']['F_history'][i]
        steps = np.arange(len(F_hist))
        ax1.plot(steps, F_hist, color=agent_colors[i],
                linewidth=1.0, alpha=0.8, label=f'Agent {i}')
        # 移动平均
        window = max(10, len(steps) // 20)
        if len(steps) > window:
            F_smooth = np.convolve(F_hist, np.ones(window) / window, mode='valid')
            ax1.plot(steps[window - 1:], F_smooth, color=agent_colors[i],
                    linewidth=1.8, alpha=0.5, linestyle='--')
    ax1.set_title('Panel 1: F_total per Agent', fontsize=13, fontweight='bold',
                 color='#e8ecf2', loc='left')
    ax1.set_xlabel('Step'); ax1.set_ylabel('F_total')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle=':')

    # ---- Panel 2: F_social per agent ----
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(n_agents):
        Fs_hist = result['per_agent']['F_social_history'][i]
        steps = np.arange(len(Fs_hist))
        ax2.plot(steps, Fs_hist, color=agent_colors[i],
                linewidth=1.0, alpha=0.8, label=f'Agent {i}')
    ax2.set_title('Panel 2: F_social per Agent', fontsize=13, fontweight='bold',
                 color='#e8ecf2', loc='left')
    ax2.set_xlabel('Step'); ax2.set_ylabel('F_social')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle=':')

    # ---- Panel 3: Cumulative Reward + Trust ----
    ax3 = fig.add_subplot(gs[1, 0])
    # 累计奖励
    for i in range(n_agents):
        rewards = result['per_agent']['rewards'][i]
        cum_reward = np.cumsum(rewards)
        ax3.plot(np.arange(len(cum_reward)), cum_reward,
                color=agent_colors[i], linewidth=1.2, alpha=0.8,
                label=f'Agent {i} reward')
    # 信任度 inset
    ax3_trust = ax3.inset_axes([0.55, 0.55, 0.42, 0.40])
    trust_data = result['per_agent']['trust_levels']
    for i in range(n_agents):
        if trust_data[i]:
            trust_vals = list(trust_data[i].values())
            trust_labels = [f'A{j}' for j in trust_data[i].keys()]
            ax3_trust.bar(np.arange(len(trust_vals)) + i * 0.25, trust_vals,
                         width=0.2, color=agent_colors[i], alpha=0.8,
                         label=f'A{i}')
    ax3_trust.set_xticks(range(len(trust_labels)))
    ax3_trust.set_xticklabels(trust_labels, fontsize=6)
    ax3_trust.set_ylim(0, 1); ax3_trust.set_ylabel('Trust', fontsize=6, color=C_DIM)
    ax3_trust.set_title('Trust Matrix', fontsize=7, color=C_DIM)
    ax3_trust.tick_params(colors=C_DIM, labelsize=6)
    ax3_trust.set_facecolor(C_BG)
    ax3.set_title('Panel 3: Reward & Trust', fontsize=13, fontweight='bold',
                 color='#e8ecf2', loc='left')
    ax3.set_xlabel('Step'); ax3.set_ylabel('Cumulative Reward')
    ax3.legend(fontsize=7, loc='upper left')
    ax3.grid(True, alpha=0.3, linestyle=':')

    # ---- Panel 4: Clusters + Summary Stats ----
    ax4 = fig.add_subplot(gs[1, 1])
    clusters = result['per_agent']['n_clusters']
    bars = ax4.bar(range(n_agents), clusters,
                  color=agent_colors[:n_agents], alpha=0.8,
                  edgecolor=C_BORDER, linewidth=0.5)
    for i, (bar, c) in enumerate(zip(bars, clusters)):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(c), ha='center', fontsize=10, color=C_TEXT)
    ax4.set_xticks(range(n_agents))
    ax4.set_xticklabels([f'Agent {i}' for i in range(n_agents)], fontsize=9)
    ax4.set_ylabel('Clusters', fontsize=9)
    ax4.set_title('Panel 4: Clusters & Summary', fontsize=13, fontweight='bold',
                 color='#e8ecf2', loc='left')
    ax4.grid(True, alpha=0.3, linestyle=':', axis='y')

    # 摘要文本
    total_rewards = result.get('total_rewards', [])
    summary = (f"w_social={w_social}  |  n_agents={n_agents}\n"
               f"Rewards: {' | '.join(f'A{i}={r:+.1f}' for i, r in enumerate(total_rewards))}\n"
               f"Total: {sum(total_rewards):+.1f}  |  Mean F: {result.get('mean_F', 0):.2f}\n"
               f"Clusters: {' | '.join(f'A{i}={c}' for i, c in enumerate(clusters))}")
    ax4.text(0.05, 0.5, summary, transform=ax4.transAxes,
            fontsize=8, color=C_DIM, va='center', fontfamily='monospace')

    fig.savefig(output_path, dpi=dpi, facecolor=C_BG, edgecolor='none',
                bbox_inches='tight')
    plt.close(fig)
    return output_path


# ============================================================
# 命令行测试
# ============================================================

if __name__ == '__main__':
    # 生成示例数据测试
    rng = np.random.default_rng(42)
    steps = 500

    # 合成数据
    F = 2.0 * np.exp(-np.arange(steps) / 200) + 0.3 + 0.1 * rng.normal(0, 1, steps)
    demo = {
        'F_history': F.tolist(),
        'F_body_history': (F * 0.7).tolist(),
        'F_social_history': np.zeros(steps).tolist(),
        'F_cognitive_history': (F * 0.3).tolist(),
        'valence_history': np.tanh(-0.5 * F).tolist(),
        'arousal_history': np.tanh(0.5 * np.abs(F)).tolist(),
        'attention_history': (0.3 + 0.2 * rng.random(steps)).tolist(),
        'actions': rng.integers(0, 4, steps).tolist(),
        'rewards': (rng.random(steps) * 2 - 0.5).tolist(),
        'pos_history': np.cumsum(rng.normal(0, 0.3, (steps, 2)), axis=0).tolist(),
        'n_clusters': 5,
        'theta_snapshots': [{'step': 0,
                             'sigma_z': 0.1, 'sigma_x': 1.0, 'decay_rate': 0.01,
                             'cluster_threshold': 0.85, 'learn_rate_l0': 0.05,
                             'w_body': 1.0, 'w_social': 1.0, 'w_cognitive': 1.0,
                             'eta_valence': 0.5, 'eta_arousal': 0.5,
                             'habituation_tau': 10.0, 'gamma': 0.95,
                             'exploration_bonus': 0.1, 'temperature': 1.0,
                             'urgency_weight': 0.3, 'meta_lr': 0.01,
                             'plasticity_decay': 0.999}],
    }

    path = build_dashboard(demo, tag='demo')
    print(f"Demo dashboard saved to: {os.path.abspath(path)}")
