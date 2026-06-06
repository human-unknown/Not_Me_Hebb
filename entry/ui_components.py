"""
ui_components.py — Rich 终端渲染组件 (v5.7)

提供 Claude Code 风格的终端 UI:
  - Header: 常驻顶栏, Agent 核心状态
  - StatusPanels: /status 展开的五面板
  - DiagReport: /diag 系统连接性报告
  - HelpPanel: /help 命令列表
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich import box
import numpy as np

console = Console()

# ================================================================
# 颜色映射
# ================================================================

def valence_color(v: float) -> str:
    """效价 → 颜色."""
    if v > 0.3: return "green"
    elif v > 0.05: return "green"
    elif v < -0.3: return "red"
    elif v < -0.05: return "red"
    return "yellow"

def valence_emoji(v: float) -> str:
    """效价 → emoji."""
    if v > 0.3: return "😊"
    elif v > 0.05: return "🙂"
    elif v < -0.3: return "😢"
    elif v < -0.05: return "😐"
    return "😐"

def bar_chart(value: float, width: int = 10, color: str = "green") -> str:
    """简易条形图."""
    filled = int(np.clip(value, 0, 1) * width)
    empty = width - filled
    return f"[{color}]{'█' * filled}{'░' * empty}[/{color}]"

def fmt_val(v: float, decimals: int = 2) -> str:
    """格式化数值, 带颜色."""
    color = "green" if v >= 0 else "red"
    return f"[{color}]{v:+.{decimals}f}[/{color}]"


# ================================================================
# Header
# ================================================================

def render_header(agent, meta: dict = None, broca=None, age: int = 0) -> Panel:
    """渲染顶部状态栏.

    Args:
        agent: Agent 实例
        meta: 持久化元数据 (session #, turns, etc.)
        broca: Broca 实例 (可选, 用于显示 trigram 网络在线生长状态)
        age: 发育年龄 0-3 (v5.7)
    """
    v = agent.valence_history[-1] if agent.valence_history else 0.0
    a = agent.arousal_history[-1] if agent.arousal_history else 0.5
    F = agent.F_history[-1] if agent.F_history else 0.0
    n_clusters = agent.net.n_clusters

    # Session info
    n_sessions = meta.get('n_sessions', 1) if meta else 1
    n_turns = meta.get('total_turns', 0) if meta else 0

    # Age display
    age_names = {0: '👶', 1: '🧒', 2: '🧑', 3: '🧠'}
    age_emoji = age_names.get(age, '🧠')

    parts = [
        f"[bold cyan]NotMe v5.7[/bold cyan]",
        f"Age:{age}{age_emoji}",
        f"Session #{n_sessions}",
        f"{n_turns} turns",
        f"F={F:.2f}",
        f"V={fmt_val(v)} {valence_emoji(v)}",
        f"A={a:.2f}",
        f"Mem: {n_clusters} clusters",
    ]
    # v5.7: 显示 trigram 网络在线生长状态
    if broca is not None:
        n_tri = broca.word_order_net.n_clusters
        n_sent = len(broca.sentences)
        parts.append(f"Tri: {n_tri}")
        if n_sent > 0:
            parts.append(f"Sent: {n_sent}")
    # Build header with alternating separator
    header_text = Text()
    for i, p in enumerate(parts):
        if i > 0:
            header_text.append("  │  ", style="dim")
        header_text.append(p)
    return Panel(header_text, box=box.HEAVY, border_style="cyan", padding=(0, 2))


# ================================================================
# Status Panels (/status)
# ================================================================

def render_language_panel(agent, understanding: dict = None,
                          speech_diag: dict = None) -> Panel:
    """语言系统面板."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("key", style="dim", width=18)
    table.add_column("value")

    # N400/P600
    if agent.language_pe_history:
        pe = agent.language_pe_history[-1]
        table.add_row("N400 (semantic PE)", f"{pe.get('semantic_pe', 0):.3f}")
        table.add_row("P600 (syntactic PE)", f"{pe.get('syntactic_pe', 0):.3f}")
        table.add_row("F_language", f"{pe.get('F_language', 0):.3f}")

    # TPJ
    if understanding:
        tpj_info = understanding.get('tpj_inference', {})
        if tpj_info:
            intent = tpj_info.get('intent', 'unknown')
            conf = tpj_info.get('confidence', 0)
            table.add_row("TPJ intent", f"{intent} (conf={conf:.2f})")
        table.add_row("Pragmatic enriched",
                      str(understanding.get('pragmatic_enriched', False)))

    # Angular Gyrus
    if understanding:
        ag_conf = understanding.get('ag_confidence', 0)
        ag_norm = understanding.get('ag_phon_norm', 0)
        if ag_conf > 0:
            table.add_row("AngularGyrus",
                          f"ph_norm={ag_norm:.2f} conf={ag_conf:.2f}"
                          f" weight={understanding.get('ag_weight', 0):.2f}")

    # Phonological Loop
    if hasattr(agent, 'phonological_loop'):
        pl = agent.phonological_loop
        pl_state = pl.get_state()
        n_chunks = pl_state.get('n_chunks', 0)
        table.add_row("Phonological Loop",
                      f"{bar_chart(n_chunks/7, 7)} {n_chunks}/7 chunks")

    # Phrase Structure
    if speech_diag:
        ps_used = speech_diag.get('phrase_structure_used', False)
        boundaries = speech_diag.get('phrase_boundaries', 0)
        if ps_used:
            table.add_row("Phrase Structure",
                          f"active ({boundaries} boundaries)")

    # AF + Motor
    if speech_diag:
        af_conf = speech_diag.get('af_confidence', 0)
        motor = speech_diag.get('motor_plan_executed', False)
        table.add_row("AF ventral", f"seed={af_conf > 0.1} ({af_conf:.2f})")
        table.add_row("Motor Cortex",
                      f"plan={motor} ({speech_diag.get('motor_plan_length', 0)} steps)")

    return Panel(table, title="[bold]Language Pipeline[/bold]",
                 border_style="blue", padding=(0, 1))


def render_perception_panel(agent) -> Panel:
    """感知系统面板 (v5.7: 视觉+听觉常开, 含处理模式)."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("key", style="dim", width=16)
    table.add_column("value")

    # Visual — v5.7: 常开 (真实图像/跨模态补全/语义代理)
    vis_result = getattr(agent, '_current_visual_result', {}) or {}
    if vis_result:
        diag = vis_result.get('diagnostics', {})
        pe = vis_result.get('PE_total', 0)
        v1_norm = diag.get('V1_mean_norm', 0)
        v2_norm = diag.get('V2_mean_norm', 0)
        v4_norm = diag.get('V4_mean_norm', 0)
        it_norm = diag.get('IT_mean_norm', 0)
        mode = diag.get('mode', 'real')
        mode_tag = {'real': '', 'crossmodal_fill': '[dim](cross)[/dim]',
                    'semantic_proxy': '[dim](proxy)[/dim]',
                    'no_input': '[dim](idle)[/dim]',
                    'error': '[red](err)[/red]'}.get(mode, '')
        table.add_row("Vision",
                      f"V1={v1_norm:.2f} V2={v2_norm:.2f} "
                      f"V4={v4_norm:.2f} IT={it_norm:.2f} PE={pe:.3f} "
                      f"{mode_tag}")
    else:
        table.add_row("Vision", "[dim]inactive[/dim]")

    # Auditory — v5.2: 常开 (真实音频/语义代理)
    aud_result = getattr(agent, '_current_auditory_result', {}) or {}
    if aud_result:
        diag = aud_result.get('diagnostics', {})
        cn = diag.get('CN_mean_norm', 0)
        soc = diag.get('SOC_ITD_std', 0)
        ic = diag.get('IC_mean_norm', 0)
        ac = diag.get('AC_mean_norm', 0)
        mode = diag.get('mode', 'real')
        mode_tag = {'real': '', 'error': '[red](err)[/red]'}.get(mode, '')
        table.add_row("Audio",
                      f"CN={cn:.2f} SOC={soc:.2f} IC={ic:.2f} AC={ac:.2f}"
                      f"{' ' + mode_tag if mode_tag else ''}")
    else:
        table.add_row("Audio", "[dim]inactive[/dim]")

    # Pain — v5.4: 常开 (伤害性+触觉闸门)
    pain_result = getattr(agent, '_current_pain_result', {}) or {}
    if pain_result:
        pi = pain_result.get('pain_intensity', 0)
        diag = pain_result.get('diagnostics', {})
        dh = diag.get('DH_gate_output', 0)
        table.add_row("Pain",
                      f"DH={dh:.2f} intensity={pi:.2f}"
                      f" allodynia={pain_result.get('allodynia', False)}"
                      f" hyper={pain_result.get('hyperalgesia', False)}")
    else:
        table.add_row("Pain", "[dim]none[/dim]")

    return Panel(table, title="[bold]Perception[/bold]",
                 border_style="magenta", padding=(0, 1))


def render_neuromod_panel(agent) -> Panel:
    """神经调节面板."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("key", style="dim", width=16)
    table.add_column("value")

    # Hypothalamus
    hypo = getattr(agent, '_hypo_result', {}) or {}
    if hypo:
        drive = hypo.get('total_drive', 0)
        hpa = hypo.get('hpa_activation', 0)
        bal = hypo.get('autonomic_balance', 0)
        symp = "SNS↑" if bal > 0.05 else ("PNS↑" if bal < -0.05 else "balanced")
        table.add_row("Hypothalamus",
                      f"drive={drive:.2f} HPA={hpa:.2f} {symp}")

    # VTA
    vta_r = getattr(agent, '_vta_result', {}) or {}
    if vta_r:
        rpe = vta_r.get('rpe', 0)
        da = vta_r.get('total_da', 0.3)
        lr_mul = vta_r.get('learn_rate_multiplier', 1.0)
        mot = vta_r.get('motivation', 0.5)
        table.add_row("VTA",
                      f"RPE={fmt_val(rpe)} DA={da:.2f} "
                      f"lr×{lr_mul:.2f} mot={mot:.2f}")

    # LC
    lc_r = getattr(agent, '_lc_result', {}) or {}
    if lc_r:
        tonic = lc_r.get('tonic_ne', 0.2)
        phasic = lc_r.get('phasic_ne', 0)
        snr = lc_r.get('snr_gain', 1.0)
        yd = lc_r.get('yd_performance', 0.5)
        table.add_row("LC",
                      f"tonic={tonic:.2f} phasic={phasic:.2f} "
                      f"SNR={snr:.2f} YD={yd:.2f}")

    return Panel(table, title="[bold]Neuromodulation[/bold]",
                 border_style="yellow", padding=(0, 1))


def render_body_panel(agent) -> Panel:
    """身体状态面板."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("key", style="dim", width=16)
    table.add_column("value")

    body = agent.body
    b = body.b
    labels = ["social", "energy", "stress", "novelty",
              "focus", "visual", "auditory", "cognitive", "tissue"]

    row_parts = []
    for i in range(9):
        if i < len(b):
            val = float(b[i])
            label = labels[i] if i < len(labels) else f"b{i}"
            row_parts.append(f"{label}={val:.2f}")
    table.add_row("Body", "  ".join(row_parts[:5]))
    if len(row_parts) > 5:
        table.add_row("", "  ".join(row_parts[5:]))

    # Setpoints
    sp = body.setpoints
    if sp is not None and len(sp) >= 9:
        sp_parts = [f"{labels[i]}={sp[i]:.2f}" for i in range(min(9, len(sp)))]
        table.add_row("Setpoints", "  ".join(sp_parts[:5]))
        if len(sp_parts) > 5:
            table.add_row("", "  ".join(sp_parts[5:]))

    deviation = body.compute_deviation()
    table.add_row("Deviation", f"{deviation:.3f}")

    return Panel(table, title="[bold]Body State[/bold]",
                 border_style="green", padding=(0, 1))


def render_free_energy_panel(agent) -> Panel:
    """自由能面板."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("key", style="dim", width=16)
    table.add_column("value")

    F = agent.F_history[-1] if agent.F_history else 0.0
    Fb = agent.F_body_history[-1] if agent.F_body_history else 0.0
    Fs = agent.F_social_history[-1] if agent.F_social_history else 0.0
    Fc = agent.F_cognitive_history[-1] if agent.F_cognitive_history else 0.0
    Fa = agent.F_accuracy_history[-1] if agent.F_accuracy_history else 0.0
    Fl = agent.F_language_history[-1] if agent.F_language_history else 0.0
    v = agent.valence_history[-1] if agent.valence_history else 0.0
    a = agent.arousal_history[-1] if agent.arousal_history else 0.5
    att = agent.attention_history[-1] if agent.attention_history else 0.5

    table.add_row("F_total", f"{F:.3f}")
    table.add_row("F_components",
                  f"body={Fb:.3f} social={Fs:.3f} cog={Fc:.3f} "
                  f"acc={Fa:.3f} lang={Fl:.3f}")
    table.add_row("Valence",
                  f"[{valence_color(v)}]{v:+.2f}[/{valence_color(v)}] "
                  f"{valence_emoji(v)}")
    table.add_row("Arousal", f"{a:.2f}")
    table.add_row("Attention", f"{att:.2f}")

    # TPN/DMN
    if hasattr(agent, 'tpn'):
        tpn_act = agent.tpn.tpn_activation
        dmn_act = agent.tpn.dmn_activation
        fatigue = agent.tpn.task_fatigue
        table.add_row("TPN↔DMN",
                      f"TPN={tpn_act:.2f} {bar_chart(tpn_act, 10, 'blue')}  "
                      f"DMN={dmn_act:.2f} {bar_chart(dmn_act, 10, 'cyan')}")
        table.add_row("Fatigue", f"{fatigue:.2f}")

    return Panel(table, title="[bold]Free Energy[/bold]",
                 border_style="red", padding=(0, 1))


def render_all_status_panels(agent, understanding: dict = None,
                             speech_diag: dict = None):
    """渲染所有状态面板."""
    console.print()
    console.print(render_language_panel(agent, understanding, speech_diag))
    console.print(render_perception_panel(agent))
    console.print(render_neuromod_panel(agent))
    console.print(render_body_panel(agent))
    console.print(render_free_energy_panel(agent))
    console.print()


# ================================================================
# Diag Report (/diag)
# ================================================================

def render_diag(agent) -> Panel:
    """系统连接性诊断报告."""
    lines = []
    def check(name, condition):
        status = "[green]OK[/green]" if condition else "[red]MISSING[/red]"
        lines.append(f"  {name}: {status}")

    # Core
    check("Hippocampus (L0)", agent.net.n_clusters >= 0)
    check("Habituation (L1)", agent.hab is not None)
    check("MoEGate (L2)", agent.moe is not None)
    check("MetaLearner (L3)", agent.meta is not None)
    lines.append("")

    # Perception
    check("Visual Hierarchy", hasattr(agent, 'visual_hierarchy'))
    check("Auditory Hierarchy", hasattr(agent, 'auditory_hierarchy'))
    check("Nociception", hasattr(agent, 'nociception_hierarchy'))
    lines.append("")

    # Neuromodulation
    check("Hypothalamus", hasattr(agent, 'hypothalamus'))
    check("VTA (Dopamine)", hasattr(agent, 'vta'))
    check("LC (Norepinephrine)", hasattr(agent, 'locus_coeruleus'))
    check("LC→RVM connection", hasattr(agent, 'locus_coeruleus')
         and hasattr(agent, 'nociception_hierarchy'))
    lines.append("")

    # Language
    check("Wernicke (comprehend)", True)
    check("TPJ (pragmatics)", hasattr(agent, 'tpj'))
    check("AF ventral (W→B)", hasattr(agent, 'arcuate_fasciculus'))
    check("Broca (speak)", True)
    check("Phrase Structure", hasattr(agent, 'phrase_structure'))
    check("Motor Cortex", hasattr(agent, 'motor_cortex'))
    check("AF dorsal (B→Motor)", hasattr(agent, 'arcuate_fasciculus'))
    check("Phonological Loop", hasattr(agent, 'phonological_loop'))
    check("Angular Gyrus (reading)", hasattr(agent, 'angular_gyrus'))
    lines.append("")

    # Attention
    check("FPN (attention)", hasattr(agent, 'fpn'))
    check("TPN↔DMN (seesaw)", hasattr(agent, 'tpn'))
    check("DMN (self-model)", hasattr(agent, 'self_model'))

    text = "\n".join(lines)
    return Panel(text, title="[bold]System Connectivity[/bold]",
                 border_style="cyan")


# ================================================================
# Help (/help)
# ================================================================

def render_help() -> Panel:
    """帮助面板."""
    commands = [
        ("/status", "展开完整状态面板"),
        ("/diag", "系统连接性诊断"),
        ("/memory", "Hebb 网络统计"),
        ("/body <dim> <val>", "手动调制身体状态 (0-8, 0-1)"),
        ("/pain <0-1>", "施加痛觉刺激"),
        ("/touch <0-1>", "Aβ触觉输入 (闸门控制)"),
        ("/speaker <name>", "设置说话人 (激活 TPJ)"),
        ("/read <text>", "纯角回阅读 (跳过 MiniLM)"),
        ("/stream start|stop|status", "摄像头+麦克风实时流"),
        ("/age [0-3]", "发育年龄 (0=婴儿 1=儿童 2=青少年 3=成人)"),
        ("/save [name]", "手动存档"),
        ("/load <name>", "加载存档"),
        ("/reset", "重置 Agent (确认后)"),
        ("/help", "显示此帮助"),
    ]
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("cmd", style="bold cyan", width=22)
    table.add_column("desc")
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    return Panel(table, title="[bold]Commands[/bold]", border_style="cyan")
