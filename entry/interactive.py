"""
interactive.py — NotMe v5.7 全系统交互模式 (Claude Code 风格终端 UI)

特性:
  - Rich 终端渲染 (面板/表格/颜色)
  - prompt_toolkit 输入 (自动补全/历史/斜杠命令)
  - 多模态同步输入 (文本+视觉+听觉+痛觉同时激活)
  - 会话持久化 (自动保存/加载)
  - 所有 v5.6/v5.7 模块全部激活 (TPJ/AngularGyrus/PhraseStructure)

启动:
  python interactive.py                # 新会话 (或加载最新存档)
  python interactive.py --fresh        # 强制全新会话
  python interactive.py --load <name>  # 加载指定存档
"""

import sys
import os
import time
import traceback
import numpy as np

# Ensure project root is on path (works whether run from root or entry/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---- Rich UI ----
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()

# ---- prompt_toolkit ----
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style

# ---- NotMe core ----
from cns.data_types import D, BodyVector, Action, ACTION_DIRECTIONS
from cns.agent import Agent
from cns.persistence import latest_save, list_saves, auto_save

# ---- Multi-modal input ----
from entry.input_bus import InputBus

# ---- UI Components ----
from entry.ui_components import (
    render_header, render_all_status_panels, render_diag, render_help,
    valence_sign, fmt_val, valence_color,
)


# ================================================================
# 命令处理器
# ================================================================

class CommandHandler:
    """处理斜杠命令和非对话输入."""

    def __init__(self, interactive):
        self.interactive = interactive  # InteractiveSession ref

    def handle(self, text: str) -> bool:
        """处理命令, 返回 True 表示已处理 (不需要走对话管道)."""
        text = text.strip()
        if not text.startswith('/'):
            return False

        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handlers = {
            '/status': self._status,
            '/diag': self._diag,
            '/memory': self._memory,
            '/body': self._body,
            '/pain': self._pain,
            '/touch': self._touch,
            '/speaker': self._speaker,
            '/read': self._read,
            '/save': self._save,
            '/load': self._load,
            '/reset': self._reset,
            '/stream': self._stream,
            '/age': self._age,
            '/help': self._help,
        }
        h = handlers.get(cmd)
        if h:
            h(arg)
            return True
        return False

    def _status(self, arg):
        render_all_status_panels(
            self.interactive.agent,
            self.interactive.last_understanding,
            self.interactive.last_speech_diag)

    def _diag(self, arg):
        console.print(render_diag(self.interactive.agent))

    def _memory(self, arg):
        agent = self.interactive.agent
        net = agent.net
        console.print(f"\n  L0 Clusters: {net.n_clusters} "
                      f"(total_activation={net.total_activation:.2f})")
        if net.n_clusters > 0:
            top5 = sorted(net.clusters, key=lambda c: c.activation,
                         reverse=True)[:5]
            for i, c in enumerate(top5):
                console.print(f"    [{i}] act={c.activation:.3f} "
                             f"count={c.count} age={c.age} "
                             f"G_ema={c.G_ema:.3f} "
                             f"norm={np.linalg.norm(c.centroid):.1f}")

        # v5.6 modules
        if hasattr(agent, 'arcuate_fasciculus'):
            af = agent.arcuate_fasciculus
            console.print(f"  AF: {af.n_ventral_clusters}v/"
                          f"{af.n_dorsal_clusters}d clusters")
        if hasattr(agent, 'angular_gyrus'):
            ag = agent.angular_gyrus
            console.print(f"  AngularGyrus: {ag.n_grapheme_clusters} glyphs, "
                          f"{len(ag._word_to_phoneme)} cached words")
        if hasattr(agent, 'tpj'):
            tpj = agent.tpj
            console.print(f"  TPJ: {tpj._n_intents} intents, "
                          f"{tpj.speaker_net.n_clusters if hasattr(tpj, 'speaker_net') else 0} "
                          f"speaker models")
        if hasattr(agent, 'phrase_structure'):
            ps = agent.phrase_structure
            console.print(f"  PhraseStructure: {ps.n_phrase_types} types, "
                          f"trained={ps._trained}")
        if hasattr(agent, 'self_model'):
            sm = agent.self_model
            console.print(f"  SelfModel: {sm.net.n_clusters} clusters, "
                          f"{sm.n_experiences} experiences")
        # v6.0 modules
        if hasattr(agent, 'semantic_memory'):
            ssm = agent.semantic_memory
            console.print(f"  SemanticMemory: {ssm.n_clusters} concepts, "
                         f"{ssm.n_facts} facts")
        if hasattr(agent, 'striatum'):
            st = agent.striatum
            console.print(f"  Striatum: {st.get_state()['n_states_known']} states, "
                         f"habit={st.global_habit_strength:.3f}")
        console.print()

    def _body(self, arg):
        try:
            dim, val = arg.split()
            dim = int(dim)
            val = float(val)
            if 0 <= dim < len(self.interactive.agent.body.b):
                self.interactive.agent.body.b[dim] = float(np.clip(val, 0, 1))
                console.print(f"  Body b[{dim}] → {val:.2f}")
            else:
                console.print(f"  [red]Invalid dim {dim}, "
                             f"valid: 0-{len(self.interactive.agent.body.b)-1}[/red]")
        except (ValueError, IndexError):
            console.print("  [red]Usage: /body <dim> <value>  "
                         "e.g. /body 2 0.9 (high stress)[/red]")

    def _pain(self, arg):
        try:
            intensity = float(arg.strip())
            self.interactive.agent.set_pain_input(intensity)
            console.print(f"  Pain intensity → {intensity:.2f}")
        except ValueError:
            console.print("  [red]Usage: /pain <0-1>[/red]")

    def _touch(self, arg):
        try:
            intensity = float(arg.strip())
            self.interactive.agent.set_pain_input(
                self.interactive.agent._current_pain_input, intensity)
            console.print(f"  Touch (Aβ) → {intensity:.2f} "
                         f"(gate control active)")
        except ValueError:
            console.print("  [red]Usage: /touch <0-1>[/red]")

    def _speaker(self, arg):
        if arg.strip():
            self.interactive.speaker_name = arg.strip()
            console.print(f"  Speaker → '{self.interactive.speaker_name}' "
                         f"(TPJ active)")
        else:
            console.print(f"  Current speaker: '{self.interactive.speaker_name}'")

    def _read(self, arg):
        if not arg.strip():
            console.print("  [red]Usage: /read <text>[/red]")
            return
        agent = self.interactive.agent
        if hasattr(agent, 'angular_gyrus'):
            ag_phon, ag_conf = agent.angular_gyrus.read(arg.strip())
            console.print(f"  AG read '{arg.strip()}': "
                         f"ph_norm={np.linalg.norm(ag_phon):.2f} "
                         f"conf={ag_conf:.2f}")
        else:
            console.print("  [red]AngularGyrus not available[/red]")

    def _save(self, arg):
        name = arg.strip() if arg.strip() else None
        path = self.interactive.agent.save(
            name=name, n_sessions=self.interactive.n_sessions,
            n_turns=self.interactive.n_turns)
        console.print(f"  [green]Saved: {path}[/green]")

    def _load(self, arg):
        name = arg.strip()
        if not name:
            saves = list_saves()
            if not saves:
                console.print("  [red]No saves found[/red]")
                return
            console.print("  Available saves:")
            for s in saves[-10:]:
                console.print(f"    {os.path.basename(s)}")
            return
        path = os.path.join(".notme/sessions", f"{name}.pkl")
        if not os.path.exists(path):
            console.print(f"  [red]Save not found: {name}[/red]")
            return
        console.print(f"  Loading {name}...")
        self.interactive._reload_from(path)

    def _reset(self, arg):
        console.print("  [yellow]Resetting Agent...[/yellow]")
        self.interactive._init_fresh()
        console.print("  [green]Agent reset complete[/green]")

    def _stream(self, arg):
        """启停摄像头+麦克风流."""
        sub = arg.strip().lower()
        interactive = self.interactive

        if sub == 'start':
            if interactive._streaming:
                console.print("  [yellow]Stream already running[/yellow]")
                return

            console.print("  [cyan]Starting camera + mic stream...[/cyan]")
            try:
                from tools.sensor_io import StreamSession
                interactive._stream_session = StreamSession(
                    camera_id=0, camera_fps=5.0,
                    camera_resolution=(128, 128),
                    mic_sample_rate=22050, mic_chunk_ms=200,
                )
                status = interactive._stream_session.start()
                if status['camera']:
                    console.print("  [green]✓[/green] Camera ready (128x128 @ 5fps)")
                else:
                    console.print("  [dim]Camera not available[/dim]")
                if status['mic']:
                    console.print("  [green]✓[/green] Microphone ready (22kHz, 200ms chunks)")
                else:
                    console.print("  [dim]Microphone not available[/dim]")

                if status['camera'] or status['mic']:
                    interactive._streaming = True
                    console.print("  [bold cyan]Stream started[/bold cyan] — "
                                  "Agent will respond to significant events")
                    console.print("  Type [bold]/stream stop[/bold] to end")
                else:
                    console.print("  [red]No sensors available[/red]")
                    interactive._stream_session.stop()
                    interactive._stream_session = None
            except Exception as e:
                console.print(f"  [red]Stream error: {e}[/red]")
                interactive._streaming = False

        elif sub == 'stop':
            if not interactive._streaming:
                console.print("  [dim]No stream running[/dim]")
                return
            interactive._streaming = False
            if interactive._stream_session:
                stats = interactive._stream_session.stats()
                interactive._stream_session.stop()
                interactive._stream_session = None
            console.print("  [green]Stream stopped[/green]")
            n_frames = stats.get('n_frames', 0) if 'stats' in dir() else 0
            console.print(f"    Frames captured: {n_frames}")

        elif sub == 'status':
            if interactive._streaming and interactive._stream_session:
                s = interactive._stream_session.stats()
                console.print(f"  Stream: [green]active[/green] ({s['n_frames']} frames)")
                cam = s['camera']
                console.print(f"    Camera: {'✓' if cam['is_open'] else '✗'} "
                              f"read={cam['frames_read']} drop={cam['frames_dropped']}")
                mic = s['mic']
                console.print(f"    Mic: {'✓' if mic['is_streaming'] else '✗'} "
                              f"chunks={mic['chunks_read']} samples={mic['total_samples']}")
            else:
                console.print("  Stream: [dim]inactive[/dim]")

        else:
            console.print("  Usage: /stream start | stop | status")

    def _age(self, arg):
        """设置或查看发育年龄."""
        arg = arg.strip()
        age_names = {0: '婴儿 (模仿)', 1: '儿童 (学习)', 2: '青少年 (生成)', 3: '成人 (自主)'}

        if not arg:
            name = age_names.get(self.interactive.age, '未知')
            n_tri = self.interactive.broca.word_order_net.n_clusters
            console.print(f"  Age: {self.interactive.age} — {name}")
            console.print(f"  Trigram clusters: {n_tri}")
            # 显示自动升级阈值
            if self.interactive.age == 0:
                console.print(f"  → 升级到 age=1 需要: 50+ trigrams")
            elif self.interactive.age == 1:
                console.print(f"  → 升级到 age=2 需要: 200+ trigrams")
            elif self.interactive.age == 2:
                console.print(f"  → 升级到 age=3 需要: 1000+ trigrams")
            return

        try:
            new_age = int(arg)
            if new_age < 0 or new_age > 3:
                console.print("  [red]Age must be 0-3[/red]")
                return
            old_age = self.interactive.age
            self.interactive.age = new_age
            name = age_names.get(new_age, '未知')
            console.print(f"  Age: {old_age} → {new_age} — [bold]{name}[/bold]")

            # 年龄变化时提示学习行为变化
            if new_age == 0:
                console.print("  [dim]学习: 仅人类输入 | 回应: 纯模仿[/dim]")
            elif new_age == 1:
                console.print("  [dim]学习: 人类+长回应 | 回应: 模仿+trigram[/dim]")
            elif new_age >= 2:
                console.print("  [dim]学习: 全部 | 回应: trigram链生成[/dim]")
        except ValueError:
            console.print("  [red]Usage: /age [0-3][/red]")
            console.print("    0 = 婴儿 (纯模仿, 只学人类输入)")
            console.print("    1 = 儿童 (模仿+trigram, 选择性学习)")
            console.print("    2 = 青少年 (trigram链生成, 正常学习)")
            console.print("    3 = 成人 (全自主生成)")

    def _help(self, arg):
        console.print(render_help())


# ================================================================
# 主会话
# ================================================================

class InteractiveSession:
    """v5.7 交互会话 — 管理 Agent 生命周期和对话循环."""

    def __init__(self, load_path: str = None, fresh: bool = False):
        self.n_turns = 0
        self.n_sessions = 1
        self.speaker_name = "human"
        self.last_understanding = None
        self.last_speech_diag = None
        self.age = 0  # v5.7: 发育年龄 (0=婴儿, 1=儿童, 2=青少年, 3=成人)

        # 实时流状态
        self._stream_session = None  # StreamSession instance (lazy init)
        self._streaming = False

        # 检查存档
        auto_path = None
        if not fresh:
            auto_path = latest_save()

        if load_path:
            self._init_from_save(load_path)
        elif auto_path and not fresh:
            console.print(f"  [dim]Found save: {os.path.basename(auto_path)}[/dim]")
            self._init_from_save(auto_path)
        else:
            self._init_fresh()

        # 安装 prompt_toolkit
        self._setup_prompt()

    def _init_fresh(self):
        """全新初始化 Agent (v6.0 纯净模式)."""
        rng = np.random.default_rng(42)
        self.agent = Agent(rng=rng, agent_id=0, n_agents=1)
        self.agent.body = BodyVector(mode='text')

        # v6.0: 使用先天配置替代手动覆盖
        from cns.innate import apply_innate_config
        apply_innate_config(self.agent)

        self.agent.record_action_consequence = lambda s: None
        ACTION_DIRECTIONS[3] = [0.0, 0.0]
        ACTION_DIRECTIONS[4] = [0.0, 0.0]
        self.n_sessions = 1
        self._do_minimal_setup()

    def _init_from_save(self, path: str):
        """从存档恢复 Agent (v6.0 纯净模式)."""
        self.agent, meta = Agent.load(path, verbose=True)
        # v6.0: Re-apply innate config (not manual overrides)
        from cns.innate import apply_innate_config
        apply_innate_config(self.agent)

        self.agent.record_action_consequence = lambda s: None
        ACTION_DIRECTIONS[3] = [0.0, 0.0]
        ACTION_DIRECTIONS[4] = [0.0, 0.0]
        self.n_sessions = meta.get('n_sessions', 1) + 1
        self.n_turns = meta.get('total_turns', 0)
        # 纯净模式: 从存档恢复后只做最小设置, 不重训练
        self._do_minimal_setup()

    def _do_minimal_setup(self):
        """v5.7 纯净模式: 最小启动 — 只创建编码器和词表, 无预训练.

        与旧 _do_warmup_and_train() 的根本区别:
          - 不喂 corpus.txt 到 L0 海马
          - 不训练 trigram 网络 (词序从互动中学习)
          - 不训练 AF / PhraseStructure / AngularGyrus / TPJ
          - Broca 以 load_corpus=False 创建 (仅保留词向量用于 TTS)
          - 杏仁核种子词汇保留 (极小的情感引导, 可被后续学习覆盖)
        """
        from environments.text_interface import TextEnvironment
        from cerebrum.frontal_lobe.broca import Broca

        # TextEnvironment 仍需要 (MiniLM 编码器用于理解输入)
        self.text_env = TextEnvironment()

        # Broca 纯净模式: 保留词表+词向量供 TTS 输出, 但无语料网络
        self.broca = Broca(text_env=self.text_env, load_corpus=False)

        # 确保杏仁核种子词汇就绪 (45个中文情感词弱先验)
        from cerebrum.limbic_system.amygdala import get_emotional_lexicon
        get_emotional_lexicon()

        console.print("  [bold green]Clean mode[/bold green] — "
                      f"{len(self.broca.word_list)} words, "
                      f"0 trigram clusters, 0 sentences")
        console.print("  All networks start empty — language grows from interaction")

    def _reload_from(self, path: str):
        """从指定存档重新加载."""
        self._init_from_save(path)
        console.print(f"  [green]Loaded: {path}[/green]")

    def _setup_prompt(self):
        """设置 prompt_toolkit 输入 (非交互环境回退到 input())."""
        completer = WordCompleter([
            '/status', '/diag', '/memory',
            '/body', '/pain', '/touch',
            '/speaker', '/read',
            '/save', '/load', '/reset', '/help',
            '/age', '/stream',
        ], ignore_case=True, sentence=True)

        history_path = '.notme/input_history.txt'
        os.makedirs('.notme', exist_ok=True)

        style = Style.from_dict({
            'prompt': 'bold cyan',
        })

        try:
            self.session = PromptSession(
                history=FileHistory(history_path),
                completer=completer,
                style=style,
            )
            self._use_prompt_toolkit = True
        except Exception:
            # Non-interactive environment (CI, pipe, etc.) — use input()
            self._use_prompt_toolkit = False

    # ================================================================
    # 对话循环
    # ================================================================

    def run(self):
        """主循环."""
        from cerebrum.limbic_system.amygdala import (
            analyze_sentiment, sentiment_to_social_signal, get_emotional_lexicon)
        from cerebrum.limbic_system.cingulate import SocialContext
        from cns.data_types import M_V1_START, D_VISUAL_V5, BINDING_END

        agent = self.agent
        text_env = self.text_env
        broca = self.broca
        social_ctx = SocialContext(tau=25.0)  # v5.7: 较慢适应 → 更多F_social波动
        emo_lexicon = get_emotional_lexicon()

        # v5.7 纯净模式: 跨模态视觉不使用预训练COCO模型
        # 视觉感知由 agent.step() 内的语义代理模式处理
        # (文字→Hebb网络检索→视觉段补全)
        vis_brain = None
        img_enc = None

        # 追踪
        last_human_vec = np.zeros(64)
        last_self_semantic = np.zeros(64, dtype=np.float32)
        last_self_sentiment = np.zeros(8, dtype=np.float32)
        t = 0
        expr_cooldown = 0
        inner_cooldown = 0
        VIS_END_V5 = M_V1_START + D_VISUAL_V5

        # v5.7: 初始身体扰动 — 确保自由能有初始动态 (避免全零)
        # 身体默认在调定点 = F_body=0 = 无驱力, 加入小扰动使系统开始运转
        if agent.body is not None and len(agent.body.b) >= 5:
            agent.body.b[0] = max(0.3, agent.body.b[0] - 0.25)  # 明显社交需求
            agent.body.b[2] = min(0.7, agent.body.b[2] + 0.15)  # 适度压力
            agent.body.b[3] = min(0.85, agent.body.b[3] + 0.3)  # 较强新颖性寻求
            agent.body.b[4] = max(0.4, agent.body.b[4] - 0.1)   # 注意力略降

        # ---- 启动信息 ----
        console.print()
        console.print(Panel(
            "[bold cyan]NotMe v5.7[/bold cyan] — "
            "自由能原理情感智能体\n"
            "[bold green]纯净模式[/bold green] — "
            "零预训练, 网络从互动中生长\n"
            "\n"
            "[dim]"
            "  发育年龄: 👶 婴儿 (纯模仿, 只学人类输入)\n"
            "  所有知识从与你的对话中在线学习\n"
            "  trigram网络 · 句记忆 · 概念词映射 全部从零开始\n"
            "  输入 /age 查看年龄, /help 查看命令, /status 查看状态[/dim]",
            border_style="cyan"))
        console.print()

        try:
            while True:
                response = ""  # 初始化为空 (v5.7: 修复非表达行动时的 NameError)
                # ---- Header ----
                meta = {'n_sessions': self.n_sessions,
                       'total_turns': self.n_turns}
                console.print(render_header(agent, meta, broca=broca, age=self.age))

                # ---- Input ----
                try:
                    if self._use_prompt_toolkit:
                        human_text = self.session.prompt(
                            [('class:prompt', '> ')],
                        )
                    else:
                        console.print(Text("> ", style="bold cyan"), end="")
                        human_text = input()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n  [yellow]Goodbye![/yellow]")
                    # v6.0: 退出前跨会话巩固
                    try:
                        result = agent.consolidate_across_sessions()
                        console.print(f"  [dim]Consolidated: "
                                     f"{result['n_extracted']} processed, "
                                     f"{result['n_new_facts']} new facts[/dim]")
                        agent.save(name=None, n_sessions=self.n_sessions,
                                  n_turns=self.n_turns)
                    except Exception:
                        pass
                    break

                human_text = human_text.strip()
                if not human_text:
                    continue

                if human_text.lower() == 'exit':
                    # v6.0: 退出前跨会话巩固 + 保存
                    console.print("  [cyan]Running cross-session consolidation...[/cyan]")
                    try:
                        result = agent.consolidate_across_sessions()
                        console.print(f"  [green]Consolidated:[/green] "
                                     f"{result['n_extracted']} processed, "
                                     f"{result['n_new_facts']} new facts")
                    except Exception as e:
                        console.print(f"  [yellow]Consolidation: {e}[/yellow]")
                    agent.save(name=None, n_sessions=self.n_sessions,
                              n_turns=self.n_turns)
                    console.print("  [green]Final state saved[/green]")
                    break

                # ---- Command handler ----
                cmd_handler = CommandHandler(self)
                if cmd_handler.handle(human_text):
                    continue

                # ---- v5.7: 实时传感器流 (摄像头+麦克风) ----
                if self._streaming and self._stream_session:
                    stream_data = self._stream_session.read()
                    if stream_data is not None:
                        if stream_data.get('frame') is not None:
                            agent.set_current_image(stream_data['frame'])
                        if stream_data.get('audio') is not None:
                            agent.set_audio_input(stream_data['audio'])

                # ---- 多模态解析 (v5.7: InputBus 统一处理) ----
                bus = InputBus.parse(human_text)
                # 继承当前 speaker 设置
                if bus.speaker_name == 'human':
                    bus.speaker_name = self.speaker_name
                human_text_clean = bus.text or ""

                # 解析待加载资源 (图像/音频文件/Mic录制)
                resolve_result = bus.resolve_pending(img_enc=img_enc)
                if resolve_result.get('errors'):
                    for err in resolve_result['errors']:
                        console.print(f"  [red]{err}[/red]")
                    continue

                # 更新 speaker_name
                self.speaker_name = bus.speaker_name

                # 音频就绪 → 通知 Agent (仅当未通过 stream 设置时)
                if bus.has_audio:
                    agent.set_audio_input(bus.audio_data)
                elif not self._streaming:
                    agent.set_audio_input(None)

                # 图像就绪 → 通知 Agent (仅当未通过 stream 设置时)
                if bus.has_image:
                    agent.set_current_image(bus.image)
                elif not self._streaming:
                    # v5.7: 非流模式下不清除已设置的图像 (允许后续使用)
                    pass

                # ---- 构建完整感知向量 (v5.7: 所有活跃通道同时填入) ----
                s, bus_info = bus.build_sensory(
                    text_env, img_enc=img_enc, vis_brain=vis_brain)
                last_human_vec = bus_info['text_vec'].copy()

                # ---- 社会上下文更新 ----
                sentiment = bus_info.get('sentiment', {})
                s_valence = sentiment.get('valence', 0.0)
                s_arousal = sentiment.get('arousal', 0.0)
                social_ctx.update(s_valence, s_arousal)

                # v5.7 纯净模式: 跨模态视觉由 agent.step() 语义代理模式处理
                # (text→Hebb网络检索→视觉段补全, 不依赖预训练COCO模型)

                # ---- 痛觉输入 → Agent ----
                if bus.has_pain:
                    agent.set_pain_input(bus.pain_intensity,
                                         abeta_input=bus.touch_intensity)

                # ---- Comprehend (v5.7: TPJ + AngularGyrus 全活跃) ----
                comp_vec, understanding = agent.comprehend(
                    last_human_vec, s[80:88].astype(np.float32),
                    speaker_name=bus.speaker_name,
                    human_text=human_text_clean)

                # ---- 多模态输入显示 ----
                input_type = getattr(bus, '_input_type', 'text')
                if input_type == 'image':
                    img_path = getattr(bus, '_pending_image_path', '?')
                    console.print(f"  [You] [img]: {img_path}")
                    if human_text_clean:
                        console.print(f"         → {human_text_clean}")
                elif input_type in ('audio_file', 'mic'):
                    dur = bus.audio_data.get('duration', 0) if bus.audio_data else 0
                    console.print(f"  [You] [audio]: {dur:.1f}s")
                    if human_text_clean:
                        console.print(f"         → {human_text_clean}")
                elif input_type == 'pain':
                    console.print(f"  [You] [pain]={bus.pain_intensity:.2f}: {human_text_clean}")
                elif input_type == 'touch':
                    console.print(f"  [You] [touch]={bus.touch_intensity:.2f}: {human_text_clean}")
                else:
                    v_icon = valence_sign(s_valence)
                    console.print(f"  [You] {v_icon}: {human_text_clean}")
                    console.print(f"         v={s_valence:+.2f} "
                                  f"a={s_arousal:.2f} "
                                  f"trust={social_ctx.trust_level:.2f} "
                                  f"mem={understanding['n_triggered_memories']}"
                                  f" TPJ={understanding.get('pragmatic_enriched', False)}"
                                  f" AG={understanding.get('ag_confidence', 0):.2f}")

                self.last_understanding = understanding

                # ---- Self audio ----
                agent.set_self_audio(last_self_semantic, last_self_sentiment)

                # ---- Agent step ----
                F_before = agent.F_body_history[-1] if agent.F_body_history else 0.0
                action = agent.step(s, t, social_ctx=social_ctx)
                F_after = agent.F_body_history[-1] if agent.F_body_history else 0.0

                # Clear audio after single use (v5.7: use bus.has_audio)
                if bus.has_audio:
                    agent.set_audio_input(None)
                # Clear image after single use
                if bus.has_image:
                    agent.set_current_image(None)

                # ---- Hebb emotional learning ----
                if human_text_clean:
                    delta_F = F_before - F_after
                    import jieba
                    input_words = [w for w in jieba.lcut(human_text_clean)
                                  if len(w.strip()) >= 1]
                    arousal_now = agent.arousal_history[-1] if agent.arousal_history else 0.5
                    emo_lexicon.learn_from_feedback(input_words, delta_F, arousal_now)

                # ---- Expression / Inner speech ----
                is_silence = not (human_text and human_text.strip())
                if action.index == 3 and is_silence:
                    inner_cooldown = 0
                    action = Action(index=4, expected_F=action.expected_F,
                                   expected_G=action.expected_G,
                                   confidence=action.confidence)

                if action.index == 3 and expr_cooldown <= 0:
                    if agent.net.n_clusters > 0:
                        top = max(agent.net.clusters, key=lambda c: c.activation)
                        if top.activation > 0.01:
                            # Build query
                            n_trig = understanding.get('n_triggered_memories', 0)
                            comp_precision = 0.15 + min(1.0, n_trig / 5.0)
                            belief_precision = 0.15 + min(1.0, top.activation * 2.0)
                            ctx_vec = agent.dialogue_ctx.get_context_vector()
                            ctx_precision = 0.15 + min(1.0, agent.dialogue_ctx.n_turns() / 5.0)
                            self_anchor = agent.self_model.get_self_anchor()
                            n_self = agent.self_model.n_experiences
                            self_precision = 0.05 + min(0.20, n_self * 0.005)

                            total_p = max(comp_precision + belief_precision
                                         + ctx_precision + self_precision, 1e-6)
                            query = (comp_vec * (comp_precision / total_p)
                                    + top.centroid[:64].astype(np.float32)
                                    * (belief_precision / total_p)
                                    + ctx_vec * (ctx_precision / total_p)
                                    + self_anchor * (self_precision / total_p))
                            query = query.astype(np.float32)

                            v = agent.valence_history[-1] if agent.valence_history else 0
                            a = agent.arousal_history[-1] if agent.arousal_history else 0
                            sa = agent.self_arousal_ema
                            temp_base = 0.5 + abs(v) * 0.8 + sa * 0.2
                            body = agent.body
                            social_need = max(0.0, body.setpoints[0] - body.b[0])
                            novelty_need = max(0.0, 0.5 - body.b[3])
                            temp = temp_base * (1.0 + social_need * 0.6 + novelty_need * 0.4)

                            # Speak (v5.7: full language pipeline)
                            words, audio, speech_diag = agent.speak(
                                broca=broca,
                                query_vec=query,
                                belief_vec=top.centroid,
                                valence=v, arousal=a,
                                temperature=temp, max_words=20,
                                use_phrase_structure=True,
                                human_text=human_text_clean,
                            )
                            self.last_speech_diag = speech_diag
                            response = "".join(words) if words else ""

                            # Self-eval
                            eval_score = None
                            if response:
                                try:
                                    resp_vec = text_env.encode_text(response).astype(np.float32)
                                    eval_result = agent.evaluate_own_response(resp_vec)
                                    eval_score = eval_result['overall_score']
                                    if not eval_result['acceptable'] and len(words) < 8:
                                        words2, audio2, _ = agent.speak(
                                            broca=broca, query_vec=query,
                                            belief_vec=top.centroid,
                                            valence=v, arousal=a,
                                            temperature=temp * 1.5,
                                            max_words=20,
                                            use_phrase_structure=False)
                                        if words2 and len("".join(words2)) > len(response):
                                            words, audio, response = words2, audio2, "".join(words2)
                                except Exception:
                                    pass

                            # Display
                            v_now = agent.valence_history[-1] if agent.valence_history else 0
                            feel = valence_sign(v_now)
                            eval_str = f' score={eval_score:.2f}' if eval_score else ''
                            console.print(f"  [Agent] {feel}: {response}")
                            console.print(f"          V={fmt_val(v_now)} "
                                         f"A={a:.2f} b0={agent.body.b[0]:.2f}"
                                         f"{eval_str} temp={temp:.2f}")
                            expr_cooldown = 1

                            # Store in dialogue memory
                            if response:
                                try:
                                    resp_vec = text_env.encode_text(response).astype(np.float32)
                                except Exception:
                                    resp_vec = np.zeros(64, dtype=np.float32)
                                agent.dialogue_ctx.add_turn(
                                    human_text=human_text_clean,
                                    human_vec=last_human_vec,
                                    human_sentiment=s[80:88].astype(np.float32),
                                    agent_response=response,
                                    agent_vec=resp_vec,
                                    agent_valence=v_now,
                                    agent_arousal=a,
                                    comprehension_vec=comp_vec)
                                agent.self_model.add_experience(
                                    response_vec=resp_vec, valence=v_now,
                                    arousal=a,
                                    self_valence_ema=agent.self_valence_ema,
                                    self_arousal_ema=agent.self_arousal_ema,
                                    self_coherence=agent.self_coherence,
                                    body_state=agent.body,
                                    comprehension_vec=comp_vec,
                                    dialogue_ctx_vec=ctx_vec)

                                # Micro consolidation
                                agent.micro_consolidate()
                                cb_result = agent.maybe_consolidate(broca=broca)
                                if cb_result and cb_result.get('phase') == 'full':
                                    console.print(f"  [sleep] {cb_result['n_turns']}t → "
                                                  f"{cb_result['n_l0_clusters']} L0, "
                                                  f"{cb_result['pruned']} pruned")

                                # Auto-save
                                auto_save(agent, self.n_turns,
                                         n_sessions=self.n_sessions,
                                         save_every=10, verbose=False)

                            # Self audio for next turn
                            if response:
                                try:
                                    last_self_semantic = text_env.encode_text(
                                        response).astype(np.float32)
                                except Exception:
                                    last_self_semantic = np.zeros(64, dtype=np.float32)
                                self_sent = analyze_sentiment(response)
                                last_self_sentiment = sentiment_to_social_signal(
                                    self_sent)[:8].astype(np.float32)

                # v5.7 纯净模式: 从真实对话中在线学习 trigram + 概念词映射
                if response and human_text_clean:
                    try:
                        broca.learn_from_interaction(
                            human_text_clean, response, age=self.age)
                        # 始终学人类输入 (不污染), Agent输出取决于年龄
                        broca.learn_sentence_online(human_text_clean)
                        if self.age >= 1 and len(response) >= 6:
                            broca.learn_sentence_online(response)
                    except Exception:
                        pass

                    # v5.7: 自动年龄升级 — 基于trigram网络生长
                    n_tri = broca.word_order_net.n_clusters
                    old_age = self.age
                    if self.age == 0 and n_tri >= 50:
                        self.age = 1
                    elif self.age == 1 and n_tri >= 200:
                        self.age = 2
                    elif self.age == 2 and n_tri >= 1000:
                        self.age = 3
                    if self.age != old_age:
                        age_names = {0: '👶婴儿', 1: '🧒儿童', 2: '🧑青少年', 3: '🧠成人'}
                        console.print(f"  [bold yellow]🎉 年龄升级![/bold yellow] "
                                     f"{age_names.get(old_age)} → {age_names.get(self.age)} "
                                     f"(trigrams: {n_tri})")

                expr_cooldown = max(0, expr_cooldown - 1)
                inner_cooldown = max(0, inner_cooldown - 1)

                # Inner speech
                if not is_silence:
                    inner_cooldown = max(inner_cooldown, 3)
                if is_silence and action.index != 3 and inner_cooldown <= 0:
                    if agent.net.n_clusters > 0:
                        top = max(agent.net.clusters, key=lambda c: c.activation)
                        if top.activation > 0.01:
                            v = agent.valence_history[-1] if agent.valence_history else 0
                            a = agent.arousal_history[-1] if agent.arousal_history else 0
                            words, _, _ = agent.speak(
                                broca=broca,
                                query_vec=s[:64].astype(np.float32),
                                belief_vec=top.centroid,
                                valence=v, arousal=a,
                                temperature=0.7 + a * 0.4,
                                max_words=10,
                                use_phrase_structure=False)
                            if words:
                                inner_text = "".join(words)
                                console.print(f"  [dim][inner] {inner_text}[/dim]")
                                try:
                                    last_self_semantic = text_env.encode_text(
                                        inner_text).astype(np.float32)
                                except Exception:
                                    pass
                                inner_sent = analyze_sentiment(inner_text)
                                last_self_sentiment = sentiment_to_social_signal(
                                    inner_sent)[:8].astype(np.float32)
                            inner_cooldown = 4

                self.n_turns += 1
                t += 1

        except KeyboardInterrupt:
            console.print("\n  [yellow]Saving before exit...[/yellow]")
        except Exception:
            console.print(f"\n  [red]Unexpected error:[/red]")
            console.print(traceback.format_exc())
        finally:
            # Final save
            try:
                path = self.agent.save(
                    n_sessions=self.n_sessions,
                    n_turns=self.n_turns)
                console.print(f"  [green]Saved: {path}[/green]")
            except Exception:
                pass
            console.print("  [dim]Goodbye![/dim]")


# ================================================================
# 入口
# ================================================================

def main():
    """入口点."""
    import argparse
    parser = argparse.ArgumentParser(description='NotMe v6.4 Interactive')
    parser.add_argument('--fresh', action='store_true',
                       help='Force fresh session (ignore saves)')
    parser.add_argument('--load', type=str, default=None,
                       help='Load specific save by name')
    parser.add_argument('--auto', action='store_true',
                       help='Autonomous mode — agent runs continuously')
    parser.add_argument('--steps', type=int, default=None,
                       help='Steps to run in --auto mode (default: infinite)')
    args = parser.parse_args()

    if args.auto:
        _run_autonomous(args)
    else:
        session = InteractiveSession(load_path=args.load, fresh=args.fresh)
        session.run()


def _run_autonomous(args):
    """v6.4: 控制台自主模式."""
    import time
    from cns.agent import Agent
    from cns.data_types import BodyVector, ACTION_DIRECTIONS
    from cns.innate import apply_innate_config
    from cns.persistence import latest_save
    from entry.autonomous import AutonomousLoop
    from cerebrum.association.internal_life import InternalLife
    from tools.telemetry import Telemetry
    from tools.reader import Reader

    # 创建或加载 Agent
    rng = np.random.default_rng(42)
    if not args.fresh:
        auto_path = latest_save()
    else:
        auto_path = None

    if auto_path and not args.fresh:
        agent, meta = Agent.load(auto_path, verbose=True)
        apply_innate_config(agent)
        agent.internal_life = InternalLife()
        agent.telemetry = Telemetry()
        agent.reader = Reader()
    else:
        agent = Agent(rng=rng, agent_id=0, n_agents=1)
        agent.body = BodyVector(mode='text')
        apply_innate_config(agent)
        agent.internal_life = InternalLife()
        agent.telemetry = Telemetry()
        agent.reader = Reader()

    agent.record_action_consequence = lambda s: None
    ACTION_DIRECTIONS[3] = [0.0, 0.0]
    ACTION_DIRECTIONS[4] = [0.0, 0.0]

    # Broca
    from environments.text_interface import TextEnvironment
    from cerebrum.frontal_lobe.broca import Broca
    te = TextEnvironment()
    broca = Broca(text_env=te, load_corpus=False)

    # Autonomous loop
    loop = AutonomousLoop(agent, broca=broca, steps_per_second=10)
    loop.reader = agent.reader
    loop.telemetry = agent.telemetry
    loop.internal_life = agent.internal_life

    print("=" * 50)
    print("  NotMe v6.4 — Autonomous Mode")
    print(f"  Clusters: {agent.net.n_clusters}")
    print(f"  Mode: {'fresh' if args.fresh else 'from save'}")
    print(f"  Steps: {'infinite' if args.steps is None else args.steps}")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    try:
        stats = loop.run(duration_steps=args.steps, blocking=True)
        print(f"\n  Run complete: {stats['total_ticks']} ticks")
        print(f"  Modes: {stats['mode_counts']}")
        print(f"  Sleep: {stats['sleep_ticks']} ticks ({100*stats['sleep_ticks']/max(1,stats['total_ticks']):.1f}%)")
    except KeyboardInterrupt:
        print("\n  Stopping...")
        loop.stop()

    # 保存
    agent.telemetry.flush()
    path = agent.save(name=f"auto_{time.strftime('%Y%m%d_%H%M%S')}")
    print(f"  Saved: {path}")


if __name__ == '__main__':
    main()
