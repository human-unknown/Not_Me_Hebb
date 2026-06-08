"""
server.py — NotMe v6.6 Web 界面 (Flask REST API + SSE 实时推送)

提供:
  - REST API: Agent 状态查询、对话、阅读控制
  - SSE 实时推送: Agent 状态 + 视觉帧 + 音频波形
  - 静态首页: 单页仪表板

启动:
  python web/server.py --port 8080

依赖: flask (pip install flask)
"""

import sys
import os
import time
import json
import threading
import base64
import io
import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask, Response, request, jsonify, send_from_directory
import flask

app = Flask(__name__, static_folder='static', static_url_path='')

# ---- Global state (shared across threads) ----
_agent = None
_broca = None
_loop = None
_loop_thread = None
_text_env = None
_save_path = None  # v6.6: auto-save path for persistence


# ============================================================
# Helpers
# ============================================================

def _get_text_env():
    global _text_env
    if _text_env is None:
        from environments.text_interface import TextEnvironment
        _text_env = TextEnvironment(load_corpus=False)
    return _text_env


def _build_status(agent) -> dict:
    """构建 Agent 完整状态快照."""
    v = agent.valence_history[-1] if agent.valence_history else 0.0
    a = agent.arousal_history[-1] if agent.arousal_history else 0.0
    body_b = agent.body.b.tolist() if agent.body else [0]*9

    # v6.6: SCN time hour — use reliable step-based clock (not TTFL phase)
    total_steps_v = agent.meta.step_count if hasattr(agent, 'meta') else 0
    try:
        from cerebrum.limbic_system.scn import SCN
        scn_hour = SCN.get_reliable_hour(total_steps_v)
    except Exception:
        scn_hour = 0.0

    # SCN/VLPO
    circa = {
        'phase': float(getattr(agent._circadian_state, 'circadian_phase', 0.0)),
        'melatonin': float(getattr(agent._circadian_state, 'melatonin', 0.0)),
        'cortisol': float(getattr(agent._circadian_state, 'cortisol', 0.0)),
        'sleep_pressure': float(getattr(agent._circadian_state, 'sleep_pressure', 0.0)),
        'sleep_propensity': float(getattr(agent._circadian_state, 'sleep_propensity', 0.0)),
        # v6.6: SCN hour from reliable step-based clock
        'scn_hour': scn_hour,
    }
    sleep = {
        'is_asleep': agent.vlpo.is_asleep if hasattr(agent, 'vlpo') else False,
        'is_rem': agent.vlpo.is_in_rem if hasattr(agent, 'vlpo') else False,
        'is_nrem': agent.vlpo.is_in_nrem if hasattr(agent, 'vlpo') else False,
        'state': agent._sleep_state.state if hasattr(agent, '_sleep_state') else 'awake',
        'phase': agent._sleep_state.phase if hasattr(agent, '_sleep_state') else 'none',
        'time_in_state': int(getattr(agent._sleep_state, 'time_in_state', 0)),
        'time_in_phase': int(getattr(agent._sleep_state, 'time_in_phase', 0)),
        'cycle_count': int(getattr(agent._sleep_state, 'sleep_cycle_count', 0)),
        'n_episodes': int(getattr(agent._sleep_state, 'n_sleep_episodes', 0)),
        'vlpo_activation': float(getattr(agent._sleep_state, 'vlpo_activation', 0.0)),
        'rem_on': float(getattr(agent._sleep_state, 'rem_on_activity', 0.0)),
        'rem_off': float(getattr(agent._sleep_state, 'rem_off_activity', 0.0)),
        'total_sleep_steps': int(getattr(agent._sleep_state, 'total_sleep_steps', 0)),
        'total_nrem_steps': int(getattr(agent._sleep_state, 'total_nrem_steps', 0)),
        'total_rem_steps': int(getattr(agent._sleep_state, 'total_rem_steps', 0)),
        'cycle_position': float(getattr(agent._sleep_state, 'cycle_position', 0.0)),
    }

    # Memory
    memory = {
        'n_clusters': agent.net.n_clusters,
        'total_activation': float(agent.net.total_activation),
        'n_semantic': (agent.semantic_memory.n_clusters
                      if hasattr(agent, 'semantic_memory') else 0),
        'n_self': agent.self_model.n_experiences,
        'trigram_count': (agent.arcuate_fasciculus.n_ventral_clusters
                         if hasattr(agent, 'arcuate_fasciculus') else 0),
    }

    # Neuro
    neuro = {
        'tonic_ne': float(agent._lc_result.get('tonic_ne', 0.2)
                         if agent._lc_result else 0.2),
        'phasic_ne': float(agent._lc_result.get('phasic_ne', 0.0)
                          if agent._lc_result else 0.0),
        'ne_snr': float(agent._lc_result.get('snr', 0.5)
                       if agent._lc_result else 0.5),
        'rpe': float(agent._vta_result.get('rpe', 0.0)
                    if agent._vta_result else 0.0),
        'da_tonic': float(agent._vta_result.get('tonic_da', 0.3)
                         if agent._vta_result else 0.3),
        'da_phasic': float(agent._vta_result.get('phasic_da', 0.0)
                          if agent._vta_result else 0.0),
        'tpn_act': float(agent.tpn.tpn_activation if hasattr(agent, 'tpn') else 0.0),
        'dmn_act': float(1.0 - (agent.tpn.tpn_activation
                               if hasattr(agent, 'tpn') else 0.3)),
        'fpn_act': float(agent.fpn.tpn_activation if hasattr(agent, 'fpn') else 0.5),
        'ach_level': float(agent.vlpo.get_ach_level() if hasattr(agent, 'vlpo') else 0.4),
    }

    # F components
    F = {
        'total': float(agent.F_history[-1] if agent.F_history else 0.0),
        'body': float(agent.F_body_history[-1] if agent.F_body_history else 0.0),
        'social': float(agent.F_social_history[-1] if agent.F_social_history else 0.0),
        'cognitive': float(agent.F_cognitive_history[-1] if agent.F_cognitive_history else 0.0),
        'accuracy': float(agent.F_accuracy_history[-1] if agent.F_accuracy_history else 0.0),
    }

    # Activity
    activity = {
        'mode': agent._last_activity if hasattr(agent, '_last_activity') else 'idle',
        'autonomous': agent._autonomous_mode if hasattr(agent, '_autonomous_mode') else False,
        'last_thought': agent._last_thought if hasattr(agent, '_last_thought') else '',
    }

    # Reader
    reader = None
    if hasattr(agent, 'reader') and agent.reader is not None:
        reader = agent.reader.get_progress()

    # Activity log (recent internal activities)
    activity_log = []
    if hasattr(agent, '_activity_log') and agent._activity_log:
        activity_log = list(agent._activity_log[-10:])

    # Developmental age
    development = {}
    if hasattr(agent, 'meta') and agent.meta is not None:
        dev_factors = agent.meta.get_developmental_factors()
        development = {
            'stage': dev_factors.get('stage', 1),
            'stage_name': dev_factors.get('stage_name', 'infant'),
            'critical_window': dev_factors.get('critical_window', 1.0),
            'plasticity': dev_factors.get('plasticity', 1.0),
            'learn_rate_scale': dev_factors.get('learn_rate_scale', 1.0),
            'is_infant': dev_factors.get('is_infant', True),
            'is_child': dev_factors.get('is_child', False),
            'is_adolescent': dev_factors.get('is_adolescent', False),
            'is_adult': dev_factors.get('is_adult', False),
        }

    # Session info (v6.6: +total_steps from meta)
    session_info = {}
    if hasattr(agent, 'telemetry') and agent.telemetry is not None:
        session_info = agent.telemetry.get_session_info()
    # Always include authoritative step count from meta learner
    session_info['total_steps'] = (
        agent.meta.step_count if hasattr(agent, 'meta') else 0)

    # v6.6: Last monologue/proactive speech
    last_monologue = ''
    if hasattr(agent, '_last_monologue') and agent._last_monologue:
        last_monologue = agent._last_monologue

    return {
        'valence': float(v),
        'arousal': float(a),
        'body': body_b,
        'circadian': circa,
        'sleep': sleep,
        'memory': memory,
        'neuro': neuro,
        'F': F,
        'activity': activity,
        'reader': reader,
        'activity_log': activity_log,
        'development': development,
        'session': session_info,
        'scn_hour': scn_hour,
        'last_monologue': last_monologue,
        'timestamp': time.time(),
    }


def _build_visual(agent) -> dict | None:
    """构建视觉数据 (摄像头帧 + 视觉通道状态)."""
    has_frame = False
    frame_b64 = None
    v1_mean = v2_mean = v4_mean = it_mean = 0.0

    # 摄像头帧
    if (hasattr(agent, '_current_image') and
        agent._current_image is not None):
        try:
            img = agent._current_image
            # Downsize to 128x128 if larger
            if img.shape[0] > 128 or img.shape[1] > 128:
                from PIL import Image
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((128, 128), Image.LANCZOS)
                buf = io.BytesIO()
                pil_img.save(buf, format='JPEG', quality=60)
                frame_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            else:
                from PIL import Image
                pil_img = Image.fromarray(img)
                buf = io.BytesIO()
                pil_img.save(buf, format='JPEG', quality=60)
                frame_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            has_frame = True
        except Exception:
            pass

    # 视觉通道强度 (v6.6: +reason)
    v1_mean = v2_mean = v4_mean = it_mean = 0.0
    vis_reason = ''
    vis_result = getattr(agent, '_current_visual_result', {})
    diag = vis_result.get('diagnostics', {})
    if diag:
        v1_mean = float(diag.get('V1_mean_norm', 0.0))
        v2_mean = float(diag.get('V2_mean_norm', 0.0))
        v4_mean = float(diag.get('V4_mean_norm', 0.0))
        it_mean = float(diag.get('IT_mean_norm', 0.0))
        vis_reason = diag.get('reason', '')

    return {
        'has_frame': has_frame,
        'frame_b64': frame_b64,
        'v1_mean': v1_mean,
        'v2_mean': v2_mean,
        'v4_mean': v4_mean,
        'it_mean': it_mean,
        'vis_reason': vis_reason,
    }


def _build_audio(agent) -> dict:
    """构建音频数据 (波形 + 频谱 + 方位)."""
    has_audio = False
    waveform = []
    mel_spectrum = []
    rms_level = 0.0
    azimuth = 0.0

    aud_data = getattr(agent, '_current_audio_data', None)
    if aud_data is not None:
        has_audio = True
        # 波形: 取最近 128 采样点
        raw_wave = aud_data.get('waveform', None)
        if raw_wave is not None:
            wf = np.asarray(raw_wave).ravel()
            # 降采样到 128 点
            if len(wf) > 128:
                step = len(wf) // 128
                wf = wf[::step][:128]
            elif len(wf) < 128:
                wf = np.pad(wf, (0, 128 - len(wf)))
            waveform = wf.tolist()
            rms_level = float(np.sqrt(np.mean(np.square(wf))) + 1e-8)

        # Mel 频谱
        spectrum = aud_data.get('spectrum', None)
        if spectrum is not None:
            spec = np.asarray(spectrum).ravel()
            if len(spec) > 32:
                step = len(spec) // 32
                spec = spec[::step][:32]
            elif len(spec) < 32:
                spec = np.pad(spec, (0, 32 - len(spec)))
            mel_spectrum = spec.tolist()

        # 方位角
        azimuth = float(aud_data.get('azimuth', 0.0))

    # 听觉通道状态
    aud_result = getattr(agent, '_current_auditory_result', {})
    aud_diag = aud_result.get('diagnostics', {})
    if aud_diag and not has_audio:
        # 语义代理模式 — 有通道活动但无真实音频
        pass

    return {
        'has_audio': has_audio,
        'waveform': waveform,
        'mel_spectrum': mel_spectrum,
        'rms_level': float(rms_level),
        'azimuth': float(azimuth),
    }


# ============================================================
# REST API
# ============================================================

@app.route('/')
def index():
    """静态首页."""
    return send_from_directory('static', 'index.html')


@app.route('/api/status')
def api_status():
    """Agent 完整状态快照."""
    if _agent is None:
        return jsonify({'error': 'Agent not initialized'}), 503
    return jsonify(_build_status(_agent))


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """发送人类消息，返回 Agent 回应."""
    if _agent is None or _broca is None:
        return jsonify({'error': 'Agent not ready'}), 503

    data = request.get_json() or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Empty text'}), 400

    global _loop
    if _loop is not None:
        result = _loop.interrupt_with_human_input(text, text_env=_get_text_env())
    else:
        result = {'response': '', 'error': 'No autonomous loop'}

    return jsonify({
        'response': result.get('response', ''),
        'valence': result.get('valence', 0.0),
        'arousal': result.get('arousal', 0.0),
    })


@app.route('/api/reading/start', methods=['POST'])
def api_reading_start():
    """开始阅读."""
    if _agent is None:
        return jsonify({'error': 'Agent not initialized'}), 503

    data = request.get_json() or {}
    file_path = data.get('file_path', '').strip()
    if not file_path:
        return jsonify({'error': 'file_path required'}), 400

    try:
        from tools.reader import Reader
        if not hasattr(_agent, 'reader') or _agent.reader is None:
            _agent.reader = Reader()
        _agent.reader.load(file_path)
        if _loop is not None:
            _loop.reader = _agent.reader
        return jsonify(_agent.reader.get_progress())
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/reading/stop', methods=['POST'])
def api_reading_stop():
    """停止阅读."""
    if _agent is not None and hasattr(_agent, 'reader') and _agent.reader is not None:
        _agent.reader.close()
    if _loop is not None:
        _loop.reader = None
    return jsonify({'status': 'stopped'})


@app.route('/api/reading/status')
def api_reading_status():
    """阅读进度."""
    if _agent is not None and hasattr(_agent, 'reader') and _agent.reader is not None:
        return jsonify(_agent.reader.get_progress())
    return jsonify({'is_active': False})


@app.route('/api/telemetry')
def api_telemetry():
    """最近遥测数据."""
    n = request.args.get('n', 200, type=int)
    if _agent is not None and hasattr(_agent, 'telemetry') and _agent.telemetry is not None:
        return jsonify({
            'recent': _agent.telemetry.get_recent_steps(n),
            'summary': _agent.telemetry.get_summary(n),
            'session': _agent.telemetry.get_session_info(),
        })
    return jsonify({'error': 'Telemetry not available'}), 503


@app.route('/api/memory')
def api_memory():
    """记忆网络概览."""
    if _agent is None:
        return jsonify({'error': 'Agent not initialized'}), 503

    net = _agent.net
    top10 = []
    if net.n_clusters > 0:
        top = sorted(net.clusters, key=lambda c: c.activation, reverse=True)[:10]
        for c in top:
            top10.append({
                'activation': float(c.activation),
                'count': int(c.count),
                'age': int(getattr(c, 'age', 0)),
                'G_ema': float(c.G_ema),
            })

    return jsonify({
        'n_clusters': net.n_clusters,
        'total_activation': float(net.total_activation),
        'n_semantic': (_agent.semantic_memory.n_clusters
                      if hasattr(_agent, 'semantic_memory') else 0),
        'n_self_model': _agent.self_model.net.n_clusters,
        'top10': top10,
    })


@app.route('/api/reading/list', methods=['POST'])
def api_reading_list():
    """列出指定目录下的文本文件."""
    data = request.get_json() or {}
    directory = data.get('directory', '.').strip()

    # v6.6: 安全 — 解析真实路径并限制在允许目录内
    try:
        real_dir = os.path.realpath(directory)
    except (ValueError, OSError):
        return jsonify({'error': 'Invalid directory path'}), 400

    # 允许的目录前缀列表
    _allowed_prefixes = [
        os.path.realpath(_PROJECT_ROOT),
        os.path.realpath(os.path.expanduser('~')),
        os.path.realpath(os.getcwd()),
    ]

    if not any(real_dir.startswith(prefix) for prefix in _allowed_prefixes):
        return jsonify({'error': 'Directory not allowed (outside safe paths)'}), 403

    try:
        files = []
        for f in os.listdir(real_dir):
            if f.endswith(('.txt', '.md', '.csv')):
                fpath = os.path.join(real_dir, f)
                if os.path.isfile(fpath):
                    files.append({
                        'name': f,
                        'path': os.path.abspath(fpath),
                        'size': os.path.getsize(fpath),
                    })
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/reading/upload', methods=['POST'])
def api_reading_upload():
    """v6.6: 上传文件 (拖入) → 自动识别并处理.

    支持: .txt/.md/.csv → 文本阅读管道
          .jpg/.png → 视觉管道
          .wav/.mp3 → 听觉管道
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext in ('.txt', '.md', '.csv', '.json', '.log', '.py',
                   '.yaml', '.yml', '.toml', '.cfg', '.ini'):
            # 文本文件 → 阅读管道
            content = file.read().decode('utf-8')
            # 截断过长内容
            max_chars = 4000
            if len(content) > max_chars:
                content = content[:max_chars]

            from tools.reader import Reader
            global _agent, _loop
            if _agent is not None:
                if not hasattr(_agent, 'reader') or _agent.reader is None:
                    _agent.reader = Reader()
                # 将内容写入临时文件供 Reader 使用
                import tempfile
                tmp_path = os.path.join(tempfile.mkdtemp(), filename)
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                _agent.reader.load(tmp_path)
                if _loop is not None:
                    _loop.reader = _agent.reader
                return jsonify({
                    'status': 'loaded',
                    'filename': filename,
                    'chars': len(content),
                    **(_agent.reader.get_progress() if _agent.reader else {}),
                })
            return jsonify({'error': 'Agent not initialized'}), 503

        elif ext in ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'):
            # 图像文件 → 视觉管道
            from PIL import Image
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((128, 128))
            frame = np.array(img, dtype=np.float32) / 255.0
            if _agent is not None:
                _agent.set_current_image(frame)
                return jsonify({'status': 'loaded', 'filename': filename,
                               'type': 'image'})
            return jsonify({'error': 'Agent not initialized'}), 503

        elif ext in ('.wav', '.mp3', '.flac', '.ogg', '.m4a'):
            # 音频文件 → 听觉管道
            import tempfile as tmp_mod
            tmp_path = os.path.join(tmp_mod.mkdtemp(), filename)
            file.save(tmp_path)
            try:
                from tools.audio_io import load_audio_file
                audio_data = load_audio_file(tmp_path)
                if audio_data is not None and _agent is not None:
                    _agent.set_audio_input(audio_data)
                    return jsonify({'status': 'loaded', 'filename': filename,
                                   'type': 'audio',
                                   'duration': audio_data.get('duration', 0)})
            except ImportError:
                return jsonify({'error': 'audio_io not available'}), 500
            except Exception as e:
                return jsonify({'error': str(e)}), 400

            return jsonify({'error': 'Audio load failed'}), 400

        else:
            return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ============================================================
# SSE (Server-Sent Events)
# ============================================================

@app.route('/api/stream')
def api_stream():
    """SSE 实时推送 Agent 状态."""
    def generate():
        while True:
            if _agent is None:
                time.sleep(0.5)
                continue

            status = _build_status(_agent)
            status['visual'] = _build_visual(_agent)
            status['audio'] = _build_audio(_agent)

            yield f"data: {json.dumps(status, ensure_ascii=False)}\n\n"
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'X-Accel-Buffering': 'no'})


@app.route('/api/stream/visual')
def api_stream_visual():
    """SSE 高频视觉帧推送."""
    def generate():
        while True:
            if _agent is None:
                time.sleep(0.2)
                continue

            visual = _build_visual(_agent)
            if visual.get('has_frame'):
                yield f"data: {json.dumps(visual, ensure_ascii=False)}\n\n"
            time.sleep(0.2)

    return Response(generate(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'X-Accel-Buffering': 'no'})


# ============================================================
# 启动逻辑
# ============================================================

def init_agent(fresh: bool = False):
    """初始化 Agent + 自主循环.

    v6.6: 默认从最新存档恢复 (持久化). --fresh 跳过存档.
    """
    global _agent, _broca, _loop, _loop_thread, _save_path

    from cns.agent import Agent
    from cns.data_types import BodyVector
    from cerebrum.frontal_lobe.broca import Broca
    from environments.text_interface import TextEnvironment
    from cns.innate import apply_innate_config
    from cns.data_types import ACTION_DIRECTIONS
    from cns.persistence import load_agent_state, restore_agent

    # v6.6: 尝试从 web 专用存档恢复
    loaded_from_save = False
    if not fresh:
        from cns.persistence import _make_path
        _save_path = _make_path('web_autosave')
        if os.path.exists(_save_path):
            print(f"  [Load] Found save: {os.path.basename(_save_path)}", flush=True)
            try:
                save_data = load_agent_state(_save_path)
                loaded_from_save = True
                print(f"  [Load] v{save_data.get('version', '?')}, "
                      f"{save_data.get('meta_learner', {}).get('step_count', 0)} steps, "
                      f"{save_data.get('total_turns', 0)} turns", flush=True)
            except Exception as e:
                print(f"  [Load] Failed: {e} — creating fresh agent", flush=True)
                loaded_from_save = False

    # Agent
    rng = np.random.default_rng(42)
    _agent = Agent(rng=rng, agent_id=0, n_agents=1)
    _agent.body = BodyVector(mode='text')
    apply_innate_config(_agent)
    _agent.record_action_consequence = lambda s: None
    ACTION_DIRECTIONS[3] = [0.0, 0.0]
    ACTION_DIRECTIONS[4] = [0.0, 0.0]

    # v6.6: 从存档恢复 Agent 状态
    if loaded_from_save:
        try:
            restore_agent(_agent, save_data, verbose=True)
        except Exception as e:
            print(f"  [Load] Restore failed: {e}", flush=True)
            loaded_from_save = False

    # Broca (纯净模式)
    te = TextEnvironment(load_corpus=False)
    _broca = Broca(text_env=te, load_corpus=False)

    # v6.4: InternalLife + Telemetry + Reader
    from cerebrum.association.internal_life import InternalLife
    from tools.telemetry import Telemetry
    from tools.reader import Reader

    _agent.internal_life = InternalLife()
    _agent.telemetry = Telemetry()
    _agent.reader = Reader()

    # 初始化传感器占位 (摄像头/麦克风默认不活跃)
    _agent._current_image = None
    _agent._current_audio_data = None
    _agent._streaming = False
    _agent._last_monologue = ''  # v6.6: 最近独白文本

    n_clusters = _agent.net.n_clusters
    steps_info = f"{_agent.meta.step_count} steps" if loaded_from_save else "0 steps"
    print(f"  Agent initialized: {n_clusters} clusters, "
          f"clean mode, {steps_info}", flush=True)

    # Autonomous loop
    from entry.autonomous import AutonomousLoop
    _loop = AutonomousLoop(_agent, broca=_broca, steps_per_second=1)
    _loop.reader = _agent.reader
    _loop.telemetry = _agent.telemetry
    _loop.internal_life = _agent.internal_life

    # v6.6: 恢复 loop 状态 (step_counter + social_ctx)
    if loaded_from_save and 'loop_state' in save_data:
        try:
            _loop.restore_from_save(save_data['loop_state'])
            print(f"  Loop: step_counter={_loop._step_counter} restored", flush=True)
        except Exception as e:
            print(f"  Loop restore: {e}", flush=True)
    # v6.6: 同步 loop._step_counter ← agent.meta.step_count (以 agent 为准)
    if _agent is not None and hasattr(_agent, 'meta') and _agent.meta is not None:
        _loop._step_counter = max(_loop._step_counter, _agent.meta.step_count)

    print("  AutonomousLoop ready", flush=True)


def start_autonomous_loop():
    """在后台线程启动自主循环."""
    global _loop, _loop_thread

    if _loop is None:
        init_agent()

    def run_loop():
        _loop.run(blocking=True)

    _loop_thread = threading.Thread(target=run_loop, daemon=True)
    _loop_thread.start()
    print("  Autonomous loop started (background thread)", flush=True)


# ---- v6.6: Auto-save ----

def save_web_state():
    """保存当前 Agent + Loop 状态到磁盘 (自动调用)."""
    global _agent, _loop, _save_path
    if _agent is None:
        return
    try:
        from cns.persistence import save_agent
        extra = {}
        if _loop is not None:
            extra['loop_state'] = _loop.get_state_for_save()
        if _save_path is None:
            from cns.persistence import _make_path
            _save_path = _make_path('web_autosave')
        path = save_agent(_agent, path=_save_path, extra=extra)
        print(f"  [Save] Agent state -> {os.path.basename(path)} "
              f"({_agent.meta.step_count} steps, "
              f"{_agent.net.n_clusters} clusters)", flush=True)
    except Exception as e:
        print(f"  [Save] Failed: {e}", flush=True)


# ---- Sensor Capture (摄像头 + 麦克风) ----

_sensor_thread = None
_sensor_running = False
_has_camera = False
_has_mic = False


def _sensor_capture_loop():
    """后台线程: 持续采集摄像头帧 + 麦克风音频，写入 agent 供 SSE 读取.

    使用 queue.Queue 进行线程安全的音频数据传输。
    摄像头帧直接写入 agent._current_image (原子赋值).
    """
    global _agent, _sensor_running, _has_camera, _has_mic
    import time as _time
    from queue import Queue, Empty

    cam = None
    mic_stream = None
    mic_sr = 22050
    mic_chunk = 512
    mic_queue = Queue(maxsize=50)  # 线程安全队列，上限 50 块

    # --- 并行打开摄像头和麦克风 (提速) ---
    print("  [Sensor] Opening camera + microphone (may take 1-3s)...", flush=True)

    cam_result = {'cam': None, 'ok': False}
    mic_result = {'stream': None, 'ok': False, 'callback': None}

    def _open_camera():
        try:
            from tools.sensor_io import CameraInput
            c = CameraInput(camera_id=0, fps=5.0, resolution=(128, 128))
            if c.open():
                cam_result['cam'] = c
                cam_result['ok'] = True
        except ImportError:
            pass
        except Exception:
            pass

    def _open_mic():
        try:
            import sounddevice as sd

            def mic_callback(indata, frames, time_info, status):
                if status:
                    return
                try:
                    mic_queue.put_nowait(indata.copy())
                except Exception:
                    pass

            stream = sd.InputStream(
                samplerate=mic_sr, channels=1, blocksize=mic_chunk,
                callback=mic_callback, dtype='float32',
                latency='low')
            stream.start()
            mic_result['stream'] = stream
            mic_result['ok'] = True
        except ImportError:
            pass
        except Exception:
            pass

    # 并行打开
    t_cam = threading.Thread(target=_open_camera, daemon=True)
    t_mic = threading.Thread(target=_open_mic, daemon=True)
    t_cam.start()
    t_mic.start()
    t_cam.join(timeout=8.0)
    t_mic.join(timeout=8.0)

    if cam_result['ok']:
        cam = cam_result['cam']
        _has_camera = True
        print("  [Sensor] Camera opened (128x128 @ 5fps, DSHOW)", flush=True)
    else:
        print("  [Sensor] No camera found (or opencv-python missing)", flush=True)

    if mic_result['ok']:
        mic_stream = mic_result['stream']
        _has_mic = True
        print(f"  [Sensor] Microphone opened ({mic_sr}Hz, mono)", flush=True)
    else:
        print("  [Sensor] No microphone (or sounddevice missing)", flush=True)

    if not _has_camera and not _has_mic:
        print("  [Sensor] No sensors available — visual/audio panels show placeholders", flush=True)
        _sensor_running = False
        return

    # 标记 streaming 活跃 (触发 autonomous loop 进入 streaming 模式)
    if _agent is not None:
        _agent._streaming = True

    print("  [Sensor] Capture loop started", flush=True)

    # --- 采集循环 ---
    last_cam_read = 0.0
    cam_interval = 0.2  # 5fps

    while _sensor_running:
        try:
            # 摄像头帧 (非阻塞)
            if _has_camera and cam is not None:
                now = _time.time()
                if now - last_cam_read >= cam_interval:
                    try:
                        frame = cam.read_frame()
                        if frame is not None and _agent is not None:
                            _agent._current_image = frame
                    except Exception:
                        pass
                    last_cam_read = now

            # 麦克风音频 (从线程安全队列中排空)
            if _has_mic:
                chunks = []
                try:
                    while True:
                        chunk = mic_queue.get_nowait()
                        chunks.append(chunk)
                except Empty:
                    pass

                if chunks and _agent is not None:
                    audio_chunk = np.concatenate(chunks, axis=0).ravel()
                    if len(audio_chunk) >= 64:
                        # 降采样到 128 点波形
                        step = max(1, len(audio_chunk) // 128)
                        waveform = audio_chunk[::step][:128].astype(np.float32)
                        # Mel 频谱近似: FFT → 32 bins
                        fft = np.abs(np.fft.rfft(audio_chunk))
                        n_fft_bins = len(fft)
                        mel = np.zeros(32, dtype=np.float32)
                        bin_step = max(1, n_fft_bins // 32)
                        for i in range(32):
                            seg = fft[i*bin_step:(i+1)*bin_step]
                            mel[i] = float(np.mean(seg)) if len(seg) > 0 else 0.0
                        mel = mel / (np.max(mel) + 1e-8)
                        rms = float(np.sqrt(np.mean(np.square(audio_chunk))) + 1e-8)
                        _agent._current_audio_data = {
                            'waveform': waveform,
                            'spectrum': mel,
                            'rms_level': rms,
                            'azimuth': 0.0,
                        }

            _time.sleep(0.05)  # 20Hz 采集循环，主动释放 GIL

        except Exception:
            _time.sleep(0.5)

    # --- 清理 ---
    _has_camera = False
    _has_mic = False
    if cam is not None:
        try:
            cam.release()
        except Exception:
            pass
    if mic_stream is not None:
        try:
            mic_stream.stop()
            mic_stream.close()
        except Exception:
            pass

    print("  [Sensor] Capture stopped", flush=True)


def start_sensors():
    """启动传感器采集线程."""
    global _sensor_thread, _sensor_running
    if _sensor_running:
        return
    _sensor_running = True
    _sensor_thread = threading.Thread(target=_sensor_capture_loop, daemon=True)
    _sensor_thread.start()


def stop_sensors():
    """停止传感器采集."""
    global _sensor_running, _sensor_thread
    _sensor_running = False
    if _sensor_thread is not None:
        _sensor_thread.join(timeout=2.0)
        _sensor_thread = None


def _kill_existing_on_port(port: int):
    """自动清理占用目标端口的陈旧进程 (Windows + Linux)."""
    import subprocess, platform
    killed = False
    try:
        if platform.system() == 'Windows':
            # 找到占用端口的 PID
            result = subprocess.run(
                ['netstat', '-ano'], capture_output=True, text=True, timeout=5)
            for line in result.stdout.splitlines():
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    pid = parts[-1]
                    if pid.isdigit() and int(pid) != os.getpid():
                        print(f"  [Cleanup] Killing stale process on port {port} (PID {pid})...", flush=True)
                        subprocess.run(['taskkill', '/F', '/PID', pid],
                                      capture_output=True, timeout=5)
                        killed = True
                        import time as _t
                        _t.sleep(0.5)
        else:
            # Linux/macOS: lsof + kill
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'], capture_output=True, text=True, timeout=5)
            pids = result.stdout.strip().split()
            for pid in pids:
                if pid.isdigit() and int(pid) != os.getpid():
                    print(f"  [Cleanup] Killing stale process on port {port} (PID {pid})...", flush=True)
                    subprocess.run(['kill', '-9', pid], capture_output=True, timeout=5)
                    killed = True
                    import time as _t
                    _t.sleep(0.3)
    except Exception:
        pass
    if killed:
        print("  [Cleanup] Port freed, continuing...", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NotMe v6.6 Web Interface')
    parser.add_argument('--port', type=int, default=8080, help='HTTP port')
    parser.add_argument('--host', default='0.0.0.0', help='Bind address')
    parser.add_argument('--fresh', action='store_true', help='Force fresh agent')
    parser.add_argument('--no-auto', action='store_true',
                       help='Disable autonomous loop')
    parser.add_argument('--no-sensors', action='store_true',
                       help='Disable camera + microphone capture')
    parser.add_argument('--dev', action='store_true',
                       help='Force Flask dev server (skip Waitress)')
    args = parser.parse_args()

    # ---- 端口冲突自动清理 ----
    _kill_existing_on_port(args.port)

    print("=" * 50)
    print("  NotMe v6.6 — Web Interface")
    print("=" * 50)

    # Init
    init_agent(fresh=args.fresh)

    # Start autonomous loop FIRST
    if not args.no_auto:
        start_autonomous_loop()

    # Print startup banner (use localhost for 0.0.0.0)
    display_host = 'localhost' if args.host in ('0.0.0.0', '::') else args.host
    print(f"\n  [Web] Open: http://{display_host}:{args.port}", flush=True)
    print("  Press Ctrl+C to stop\n", flush=True)

    # Start sensors AFTER everything else is ready (avoids GIL contention during init)
    if not args.no_sensors:
        # Small delay to let agent stabilize
        import time as _stime
        _stime.sleep(0.3)
        start_sensors()

    # Use production WSGI server (Waitress) when available,
    # fall back to Flask dev server with warning.
    use_waitress = False
    if not args.dev:
        try:
            from waitress import serve
            use_waitress = True
        except ImportError:
            print("  [INFO] Waitress not installed. Using Flask dev server.")
            print("  [INFO] Install with: pip install waitress")
            print("  [INFO] Or use --dev to suppress this message.\n")

    # ---- 注册退出清理 (确保 Ctrl+C / 终端关闭 / kill 都能清理) ----
    import atexit as _atexit
    import signal as _signal

    def _cleanup():
        """退出前清理所有资源."""
        stop_sensors()
        # v6.6: 保存状态到磁盘
        save_web_state()
        if _loop:
            try:
                _loop.stop()
            except Exception:
                pass
        if _agent and _agent.telemetry:
            try:
                _agent.telemetry.flush()
            except Exception:
                pass

    _atexit.register(_cleanup)

    # 捕获 SIGTERM (kill 命令) 和 SIGINT (Ctrl+C)
    def _sig_handler(signum, frame):
        print("\n  Shutting down...", flush=True)
        _cleanup()
        print("  Goodbye!", flush=True)
        os._exit(0)

    _signal.signal(_signal.SIGTERM, _sig_handler)
    _signal.signal(_signal.SIGINT, _sig_handler)

    try:
        if use_waitress:
            print("  Using Waitress production WSGI server")
            serve(app, host=args.host, port=args.port, threads=4)
        else:
            app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass  # 信号处理器已处理


if __name__ == '__main__':
    main()
