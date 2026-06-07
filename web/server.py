"""
server.py — NotMe v6.4 Web 界面 (Flask REST API + SSE 实时推送)

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

    # SCN/VLPO
    circa = {
        'phase': float(getattr(agent._circadian_state, 'circadian_phase', 0.0)),
        'melatonin': float(getattr(agent._circadian_state, 'melatonin', 0.0)),
        'cortisol': float(getattr(agent._circadian_state, 'cortisol', 0.0)),
        'sleep_pressure': float(getattr(agent._circadian_state, 'sleep_pressure', 0.0)),
        'sleep_propensity': float(getattr(agent._circadian_state, 'sleep_propensity', 0.0)),
    }
    sleep = {
        'is_asleep': agent.vlpo.is_asleep if hasattr(agent, 'vlpo') else False,
        'is_rem': agent.vlpo.is_in_rem if hasattr(agent, 'vlpo') else False,
        'state': agent._sleep_state.state if hasattr(agent, '_sleep_state') else 'awake',
        'time_in_state': int(getattr(agent._sleep_state, 'time_in_state', 0)),
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
        'rpe': float(agent._vta_result.get('rpe', 0.0)
                    if agent._vta_result else 0.0),
        'tpn_act': float(agent.tpn.tpn_activation if hasattr(agent, 'tpn') else 0.0),
        'dmn_act': float(1.0 - (agent.tpn.tpn_activation
                               if hasattr(agent, 'tpn') else 0.3)),
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

    # 视觉通道强度
    vis_result = getattr(agent, '_current_visual_result', {})
    diag = vis_result.get('diagnostics', {})
    if diag:
        v1_mean = float(diag.get('V1_mean_norm', 0.0))
        v2_mean = float(diag.get('V2_mean_norm', 0.0))
        v4_mean = float(diag.get('V4_mean_norm', 0.0))
        it_mean = float(diag.get('IT_mean_norm', 0.0))

    return {
        'has_frame': has_frame,
        'frame_b64': frame_b64,
        'v1_mean': v1_mean,
        'v2_mean': v2_mean,
        'v4_mean': v4_mean,
        'it_mean': it_mean,
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
    try:
        files = []
        for f in os.listdir(directory):
            if f.endswith(('.txt', '.md', '.csv')):
                fpath = os.path.join(directory, f)
                if os.path.isfile(fpath):
                    files.append({
                        'name': f,
                        'path': os.path.abspath(fpath),
                        'size': os.path.getsize(fpath),
                    })
        return jsonify({'files': files})
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
    """初始化 Agent + 自主循环."""
    global _agent, _broca, _loop, _loop_thread

    from cns.agent import Agent
    from cns.data_types import BodyVector
    from cerebrum.frontal_lobe.broca import Broca
    from environments.text_interface import TextEnvironment
    from cns.innate import apply_innate_config
    from cns.data_types import ACTION_DIRECTIONS

    # Agent
    rng = np.random.default_rng(42)
    _agent = Agent(rng=rng, agent_id=0, n_agents=1)
    _agent.body = BodyVector(mode='text')
    apply_innate_config(_agent)
    _agent.record_action_consequence = lambda s: None
    ACTION_DIRECTIONS[3] = [0.0, 0.0]
    ACTION_DIRECTIONS[4] = [0.0, 0.0]

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

    print(f"  Agent initialized: {_agent.net.n_clusters} clusters, "
          f"clean mode, 0 trigrams")

    # Autonomous loop
    from entry.autonomous import AutonomousLoop
    _loop = AutonomousLoop(_agent, broca=_broca, steps_per_second=1)
    _loop.reader = _agent.reader
    _loop.telemetry = _agent.telemetry
    _loop.internal_life = _agent.internal_life

    print("  AutonomousLoop ready")


def start_autonomous_loop():
    """在后台线程启动自主循环."""
    global _loop, _loop_thread

    if _loop is None:
        init_agent()

    def run_loop():
        _loop.run(blocking=True)

    _loop_thread = threading.Thread(target=run_loop, daemon=True)
    _loop_thread.start()
    print("  Autonomous loop started (background thread)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NotMe v6.4 Web Interface')
    parser.add_argument('--port', type=int, default=8080, help='HTTP port')
    parser.add_argument('--host', default='0.0.0.0', help='Bind address')
    parser.add_argument('--fresh', action='store_true', help='Force fresh agent')
    parser.add_argument('--no-auto', action='store_true',
                       help='Disable autonomous loop')
    parser.add_argument('--dev', action='store_true',
                       help='Force Flask dev server (skip Waitress)')
    args = parser.parse_args()

    print("=" * 50)
    print("  NotMe v6.4 — Web Interface")
    print("=" * 50)

    # Init
    init_agent(fresh=args.fresh)

    # Start autonomous loop
    if not args.no_auto:
        start_autonomous_loop()

    print(f"\n  Opening http://{args.host}:{args.port}")
    print("  Press Ctrl+C to stop\n")

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

    try:
        if use_waitress:
            print("  Using Waitress production WSGI server")
            serve(app, host=args.host, port=args.port, threads=4)
        else:
            app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        if _loop:
            _loop.stop()
        if _agent and _agent.telemetry:
            _agent.telemetry.flush()
        print("  Goodbye!")


if __name__ == '__main__':
    main()
