"""
autonomous.py — 自主时间流引擎 (v6.4)

管理 Agent 在无人交互时的自主时间推进。核心组件:
  - AutonomousLoop: 活动调度 + 模式切换
  - 支持 Web 模式 (后台线程) 和控制台模式 (阻塞循环)

Agent 的一天:

  觉醒期:
    ├─ 有活跃 Reader?      → reading 模式
    ├─ 传感器流活跃?       → streaming 模式
    ├─ DMN 主导?           → wandering (走神)
    ├─ TPN 主导 + 足够记忆? → monologue (内部独白)
    ├─ 负效价+高唤醒?      → rumination (情绪反刍)
    └─ 默认                → idle (轻量 step)

  睡眠期:
    └─ 跳过所有自主活动, VLPO 状态机自行推进

用法:
  # 控制台模式
  loop = AutonomousLoop(agent, broca=broca)
  loop.run(duration_steps=1000)

  # Web 后台线程模式
  loop = AutonomousLoop(agent, broca=broca)
  thread = threading.Thread(target=loop.run, kwargs={'blocking': True})
  thread.start()
"""

import time
import threading
import numpy as np
from typing import Optional


# ============================================================
# 常量
# ============================================================

DEFAULT_STEPS_PER_SECOND = 1     # 正常自主模式: 1 step/s (≈30s biological per step)
REALTIME_STEPS_PER_SECOND = 1.0 / 30.0  # 实时模式: 1 step ≈ 30s

# 模式切换阈值
DMN_DOMINANT_THRESHOLD = 0.45    # DMN 激活超此值 → 走神倾向
TPN_ACTIVE_THRESHOLD = 0.35      # TPN 激活超此值 → 任务倾向
MIN_CLUSTERS_FOR_MONOLOGUE = 5   # 内部独白最低集群数
RUMINATION_CHECK_INTERVAL = 20   # 每 N 步检查一次反刍条件

# 疲劳恢复
REST_RECOVERY_STEPS = 50         # 疲劳后休息步数


# ============================================================
# AutonomousLoop
# ============================================================

class AutonomousLoop:
    """自主时间流引擎。

    Attributes:
        agent: Agent 实例
        broca: Broca 实例 (内部独白需要)
        mode: 当前活动模式
        steps_per_second: 时间推进速度
        _running: 后台运行标志
    """

    def __init__(self, agent, broca=None,
                 steps_per_second: float = DEFAULT_STEPS_PER_SECOND):
        self.agent = agent
        self.broca = broca
        self.mode: str = 'idle'
        self.steps_per_second = steps_per_second

        # 子组件 (可外部注入)
        self.reader = getattr(agent, 'reader', None)
        self.telemetry = getattr(agent, 'telemetry', None)
        self.internal_life = getattr(agent, 'internal_life', None)

        # 状态
        self._running: bool = False
        self._step_counter: int = 0
        self._paused: bool = False  # 人类介入时暂停
        self._last_human_time: float = 0.0
        self._rest_counter: int = 0  # 疲劳休息倒计时

        # 追踪
        self.activity_history: list[str] = []
        self._rumination_check_counter: int = 0

        # 标记 Agent 处于自主模式
        agent._autonomous_mode = True

    # ---- 主循环 ----

    def run(self, duration_steps: int = None,
            blocking: bool = True,
            tick_callback=None) -> dict:
        """启动自主时间流。

        Args:
            duration_steps: 运行步数 (None = 无限)
            blocking: True = 阻塞运行, False = 单步非阻塞 (由外部循环调用)
            tick_callback: 每步回调 tick_callback(mode, stats)

        Returns:
            运行统计 dict
        """
        if blocking:
            return self._run_blocking(duration_steps, tick_callback)
        else:
            return self._run_single_tick(tick_callback)

    def _run_blocking(self, duration_steps: int = None,
                      tick_callback=None) -> dict:
        """阻塞运行自主循环。"""
        self._running = True
        start_time = time.time()
        stats = {'total_ticks': 0, 'mode_counts': {},
                 'sleep_ticks': 0, 'error_ticks': 0}

        try:
            while self._running:
                if duration_steps is not None and stats['total_ticks'] >= duration_steps:
                    break

                tick_result = self.tick()
                stats['total_ticks'] += 1
                mode = tick_result.get('mode', 'idle')
                stats['mode_counts'][mode] = stats['mode_counts'].get(mode, 0) + 1
                if tick_result.get('is_asleep'):
                    stats['sleep_ticks'] += 1
                if tick_result.get('error'):
                    stats['error_ticks'] += 1

                if tick_callback:
                    tick_callback(mode, tick_result)

                # 时间控制
                if self.steps_per_second > 0:
                    time.sleep(1.0 / self.steps_per_second)

        except KeyboardInterrupt:
            stats['interrupted'] = True
        finally:
            self._running = False

        stats['elapsed_seconds'] = time.time() - start_time
        stats['avg_steps_per_sec'] = (stats['total_ticks']
                                      / max(stats['elapsed_seconds'], 0.001))
        return stats

    def _run_single_tick(self, tick_callback=None) -> dict:
        """单步运行 (用于 Web 后台线程)。"""
        tick_result = self.tick()
        if tick_callback:
            tick_callback(self.mode, tick_result)
        return tick_result

    def tick(self) -> dict:
        """单步自主推进。

        Returns:
            dict with mode, activity, stats
        """
        agent = self.agent
        self._step_counter += 1

        # ---- 0. 人类介入检查 ----
        if self._paused:
            return {'mode': 'paused', 'activity': 'paused',
                    'step': self._step_counter, 'is_asleep': agent.vlpo.is_asleep}

        # ---- 1. 睡眠期 → 轻量 step + 跳过所有活动 ----
        if agent.vlpo.is_asleep:
            stats = agent.light_step(self._step_counter, activity='sleeping')
            if self.telemetry:
                self.telemetry.record_step(agent, activity='sleeping')
            return {
                'mode': 'sleeping',
                'activity': 'sleeping',
                'step': self._step_counter,
                'is_asleep': True,
                'sleep_state': agent._sleep_state.state,
                'stats': stats,
            }

        # ---- 2. 疲劳恢复 ----
        if self._rest_counter > 0:
            self._rest_counter -= 1
            stats = agent.light_step(self._step_counter, activity='resting')
            if self.telemetry:
                self.telemetry.record_step(agent, activity='resting')
            return {
                'mode': 'resting',
                'activity': 'resting',
                'step': self._step_counter,
                'is_asleep': False,
                'rest_remaining': self._rest_counter,
                'stats': stats,
            }

        # ---- 3. 决定活动模式 ----
        mode = self._decide_mode()
        self.mode = mode
        self.activity_history.append(mode)
        if len(self.activity_history) > 200:
            self.activity_history = self.activity_history[-200:]

        # ---- 4. 执行活动 ----
        result = self._execute_mode(mode)

        # ---- 5. 遥测 ----
        if self.telemetry:
            self.telemetry.record_step(agent, activity=mode)

        return result

    def _decide_mode(self) -> str:
        """决定当前活动模式。

        优先级:
          1. 活跃的 Reader → reading
          2. 传感器流 → streaming
          3. 情绪反刍条件 → rumination
          4. DMN 主导 → wandering
          5. TPN 主导 + 足够记忆 → monologue
          6. 默认 → idle
        """
        agent = self.agent

        # 1. Reading?
        if self.reader is not None and self.reader.state.is_active:
            if self.reader.is_active and agent.body is not None:
                if self.reader.should_read(
                    agent.body.b,
                    agent.tpn.tpn_activation if hasattr(agent, 'tpn') else 0.3
                ):
                    return 'reading'
                elif self.reader.state.is_paused:
                    # 尝试恢复
                    if self.reader.try_resume(
                        agent.body.b,
                        agent.tpn.tpn_activation if hasattr(agent, 'tpn') else 0.3
                    ):
                        return 'reading'
                # 书已完成
                if self.reader.is_finished:
                    return self._fallback_mode()

        # 2. Streaming? (传感器活跃)
        if hasattr(agent, '_streaming') and agent._streaming:
            return 'streaming'

        # 3. Rumination?
        self._rumination_check_counter += 1
        if self._rumination_check_counter >= RUMINATION_CHECK_INTERVAL:
            self._rumination_check_counter = 0
            v = agent.valence_history[-1] if agent.valence_history else 0.0
            a = agent.arousal_history[-1] if agent.arousal_history else 0.0
            if v < -0.2 and a > 0.5:
                return 'rumination'

        # 4-6. TPN/DMN 平衡
        return self._fallback_mode()

    def _fallback_mode(self) -> str:
        """基于 TPN/DMN 跷跷板的回退模式选择。"""
        agent = self.agent
        tpn_act = agent.tpn.tpn_activation if hasattr(agent, 'tpn') else 0.3
        dmn_act = 1.0 - tpn_act

        if dmn_act > DMN_DOMINANT_THRESHOLD:
            # DMN 主导 → 走神
            if agent.net.n_clusters >= 3:
                return 'wandering'
            return 'idle'
        elif tpn_act > TPN_ACTIVE_THRESHOLD:
            # TPN 主导 → 内部独白 (如果有足够记忆)
            if agent.net.n_clusters >= MIN_CLUSTERS_FOR_MONOLOGUE and self.broca is not None:
                return 'monologue'
            return 'idle'
        else:
            return 'idle'

    def _execute_mode(self, mode: str) -> dict:
        """执行指定活动模式。"""
        agent = self.agent

        if mode == 'reading':
            return self._do_reading()
        elif mode == 'wandering':
            return self._do_wandering()
        elif mode == 'monologue':
            return self._do_monologue()
        elif mode == 'rumination':
            return self._do_rumination()
        elif mode == 'streaming':
            return self._do_streaming()
        else:
            return self._do_idle()

    def _do_reading(self) -> dict:
        """执行阅读活动。"""
        agent = self.agent
        sentence = self.reader.next_sentence()
        result = {'mode': 'reading', 'step': self._step_counter,
                  'is_asleep': False, 'has_sentence': False}

        if sentence is not None:
            # 理解句子
            try:
                from environments.text_interface import TextEnvironment
                te = TextEnvironment(load_corpus=False)
                sent_vec = te.encode_text(sentence)[:64].astype(np.float32)
            except Exception:
                sent_vec = np.zeros(64, dtype=np.float32)
                sent_vec[:min(64, len(sentence))] = 0.01

            # 通过 comprehend 管线 (不含 TPJ/AG)
            try:
                comp_vec, understanding = agent.comprehend(
                    sent_vec,
                    human_sentiment=np.zeros(8, dtype=np.float32),
                    speaker_name="book",
                    human_text=sentence,
                )
                comprehension = min(1.0,
                    understanding.get('n_triggered_memories', 0) / 5.0)
            except Exception:
                comp_vec = sent_vec
                comprehension = 0.1

            # light_step with text input
            stats = agent.light_step(self._step_counter,
                                     activity='reading',
                                     text_input=sentence)

            # Log reading activity
            if hasattr(agent, '_log_activity'):
                agent._log_activity('reading', f"📖 {sentence[:80]}")

            # 记录理解深度
            self.reader.record_comprehension(comprehension)

            if self.telemetry:
                self.telemetry.record_reading(
                    sentence, comprehension,
                    self.reader.cognitive_load,
                    self.reader.progress,
                )

            result['has_sentence'] = True
            result['comprehension'] = comprehension
            result['sentence_len'] = len(sentence)
            result['stats'] = stats
        else:
            # 没有更多句子了
            stats = agent.light_step(self._step_counter, activity='idle')
            result['stats'] = stats

        return result

    def _do_wandering(self) -> dict:
        """执行走神活动。"""
        agent = self.agent
        result = {'mode': 'wandering', 'step': self._step_counter,
                  'is_asleep': False}

        if self.internal_life is not None:
            thought_result = self.internal_life.mind_wander(agent)
            result.update(thought_result)
        else:
            # 回退: agent.internal_thought
            thought_result = agent.internal_thought('wander')
            result.update(thought_result)

        # 轻量 step (传播走神的情感效应)
        stats = agent.light_step(self._step_counter, activity='wandering')
        result['stats'] = stats
        return result

    def _do_monologue(self) -> dict:
        """执行内部独白活动。"""
        agent = self.agent
        result = {'mode': 'monologue', 'step': self._step_counter,
                  'is_asleep': False}

        if self.internal_life is not None and self.broca is not None:
            thought_result = self.internal_life.internal_monologue(
                agent, broca=self.broca)
            result.update(thought_result)
        else:
            thought_result = agent.internal_thought('monologue', broca=self.broca)
            result.update(thought_result)

        # 轻量 step
        text = result.get('text', '')
        stats = agent.light_step(self._step_counter, activity='monologue',
                                 text_input=text if text else None)
        result['stats'] = stats
        return result

    def _do_rumination(self) -> dict:
        """执行情绪反刍活动。"""
        agent = self.agent
        result = {'mode': 'rumination', 'step': self._step_counter,
                  'is_asleep': False}

        if self.internal_life is not None:
            thought_result = self.internal_life.emotional_rumination(agent)
            result.update(thought_result)
        else:
            thought_result = agent.internal_thought('rumination')
            result.update(thought_result)

        stats = agent.light_step(self._step_counter, activity='rumination')
        result['stats'] = stats
        return result

    def _do_streaming(self) -> dict:
        """执行传感器流活动。"""
        agent = self.agent

        # 如果有当前摄像头帧，使用全 step()
        if (hasattr(agent, '_current_image') and
            agent._current_image is not None):
            sensory = np.zeros(516, dtype=np.float32)
            action = agent.step(sensory, self._step_counter)
            result = {'mode': 'streaming', 'step': self._step_counter,
                      'is_asleep': False, 'has_frame': True,
                      'action': action.index}
        else:
            stats = agent.light_step(self._step_counter, activity='streaming')
            result = {'mode': 'streaming', 'step': self._step_counter,
                      'is_asleep': False, 'has_frame': False,
                      'stats': stats}
        return result

    def _do_idle(self) -> dict:
        """执行空闲活动 (最轻量)。"""
        agent = self.agent
        stats = agent.light_step(self._step_counter, activity='idle')
        return {
            'mode': 'idle',
            'activity': 'idle',
            'step': self._step_counter,
            'is_asleep': False,
            'stats': stats,
        }

    # ---- 控制接口 ----

    def pause(self):
        """暂停自主循环 (人类介入)。"""
        self._paused = True
        self._last_human_time = time.time()

    def resume(self):
        """恢复自主循环。"""
        self._paused = False

    def stop(self):
        """停止自主循环。"""
        self._running = False

    def interrupt_with_human_input(self, human_text: str,
                                    text_env=None,
                                    social_ctx=None) -> dict:
        """人类介入 → 临时切换到交互模式，处理输入后恢复自主。

        Args:
            human_text: 人类输入文本
            text_env: TextEnvironment 实例 (可选)
            social_ctx: SocialContext 实例 (可选)

        Returns:
            dict with agent_response, comprehension, etc.
        """
        self.pause()
        agent = self.agent

        # ---- 睡眠唤醒检查 ----
        was_asleep = agent.vlpo.is_asleep
        sleep_state_before = None
        if was_asleep:
            sleep_state_before = agent._sleep_state.state if hasattr(agent, '_sleep_state') else 'nrem_n2'
            # 社会刺激 → 直接唤醒 (人是社会性动物, 有人跟你说话≈紧急唤醒信号)
            # 模拟: NE 瞬间飙升 → 觉醒中枢激活 → VLPO 被抑制
            agent.vlpo._is_asleep = False
            agent.vlpo.vlpo_activation = 0.0
            agent.vlpo.arousal_center_activity = 0.6
            agent.vlpo._transition_pending = 0
            if hasattr(agent, '_sleep_state'):
                agent._sleep_state.state = 'awake'
                agent._sleep_state.phase = 'none'
                agent._sleep_state.vlpo_activation = 0.0
                agent._sleep_state.rem_on_activity = 0.0
                agent._sleep_state.rem_off_activity = 0.0
            # 高唤醒启动 (模拟"被叫醒"的迷蒙状态)
            agent.body.b[4] = min(0.9, agent.body.b[4] + 0.2)  # 专注度↑

        try:
            from environments.text_interface import TextEnvironment
            from cerebrum.limbic_system.amygdala import (
                analyze_sentiment, sentiment_to_social_signal)
            from cerebrum.limbic_system.cingulate import SocialContext

            if text_env is None:
                text_env = TextEnvironment(load_corpus=False)
            if social_ctx is None:
                social_ctx = SocialContext(tau=25.0)

            # 编码
            human_vec = text_env.encode_text(human_text)[:64].astype(np.float32)
            sentiment = analyze_sentiment(human_text)
            social_signal = sentiment_to_social_signal(sentiment)[:8].astype(np.float32)
            social_ctx.update(sentiment.get('valence', 0.0),
                            sentiment.get('arousal', 0.0))

            # 理解
            comp_vec, understanding = agent.comprehend(
                human_vec, social_signal,
                speaker_name="human",
                human_text=human_text,
            )

            # 全管线 step
            F_before = agent.F_body_history[-1] if agent.F_body_history else 0.0

            sensory = np.zeros(516, dtype=np.float32)
            sensory[:64] = human_vec
            sensory[80:88] = social_signal

            action = agent.step(sensory, self._step_counter,
                               social_ctx=social_ctx)
            F_after = agent.F_body_history[-1] if agent.F_body_history else 0.0

            # 情感学习
            import jieba
            input_words = [w for w in jieba.lcut(human_text)
                          if len(w.strip()) >= 1]
            arousal_now = agent.arousal_history[-1] if agent.arousal_history else 0.5
            delta_F = F_before - F_after

            # 生成回应
            response = ""
            if action.index == 3 and self.broca is not None:
                if agent.net.n_clusters > 0:
                    top = max(agent.net.clusters, key=lambda c: c.activation)
                    if top.activation > 0.01:
                        n_trig = understanding.get('n_triggered_memories', 0)
                        comp_precision = 0.15 + min(1.0, n_trig / 5.0)
                        ctx_vec = agent.dialogue_ctx.get_context_vector()
                        ctx_precision = 0.15 + min(1.0,
                            agent.dialogue_ctx.n_turns() / 5.0)
                        self_anchor = agent.self_model.get_self_anchor()
                        n_self = agent.self_model.n_experiences
                        self_precision = 0.05 + min(0.20, n_self * 0.005)

                        total_p = max(comp_precision + ctx_precision
                                     + self_precision, 1e-6)
                        query = (comp_vec * (comp_precision / total_p)
                                + ctx_vec * (ctx_precision / total_p)
                                + self_anchor * (self_precision / total_p))
                        query = query.astype(np.float32)

                        v = agent.valence_history[-1] if agent.valence_history else 0
                        a = agent.arousal_history[-1] if agent.arousal_history else 0
                        sa = agent.self_arousal_ema
                        temp = 0.5 + abs(v) * 0.8 + sa * 0.2

                        words, _, speech_diag = agent.speak(
                            broca=self.broca,
                            query_vec=query,
                            belief_vec=top.centroid,
                            valence=v, arousal=a,
                            temperature=temp, max_words=20,
                            use_phrase_structure=True,
                            human_text=human_text,
                        )
                        response = "".join(words) if words else ""

            # 对话记忆
            if response:
                try:
                    resp_vec = text_env.encode_text(response).astype(np.float32)
                except Exception:
                    resp_vec = np.zeros(64, dtype=np.float32)
                agent.dialogue_ctx.add_turn(
                    human_text=human_text,
                    human_vec=human_vec,
                    human_sentiment=social_signal,
                    agent_response=response,
                    agent_vec=resp_vec,
                    agent_valence=agent.valence_history[-1] if agent.valence_history else 0,
                    agent_arousal=agent.arousal_history[-1] if agent.arousal_history else 0,
                    comprehension_vec=comp_vec,
                )

            # 遥测
            if self.telemetry:
                self.telemetry.record_dialogue({
                    'human_text_len': len(human_text),
                    'agent_response_len': len(response),
                    'comprehension_depth': understanding.get(
                        'n_triggered_memories', 0) / 5.0,
                    'n_triggered_memories': understanding.get(
                        'n_triggered_memories', 0),
                    'valence_before': float(F_before),
                    'valence_after': float(F_after),
                    'arousal_before': (agent.arousal_history[-2]
                                      if len(agent.arousal_history) >= 2 else 0.5),
                    'arousal_after': float(arousal_now),
                    'eval_score': None,
                })

            # Log activity
            if hasattr(agent, '_log_activity'):
                agent._log_activity('chat', f"人类: {human_text[:60]}")

            # 睡眠唤醒通知
            if was_asleep and response:
                # 自然语言提示: 被叫醒后可能比较迷糊
                wake_notice = "[被唤醒] "
                response = wake_notice + response

            return {
                'response': response,
                'comprehension': understanding,
                'valence': agent.valence_history[-1] if agent.valence_history else 0,
                'arousal': agent.arousal_history[-1] if agent.arousal_history else 0,
                'was_asleep': was_asleep,
                'sleep_state_before': sleep_state_before,
            }

        finally:
            self.resume()

    # ---- 状态查询 ----

    def get_state(self) -> dict:
        """获取自主循环状态摘要。"""
        mode_counts = {}
        for m in self.activity_history[-200:]:
            mode_counts[m] = mode_counts.get(m, 0) + 1

        return {
            'mode': self.mode,
            'step_counter': self._step_counter,
            'is_running': self._running,
            'is_paused': self._paused,
            'mode_counts': mode_counts,
            'reader_progress': self.reader.get_progress() if self.reader else None,
            'internal_life': (self.internal_life.get_state()
                            if self.internal_life else None),
        }

    def get_state_for_save(self) -> dict:
        """可序列化状态 (用于持久化)。"""
        return {
            'mode': self.mode,
            'step_counter': self._step_counter,
            'is_paused': self._paused,
            'rest_counter': self._rest_counter,
            'activity_history': self.activity_history[-50:],
        }

    def restore_from_save(self, data: dict):
        """从持久化数据恢复。"""
        if not data:
            return
        self.mode = data.get('mode', 'idle')
        self._step_counter = data.get('step_counter', 0)
        self._paused = data.get('is_paused', False)
        self._rest_counter = data.get('rest_counter', 0)
        self.activity_history = data.get('activity_history', [])
