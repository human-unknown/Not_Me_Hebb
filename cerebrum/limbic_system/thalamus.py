"""
thalamus.py — 丘脑 (Thalamus)  [待实现]

对应脑区: 丘脑诸核团
所属层级: 大脑 → 边缘系统 → 丘脑 (注: 丘脑是间脑结构，功能上常与边缘系统关联)

功能职责:
  - 感觉中继站 — 除嗅觉外的所有感觉在此中继 → 皮层
  - 感觉门控 — 调节信息流 (选择性传递/抑制)
  - 意识调节 — 丘脑-皮层节律 (睡眠纺锤波、觉醒)
  - 注意力 — 丘脑网状核(TRN) → 探照灯效应
  - 运动中继 — 基底节→丘脑→皮层运动回路

核心核团:
  感觉中继核:
    外侧膝状体 (LGN)         — 视觉中继 (视网膜→V1)
    内侧膝状体 (MGN)         — 听觉中继 (下丘→A1)
    腹后外侧核 (VPL)         — 躯体感觉中继 (身体→S1)
    腹后内侧核 (VPM)         — 面部感觉中继 (三叉神经→S1)
  运动核:
    腹前核 (VA)              — 基底节→运动皮层
    腹外侧核 (VL)            — 小脑→运动皮层
  联合核:
    丘脑枕 (Pulvinar)        — 视觉注意、多感官整合
    背内侧核 (MD)            — 前额叶中继
  非特异性核:
    丘脑网状核 (TRN)         — 感觉门控 (GABA 抑制性)
    板内核 (IL)              — 觉醒/意识

在 NotMe 中的待实现功能:
  1. 感觉门控: 根据注意力/唤醒度调节传入信息的增益
  2. 丘脑-皮层节律: 调制感知处理的时域动态
  3. Pulvinar 捷径: 低空间频率快速视觉通路 (已部分实现)
  4. 注意力探照灯: TRN → 选择性增强目标信号

当前状态:
  感觉信息目前直接流入 L0，没有门控层。
  已有 pulvinar 通路的初步实现 (在 visual_pathway.py 中)。

接口设计 (预留):
  class Thalamus:
      def gate_sensory(sensory_input, attention, arousal) -> gated_input
      def relay_visual(retina_output, gate) -> v1_input
      def relay_auditory(cochlear_output, gate) -> a1_input
      def relay_motor(basal_ganglia_output, cerebellum_output) -> motor_cortex_input
      def trn_modulation(attention_focus) -> gate_pattern

参考:
  - Sherman, S. M., & Guillery, R. W. (2002). The role of the thalamus in the
    flow of information to the cortex.
  - Crick, F. (1984). Function of the thalamic reticular complex: The
    searchlight hypothesis.

TODO 清单:
  [ ] SensoryGate: 感觉门控机制
  [ ] LGN: 外侧膝状体 (视觉中继)
  [ ] MGN: 内侧膝状体 (听觉中继)
  [ ] Pulvinar: 丘脑枕 (视觉注意)
  [ ] TRN: 丘脑网状核 (注意探照灯)
  [ ] Thalamocortical: 丘脑-皮层节律
"""

# 占位: 丘脑将在未来版本实现
# 当前感觉中继是直通 (passthrough) 的
