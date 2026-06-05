"""
pallidum.py — 苍白球 (Globus Pallidus)  [待实现]

对应脑区: 外侧苍白球 (GPe) + 内侧苍白球 (GPi)
所属层级: 大脑 → 基底神经节 → 苍白球

功能职责:
  外侧苍白球 (GPe):
    - 间接通路中转站 — D2 MSN → GPe → STN
    - 节律生成 — 与 STN 构成起搏器 (beta 振荡, 帕金森相关)
  内侧苍白球 (GPi):
    - 基底节主要输出核 — 抑制性 GABA 投射到丘脑
    - 直接通路: 皮层→纹状体→GPi(抑制)→丘脑(去抑制)→皮层(Go)
    - 间接通路: 皮层→纹状体→GPe→STN→GPi(增强)→丘脑(抑制)→皮层(NoGo)

在 NotMe 中的待实现功能:
  1. GPi 输出门控: 对每个候选动作输出抑制/去抑制信号
  2. GPe-STN 振荡: 节律性动作时间调制
  3. 动作选择竞争: 多动作并行提议 → GPi 选择最强的

当前状态:
  动作门控由 MoEGate (action_gating.py) 实现，没有明确的 GPi/GPe 分离。

接口设计 (预留):
  class GlobusPallidus:
      def gpi_output(direct_input, indirect_input) -> thalamic_inhibition
      def gpe_relay(striatal_d2) -> stn_input
      def action_selection(competing_actions) -> selected_action

参考:
  - DeLong, M. R. (1990). Primate models of movement disorders of basal
    ganglia origin.
  - Nambu, A. (2004). A new dynamic model of the cortico-basal ganglia loop.

TODO 清单:
  [ ] GPi: 内侧苍白球输出门
  [ ] GPe: 外侧苍白球中继
  [ ] Oscillator: GPe-STN 振荡器
  [ ] WinnerTakeAll: 动作竞争选择
"""

# 占位: 苍白球将在未来版本实现
# 当前动作门控由 cerebrum.basal_ganglia.action_gating.MoEGate 实现
