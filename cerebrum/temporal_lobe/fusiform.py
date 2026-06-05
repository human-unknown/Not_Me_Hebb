"""
fusiform.py — 梭状回 (Fusiform Gyrus)  [待实现]

对应脑区: BA37 (梭状回面部区 FFA + 视觉词形区 VWFA)
所属层级: 大脑 → 颞叶 → 梭状回

功能职责:
  FFA (Fusiform Face Area):
    - 面孔识别 — 整体加工 (holistic)，非部分特征拼合
    - 面孔身份 — "这是谁"
    - 面孔表情 — 情绪解读 (与杏仁核协作)
    - 面孔熟悉度 — 熟悉 vs 陌生

  VWFA (Visual Word Form Area):
    - 文字识别 — 字形→语音/语义映射
    - 阅读习得 — 学习后激活 (先天性盲人无此区)
    - 跨语言通用 — 中文/英文/日文等均激活此区

在 NotMe 中的待实现功能:
  1. 面孔编码: 输入图像 → 面孔特征向量
  2. 面孔记忆: Hebb 网络中存储/识别已知面孔
  3. 表情识别: 与杏仁核协作 → 情感传染
  4. 文字识别: 图像中的文字 → 文本 (与韦尼克区协作)

当前状态:
  面孔处理和表情识别完全缺失。这是社交 Agent 的关键缺失组件。

接口设计 (预留):
  class FusiformGyrus:
      # FFA
      def face_encode(image_region) -> face_vector
      def face_recognize(face_vector) -> (identity, familiarity)
      def emotion_decode(face_vector) -> emotion_vector
      # VWFA
      def word_form_encode(text_image) -> orthographic_vector

参考:
  - Kanwisher, N., McDermott, J., & Chun, M. M. (1997). The fusiform face area.
  - Dehaene, S., & Cohen, L. (2011). The unique role of the visual word form
    area in reading.

TODO 清单:
  [ ] FaceDetector: 面孔检测 (Haar/HOG/MTCNN)
  [ ] FaceEncoder: 面孔特征编码
  [ ] FaceMemory: Hebb 面孔记忆网络
  [ ] EmotionDecoder: 面部表情→情绪
  [ ] VWFA: 视觉词形区
"""

# 占位: 梭状回将在未来版本实现
# 当前项目不进行面孔/文字图像识别
