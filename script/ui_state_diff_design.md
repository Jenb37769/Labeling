# UI 状态差分设计

## 目标

`process_video.py` 不再做普通图像去重，而是做基于 UI 状态的去重。

- 画布内容本身的变化，通常不应该触发保留。
- 顶部栏、左侧工具栏、右侧面板、画布交互覆盖物的变化，应该触发保留。
- 即使只有少量 UI 变化，只要这些变化置信度高，也应该保留。

## 整体流程

1. 从 `video/unlabel` 中读取按文件名排序后的第一个视频。
2. 将该视频移动到 `video/label`。
3. 每隔 `FRAME_INTERVAL` 帧抽取一张图片，并保存到 `image/unlabel`。
4. 对抽出的图片运行 OmniParser。
5. 过滤明显无关的噪声元素。
6. 将当前帧的 OmniParser JSON 与上一张保留帧的 JSON 做比较。
7. 根据 UI 状态变化规则决定当前帧是保留还是丢弃。
8. 将 OmniParser JSON 和 diff JSON 保存在 `image/process/<video_stem>/` 下。

## OmniParser 元素标准化

每个标准化后的元素保留这些核心字段：

- `name`
- `bbox`
- `center`
- `raw_type`
- `clickable`
- `confidence`
- `region`
- `source`

## 噪声过滤

先过滤掉通常不属于 Photoshop 标注目标的内容：

- 底部任务栏和播放器控制区
- 长字幕文本
- 顶部视频标题或全屏提示横幅
- 明显属于 YouTube 的提示元素

这一步的目标不是做到绝对精确，而是尽量减少重复帧判断时的噪声。

## 区域划分

每个元素会被分配到以下区域之一：

- `top_bar`
- `left_toolbar`
- `right_panel`
- `canvas_overlay`
- `other`

其中 `other` 的权重会被刻意压低，避免边缘噪声或无关元素把变化分数抬高。

## 元素匹配

系统不会直接比较 JSON 文本，而是先在两帧之间做元素匹配。

匹配信号包括：

- `bbox IoU`
- 中心点距离
- 文本相似度
- 区域一致性

只有匹配分数高于阈值的两个元素，才会被视为同一个元素。

## 变化类型

元素匹配完成后，变化分为三类：

- `added`
- `removed`
- `modified`

其中 `modified` 主要用于表示：

- 位置明显变化
- 所属区域变化
- 画布交互覆盖物发生移动

## 加权变化分数

每个变化事件的分数由以下 4 部分共同决定：

- `change_type_weight`
- `type_weight`
- `region_weight`
- `confidence`

当前默认偏好如下：

- `icon > text`
- `left_toolbar / top_bar / right_panel / canvas_overlay > other`
- `added / removed > modified`
- `high confidence > low confidence`

## 保留规则

当前帧只要满足以下任意一个条件，就会被保留：

1. `total_change_score` 足够高。
2. `significant_change_count` 足够多。
3. 虽然变化不多，但 `high_conf_change_count` 和 `high_conf_change_score` 足够高。
4. 在重要区域出现了新的高置信元素。

这套规则同时覆盖两种目标：

- 当很多 UI 元素发生变化时保留。
- 当只有少量 UI 元素变化，但这些变化很可信时也保留。

## 优先调参项

后续最值得优先调的参数包括：

- `FRAME_INTERVAL`
- `HIGH_CONFIDENCE_THRESHOLD`
- `TOTAL_CHANGE_KEEP_THRESHOLD`
- `SIGNIFICANT_CHANGE_COUNT_THRESHOLD`
- `REGION_WEIGHTS`
- `BOTTOM_NOISE_START_RATIO`
- `SUBTITLE_START_RATIO`

如果后面发现画布交互覆盖物仍然漏检，优先增强 `canvas_overlay` 的规则，而不是退回到整图像素相似度。
