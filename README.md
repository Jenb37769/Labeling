# Photoshop GUI Agent 标注

这是一个把 Photoshop 教学视频逐步整理成 GUI 标注数据的本地流水线仓库。当前主流程已经拆成三层目录：

1. `processing/1.final_unlabel`
2. `processing/2.prelabel`
3. `processing/3.final_label`

整体链路是：

1. 从视频中抽帧并用 OmniParser 解析元素，输出到 `1.final_unlabel`
2. 使用 LLM 对候选元素做预标注，输出到 `2.prelabel`
3. 浏览器标注工具直接读取 `2.prelabel`，人工确认后保存到 `3.final_label`

当前默认运行环境为 Windows + PowerShell。

## 目录结构

- `video/unlabel/`：待处理的视频
- `script/fast_process.py`：主抽帧与筛帧流程
- `script/process_video.py`：单进程参考实现
- `script/prelabel.py`：单元素预标注脚本
- `script/test_prelabel_batch.py`：批量预标注脚本，直接写入 `processing/2.prelabel`
- `script/prelabel_read.py`：辅助转换脚本
- `processing/1.final_unlabel/`：抽帧结果与 OmniParser 原始输出
- `processing/2.prelabel/`：LLM 预标注结果，当前标注网站直接读取这里的 `data/` 和 `ima/`
- `processing/3.final_label/`：人工保存后的最终标注
- `label/`：浏览器标注 UI 和服务端
- `UNLOCK.py`：将已保存数据退回重新标注
- `DELETE.py`：删除未锁定数据
- `OmniParser/`：模型及相关依赖
- `hist.json`：历史统计与预填数据
- `load_counts.json`：图片加载次数统计
- `trans.py`：把带 `prelabel` 的 OmniParser JSON 转成标注可读 JSON 的辅助脚本

## 环境配置

推荐：

- Windows 10/11
- Python 3.12
- Conda

安装依赖：

```powershell
conda create -y -n omni python=3.12
conda activate omni
pip install -r .\OmniParser\requirements.txt
pip install pillow requests
```

## OmniParser 权重

需要提前准备 OmniParser v2 权重。

```powershell
cd g:\grounding\OmniParser
huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/train_args.yaml --local-dir weights
huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/model.pt --local-dir weights
huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/model.yaml --local-dir weights
huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/config.json --local-dir weights
huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/generation_config.json --local-dir weights
huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/model.safetensors --local-dir weights
Move-Item .\weights\icon_caption .\weights\icon_caption_florence
cd g:\grounding
```

## 第一步：视频抽帧到 `1.final_unlabel`

将视频放入 `video/unlabel/`，执行：

```powershell
python g:\grounding\script\fast_process.py
```

输出：

- `processing/1.final_unlabel/ima/*.png`：抽帧图片
- `processing/1.final_unlabel/data/*_omniparser.json`：OmniParser 元素结果
- `processing/1.final_unlabel/total/*_overlay.png`：原始可视化 overlay

说明：

- 这一步 GPU 占用较高
- 默认更适合较强显卡环境
- `GPU_WORKER_COUNT` 等参数可在相关脚本中调整

## 第二步：预标注到 `2.prelabel`

### 单元素预标注

```powershell
python g:\grounding\script\prelabel.py --stem "your_frame_stem"
```

### 批量预标注

当前推荐使用批量脚本：

```powershell
python g:\grounding\script\test_prelabel_batch.py --stem "your_frame_stem"
```

默认行为：

- 输入读取 `processing/1.final_unlabel`
- 输出直接写入 `processing/2.prelabel`
- `ima/`：原始图片
- `data/`：网站可直接读取的预标注 JSON
- `total/`：只画框的可视化图

当前批量预标注会：

- 把候选元素按 batch 发送给 LLM
- 每个 batch 发 `1` 张总览图和 `N` 张 crop 图
- 返回 `validity`、`type`、`name`、`clickable`、`instruction`、`reason`、`confidence`
- 只将 `validity == valid` 的元素写入最终 `2.prelabel/data`

支持中断保存：

- 脚本每完成一个 batch 就会刷新一次 `2.prelabel`
- 中途 `Ctrl+C` 也会保留当前已经完成的进度

## 第三步：浏览器标注

启动服务：

```powershell
python g:\grounding\label\server.py
```

浏览器打开：

```text
http://127.0.0.1:8765
```

当前标注服务读取：

- `processing/2.prelabel/ima`
- `processing/2.prelabel/data`

保存后写入：

- `processing/3.final_label/ima`
- `processing/3.final_label/data`

## 标注行为说明

- 保存后即锁定为只读，不允许再次修改
- 顶部显示当前图片加载次数
- 加载图片时会参考 `hist.json` 做历史预填

### 预填匹配规则

- `type` 相同
- 宽 / 高 / 面积误差 <= 10%
- 中心点距离 <= `min(width,height) * 8%`
- 多个命中时优先取 `count` 最大的项

### 框颜色

- 浅黄色：未匹配
- 蓝色：历史预填命中
- 浅绿色：已确认 / 已修改
- 红色：当前选中

### 快捷键

- `Delete`：删除当前框
- `←`：上一个框
- `→`：下一个框
- `Ctrl/Cmd+S`：保存

## 已保存数据回退

```powershell
python g:\grounding\UNLOCK.py --dry-run
python g:\grounding\UNLOCK.py
```

## hist 统计规则

每条 hist 记录的唯一性由以下条件共同决定：

- `type`
- `name`
- `raw_type`
- `region`
- `clickable`
- size（宽 / 高 / 面积误差 <= 10%）
- center 距离 <= `min(width,height) * 8%`

## 备注

- `trans.py` 可用于把带 `prelabel` 的 OmniParser JSON 手动转换成标注可读 JSON
- `script/prelabel_read.py` 目前更适合作为辅助工具，不是主流程入口

## 反馈

如果你发现问题或准备继续调整这条流水线，建议连同样例图片、对应 JSON 和复现步骤一起保存，后续排查会更快。
