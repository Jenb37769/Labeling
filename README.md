# Photoshop GUI Agent 标注

这是一个完整的 Photoshop 教学视频到 GUI 标注数据的流水线，包含：
1. 视频抽帧 + 基于 UI 状态的差分过滤
2. 浏览器标注工具（带历史预填、只读锁定、审阅辅助）

当前流程以 Windows + PowerShell 本地运行作为默认假设。

## 目录结构

- `video/unlabel/`：待处理的视频
- `script/fast_process.py`：主流程
- `script/process_video.py`：单进程参考实现
- `final_unlabel/`：抽帧结果与 OmniParser 输出
- `final_label/`：已保存标注
- `label/`：浏览器标注 UI + 服务端
- `UNLOCK.py`：解锁已保存数据回到未标注
- `OmniParser/`：模型和依赖
- `hist.json`：历史 icon 统计与预填数据
- `load_counts.json`：图片加载次数

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
pip install pillow
```

## OmniParser 权重

需要提前准备 OmniParser v2 权重。

下载权重：
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

## 运行：视频抽帧

将视频放入 `video/unlabel/`，执行：
```powershell
python g:\grounding\script\fast_process.py
```

输出：
- `final_unlabel/ima/*.png`：抽帧图片
- `final_unlabel/data/*_omniparser.json`：元素解析结果
- `final_unlabel/total/*_overlay.png`：可选 overlay

在运行之后，视频将会被放在`video/label/`，表明已经被完整的处理过。视频处理期间，需要大量GPU算力，不建议4080及以下卡
在使用的过程中将GPU_WORKER_COUNT设置为2，具体设置见fast_process.py。

## 运行：标注工具

启动服务：
```powershell
python g:\grounding\label\server.py
```

浏览器打开：
```
http://127.0.0.1:8765
```

### 标注行为说明（重要）

- **保存后即锁定为只读**，不可再次修改。
- 顶部显示当前图片加载次数。
- 加载未标注图片时，会从 `hist.json` 的前 1000 条记录做预填：
  - 匹配条件：`type` 相同 + 宽/高/面积误差 <= 10%
  - 中心点距离 <= `min(width,height) * 8%`
  - 多个命中时：选 `count` 最大的，若相同取最前面
  - 命中后自动预填 `name/raw_type/region/clickable`，并将框标成蓝色

### 框颜色规则

- **浅黄色**：未匹配
- **蓝色**：历史预填匹配成功
- **浅绿色**：已读/已修改
- **红色**：当前选中（最高优先级）

只有当所有颜色均为红色或绿色是才能被保存，在1.1版中，不允许对已经保存的图片再次进行修改。

### 快捷键

- `Delete`：删除当前框
- `←`：上一个框
- `→`：下一个框
- `Ctrl/Cmd+S`：保存

## 解锁已保存数据

```powershell
python g:\grounding\UNLOCK.py --dry-run
python g:\grounding\UNLOCK.py
```

## hist 统计规则

每条 hist 记录的唯一性由以下条件共同决定：
- `type` / `name` / `raw_type` / `region` / `clickable`
- size（宽/高/面积误差 <= 10%）
- center 距离 <= `min(width,height) * 8%`

## 反馈

发现问题或有改进建议，请截图并提供复现步骤，及时联系彭远睿。
