# Photoshop GUI Agent

- 每次开始标注之前，请习惯性的拉取仓库，或者查看这个README文件，如果没有代码版本上的更新，则可以不用拉取。

- 标注过程的结果，请提交在网盘。

- 请不要修改任何代码。

- 如果发现使用过程中有任何BUG或者对某一个操作有改进意见，请截图并发送到邮箱2769832297@qq.com或者以任何方式联系彭远睿。


# Version

1.0     3月29日     完成项目基本搭建，上传到Gitee        彭远睿


# Environment
环境启动 请自行安装配套环境 Ominiparser需要的环境在 GROUNDING/Ominiparser/requirements.txt文件中表明

env环境样例启动行 请实际根据环境配置

```python
conda activate g:\grounding\OmniParser\.omni_env
```

# Run

## process video

```python
python g:\grounding\script\fast_process.py
```
## label

```python
python g:\grounding\label\server.py
```
然后用浏览器打开http://127.0.0.1:8765


