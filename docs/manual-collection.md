# 手工快速采集

只读取传感器图像，由人手工按压和选择标签，不连接机械臂，不加载识别模型，不需要空载校准或确认勾选。

## 开始

已有 Ubuntu 运行环境时，在项目目录的桌面终端直接运行：

```bash
bash tools/start_manual_collect_ubuntu.sh
```

默认自动查找 VBTS 传感器。窗口获得焦点后，用英文输入法操作：**选标签 → 按压 → 空格保存**。也可启动时指定标签，直接按压拍摄：

```bash
bash tools/start_manual_collect_ubuntu.sh --label rat
```

| 按键 | 操作 |
| --- | --- |
| `1 2 3 4 5 6 7 8 9 0 - =` | 鼠、牛、虎、兔、龙、蛇、马、羊、猴、鸡、狗、猪 |
| `U` / `N` | 未知物体 / 空载 |
| 空格 | 保存当前标签的一张图像 |
| `Z` / 退格（Backspace） | 撤回本会话最近一张，可连续撤回多张；同时停止连拍 |
| `R` | 开始连拍，默认每 0.2 秒最多一张 |
| `Esc` | 停止连拍 |
| `Q` 或关闭窗口 | 退出并等待已接受的保存任务完成 |

连拍由按键控制，不自动判断接触；压好后按 R，松开或换物体前按 Esc。切换标签自动停止连拍。界面显示本类、本会话已成功保存数量及待写入数量。

采错类别时：**按 Z 撤回 → 选择正确标签 → 重新按压，空格补拍**。撤回按本会话的采集顺序执行，即使已切换标签，也会撤回最后实际采集的图片。正在写入的样本会先完成，再执行撤回；连续按 Z 可逐张撤回。退出时会等待已接受的保存和撤回操作完成。

撤回成功后同步减少类别及会话计数，并从 `samples.jsonl` 移除对应记录，后续训练不会读取该样本。原图和原标签记录保留在当前会话的 `_undone/<原标签>/` 中，便于找回；不会直接改标，也不会撤回以前会话的数据。若出现“原图归档失败”，该样本仍已从训练清单移除，但图片保留在原位置。

原识别入口和机械臂采集入口保持原样。启动前退出正在使用传感器的其他 TouchSign 程序；新入口复用桌面进程锁，避免同时占用传感器。

## 保存内容

默认目录为 `data/manual/<时间_会话ID>/<英文标签>/*.png`。每次启动新建会话，不覆盖已有图片；`session.json` 记录传感器与旋转设置，`samples.jsonl` 逐行记录标签、采集时间、帧编号和图像路径（相对本会话目录）。

保存完整分辨率、无文字叠加的无损 PNG，**相对摄像头原始画面旋转 180°**，预览与保存方向一致。旋转复用 `CameraStream` 的现有处理，只执行一次。图像写入后台线程，最多保留四个待完成任务；磁盘忙时提示等待，连拍不会积压无限多帧。信号中断时停止连拍；写盘失败时提示错误并停止接受新样本。

这是独立的人工标注原图数据，不生成空载基准、差分图或自动接触判定。`train.py --input raw` 已支持直接读取此格式，训练不会再次旋转图像。原来的 `collect.py` 和 GUI 采集格式保持不变。

## 训练

使用独立 `.venv-train`，保持运行环境 `.venv-runtime` 不变。Ubuntu GPU 环境安装：

```bash
python3 -m venv .venv-train
env -u PYTHONPATH .venv-train/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128'
env -u PYTHONPATH .venv-train/bin/python -m pip install -r requirements-training-ubuntu.txt
```

**只有确认每张图都松开物体后重新按压拍摄，才使用 `--manual-grouping frame`。** 同一次按压的连拍图不能按独立样本随机划分；默认 `session` 将整个会话保留在同一侧，需要多个独立会话。空载图即使在 `frame` 模式下也按会话成组，防止同一空载画面泄漏到验证集。

```bash
env -u PYTHONPATH .venv-train/bin/python train.py \
  --data data/manual --input raw --manual-grouping frame \
  --model models/raw/best.pt --device 0 --epochs 80 --batch 32 --workers 4
```

仅当暂缺 `unknown` 样本、需要先微调生肖识别时，追加 `--allow-missing-unknown`。模型仍保持原有 14 类顺序，报告会记录缺失类别；十二生肖和空载样本仍然必需。缺少未知物体或独立空载验证样本时，模型不会通过正式发布检查，应补采后再验证拒识效果。

新训练输出到 `runs/` 下独立目录，包括 PyTorch 权重、`best.onnx`、配套 `best.json`、验证报告和可追溯的数据划分记录。验证报告分别列出首选类别准确率与经过拒识阈值后的正确接受率；同一验证集用于挑选权重和阈值，不能代替独立实机测试。

用上一轮权重续训时，可增加 `--reference-split 上一轮运行目录/dataset`，按图像像素哈希保留旧训练/验证分配，新增数据按完整按压分组。参考目录须与所用权重对应，本次输入须包含原有样本；不能将旧训练图重新划入验证集来评估续训效果。新增图若属于旧按压，会继承该按压的分配；每类仅新增两次独立按压时，两侧各保留一次以检查新采集条件，因此总体比例可能略偏离 80/20。参考样本缺失、改标或同次按压跨两侧时会报错，需要先核对数据。

少量新样本续训时，可用 `--learning-rate 0.0001 --warmup-bias-lr 0` 降低更新幅度；默认值仍为 `0.001` 和 `0.1`。只有一次新增按压时，该按压会进入训练，不能同时作为验证成绩。续训后应对照旧验证集，并检查正确类别的置信度是否仍达到 GUI 的确认门槛。

## 可选参数

```bash
# 查找实际设备编号
bash tools/start_manual_collect_ubuntu.sh --list-cameras
# 指定传感器、标签、输出位置与连拍间隔
bash tools/start_manual_collect_ubuntu.sh --camera 8 --label rat --output data/manual --interval 0.1
```

`8` 仅为示例，以设备列表为准。可用 `--config` 指定传感器配置；若提示缺少中文字体，使用 `--font /路径/中文字体.ttc`。无需安装新的 Python 依赖。
