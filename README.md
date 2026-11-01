# TouchSign · 触象

视触觉十二生肖识别，支持原图与差分图训练、CPU ONNX 推理及单文件 Windows EXE。

正式识别流程检查模型有效性、接触稳定性、拒识阈值与相机状态。下面的原图单文件版使用已训练模型的预览模式。

## 原图识别单文件版

构建产物为 `dist/TouchSign.exe`，内置 Python 运行环境、原图 ONNX 权重及配套清单。使用时只需复制这一个 EXE，无需安装 Python。双击后打开独立的 Native 窗口，使用 Windows WebView2 显示界面。首次启动需要等待内置文件解压。

2026-11-01 本机构建大小为 111.1 MiB。已通过 61 项自动测试，验证了 Native 窗口启动、模型加载、相机连接及关闭退出。单文件归档内的权重与原始导出逐字节核对一致，已有生肖、未知物体、空载图像回放结果正确。现场传感器识别效果仍需实物测试。

连接传感器，确认表面空载，点击「空载校准」，校准完成后压稳生肖牌子。默认显示原始画面。关闭 TouchSign 窗口即可退出程序并释放相机。

本版内置 `models/raw/best.onnx`，来自 `raw_20261031_155549_285715` 训练批次，输入格式为 `raw_bgr_v1`。保留了原始的 `unvalidated` 状态，界面采用「模型预览」，直接显示首选类别并跳过拒识阈值。该批次验证集缺少空载样本，使用原有拒识阈值时生肖正确接受率为 62.9%。仍需补采独立数据以验证正式识别表现。

GitHub 仓库包含代码、训练权重 `models/raw/best.pt`、推理权重 `best.onnx` 和配套 `best.json`。原始采集图像保留在本机。

从源码运行同一个原图界面可执行下面的命令。

```powershell
.\.venv-runtime\Scripts\python.exe main.py --config config/raw/config.yaml
```

## 启动独立采集界面

在项目目录运行：

```powershell
.\.venv-runtime\Scripts\python.exe collect.py
```

命令会直接打开 OpenCV 的 **TouchSign Capture** 窗口，默认读取相机 0。如果主识别界面正在使用传感器，请先在主界面断开设备。

[查看 OpenCV 采集窗口截图（使用模拟传感器）](docs/collection-shortcuts.png)

**空载按 C → 按类别键 → 压稳实物 → 按空格。** 例如按 `1` 选鼠，压稳鼠牌后，点按一次空格保存一张。保持鼠牌按压，松开空格再按即可保存下一张，窗口会更新本类张数。

| 按键 | 操作 |
| --- | --- |
| `C` | 确认表面完全空载并校准，新建采集会话；成功后显示差分图 |
| `1`–`9` | 鼠、牛、虎、兔、龙、蛇、马、羊、猴 |
| `0`、`-`、`=` | 鸡、狗、猪 |
| `U` / `N` | 未知物体接触 / 无物体接触（空载） |
| `←` / `→` | 选择上一类或下一类，选中类别会高亮 |
| `Space` | 每次独立按下保存一张符合当前标签的新帧，长按只触发一次 |
| `R` / `Esc` | `R` 开始或停止连拍，`Esc` 停止连拍。切换标签也会停止连拍 |
| `V` | 在原图和差分图之间切换预览 |
| `Q` / 关闭窗口 | 退出采集并释放相机 |

多相机环境可用 `--camera` 指定传感器编号。中文标签、状态和快捷键由 Pillow 使用系统中文字体绘制，Windows 优先使用微软雅黑。需要指定字体时，传入支持简体中文的 TTF、TTC 或 OTF 文件。

```powershell
.\.venv-runtime\Scripts\python.exe collect.py --camera 1 --font 'C:\Windows\Fonts\msyh.ttc'
```

连拍参数可在启动时设置。`--target` 是本轮目标张数，`--interval` 是采样最短间隔（秒），`--per-press` 是单次接触保存上限。`--notes` 可填写本次校准备注。使用 `--summary` 可查看已有数据统计。

按拍摄键即表示你已核对当前标签与实物相符。生肖和 `unknown` 必须稳定接触，`no_contact` 需要稳定空载。手动单拍只要相机已有新帧，即使画面相同也能再次保存。自动连拍会过滤相似图像，空载连拍至少间隔 2 秒。同一次按压已按某类保存后，需松开实物再换其他类别。每个样本仍保存原图、差分图及 `session_id` / `episode_id`。不同批次重新空载校准。

类别是十二生肖加 `unknown`（真实接触的非生肖物体）、`no_contact`（空载）。未知物体建议覆盖手指、平板、硬币、普通纹理和相似图案，验证时尽量包含训练未见过的物体。

默认数据保存至 `data/raw/<session>/`，同时保存无损原图、基准图、无损差分图、采集设置和逐帧清单。清单的 `image` 字段指向**差分图**，`raw_image` 指向原图。支持中文路径。

采集时让每个类别都有多个独立按压或空载段。训练程序按所选图像的解码像素去重，差分模式还核对差分图与对应原图、基准图是否一致。

## 差分处理与拒识

差分的唯一格式为 `absdiff_bgr_v1`：

```python
difference = cv2.absdiff(current_bgr_uint8, empty_baseline_bgr_uint8)
```

差分保持三通道和原始数值。训练和在线推理采用相同的全图等比例缩放、补边、BGR→RGB 和除以 255。OpenCV 窗口在独立画布上绘制预览和文字。加载模型时，程序读取 `best.json` 的 `input_kind`，给差分模型输入差分图，给原图模型输入相机原图。

接触检测对空载噪声进行估计，使用差分变化面积、变化量、稳定时长与多帧门控。全局图像变化过大时拒识，不将任意陌生画面直接判作有效接触。空载基准不会自动更新，以免把持续按压吸收成背景；换相机、重连、分辨率改变或断流需要重新校准。

YOLO11n 分类包含 14 类；模型还输出特征向量。只有通过以下所有检查才显示生肖：

- 相机帧新鲜、基准有效、确有光学接触变化且稳定；
- 预测为已知生肖，而非 `unknown` / `no_contact`；
- 置信度、第一与第二类别的差距、与真实训练特征原型的距离符合校准阈值；
- 连续多个新帧一致确认。重复帧不能增加确认次数，释放后立即清空结果。

每次接触只记一条日志。同一次接触中结果发生生肖变化，会要求松开重新接触，不沿用上一生肖的记录。图像方法不能证明物理接触或保证拒绝所有未来未知物体；需要用独立实物测试评估误接收率，并持续补采失败案例。界面不再显示虚构的法向力、温度和演示历史。

## YOLO 训练、导出与发布

训练环境沿用 `.venv-train`；训练允许使用 PyTorch，**运行和打包不使用它**。

```powershell
# 可选。核对采集图像并查看 80/20 划分及各类数量
.\.venv-train\Scripts\python.exe train.py --data data/raw --prepare-only

# 使用第一张 NVIDIA 显卡训练，训练结束自动导出 ONNX
.\.venv-train\Scripts\python.exe train.py --data data/raw --epochs 80 --batch 32 --device 0

# 用相同训练设置比较原图效果
.\.venv-train\Scripts\python.exe train.py --data data/raw --input raw --model yolo11n-cls.pt --epochs 80 --batch 32 --device 0 --seed 42

# 若要在验证通过后接入 GUI，可在训练命令末尾加 --deploy
```

默认使用 `yolo11n-cls.pt`，首次运行可能下载预训练权重；也可用 `--model` 指向本地权重。`--help` 查看所有参数。

`--input raw` 直接读取已有清单的 `raw_image` 训练和验证，结果单独保存到 `runs/raw_<时间>/`。比较两次训练的 `fit/results.csv` 中的验证 Top-1 和 Top-5。原图模型也可通过 `main.py --preview-model` 加载对应的 `best.onnx`。

程序只创建训练集和验证集，目标图像数分别为 **80% 和 20%**。优先按完整校准会话分组。会话不足时按完整按压或空载段分组，同一段的所有帧都在同一侧。只有一个独立段的类别留在训练集，验证报告会列出缺少的类别。为保持分组完整，实际比例以 `dataset/split.json` 为准。训练和预检使用相同的默认随机种子。

每次训练在 `runs/<时间>/` 生成训练权重 `fit/weights/best.pt`、`best.onnx`、配套 `best.json` 和 `report.json`。验证集用于选择训练轮次和拒识阈值。报告的接受率、错认率和混淆统计均来自该验证集。

默认发布要求验证集有全部 14 类，且每个生肖正确接受率 ≥80%、生肖错认率为 0、负样本误接收率为 0。缺少某类独立验证样本时仍可训练并导出供分析的 ONNX。补采后重新训练，使最佳轮次和拒识阈值与新的数据划分对应。

通过后 `--deploy` 将配对文件写入 `models/best.onnx` 与 `models/best.json`，并备份旧版本。GUI 点击“重新加载模型”即可；也可以通过“模型路径”加载其他通过验证的配套模型。清单校验输入方式、类别、模型哈希及验证状态，普通ImageNet权重、原始导出ONNX和合成烟测模型不能直接冒充生产权重。

仅验证工程链路：

```powershell
.\.venv-train\Scripts\python.exe train.py --smoke
```

这会实际训练1轮、导出和验证，但数据为合成，输出仅在 `.cache/model-smoke/`，不能发布。

## 主GUI与独立推理

要在 GUI 试用训练刚导出的权重，可直接预览本轮 ONNX。保持同目录的 `best.onnx` 和 `best.json` 配对：

```powershell
.\.venv-runtime\Scripts\python.exe main.py --preview-model runs/20261031_153518_710111/best.onnx
```

在浏览器中点击「空载校准」，待提示校准完成后压稳牌子。模型预览显示 14 类中分数最高的一类，包括未知物体和空载，跳过拒识阈值。也可在「模型路径」中填写 ONNX 路径并勾选预览。预览结果只显示在当前页面，正式识别使用下方的常规入口。

```powershell
.\.venv-runtime\Scripts\python.exe main.py
```

主界面默认 `http://127.0.0.1:8080`。连接传感器 → 空载校准 → 接触牌子。默认显示差分图，可切换原图。没有有效权重时显示模型未就绪；接触检测和差分预览仍可使用。识别记录保存至 `logs/recognition.csv`，只记录真正接受的接触。

不启动网页的独立推理：先确认表面空载，再运行：

```powershell
.\.venv-train\Scripts\python.exe infer.py --camera 0 --calibrate-empty
# 无窗口验收示例，20帧后退出
.\.venv-train\Scripts\python.exe infer.py --camera 0 --calibrate-empty --headless --max-frames 20
```

`infer.py` 只导入 NumPy、OpenCV、YAML 与 ONNX Runtime，不导入训练框架；可使用干净运行环境启动。打开窗口后按 Q / Esc 退出。相机重新连接或基准失效后必须重新确认空载。

## EXE 准备

```powershell
uv venv --python 3.12 .venv-runtime
uv pip install --python .venv-runtime\Scripts\python.exe -r requirements-build.txt
.\tools\build_exe.ps1
```

构建脚本检查干净的 CPU 推理环境，验证原图模型及其清单，并将它们内嵌到 `dist/TouchSign.exe`。默认使用 `config/raw/config.yaml` 启动配置。构建结束后逐字节核对归档内的权重和配置，并检查训练依赖是否被排除。

原图双输出 ONNX 约 5.93 MiB。完整 EXE 还包含 Python、OpenCV、ONNX Runtime、NiceGUI 和窗口依赖，脚本会打印实际大小。启动时自动将内部依赖解压到临时目录，缓存和日志写入 `%LOCALAPPDATA%/TouchSign`。EXE 默认以 Native 模式启动，目标机需要 Windows WebView2 Runtime。

需要复现本机依赖版本时，可用 `requirements-runtime-lock.txt` 替代 `requirements-build.txt` 安装。旧版目录构建的验收记录保存在开发验证文档中。

Windows 相机和发布包的检查步骤见[部署指南](docs/camera-permissions-and-exe-deployment.md)。

## 配置与验证

`config/config.yaml` 可调接触阈值、稳定时间、帧龄、模型路径、确认帧数及日志位置。差分阈值需要在真实正例和未知物体上验收，不能只凭空载测试确定。

```powershell
.\.venv-train\Scripts\python.exe -m unittest discover -s tests -t . -v
.\.venv-train\Scripts\python.exe main.py --list-cameras
```

早期实机验收使用 640×480 传感器画面，30 帧空载校准后连续检测 100 帧，接触误触发 0 次。训练后的生肖正确接受率和未知物体误接受率见对应训练目录的 `report.json`。[开发验证记录](docs/implementation-validation.md)

实现入口：`collect.py` / `train.py` / `infer.py` / `main.py`；核心算法在 `core/`，训练专用代码在 `training/`，真实界面在 `gui/`。

技术参考：[Ultralytics分类文档](https://docs.ultralytics.com/tasks/classify/)、[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)。
