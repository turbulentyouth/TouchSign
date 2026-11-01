# 差分图采集、训练、推理与EXE开发验证

日期：2026-10-31，Asia/Shanghai。

## 已实现

| 环节 | 入口与行为 |
| --- | --- |
| 独立采集 | `collect.py` 直接打开 OpenCV 窗口。按键校准、选类、单拍或连拍，V 切换原图和差分，保存 PNG、基准及 session/episode 清单 |
| 接触门控 | 多帧中值空载基准、噪声阈值、变化面积、全局变化拒绝、稳定时间、释放即时清除 |
| YOLO训练 | `train.py` 训练 14 类 YOLO11n 分类模型。训练与验证目标比例为 80/20，按完整会话或完整按压分组 |
| 导出与拒识 | ONNX 输出类别概率和特征向量，使用训练集建立类别原型，使用验证集选择并评估拒识阈值 |
| 轻量推理 | `infer.py`及`core/inference.py`，CPU ONNX Runtime，不导入PyTorch、Ultralytics |
| 主GUI | `main.py`，默认差分，可切原图，空载校准、模型路径、重载，真实状态和CSV日志 |
| EXE | `TouchSign.spec`及`tools/build_exe.ps1`；干净运行环境，训练依赖和CUDA模块审计，应用数据写入用户目录 |

采集、训练、推理均使用 `cv2.absdiff(current, baseline)` 的 uint8 BGR三通道绝对差分，不做逐帧幅值拉伸。训练读取时会核验保存图像与对应原图/基准的逐像素一致性。

## 真实传感器验证

用户明确确认了传感器空载，并选择暂不进行生肖牌子接触测试。

- `icspring camera`，成功读取640×480×3图像；DirectShow和MSMF均可取帧。
- 独立脚本先预热30帧，以30帧空载图建立中值基准，再检查100帧。
- 100帧中接触误触发 **0次**，最大变化面积比例 **0.0**，最大归一化差分分数 **0.0085002161**。
- 保存了10张真实空载原图与差分图，属于一个session和一个空载episode，未拆分成虚假的独立批次。
- 现场数据路径：`data/raw/20261031_120120_8ccefe2d6e/`。
- 最终 `infer.py --headless --calibrate-empty --max-frames 20` 实机执行成功，状态为 `no_contact`，`zodiac=null`，`new_record=false`。缺少模型时没有生成任何生肖。

以上只证明当时空载状态的拒识行为，不代表生肖识别率、不同压力下接触召回率或未知物体总体误接收率。

## 模型工程烟测

实际执行 `train.py --smoke`，完成YOLO11n分类CPU训练1轮、ONNX导出、特征提取、阈值校准与留出评估。此流程使用带明确标记的合成数据和64像素输入，验证代码，不评估真实传感器准确率。

- ONNX模型约 **5.93 MiB**。
- 对5个输入比对PyTorch与ONNX两个输出，最大绝对误差 **2.98e-8**。
- ONNX checker通过；工程产物标记 `synthetic=true`、`status=unvalidated`，运行时明确拒绝作为生产模型加载。
- 报告：`.cache/model-smoke/20261031_115900_099412/report.json`。
- 真实空载样本已经通过训练输入真实性检查；缺失12生肖、未知物体和足够独立session时，训练入口会列出不足并终止。

## 自动检查

在项目目录运行测试。

```powershell
.\.venv-train\Scripts\python.exe -m unittest discover -s tests -t . -v
```

测试覆盖单拍和连拍、空载采样、真实差分保存、设备重连、像素去重、训练集合隔离、陌生特征拒识、重复帧确认和释放后的识别状态。

源码编译检查通过；采集测试也在不含训练框架的`.venv-runtime`中通过。干净环境导入主GUI、采集、识别状态机和独立推理后，`sys.modules`中没有torch、torchvision或ultralytics。

## GUI与EXE

- 独立采集入口在 Windows OpenCV 原生窗口中从相机 0 预览了 8 帧不同的实机图像，分辨率为 1600 × 1200。窗口使用微软雅黑，实机预览期间未采样入库。
- 已在 Windows 原生 OpenCV 窗口中使用模拟传感器采集样本，核对了中文生肖、快捷键、保存反馈和计数。字体检查可以识别缺少中文字形的字体，并提示通过 `--font` 选择本机字体。[原生窗口截图](collection-shortcuts.png)
- 主GUI完整ASGI页面返回200，校准/差分/重载控件存在；空白静态图返回200，无帧快照返回503。
- 创建了干净`.venv-runtime`，仅安装运行及打包依赖；版本锁定在 `requirements-runtime-lock.txt`。
- 此前实际构建的主识别程序 `dist/TouchSign/TouchSign.exe` 的发布目录约 **273.4 MiB、698个文件**，未包含真实训练模型。EXE本体约14.3MiB，必须与整个`_internal`目录一起分发。
- 打包模块归档及外部文件检查未发现PyTorch、torchvision、Ultralytics、训练代码或CUDA运行库。
- EXE `--help` 正常；完整EXE启动后，主页HTTP200（106095字节）、静态资源HTTP200、真实相机快照HTTP200（80338字节）。日志中没有异常或缺失依赖。
- 上述测试服务已停止，未留下占用传感器的验收进程。

Windows 安装路径、模型放置和设备连接说明见[部署指南](camera-permissions-and-exe-deployment.md)。

## 当前边界与下一步

早期实机验收使用了空载样本。后续训练使用新采集的生肖、未知物体和空载数据。验证集覆盖全部类别且通过拒识门槛后，配对的 `models/best.onnx` 和 `best.json` 可接入 GUI。

当前没有真实生肖或未知物体的准确率结论，没有完成无Python目标电脑、只读安装目录、原生WebView2窗口及不同USB设备的跨机验收。有限负样本中零误接收不能保证所有未来未知物体都能被拒绝；应保留独立实物与环境测试，并补采失败案例。
