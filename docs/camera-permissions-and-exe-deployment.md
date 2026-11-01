# Windows 采集与识别程序部署

## 采集窗口

在项目目录运行 `collect.py`，OpenCV 会打开 **TouchSign Capture** 窗口。默认相机编号为 0，多相机环境可用 `--camera 1` 指定其他设备。传感器若被主识别界面或系统相机程序占用，请先释放设备，再启动采集。确认 Windows 的相机隐私设置允许桌面应用访问相机。

```powershell
.\.venv-runtime\Scripts\python.exe collect.py --camera 0
```

窗口中的中文使用系统字体。可用 `--font 'C:\Windows\Fonts\msyh.ttc'` 指定微软雅黑，也可传入其他含简体中文字形的 TTF、TTC 或 OTF 文件。关闭窗口或按 `Q` 释放设备。

## 主识别程序 EXE

用 `requirements-build.txt` 创建干净的 Windows 运行及打包环境。使用 `tools/build_exe.ps1` 构建 CPU ONNX 原图预览版。构建输入为 `models/raw/best.onnx`、同名配套 `best.json` 和 `config/raw/config.yaml`。具体命令见[项目说明](../README.md)。

将 `dist/TouchSign.exe` 单个文件分发给用户，权重、界面资源和运行依赖已经内嵌。缓存及识别记录写入 `%LOCALAPPDATA%/TouchSign`。

普通用户直接启动 EXE 后，主识别界面默认在系统浏览器中显示。主界面可选的 `--native` 模式使用 WebView2。目标机应具备与 ONNX Runtime 匹配的 Visual C++ 运行库。

保持 TouchSign 控制台窗口打开以维持服务，关闭该窗口退出程序。首次启动会自动解压依赖，请等待浏览器界面出现。先确认传感器空载并校准，再压稳生肖牌子。内置原图模型尚未达到正式发布验证标准，因此界面显示「模型预览」。

## 目标机检查

1. 在普通桌面账户下打开传感器，完成空载校准。检查相机画面、中文显示、样本和日志保存位置。
2. 分别验证稳定生肖接触、未知物体、空载和断流。确认退出后其他程序能重新打开设备。
3. 构建脚本会将 EXE 的完整性标签设置为 Medium。本机曾因输出继承 Low 标签，在双击时出现 `Could not create temporary directory!` 并立即退出。修正标签后，已使用标准用户临时目录、默认 8080 端口和正常相机扫描验证启动成功。验收时应保留默认临时目录，避免仅用隔离目录测试。

历史本机验收结果见[开发验证记录](implementation-validation.md)。
