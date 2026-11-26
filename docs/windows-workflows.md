# Windows 免安装采集、训练和识别

完整目录是 `dist/TouchSign-Windows`，可复制到 Windows 10/11 x64，例如 `D:\TouchSign-Windows`。使用 ZIP 时先完整解压。运行时无需安装 Python、CUDA、PyTorch 或修改环境变量。普通用户权限即可；目录需可写，建议使用较短路径。

| 双击入口 | 操作 |
| --- | --- |
| `01_人工采集.bat` | 先看画面确认传感器，Tab 切换相机、Enter 确认；选生肖、人工按压、空格保存；R 连拍、Esc 停止、Z 撤回 |
| `02_一键训练.bat` | 自动合并历史数据、人工采集和纠错样本，小步续训并比较新旧模型，无需填写参数 |
| `03_识别与纠错.bat` | 打开本机浏览器，选传感器、空载校准、按压识别；误判时保存正确生肖与位置备注 |
| `04_回退上次模型.bat` | 退出其他窗口后，切换到上一次启用的模型及对应训练数据记录 |
| `05_运行自检.bat` | 检查依赖、原图、模型哈希、PyTorch/ONNX 输出、中文字体及 Windows 进程锁 |

退出识别时关闭命令窗口或按 Ctrl+C；仅关闭浏览器标签页不会停止服务。采集、识别、训练使用同一互斥锁，避免同时占用传感器或训练时修改输入。

保留 NERO 代码；这三个人工流程不控制机械臂，不需要连接 NERO，也不引入 RealMan SDK。参考目录中的 `.venv` 绑定原电脑 Python 路径，不能直接搬用。本包采用 [Python 官方嵌入式分发](https://docs.python.org/3.12/using/windows.html#the-embeddable-package)，直接展开锁定的 Windows wheel，附带 Microsoft 应用本地 C++ 运行库。使用系统自带浏览器，不需要 WebView2 环境配置。

## 数据和训练保护

随包使用本次最新原图模型，来源 `corrections_final_raw_20261126_121349`。模型与训练权重哈希见 [release.json](../models/raw/release.json)。模型依旧采用预览模式、50% 置信度、接触稳定及连续帧确认。

- `data/seed`：423 张原图，固定分配为 334 张训练、89 张验证；334 张包含本次 15 张纠错。
- `data/regression`：13 张公鸡回归检查图，始终不参与训练。
- `data/manual`、`data/corrections`：新的人工采集和识别纠错数据。
- `runs/easy_…`：本轮数据快照、训练日志、权重、ONNX、划分和逐图比较。
- `models/active.json`：原子切换当前版本及上一版本记录；随包原模型一直保留。

新采集按完整会话分组，纠错按记录的实际按压分组；旧图保留原分组和划分。像素哈希检测重复与标签冲突；撤回的数据不进入训练。缺图、损坏或原标签改变时停止，不擅自重新划分验证集。

一键训练使用 CPU、16 轮、学习率 0.00005、workers=0，无需显卡环境。自动重放旧数据；保留原固定验证集；导出后校验 PyTorch/ONNX 一致性。每一张原来识别正确的旧图都不能变错，原来达到 50% 的不能跌破门槛，正确置信度不能下降超过 15 个百分点。同时，新增样本达到 50% 的正确数必须提升，才启用候选模型。这些检查既比较当前版本，也比较随包原模型，防止多轮小幅退步累积。

检查不通过会保存结果并继续使用原模型；中断训练不改变当前模型。没有新数据则直接退出。采集不同位置时建议分多次独立会话，避免把一次连拍误当作独立验证证据。

新增样本的回放含训练数据，报告明确标为拟合检查。固定验证数据也参与选权重，不能据此宣称独立泛化能力。当前未知物体和空载独立验证不足，自动切换仅用于人工预览，不绕过正式发布门槛。

## 构建与验证

构建器是 `tools/build_windows_portable.py`；依赖输入为 `requirements-windows-portable.in`，锁定结果为 `requirements-windows-portable-lock.txt`。先在构建机下载 Python 3.12.10 x64 嵌入式 ZIP、锁定 Windows wheels，并从 Microsoft 官方 VC Redist 提取 x64 应用本地运行库，再运行构建器。目标电脑不执行安装命令。

构建产物的 `BUILD.json` 记录原始下载来源、依赖哈希和数据数量，`SHA256SUMS.txt` 用于交付完整性检查。训练数据、运行时二进制和 ZIP 只进入本地交付包，不加入 Git。

已完成 Linux 全套自动测试（278 项，1 项跳过），并在便携数据快照上完成真实 CPU 单轮训练、ONNX 导出及输出一致性检查。该工程试训没有启用到识别。包内 Windows Python 已在 Wine 中通过整包自检、模型输出对比和合成数据训练/ONNX 导出；Wine 测试单独关闭了其未实现的 OpenMP NUMA 查询，目标 Windows 不需要此设置。测试记录及目标平台限制见交付包的 `VALIDATION.json`；Linux 与 Wine 均不替代 Windows 真机 USB 传感器、相机权限和浏览器交互验收。

操作者短说明见 [windows-portable-quickstart.txt](windows-portable-quickstart.txt)。遇到问题保留 `logs/last-error.txt` 和本轮 `runs` 日志；备份时复制整个目录，包含当前模型对应的 `runs` 和 `data`。
