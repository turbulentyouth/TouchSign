# Ubuntu 数据采集与标注

采集界面沿用 `bash tools/start_ubuntu.sh` 的传感器预览、机械臂连接、示教、点位、夹爪参数以及停止/收回操作。右侧辨识结果改为人工标签与采集统计。原识别命令、模型、配置和旧 `collect.py` 快捷采集窗口继续可用。

## 启动

先在原程序点击“退出程序”，或在其终端按 Ctrl+C，等待程序退出，再运行：

```bash
bash tools/start_collect_ubuntu.sh
```

复用现有 `.venv-runtime`，不需要安装新依赖。默认读取 `config/ubuntu/config.yaml`，浏览器地址为 `http://127.0.0.1:8081`。两个 Ubuntu GUI 入口共用进程锁，避免竞争相机和 CAN；仅关闭浏览器页面不会退出后台服务。

```bash
# 指定保存目录及会话备注
bash tools/start_collect_ubuntu.sh --output data/raw --notes '操作者A / 第一批生肖牌'

# 可选：每轮目标、自动采样间隔、端口
bash tools/start_collect_ubuntu.sh --samples-per-cycle 3 --interval 0.5 --port 8081

# 只启动服务，不自动打开浏览器
bash tools/start_collect_ubuntu.sh --no-show

# 汇总所有已有会话，不打开相机或机械臂
bash tools/start_collect_ubuntu.sh --output data/raw --summary
```

## 采集一轮

1. 连接 VBTS，移开传感器上的所有物体。可先填写右侧“会话备注”，再点击“确认空载并校准”。每次校准新建独立会话，并清空所选标签。
2. 在“展开机械臂设置”中，按原来的方式选择 NERO、连接、示教和保存检测点/收回点，再结束示教。沿用原有已保存点位和夹爪参数；采集时需要改参数，仍使用这里的设置。
3. 选择真实物体的标签即可采集，无需额外勾选确认。支持十二生肖、未知物体和空载。标签由操作者提供，采集过程不加载识别模型，也不根据预测自动打标签。
4. 点击“采集（运行一轮）”。依次到检测点、按原限速闭合、确认夹爪到位、等待稳定接触、保存样本、张开、确认接触解除、返回收回点。到位前不采样，释放和收回阶段不写入样本。
5. 查看本轮结果、当前会话总数和各类数量。下一轮可更换姿态或实物；更换标签后可直接采集，运行期间标签与采集参数锁定。

默认每轮保存 **1 张**，优先增加独立接触次数、位置和姿态。每轮目标可设 1–20 张，但自动采集会跳过与上一张过于相似的画面，固定夹持不保证达到多张目标。等待上限沿用 `robot.recognize_timeout`（默认 8 秒）；超时保留已经保存的样本，随后按原流程松开并收回，不继续收紧夹爪。

“保存一张（手动补采）”用于机械臂空闲时采集当前新帧，同样要求校准、选择标签和有效接触；每次点击最多保存一张。采集 `no_contact` 时，移开所有物体、等待空载稳定后使用此按钮，不运行机械臂。`unknown` 应放置非生肖物体并形成稳定接触。

“停止本轮并收回”继续使用原有停止流程；之前已经保存的样本保留。断流、设备重连或切换后，旧基准和所选标签失效，需要重新空载校准并选择标签。写盘失败会停止采集并显示错误，沿用原有机械臂故障保持逻辑。离线模拟可演练动作和按钮，但不会生成训练样本。

## 保存格式与训练

默认保存到项目的 `data/raw/`，每次校准使用新的目录，不覆盖已有会话：

```text
data/raw/<session_id>/
  session.json            # 相机、备注、接触设置、图像尺寸、基准哈希
  baseline.png            # 空载基准，无损 BGR
  raw/<label>/*.png        # 相机原图，沿用当前链路的 180° 方向处理
  difference/<label>/*.png # 原图与本会话基准的绝对差分
  manifest.jsonl          # 人工标签、会话、接触段、帧序号、接触与稳定状态等
```

复用 `CollectionController` 和 `DatasetWriter`，与旧采集清单完全兼容。`image` 指向差分图，`raw_image` 指向原图，图像不经过 GUI 的 JPEG 预览压缩。界面数字是当前会话统计；所有会话总量使用 `--summary` 查看。重新校准会使当前会话计数从零开始，已有数据仍在磁盘。

在已有训练环境中，原图训练仍使用：

```bash
python train.py --data data/raw --input raw --prepare-only
python train.py --data data/raw --input raw --model yolo11n-cls.pt --epochs 80 --batch 32 --device 0 --seed 42
```

应为各类别采集多个独立接触段，并跨会话采集，供现有训练程序按会话/接触段划分数据。不要将同一次按压的连续帧随机分到训练集和验证集。

## 实现与验证

- 独立启动：`collect_gui.py`、`tools/start_collect_ubuntu.sh`。
- GUI 复用：`gui/collection_app.py` 继承 `TouchSignView` 和 `RobotPanel`；原入口默认仍使用原类。
- 机械臂复用：`CollectionRobotService` 仅替换到位后的识别等待阶段；移动、闭合、释放、收回、停止和断开沿用 `RobotService`。
- 接触与存储：`CollectionEngine` 适配原接触检测和数据采集控制器，不加载 ONNX，也不生成识别记录。
- 日志：`~/.local/share/TouchSign/logs/collection.log`，设置 `XDG_DATA_HOME` 时随应用数据目录变化。

验证命令：`.venv-runtime/bin/python -m unittest tests.test_collection_gui tests.test_collection tests.test_collection_input tests.test_robot_panel tests.test_robot_cycle tests.test_stop_and_return tests.test_ubuntu tests.test_browser_startup`。离线验证不能替代实物采集时对接触阈值、相机曝光及机械臂点位的验证。
