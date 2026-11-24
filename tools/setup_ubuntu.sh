#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv-runtime/bin/python ]]; then
    python3 -m venv .venv-runtime || {
        echo '请先安装：sudo apt install python3-venv python3-pip libgl1 libglib2.0-0 fonts-noto-cjk v4l-utils can-utils' >&2
        exit 1
    }
fi
env -u PYTHONPATH .venv-runtime/bin/python -m pip install -r requirements-ubuntu.txt
echo '安装完成。运行 bash tools/start_ubuntu.sh 启动中文界面。'
