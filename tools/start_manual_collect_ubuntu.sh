#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv-runtime/bin/python ]]; then
    echo '请先运行 bash tools/setup_ubuntu.sh 安装依赖。' >&2
    exit 1
fi
exec env -u PYTHONPATH .venv-runtime/bin/python collect_manual.py "$@"
