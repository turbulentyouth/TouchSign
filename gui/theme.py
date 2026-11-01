"""页面主题与装饰层。

视觉设计系统原本写在 ``gui/style.css`` 中，面向的是独立 HTML 页面；NiceGUI
会在 body 内再包一层 Quasar 布局，因此这里补充少量宿主样式并注入全局装饰。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from nicegui import ui

from gui.dom import Text, element

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STYLE_PATH = PROJECT_ROOT / 'gui' / 'style.css'


@lru_cache(maxsize=1)
def _design_css() -> str:
    return STYLE_PATH.read_text(encoding='utf-8')


_HOST_CSS = """
/* ---- NiceGUI host document ---------------------------------------------- */
html,
body {
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
}

#app,
.q-layout,
.q-page-container,
.q-page {
  width: 100%;
  height: 100%;
  min-height: 0;
  margin: 0;
  padding: 0;
}

.nicegui-content {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 0;
  margin: 0;
  padding: 0;
  gap: 0;
  overflow: hidden;
}

/* Quasar 的排版规则给 h3 设置了 3.125rem 行高，会撑高候选/最近识别标题。 */
.candidates-head h3,
.recent-head h3 {
  line-height: 1.25;
}

/* 原生标签继承 Quasar 的 1.5 行高；候选行按设计稿收紧。 */
.candidate-row {
  line-height: 1.2;
  min-height: 15px;
}

/* 窄屏会滚动，深色滚动条更符合整体视觉。 */
* {
  scrollbar-width: thin;
  scrollbar-color: rgba(212, 181, 120, 0.3) rgba(3, 10, 8, 0.35);
}

::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: rgba(3, 10, 8, 0.35);
}

::-webkit-scrollbar-thumb {
  border-radius: 4px;
  background: rgba(212, 181, 120, 0.26);
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(212, 181, 120, 0.44);
}

/* Keep the fixed-height console layout on short or narrow screens. */
@media (max-width: 1180px) {
  html,
  body {
    height: auto;
    min-height: 100%;
  }

  .q-page,
  .nicegui-content {
    height: auto;
    min-height: 100vh;
    overflow: visible;
  }
}

/* ---- Runtime states ----------------------------------------------------- */
.app.is-sensor-offline .sensor-view > img {
  filter: grayscale(1) brightness(0.42) contrast(1.06);
}

.app.is-sensor-offline .scan-line {
  opacity: 0;
}

.app.is-sensor-offline .feed-readout {
  opacity: 0.55;
}

.live-chip.is-offline {
  border-color: rgba(217, 94, 70, 0.28);
  color: var(--vermillion-bright);
  background: rgba(90, 30, 22, 0.16);
}

.live-chip.is-offline i {
  background: var(--vermillion-bright);
  box-shadow: 0 0 8px rgba(217, 94, 70, 0.8);
}

.system-center.is-offline {
  border-color: rgba(217, 94, 70, 0.16);
}

.system-center.is-offline .system-pulse {
  border-color: rgba(217, 94, 70, 0.28);
}

.system-center.is-offline .system-pulse i {
  background: var(--vermillion-bright);
  box-shadow: 0 0 9px rgba(217, 94, 70, 0.75);
}

.system-center.is-offline .system-copy strong {
  color: #d7a79b;
}

.match-chip.is-rejected {
  border-color: rgba(217, 94, 70, 0.25);
  color: #c88773;
  background: rgba(90, 30, 22, 0.15);
}

.match-chip.is-rejected i {
  background: var(--vermillion-bright);
  box-shadow: 0 0 6px rgba(217, 94, 70, 0.7);
}

.result-state.is-rejected {
  color: var(--vermillion-bright);
}

.result-state.is-rejected i {
  background: var(--vermillion-bright);
  box-shadow: 0 0 8px rgba(217, 94, 70, 0.6);
}

.result-seal.is-updating {
  animation: seal-stamp 0.55s var(--ease);
}

.name-line h2.is-updating {
  animation: title-in 0.4s var(--ease);
}

@keyframes seal-stamp {
  0% {
    transform: scale(0.94);
    opacity: 0.5;
    filter: brightness(1.5);
  }
  58% {
    transform: scale(1.025);
    opacity: 1;
    filter: brightness(1);
  }
  100% {
    transform: scale(1);
  }
}

@keyframes title-in {
  from {
    opacity: 0.2;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
"""


def apply_theme() -> None:
    """注入设计系统与宿主样式，并绘制页面装饰。"""
    ui.add_css(_design_css())
    ui.add_css(_HOST_CSS)
    _build_decorations()


def _build_decorations() -> None:
    with element('div', 'atmosphere').props('aria-hidden="true"'):
        element('span', 'ambient-orbit ambient-orbit--one')
        element('span', 'ambient-orbit ambient-orbit--two')
        Text('span', '象', 'ambient-character')
        element('span', 'grain')

    with element('div', 'page-frame').props('aria-hidden="true"'):
        for corner in ('tl', 'tr', 'bl', 'br'):
            element('i', f'frame-corner frame-corner--{corner}')
