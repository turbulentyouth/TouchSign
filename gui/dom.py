"""NiceGUI DOM 原语。

界面原型是按明确标签与 class 编写的 CSS 设计系统，因此这里不直接使用
NiceGUI 的 ``ui.label``（固定为 div）或 Quasar 组件，而是生成与原型一致的
原生标签，确保 ``gui/style.css`` 可以原样复用。
"""

from __future__ import annotations

from html import escape

from nicegui import ui
from nicegui.elements.mixins.text_element import TextElement


class Text(TextElement):
    """可指定标签的纯文本元素（span / strong / small / h2 / option ...）。"""

    def __init__(self, tag: str, text: str = '', classes: str = '') -> None:
        super().__init__(tag=tag, text=text)
        if classes:
            self.classes(classes)

    def set_text(self, text: str) -> None:
        self.text = text


def text(tag: str, content: str = '', classes: str = '') -> Text:
    return Text(tag, content, classes)


def markup(tag: str, html: str, classes: str = '') -> ui.html:
    """生成允许内嵌标记（i / small / em / b）的文本元素。"""
    element = ui.html(html, tag=tag)
    if classes:
        element.classes(classes)
    return element


def element(tag: str, classes: str = '') -> ui.element:
    element_ = ui.element(tag)
    if classes:
        element_.classes(classes)
    return element_


def svg(
    inner: str,
    classes: str = '',
    viewbox: str = '0 0 24 24',
    preserve_aspect_ratio: str = '',
) -> ui.html:
    element_ = ui.html(inner, tag='svg')
    if classes:
        element_.classes(classes)
    element_.props(f'viewBox="{viewbox}"')
    if preserve_aspect_ratio:
        element_.props(f'preserveAspectRatio="{preserve_aspect_ratio}"')
    return element_


def image(src: str, alt: str = '', classes: str = '') -> ui.element:
    element_ = ui.element('img')
    element_.props(f'src="{escape(src, quote=True)}" alt="{escape(alt, quote=True)}"')
    if classes:
        element_.classes(classes)
    return element_


def esc(value: str) -> str:
    """转义动态文本，供 innerHTML 类元素使用。"""
    return escape(value, quote=False)
