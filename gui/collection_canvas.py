"""Chinese HUD for the OpenCV collector. Sensor pixels and labels stay separate."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.zodiac import ZODIAC, ZODIAC_KEYS

LABEL_SHORTCUTS = dict(zip(ZODIAC_KEYS, '1234567890-=', strict=True))
LABEL_SHORTCUTS.update(unknown='U', no_contact='N')
_NAMES = {item.key: item.glyph for item in ZODIAC} | {'unknown': '未知物体', 'no_contact': '空载'}


def _chinese_font(requested: Path | str | None) -> Path:
    if requested is not None:
        candidates = [Path(requested).expanduser()]
    else:
        windows_fonts = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
        candidates = [
            windows_fonts / 'msyh.ttc', windows_fonts / 'NotoSansSC-VF.ttf',
            windows_fonts / 'simhei.ttf', windows_fonts / 'simsun.ttc',
            Path('/System/Library/Fonts/PingFang.ttc'),
            Path('/System/Library/Fonts/STHeiti Medium.ttc'),
            Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
            Path('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'),
        ]
    required = ''.join(_NAMES.values()) + '触象采集选择类别稳定接触校准按键已保存'
    for path in candidates:
        if not path.is_file():
            continue
        try:
            font = ImageFont.truetype(str(path), 18)
            # FreeType draws the same .notdef glyph for unsupported characters.
            missing = bytes(font.getmask(chr(0x10FFFF)))
            if all(bytes(font.getmask(ch)) != missing for ch in set(required)):
                return path
        except OSError:
            continue
    which = f'字体 {requested} 无法使用或缺少中文字符。' if requested else '没有找到可用的中文字体。'
    raise ValueError(which + '请使用 --font 指定支持简体中文的 TTF、TTC 或 OTF 字体文件。')


class Canvas:
    """Render a fixed logical canvas to BGR for cv2.imshow; preserve the preview's aspect ratio."""

    size = (1100, 650)
    preview = (24, 97, 733, 530)

    def __init__(self, font_path: Path | str | None = None) -> None:
        self.font_path = _chinese_font(font_path)
        self.fonts = {size: ImageFont.truetype(str(self.font_path), size) for size in (16, 18, 22, 30, 40)}
        self._previous_state = None
        self._background = None

    def _fit(self, text: str, size: int, available_width: int) -> str:
        text = ' '.join(text.splitlines())
        font = self.fonts[size]
        if font.getlength(text) <= available_width:
            return text
        left, right = 0, len(text)
        while left < right:
            middle = (left + right + 1) // 2
            if font.getlength(text[:middle] + '…') <= available_width:
                left = middle
            else:
                right = middle - 1
        return text[:left] + '…'

    def _draw_hud(self, label, total, class_count, contact, message, diff, recording, calibrated, camera_index, live):
        ink, muted, green = '#183a30', '#466057', '#176955'
        image = Image.new('RGB', self.size, '#f1f5f2')
        draw = ImageDraw.Draw(image)

        def text(x, y, value, size=18, color=ink, anchor='lt'):
            draw.text((x, y), value, font=self.fonts[size], fill=color, anchor=anchor)

        text(24, 18, '触象  /  快捷采集', 30)
        header = f'相机 {camera_index}      ' + ('已校准' if calibrated else '待空载校准')
        text(1076, 26, header, 18, muted, 'rt')
        draw.line((24, 61, 1076, 61), fill='#cbd8d0')
        text(24, 71, '实时差分图' if diff else '传感器原图', 18)
        x0, y0, x1, y1 = self.preview
        draw.rectangle((x0, y0, x1-1, y1-1), fill='#14231f')
        if not live:
            text((x0+x1)//2, (y0+y1)//2, '等待有效图像', 22, '#b8c9c2', 'mm')

        x = 757
        text(x, 71, '当前标签', 18, muted)
        text(x, 99, _NAMES.get(label, '未选类别'), 40)
        text(1076, 106, '连拍中' if recording else '单拍', 22, green, 'rt')
        text(x, 155, f'本类 {class_count} 张    本会话 {total} 张', 18)
        draw.line((x, 191, 1076, 191), fill='#cbd8d0')
        text(x, 205, '按键选类', 18, muted)
        for i, (key, shortcut) in enumerate(LABEL_SHORTCUTS.items()):
            if i < 12:
                left, top, width = x + (i % 4)*81, 235 + (i // 4)*43, 76
            else:
                left, top, width = x + (i-12)*162, 364, 157
            selected = key == label
            draw.rounded_rectangle((left, top, left+width, top+35), radius=5,
                                   fill=green if selected else '#ffffff', outline='#bfd0c5')
            text(left+width/2, top+17, f'{shortcut}  {_NAMES[key]}', 18,
                 '#ffffff' if selected else ink, 'mm')
        text(x, 414, '←  上一类      →  下一类', 16, muted)
        draw.line((x, 451, 1076, 451), fill='#cbd8d0')
        text(x, 466, '接触状态', 16, muted)
        text(x, 493, self._fit(contact, 18, 320), 18)

        text(24, 552, self._fit(message, 18, 1052), 18, green)
        draw.line((24, 593, 1076, 593), fill='#cbd8d0')
        text(24, 610, 'C 空载校准     空格 拍一张     R 开关连拍     Esc 停止连拍     V 切换视图     Q 退出', 18)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def render(self, frame: np.ndarray | None, *, label: str | None, total: int, class_count: int,
               contact: str, message: str, diff: bool, recording: bool, calibrated: bool,
               camera_index: int) -> np.ndarray:
        state = (label, total, class_count, contact, message, diff, recording, calibrated,
                 camera_index, frame is not None)
        if state != self._previous_state:
            self._background = self._draw_hud(*state)
            self._previous_state = state
        canvas = self._background.copy()
        if frame is not None:
            height, width = frame.shape[:2]
            x0, y0, x1, y1 = self.preview
            scale = min((x1-x0)/width, (y1-y0)/height)
            w, h = max(1, round(width*scale)), max(1, round(height*scale))
            x, y = x0 + (x1-x0-w)//2, y0 + (y1-y0-h)//2
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            canvas[y:y+h, x:x+w] = cv2.resize(frame, (w, h), interpolation=interpolation)
        return canvas
