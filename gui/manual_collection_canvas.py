"""Small Chinese keyboard UI for manual sensor capture."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core.zodiac import ZODIAC
from gui.collection_canvas import Canvas, LABEL_SHORTCUTS

NAMES = {item.key: item.glyph for item in ZODIAC} | {'unknown': '未知物体', 'no_contact': '空载'}


class ManualCanvas(Canvas):
    """Reuse the collector's Chinese fonts, text fitting and preview geometry."""

    def _background_for(self, label, total, class_count, recording, pending, message, camera_index, live):
        ink, muted, green = '#183a30', '#466057', '#176955'
        image = Image.new('RGB', self.size, '#f1f5f2')
        draw = ImageDraw.Draw(image)

        def text(x, y, value, size=18, color=ink, anchor='lt'):
            draw.text((x, y), value, font=self.fonts[size], fill=color, anchor=anchor)

        text(24, 18, '触象  /  手工快速采集', 30)
        text(1076, 26, f'相机 {camera_index}    旋转 180° 保存', 18, muted, 'rt')
        draw.line((24, 61, 1076, 61), fill='#cbd8d0')
        text(24, 71, '实时画面 · 与保存方向一致', 18)
        x0, y0, x1, y1 = self.preview
        draw.rectangle((x0, y0, x1-1, y1-1), fill='#14231f')
        if not live:
            text((x0+x1)//2, (y0+y1)//2, '等待传感器图像', 22, '#b8c9c2', 'mm')

        x = 757
        text(x, 71, '当前标签 · 按键选类', 18, muted)
        text(x, 104, NAMES.get(label, '未选标签'), 40)
        text(1076, 76, '连拍中' if recording else '单拍', 18, green, 'rt')
        text(x, 164, f'本类 {class_count} 张    本会话 {total} 张', 18)
        text(x, 196, f'待处理 {pending} 项（保存 / 撤回）' if pending else '已写入磁盘', 16, muted)
        for i, (key, shortcut) in enumerate(LABEL_SHORTCUTS.items()):
            if i < 12:
                left, top, width = x + (i % 4)*81, 235 + (i // 4)*43, 76
            else:
                left, top, width = x + (i-12)*162, 364, 157
            selected = key == label
            draw.rounded_rectangle((left, top, left+width, top+35), radius=5,
                                   fill=green if selected else '#ffffff', outline='#bfd0c5')
            text(left+width/2, top+17, f'{shortcut}  {NAMES[key]}', 18,
                 '#ffffff' if selected else ink, 'mm')
        text(x, 421, '1  选标签', 18)
        text(x, 453, '2  手工按压', 18)
        text(x, 485, '3  空格保存', 18)
        text(x, 517, '采错按 Z 撤回，再选标签补拍', 16, muted)
        text(24, 552, self._fit(message, 18, 1052), 18, green)
        draw.line((24, 593, 1076, 593), fill='#cbd8d0')
        text(24, 610, '空格 拍一张     Z / 退格 撤回     R 连拍     Esc 停止     Q 退出并等待完成', 18)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def render_manual(self, frame, *, label, total, class_count, recording, pending, message, camera_index):
        state = (label, total, class_count, recording, pending, message, camera_index, frame is not None)
        if state != self._previous_state:
            self._background = self._background_for(*state)
            self._previous_state = state
        canvas = self._background.copy()
        if frame is not None:
            height, width = frame.shape[:2]
            x0, y0, x1, y1 = self.preview
            scale = min((x1-x0)/width, (y1-y0)/height)
            w, h = max(1, round(width*scale)), max(1, round(height*scale))
            x, y = x0 + (x1-x0-w)//2, y0 + (y1-y0-h)//2
            canvas[y:y+h, x:x+w] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return canvas
