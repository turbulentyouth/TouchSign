"""十二生肖类别元数据。

本模块同时承担两个职责：

* 十二生肖的固定展示顺序；实际14类模型顺序由模型manifest显式映射；
* UI 展示信息（字形、地支、五行、方位、时辰、诗句等）的唯一来源。

模型权重尚未就绪时，``baseline_confidence`` 仅用于演示界面的初始结果，
不代表任何真实推理输出。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Zodiac:
    """单个生肖的固定元数据。"""

    key: str
    glyph: str
    title: str
    pinyin: str
    english: str
    branch: str
    index: int
    element: str
    direction: str
    direction_en: str
    hour: str
    poem: str
    baseline_confidence: float

    @property
    def nature(self) -> str:
        """五行与方位的组合描述，例如“属水 · 居北”。"""
        return f"属{self.element} · 居{self.direction}"

    @property
    def display_index(self) -> str:
        """两位序号，例如 “01”。"""
        return f"{self.index:02d}"


ZODIAC: tuple[Zodiac, ...] = (
    Zodiac('rat', '鼠', '子鼠', 'ZI', 'RAT', '子', 1, '水', '北', 'NORTH', '子时', '灵鼠衔金，机巧通灵', 98.7),
    Zodiac('ox', '牛', '丑牛', 'CHOU', 'OX', '丑', 2, '土', '东北', 'NORTHEAST', '丑时', '厚德载物，耕耘有时', 97.9),
    Zodiac('tiger', '虎', '寅虎', 'YIN', 'TIGER', '寅', 3, '木', '东北', 'NORTHEAST', '寅时', '风从虎啸，威镇山林', 98.1),
    Zodiac('rabbit', '兔', '卯兔', 'MAO', 'RABBIT', '卯', 4, '木', '东', 'EAST', '卯时', '月华凝露，静中藏慧', 97.6),
    Zodiac('dragon', '龙', '辰龙', 'CHEN', 'DRAGON', '辰', 5, '土', '东南', 'SOUTHEAST', '辰时', '云起龙骧，万象更新', 96.8),
    Zodiac('snake', '蛇', '巳蛇', 'SI', 'SNAKE', '巳', 6, '火', '东南', 'SOUTHEAST', '巳时', '灵蛇盘玉，幽微见真', 96.4),
    Zodiac('horse', '马', '午马', 'WU', 'HORSE', '午', 7, '火', '南', 'SOUTH', '午时', '天马行空，一往无前', 97.2),
    Zodiac('goat', '羊', '未羊', 'WEI', 'GOAT', '未', 8, '土', '西南', 'SOUTHWEST', '未时', '温润如玉，和合生祥', 95.9),
    Zodiac('monkey', '猴', '申猴', 'SHEN', 'MONKEY', '申', 9, '金', '西南', 'SOUTHWEST', '申时', '灵猴献瑞，机变无穷', 96.7),
    Zodiac('rooster', '鸡', '酉鸡', 'YOU', 'ROOSTER', '酉', 10, '金', '西', 'WEST', '酉时', '金鸡报晓，光启东方', 97.1),
    Zodiac('dog', '狗', '戌狗', 'XU', 'DOG', '戌', 11, '土', '西北', 'NORTHWEST', '戌时', '忠守其门，信义相随', 96.3),
    Zodiac('pig', '猪', '亥猪', 'HAI', 'PIG', '亥', 12, '水', '西北', 'NORTHWEST', '亥时', '亥水生财，丰饶安泰', 95.7),
)

ZODIAC_BY_KEY: dict[str, Zodiac] = {item.key: item for item in ZODIAC}
ZODIAC_BY_GLYPH: dict[str, Zodiac] = {item.glyph: item for item in ZODIAC}
ZODIAC_KEYS: tuple[str, ...] = tuple(item.key for item in ZODIAC)


def next_zodiac(zodiac: Zodiac) -> Zodiac:
    """返回十二生肖循环中的下一个。"""
    return ZODIAC[zodiac.index % len(ZODIAC)]


def zodiac_at(offset: int) -> Zodiac:
    """按 0 起始的类别索引取生肖（用于模型类别映射）。"""
    return ZODIAC[offset % len(ZODIAC)]
