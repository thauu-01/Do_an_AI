# config.py
import pygame

WIDTH, HEIGHT = 1000, 600
GRAY = (30, 30, 30)
WHITE = (255, 255, 255)
SAND = (194, 178, 128)
LIGHT_SAND = (224, 205, 145)
SELECTED_COLOR = (255, 236, 184)
BACK_BUTTON_COLOR = (120, 90, 40)
BACK_BUTTON_HOVER = (160, 120, 70)
TEXT_COLOR = (50, 40, 20)
HEADER_COLOR = (255, 230, 180)
BORDER_COLOR = (255, 255, 255)
FPS = 60

BLOCK_SIZE = 8
BAR_WIDTH, BAR_HEIGHT = 40, 3

CELL_SIZE = 50
WIDTH_MAZE = 21
HEIGHT_MAZE = 15
LIGHT_RADIUS = 150

GRADIENT_COLORS = [
    (255, 255, 0), (255, 191, 0), (255, 128, 0), (204, 85, 0), (139, 69, 19)
]


def get_gradient_color(ratio):
    pos = ratio * (len(GRADIENT_COLORS) - 1)
    i = int(pos)
    if i >= len(GRADIENT_COLORS) - 1:
        return GRADIENT_COLORS[-1]
    t = pos - i
    c1, c2 = GRADIENT_COLORS[i], GRADIENT_COLORS[i + 1]
    return tuple(int(c1[j] + (c2[j] - c1[j]) * t) for j in range(3))
