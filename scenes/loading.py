# scenes/loading.py 
import pygame
import time
import os
from config import WIDTH, HEIGHT, FPS, GRAY, WHITE, BLOCK_SIZE, BAR_WIDTH, BAR_HEIGHT, get_gradient_color


class LoadingScene:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 24, bold=True)
        self.done_font = pygame.font.SysFont("consolas", 36, bold=True)

        self.bar_px_w = BAR_WIDTH * BLOCK_SIZE
        self.bar_px_h = BAR_HEIGHT * BLOCK_SIZE
        self.bar_x = (WIDTH - self.bar_px_w) // 2
        self.bar_y = HEIGHT // 2 + 20

        self.progress = 0
        self.done_alpha = 0
        self.paused = False
        self.pause_duration = 1.0
        self.pause_start = 0

        base_path = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base_path, "..", "images", "loading.png")

        self.loading_img = pygame.image.load(img_path).convert_alpha()
        self.loading_img = pygame.transform.smoothscale(
            self.loading_img, (250, 180))
        self.loading_rect = self.loading_img.get_rect(midtop=(WIDTH // 2, 180))

        self.finished = False

    def update(self):
        self.clock.tick(FPS)
        self.screen.fill(GRAY)
        self.screen.blit(self.loading_img, self.loading_rect)

        pygame.draw.rect(
            self.screen, WHITE,
            pygame.Rect(self.bar_x - 2, self.bar_y - 2,
                        self.bar_px_w + 4, self.bar_px_h + 4),
            width=2
        )

        fill_blocks = int((self.progress / 100) * BAR_WIDTH)
        for row in range(BAR_HEIGHT):
            for col in range(fill_blocks):
                x = self.bar_x + col * BLOCK_SIZE
                y = self.bar_y + row * BLOCK_SIZE
                color = get_gradient_color(col / BAR_WIDTH)
                pygame.draw.rect(self.screen, color,
                                 (x, y, BLOCK_SIZE, BLOCK_SIZE))

        percent = self.font.render(f"{int(self.progress)}%", True, WHITE)
        self.screen.blit(percent, percent.get_rect(
            center=(WIDTH // 2, self.bar_y + self.bar_px_h // 2)))

        if not self.paused:
            self.progress += 0.8 if self.progress < 70 else 0.4 if self.progress < 90 else 0.1
            if self.progress >= 100:
                self.progress = 100
                self.paused = True
                self.pause_start = time.time()
                self.done_alpha = 0
        else:
            elapsed = time.time() - self.pause_start
            if elapsed < self.pause_duration:
                self.done_alpha = min(
                    255, int((elapsed / self.pause_duration) * 255))
                done = self.done_font.render("Done!", True, WHITE)
                done.set_alpha(self.done_alpha)
                self.screen.blit(done, done.get_rect(
                    center=(WIDTH // 2, self.bar_y + self.bar_px_h + 40)))
            else:
                self.finished = True

        pygame.display.flip()

    def is_finished(self):
        return self.finished  