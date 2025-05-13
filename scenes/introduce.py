# scenes/introduce.py
import pygame
import sys
import os
from config import WIDTH, HEIGHT, FPS

class IntroduceScene:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()

        base_path = os.path.dirname(os.path.abspath(__file__))
        bg_path = os.path.join(base_path, "..", "images", "scene1.png")
        intro_path = os.path.join(base_path, "..", "images", "intro.png")
        self.bg_original = pygame.image.load(bg_path).convert_alpha()
        self.introduce_original = pygame.image.load(intro_path).convert_alpha()

                
        music_path = os.path.join(base_path, "..", "sounds", "loading.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load(music_path)
        pygame.mixer.music.play(-1)

        self.bg_speed = 0.5
        self.bg_x = 0

        self.elapsed_time = 0
        self.darken_start = 10.0  
        self.darken_duration = 3.0

        self.resize_images()

        self.finished = False

    def resize_images(self):
        self.bg_img = pygame.transform.scale(self.bg_original, (WIDTH, HEIGHT))
        self.intro_img = pygame.transform.scale(self.introduce_original, (500, 500))

    def update(self):
        dt = self.clock.tick(FPS) / 1000.0
        self.elapsed_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                global WIDTH, HEIGHT
                WIDTH, HEIGHT = event.w, event.h
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                self.resize_images()

        self.bg_x -= self.bg_speed
        if self.bg_x <= -WIDTH:
            self.bg_x = 0

        self.screen.blit(self.bg_img, (self.bg_x, 0))
        self.screen.blit(self.bg_img, (self.bg_x + WIDTH, 0))

        intro_x = (WIDTH - 500) // 2
        intro_y = (HEIGHT - 500) // 2 
        self.screen.blit(self.intro_img, (intro_x, intro_y))

        if self.elapsed_time >= self.darken_start:
            progress = (self.elapsed_time - self.darken_start) / self.darken_duration
            progress = min(progress, 1)
            
            volume = max(0.0, 1.0 - progress)
            pygame.mixer.music.set_volume(volume)

            overlay = pygame.Surface((WIDTH, HEIGHT))
            overlay.fill((0, 0, 0))
            overlay.set_alpha(int(progress * 255))
            self.screen.blit(overlay, (0, 0))

            if progress >= 1:
                self.finished = True
                pygame.mixer.music.stop()

        pygame.display.flip()

    def is_finished(self):
        return self.finished
