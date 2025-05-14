import pygame
import sys
import os
import numpy as np
from config import WIDTH, HEIGHT, SAND, LIGHT_SAND, SELECTED_COLOR, BACK_BUTTON_COLOR, BACK_BUTTON_HOVER, TEXT_COLOR, HEADER_COLOR, BORDER_COLOR, FPS


class MenuScene:
    def __init__(self, screen, has_saved_game=False, saved_state=None):
        self.screen = screen
        self.WIDTH, self.HEIGHT = WIDTH, HEIGHT
        self.clock = pygame.time.Clock()
        self.saved_state = saved_state

        # Only allow saved game if state exists and game is not won
        self.has_saved_game = (
            has_saved_game and
            saved_state is not None and
            isinstance(saved_state, dict) and
            "game_won" in saved_state and
            not saved_state["game_won"]
        )

        self.old_map = None
        if self.has_saved_game and "config" in saved_state and "maze" in saved_state:
            if isinstance(saved_state["maze"], np.ndarray) and saved_state["maze"].shape == (15, 21):
                self.old_map = saved_state["maze"]

        if not pygame.font.get_init():
            pygame.font.init()

        try:
            self.font = pygame.font.SysFont("Papyrus", 28) or pygame.font.SysFont(
                None, 28)
            self.header_font = pygame.font.SysFont(
                "Papyrus", 40, bold=True) or pygame.font.SysFont(None, 40, bold=True)
        except pygame.error as e:
            sys.exit(1)

        music_path = os.path.join(os.path.dirname(
            __file__), "..", "sounds", "intro.mp3")
        try:
            pygame.mixer.music.load(music_path)
            pygame.mixer.music.set_volume(0.5)
            pygame.mixer.music.play(-1)
        except pygame.error:
            pass

        click_sound_path = os.path.join(os.path.dirname(
            __file__), "..", "sounds", "click_sound.mp3")
        try:
            self.click_sound = pygame.mixer.Sound(click_sound_path)
        except pygame.error:
            self.click_sound = None

        self.current_scene = "main_menu"
        self.scene_history = []

        self.selection = {
            "mode": None,
            "difficulty": None,
            "map": None,
            "human_algorithm": None,
            "mummy_algorithm": None,
            "old_map": None
        }

        self.selected_button_indices = {
            "main_menu": 0,
            "choose_mode": 0,
            "choose_human_algorithm": 0,
            "choose_mummy_algorithm": 0,
            "choose_difficulty": 0,
            "choose_map": 0,
        }

        self.scene_titles = {
            "main_menu": "Main Menu",
            "choose_mode": "Choose Game Mode",
            "choose_human_algorithm": "Choose Algorithm for Human",
            "choose_mummy_algorithm": "Choose Algorithm for Mummy",
            "choose_difficulty": "Choose Difficulty",
            "choose_map": "Choose Map"
        }

    class Button:
        def __init__(self, text, x, y, w, h, callback, font, colors, click_sound, is_back=False, enabled=True):
            self.text = text
            self.rect = pygame.Rect(x, y, w, h)
            self.callback = callback
            self.click_sound = click_sound
            self.selected = False
            self.font = font
            self.enabled = enabled
            self.default_color = colors["back_color"] if is_back else colors["normal"]
            self.hover_color = colors["hover_back"] if is_back else colors["hover"]
            self.selected_color = colors["selected"]
            self.disabled_color = (100, 100, 100)
            self.text_color = colors["text"]
            self.border_color = colors["border"]

        def draw(self, surface):
            mouse_pos = pygame.mouse.get_pos()
            if not self.enabled:
                color = self.disabled_color
            elif self.selected:
                color = self.selected_color
            elif self.rect.collidepoint(mouse_pos):
                color = self.hover_color
            else:
                color = self.default_color

            pygame.draw.rect(surface, color, self.rect, border_radius=12)
            pygame.draw.rect(surface, self.border_color,
                             self.rect, 2, border_radius=12)

            label = self.font.render(self.text, True, self.text_color)
            surface.blit(label, (
                self.rect.centerx - label.get_width() // 2,
                self.rect.centery - label.get_height() // 2
            ))

        def handle_event(self, event):
            if event.type == pygame.MOUSEBUTTONDOWN and self.enabled:
                if self.rect.collidepoint(pygame.mouse.get_pos()):
                    if self.click_sound:
                        self.click_sound.play()
                    return self.callback()
            return None

    def switch_scene(self, name):
        self.scene_history.append(self.current_scene)
        self.current_scene = name
        self.selected_button_indices[name] = 0

    def go_back(self):
        if self.scene_history:
            self.current_scene = self.scene_history.pop()
        return None

    def create_button(self, text, x, y, callback, is_back=False, enabled=True):
        return self.Button(
            text, x, y, 260, 50, callback,
            self.font,
            {
                "normal": SAND,
                "hover": LIGHT_SAND,
                "selected": SELECTED_COLOR,
                "back_color": BACK_BUTTON_COLOR,
                "hover_back": BACK_BUTTON_HOVER,
                "text": TEXT_COLOR,
                "border": BORDER_COLOR
            },
            self.click_sound,
            is_back,
            enabled
        )

    def update_old_map(self, saved_state):
        if (
            saved_state and
            isinstance(saved_state, dict) and
            "config" in saved_state and
            "maze" in saved_state and
            not saved_state.get("game_won", False)
        ):
            if isinstance(saved_state["maze"], np.ndarray) and saved_state["maze"].shape == (15, 21):
                self.old_map = saved_state["maze"]
            else:
                self.old_map = None
        else:
            self.old_map = None

    def main_menu(self):
        buttons = [
            self.create_button("Start Game", (self.WIDTH - 260) // 2, 200,
                               lambda: self.switch_scene("choose_mode")),
        ]
        if self.has_saved_game:
            buttons.append(self.create_button(
                "Continue Game", (self.WIDTH - 260) // 2, 300, self.continue_game))
        buttons.append(self.create_button(
            "Quit", (self.WIDTH - 260) // 2, 400 if self.has_saved_game else 300, lambda: sys.exit()))
        return buttons

    def continue_game(self):
        if (
            self.saved_state and
            isinstance(self.saved_state, dict) and
            all(k in self.saved_state for k in ["maze", "players", "mummy", "item_data", "goal", "config"]) and
            not self.saved_state.get("game_won", False)
        ):
            return ("continue", self.saved_state)
        return None

    def choose_mode(self):
        return [
            self.create_button("Player vs AI", (self.WIDTH - 260) // 2, 180,
                               lambda: self.select_mode("Player vs AI")),
            self.create_button("AI vs AI", (self.WIDTH - 260) // 2, 260,
                               lambda: self.select_mode("AI vs AI")),
            self.create_button("Back", (self.WIDTH - 260) //
                               2, 340, self.go_back, is_back=True)
        ]

    def choose_human_algorithm(self):
        return [
            self.create_button("DFS", self.WIDTH // 2 - 300, 160,
                               lambda: self.select_human_algorithm("DFS")),
            self.create_button("A*", self.WIDTH // 2 - 300, 240,
                               lambda: self.select_human_algorithm("A*")),
            self.create_button("Backtracking", self.WIDTH // 2 - 300, 320,
                               lambda: self.select_human_algorithm("Backtracking")),
            self.create_button("Q-Learning", self.WIDTH // 2 + 40, 160,
                               lambda: self.select_human_algorithm("Q-Learning")),
            self.create_button("No Observation", self.WIDTH // 2 + 40, 240,
                               lambda: self.select_human_algorithm("BFS-NoObs")),
            self.create_button("Simulated Algorithm", self.WIDTH // 2 + 40, 320,
                               lambda: self.select_human_algorithm("SA")),
            self.create_button("Back", (self.WIDTH - 260) //
                               2, 420, self.go_back, is_back=True)
        ]

    def choose_mummy_algorithm(self):
        return [
            self.create_button("DFS", (self.WIDTH - 260) // 2, 160,
                               lambda: self.select_mummy_algorithm("DFS")),
            self.create_button("A*", (self.WIDTH - 260) // 2, 240,
                               lambda: self.select_mummy_algorithm("A*")),
            self.create_button("Backtracking", (self.WIDTH - 260) // 2, 320,
                               lambda: self.select_mummy_algorithm("Backtracking")),
            self.create_button("Simulated Algorithm", (self.WIDTH - 260) // 2, 400,
                               lambda: self.select_mummy_algorithm("SA")),
            self.create_button("Back", (self.WIDTH - 260) //
                               2, 480, self.go_back, is_back=True)
        ]

    def choose_difficulty(self):
        return [
            self.create_button("Easy", (self.WIDTH - 260) // 2, 160,
                               lambda: self.select_difficulty("Easy")),
            self.create_button("Medium", (self.WIDTH - 260) // 2, 240,
                               lambda: self.select_difficulty("Medium")),
            self.create_button("Hard", (self.WIDTH - 260) // 2, 320,
                               lambda: self.select_difficulty("Hard")),
            self.create_button("Back", (self.WIDTH - 260) //
                               2, 400, self.go_back, is_back=True)
        ]

    def choose_map(self):
        return [
            self.create_button("Old Map", (self.WIDTH - 260) // 2, 200,
                               lambda: self.select_map("Old Map"), enabled=self.old_map is not None),
            self.create_button("New Map", (self.WIDTH - 260) // 2, 280,
                               lambda: self.select_map("New Map")),
            self.create_button("Back", (self.WIDTH - 260) //
                               2, 360, self.go_back, is_back=True)
        ]

    def select_mode(self, mode):
        self.selection["mode"] = mode
        if mode == "Player vs AI":
            self.selection["human_algorithm"] = None
            self.switch_scene("choose_mummy_algorithm")
        elif mode == "AI vs AI":
            self.switch_scene("choose_human_algorithm")

    def select_human_algorithm(self, algo):
        self.selection["human_algorithm"] = algo
        self.switch_scene("choose_mummy_algorithm")

    def select_mummy_algorithm(self, algo):
        self.selection["mummy_algorithm"] = algo
        self.switch_scene("choose_difficulty")

    def select_difficulty(self, difficulty):
        self.selection["difficulty"] = difficulty
        self.switch_scene("choose_map")

    def select_map(self, map_choice):
        self.selection["map"] = map_choice
        if map_choice == "Old Map":
            if self.old_map is not None and isinstance(self.old_map, np.ndarray) and self.old_map.shape == (15, 21):
                self.selection["old_map"] = self.old_map.copy()
            else:
                self.selection["map"] = "New Map"
                self.selection["old_map"] = None
        else:
            self.selection["old_map"] = None
        self.running = False
        return ("start_game", self.selection)

    def run(self):
        self.running = True
        scene_buttons = {
            "main_menu": self.main_menu,
            "choose_mode": self.choose_mode,
            "choose_human_algorithm": self.choose_human_algorithm,
            "choose_mummy_algorithm": self.choose_mummy_algorithm,
            "choose_difficulty": self.choose_difficulty,
            "choose_map": self.choose_map
        }

        while self.running:
            base_path = os.path.dirname(os.path.abspath(__file__))
            bg_path = os.path.join(base_path, "..", "images", "scene2.png")
            try:
                background = pygame.image.load(bg_path).convert()
                background = pygame.transform.scale(
                    background, (self.WIDTH, self.HEIGHT))
                self.screen.blit(background, (0, 0))
            except pygame.error:
                self.screen.fill((0, 0, 0))

            events = pygame.event.get()
            buttons = scene_buttons[self.current_scene]()
            selected_index = self.selected_button_indices[self.current_scene]

            title_text = self.scene_titles.get(self.current_scene, "")
            header = self.header_font.render(title_text, True, HEADER_COLOR)
            self.screen.blit(header, (self.WIDTH // 2 -
                             header.get_width() // 2, 60))

            for i, button in enumerate(buttons):
                button.selected = (i == selected_index)
                button.draw(self.screen)

            for event in events:
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.selected_button_indices[self.current_scene] = (
                            selected_index - 1) % len(buttons)
                    elif event.key == pygame.K_DOWN:
                        self.selected_button_indices[self.current_scene] = (
                            selected_index + 1) % len(buttons)
                    elif event.key == pygame.K_RETURN and buttons[selected_index].enabled:
                        if self.click_sound:
                            self.click_sound.play()
                        result = buttons[selected_index].callback()
                        if result:
                            self.running = False
                            try:
                                pygame.mixer.music.stop()
                            except pygame.error:
                                pass
                            return result
                for button in buttons:
                    result = button.handle_event(event)
                    if result:
                        self.running = False
                        try:
                            pygame.mixer.music.stop()
                        except pygame.error:
                            pass
                        return result

            pygame.display.flip()
            self.clock.tick(FPS)

        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        return None
