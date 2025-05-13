import numpy as np
import random
import pygame
import os
import math
import heapq
import copy
from scenes.algorithm import heuristic, get_valid_moves, find_path_astar, find_path_dfs, find_path_sa, a_star, find_path_backtracking, find_path_qlearning, search_no_observation_bfs_maze
import ctypes
import sys


def maingame(config, saved_state=None):
    print("Starting maingame with config:", config,
          "and saved_state:", saved_state is not None)
    # Kiểm tra config hợp lệ
    if not isinstance(config, dict):
        print("Error: Invalid config, expected a dictionary, got:", config)
        return None

    # Lấy cấu hình game với giá trị mặc định
    mode = config.get("mode", "AI vs AI")
    difficulty = config.get("difficulty", "Easy")
    map_name = config.get("map", "New Map")
    human_algorithm = config.get("human_algorithm", None)
    mummy_algorithm = config.get("mummy_algorithm", "A*")
    old_map = config.get("old_map", None)  # Lấy Old Map từ config

    # Kiểm tra các giá trị cần thiết
    if mode not in ["Player vs AI", "AI vs AI"]:
        print(f"Error: Invalid mode '{mode}', defaulting to 'AI vs AI'")
        mode = "AI vs AI"
    if mummy_algorithm not in ["A*", "DFS", "SA", "Backtracking"]:
        print(
            f"Error: Invalid mummy_algorithm '{mummy_algorithm}', defaulting to 'A*'")
        mummy_algorithm = "A*"
    if map_name not in ["Old Map", "New Map"]:
        print(f"Error: Invalid map '{map_name}', defaulting to 'New Map'")
        map_name = "New Map"
    if mode == "Player vs AI" and human_algorithm is not None:
        print(
            f"Warning: human_algorithm should be None in Player vs AI mode, got '{human_algorithm}', setting to None")
        human_algorithm = None
    elif mode == "AI vs AI" and human_algorithm not in ["A*", "DFS", "SA", "Q-Learning", "Backtracking", "BFS-NoObs"]:
        print(
            f"Error: Invalid human_algorithm '{human_algorithm}', defaulting to 'A*'")
        human_algorithm = "A*"

    # Thiết lập kích thước mê cung và cửa sổ
    width_maze, height_maze = 21, 15
    width, height = 1540, 800  # Kích thước cố định
    cell_size = 50

    # Lấy độ phân giải màn hình để kiểm tra
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    print(f"Screen resolution: {screen_width}x{screen_height}")
    print(f"Requested window size: {width}x{height}")

    # Khởi tạo Pygame với kích thước cố định
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Maze Game')

    # Đặt vị trí cửa sổ ở góc trên bên trái
    window_handle = pygame.display.get_wm_info()['window']
    user32.SetWindowPos(window_handle, 0, -10, 0, width, height, 0)
    print(f"Window positioned at: (0, 0)")

    # Lấy kích thước thực tế của cửa sổ để debug
    real_width, real_height = screen.get_size()
    print(f"Actual window size: {real_width}x{real_height}")

    # Thiết lập font chữ
    menu_font = pygame.font.SysFont(
        "Press Start 2P", 24, bold=True) or pygame.font.SysFont("consolas", 24, bold=True)
    score_font = pygame.font.SysFont(
        "Press Start 2P", 14, bold=True) or pygame.font.SysFont("consolas", 14, bold=True)
    win_font = pygame.font.SysFont(
        "Press Start 2P", 120, bold=True) or pygame.font.SysFont("consolas", 120, bold=True)
    game_over_font = pygame.font.SysFont(
        "Press Start 2P", 60, bold=True) or pygame.font.SysFont("consolas", 60, bold=True)
    button_font = pygame.font.SysFont(
        "Press Start 2P", 30, bold=True) or pygame.font.SysFont("consolas", 30, bold=True)
    notification_font = pygame.font.SysFont(
        "Press Start 2P", 40, bold=True) or pygame.font.SysFont("consolas", 40, bold=True)

    # Nhạc nền
    music_on = True
    music_path = os.path.join(os.path.dirname(
        __file__), "..", "sounds", "game.mp3")
    if not os.path.exists(music_path):
        print(f"Error: Music file {music_path} not found")
        music_on = False
    else:
        try:
            pygame.mixer.music.load(music_path)
            pygame.mixer.music.set_volume(0.5)
            pygame.mixer.music.play(-1)
        except pygame.error as e:
            print(f"Error loading game music: {e}")
            music_on = False

    # Tải âm thanh nhặt item
    click_sound_path = os.path.join(os.path.dirname(
        __file__), "..", "sounds", "click_sound.mp3")
    click_sound = None
    if os.path.exists(click_sound_path):
        try:
            click_sound = pygame.mixer.Sound(click_sound_path)
        except pygame.error as e:
            print(f"Error loading click sound: {e}")

    # Lấy đường dẫn thư mục chứa script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Tải hình ảnh nhân vật từ sprite sheet nhathamhiem.png
    running_frames = []
    player_sprite_path = os.path.join(
        base_dir, "..", "images", "nhathamhiem.png")
    if not os.path.exists(player_sprite_path):
        print(
            f"Critical error: Sprite sheet {player_sprite_path} not found. Exiting...")
        pygame.quit()
        sys.exit(1)
    try:
        sprite_sheet = pygame.image.load(player_sprite_path).convert_alpha()
        sprite_sheet = pygame.transform.scale(sprite_sheet, (200, 200))
        frame_size = 50
        for row in range(4):
            for col in range(2):
                frame = sprite_sheet.subsurface(pygame.Rect(
                    col * frame_size, row * frame_size, frame_size, frame_size))
                running_frames.append(frame)
    except pygame.error as e:
        print(f"Error loading sprite sheet {player_sprite_path}: {e}")
        for _ in range(8):
            frame = pygame.Surface((50, 50), pygame.SRCALPHA)
            frame.fill((255, 0, 0))
            running_frames.append(frame)

    # Tải hình ảnh mummy
    mummy_frames = []
    for i in range(1, 5):
        mummy_path = os.path.join(
            base_dir, "..", "images", "mummy", f"mummy{i}.png")
        if os.path.exists(mummy_path):
            try:
                frame = pygame.image.load(mummy_path).convert_alpha()
                frame = pygame.transform.scale(frame, (50, 50))
                mummy_frames.append(frame)
            except pygame.error as e:
                print(f"Error loading {mummy_path}: {e}")
                frame = pygame.Surface((50, 50), pygame.SRCALPHA)
                frame.fill((139, 69, 19))
                mummy_frames.append(frame)
        else:
            print(f"Warning: File {mummy_path} not found")
            frame = pygame.Surface((50, 50), pygame.SRCALPHA)
            frame.fill((139, 69, 19))
            mummy_frames.append(frame)

    # Tải hình ảnh item
    item_images = []
    for i in range(1, 6):
        path = os.path.join(base_dir, "..", "images", "items", f"item{i}.png")
        if os.path.exists(path):
            try:
                image = pygame.image.load(path).convert_alpha()
                image = pygame.transform.scale(image, (cell_size, cell_size))
                item_images.append(image)
            except pygame.error as e:
                print(f"Error loading {path}: {e}")
                placeholder = pygame.Surface(
                    (cell_size, cell_size), pygame.SRCALPHA)
                placeholder.fill((0, 255, 0))
                item_images.append(placeholder)
        else:
            print(f"Warning: File {path} not found")
            placeholder = pygame.Surface(
                (cell_size, cell_size), pygame.SRCALPHA)
            placeholder.fill((0, 255, 0))
            item_images.append(placeholder)

    # Tải hình ảnh khiên
    shield_image_path = os.path.join(base_dir, "..", "images", "shield.png")
    if os.path.exists(shield_image_path):
        try:
            shield_image = pygame.image.load(shield_image_path)
            shield_image = pygame.transform.scale(
                shield_image, (cell_size, cell_size))
        except pygame.error as e:
            print(f"Error loading {shield_image_path}: {e}")
            shield_image = pygame.Surface(
                (cell_size, cell_size), pygame.SRCALPHA)
            shield_image.fill((0, 0, 255))
    else:
        print(f"Warning: File {shield_image_path} not found")
        shield_image = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
        shield_image.fill((0, 0, 255))

    # Thiết lập ánh sáng
    LIGHT_RADIUS = 150
    light_mask = pygame.Surface(
        (width_maze * cell_size, height_maze * cell_size), pygame.SRCALPHA)
    light_mask.fill((0, 0, 0, 150))

    # Tải hình ảnh menu và scale để vừa chiều cao
    image_info_path = os.path.join(base_dir, "..", "images", "menu1.png")
    if os.path.exists(image_info_path):
        try:
            image_info = pygame.image.load(image_info_path)
            image_info = pygame.transform.scale(
                image_info, (500, height_maze * cell_size))
        except pygame.error as e:
            print(f"Error loading {image_info_path}: {e}")
            image_info = pygame.Surface(
                (500, height_maze * cell_size), pygame.SRCALPHA)
            image_info.fill((0, 0, 255))
    else:
        print(f"Warning: File {image_info_path} not found")
        image_info = pygame.Surface(
            (500, height_maze * cell_size), pygame.SRCALPHA)
        image_info.fill((0, 0, 255))

    # Hàm apply_action_maze để chuyển đổi hành động thành tọa độ mới
    def apply_action_maze(position, action):
        x, y = position
        if action == "UP":
            return (x - 1, y)
        elif action == "DOWN":
            return (x + 1, y)
        elif action == "LEFT":
            return (x, y - 1)
        elif action == "RIGHT":
            return (x, y + 1)
        return (x, y)  # Trả về vị trí hiện tại nếu hành động không hợp lệ

    # Class Button cho menu
    class Button:
        def __init__(self, text, x, y, w, h, callback):
            self.text = text
            self.rect = pygame.Rect(x, y, w, h)
            self.default_color = (120, 90, 40)
            self.hover_color = (160, 120, 70)
            self.click_color = (100, 70, 30)
            self.clicked = False
            self.callback = callback

        def draw(self, surface):
            mouse_pos = pygame.mouse.get_pos()
            if self.clicked:
                color = self.click_color
            elif self.rect.collidepoint(mouse_pos):
                color = self.hover_color
            else:
                color = self.default_color

            pygame.draw.rect(surface, color, self.rect, border_radius=8)
            pygame.draw.rect(surface, (255, 255, 255),
                             self.rect, 2, border_radius=8)
            label = menu_font.render(self.text, True, (255, 255, 255))
            surface.blit(label, (self.rect.centerx - label.get_width() //
                         2, self.rect.centery - label.get_height() // 2))

        def handle_event(self, event):
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.rect.collidepoint(pygame.mouse.get_pos()):
                    self.clicked = True
                    return self.callback()
            elif event.type == pygame.MOUSEBUTTONUP:
                self.clicked = False
            return None

    # Class Player
    class Player:
        def __init__(self, x, y, frames):
            self.grid_x = x
            self.grid_y = y
            self.quantity = 0
            self.pixel_x = x * cell_size
            self.pixel_y = y * cell_size
            self.frames = frames
            self.current_frame = 0
            self.animation_timer = 0
            self.animation_speed = 0.1
            self.speed = 120
            self.target_pixel = None
            self.items_collected = 0
            self.direction = "down"
            self.has_shield = False
            self.score_notifications = []

        def move_to(self, x, y):
            dx = x - self.grid_x
            dy = y - self.grid_y
            if dx == 1:
                self.direction = "down"
            elif dx == -1:
                self.direction = "up"
            elif dy == 1:
                self.direction = "right"
            elif dy == -1:
                self.direction = "left"

            self.grid_x = x
            self.grid_y = y
            self.target_pixel = (x * cell_size, y * cell_size)

        def update(self, delta_time):
            if self.target_pixel:
                dx = self.target_pixel[0] - self.pixel_x
                dy = self.target_pixel[1] - self.pixel_y
                distance = (dx ** 2 + dy ** 2) ** 0.5

                if distance < 5:
                    self.pixel_x, self.pixel_y = self.target_pixel
                    self.target_pixel = None
                    self.grid_x = int(self.pixel_x // cell_size)
                    self.grid_y = int(self.pixel_y // cell_size)
                    print(
                        f"Player updated to grid position: ({self.grid_x}, {self.grid_y})")
                else:
                    move_dist = self.speed * delta_time
                    self.pixel_x += move_dist * dx / distance
                    self.pixel_y += move_dist * dy / distance

                self.animation_timer += delta_time
                if self.animation_timer >= self.animation_speed:
                    self.animation_timer = 0
                    self.current_frame = (self.current_frame + 1) % 2

        def draw(self, screen, offset_x, offset_y):
            if self.direction == "down":
                frame_index = 0 + self.current_frame
            elif self.direction == "left":
                frame_index = 2 + self.current_frame
            elif self.direction == "right":
                frame_index = 4 + self.current_frame
            elif self.direction == "up":
                frame_index = 6 + self.current_frame

            frame = self.frames[frame_index]
            frame_rect = frame.get_rect(center=(
                self.pixel_y + offset_x + cell_size // 2, self.pixel_x + offset_y + cell_size // 2))
            screen.blit(frame, frame_rect.topleft)

        def save_state(self):
            return {
                "grid_x": self.grid_x,
                "grid_y": self.grid_y,
                "pixel_x": self.pixel_x,
                "pixel_y": self.pixel_y,
                "current_frame": self.current_frame,
                "animation_timer": self.animation_timer,
                "target_pixel": self.target_pixel,
                "items_collected": self.items_collected,
                "direction": self.direction,
                "has_shield": self.has_shield,
                "score_notifications": self.score_notifications
            }

        def load_state(self, state):
            try:
                self.grid_x = state["grid_x"]
                self.grid_y = state["grid_y"]
                self.pixel_x = state["pixel_x"]
                self.pixel_y = state["pixel_y"]
                self.current_frame = state["current_frame"]
                self.animation_timer = state["animation_timer"]
                self.target_pixel = state["target_pixel"]
                self.items_collected = state["items_collected"]
                self.direction = state["direction"]
                self.has_shield = state["has_shield"]
                self.score_notifications = state["score_notifications"]
            except KeyError as e:
                print(f"Error loading player state: Missing key {e}")
                raise

    # Class Mummy
    class Mummy:
        def __init__(self, x, y, frames):
            self.grid_x = x
            self.grid_y = y
            self.pixel_x = x * cell_size
            self.pixel_y = y * cell_size
            self.frames = frames
            self.current_frame = 0
            self.animation_timer = 0
            self.animation_speed = 0.1
            self.speed = 80
            self.target_pixel = None
            self.move_cooldown = 0

        def move_to(self, x, y):
            self.grid_x = x
            self.grid_y = y
            self.target_pixel = (x * cell_size, y * cell_size)

        def update(self, delta_time):
            if self.move_cooldown > 0:
                self.move_cooldown -= delta_time

            if self.target_pixel:
                dx = self.target_pixel[0] - self.pixel_x
                dy = self.target_pixel[1] - self.pixel_y
                distance = (dx ** 2 + dy ** 2) ** 0.5

                if distance < 10:
                    self.pixel_x, self.pixel_y = self.target_pixel
                    self.target_pixel = None
                    self.grid_x = int(self.pixel_x // cell_size)
                    self.grid_y = int(self.pixel_y // cell_size)
                else:
                    move_dist = self.speed * delta_time
                    self.pixel_x += move_dist * dx / distance
                    self.pixel_y += move_dist * dy / distance

                self.animation_timer += delta_time
                if self.animation_timer >= self.animation_speed:
                    self.animation_timer = 0
                    self.current_frame = (
                        self.current_frame + 1) % len(self.frames)

        def draw(self, screen, offset_x, offset_y):
            if mummy_active:
                screen.blit(self.frames[self.current_frame],
                            (self.pixel_y + offset_x, self.pixel_x + offset_y))

        def save_state(self):
            return {
                "grid_x": self.grid_x,
                "grid_y": self.grid_y,
                "pixel_x": self.pixel_x,
                "pixel_y": self.pixel_y,
                "current_frame": self.current_frame,
                "animation_timer": self.animation_timer,
                "target_pixel": self.target_pixel,
                "move_cooldown": self.move_cooldown
            }

        def load_state(self, state):
            try:
                self.grid_x = state["grid_x"]
                self.grid_y = state["grid_y"]
                self.pixel_x = state["pixel_x"]
                self.pixel_y = state["pixel_y"]
                self.current_frame = state["current_frame"]
                self.animation_timer = state["animation_timer"]
                self.target_pixel = state["target_pixel"]
                self.move_cooldown = state["move_cooldown"]
            except KeyError as e:
                print(f"Error loading mummy state: Missing key {e}")
                raise

    # Hàm tạo mê cung
    def generate_maze(width, height, extra_paths=30):
        maze = np.zeros((height, width), dtype=int)
        start_x, start_y = random.randrange(
            1, height, 2), random.randrange(1, width, 2)
        maze[start_x, start_y] = 1

        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        stack = [(start_x, start_y)]

        while stack:
            x, y = stack[-1]
            random.shuffle(directions)
            found = False
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and maze[nx, ny] == 0:
                    maze[x + dx // 2, y + dy // 2] = 1
                    maze[nx, ny] = 1
                    stack.append((nx, ny))
                    found = True
                    break
            if not found:
                stack.pop()

        for _ in range(extra_paths):
            rx, ry = random.randrange(
                1, height - 1, 2), random.randrange(1, width - 1, 2)
            neighbors = [(rx + dx, ry + dy) for dx, dy in directions if 0 <=
                         rx + dx < height and 0 <= ry + dy < width]
            if neighbors:
                nx, ny = random.choice(neighbors)
                maze[(rx + nx) // 2, (ry + ny) // 2] = 1

        return maze

    # Hàm đặt vật phẩm
    def place_items(maze, num_items=5):
        items = []
        available_positions = [(i, j) for i in range(maze.shape[0]) for j in range(maze.shape[1])
                               if maze[i, j] == 1 and (i, j) != (1, 1) and (i, j) != (height_maze-2, width_maze-2)]
        for _ in range(min(num_items, len(available_positions))):
            pos = random.choice(available_positions)
            items.append(pos)
            available_positions.remove(pos)
        return items

    # Hàm đặt khiên
    def place_shield(maze):
        available_positions = [(i, j) for i in range(maze.shape[0]) for j in range(maze.shape[1])
                               if maze[i, j] == 1 and (i, j) != (1, 1) and (i, j) != (height_maze-2, width_maze-2)]
        valid_positions = [
            pos for pos in available_positions if get_valid_moves(maze, pos)]
        if not valid_positions:
            print("Warning: No valid shield positions with valid moves")
            return []
        pos = random.choice(valid_positions)
        return [pos]

    # Hàm đặt bẫy
    def place_trap(maze):
        available_positions = [(i, j) for i in range(maze.shape[0]) for j in range(maze.shape[1])
                               if maze[i, j] == 1 and (i, j) != (1, 1) and (i, j) != (height_maze-2, width_maze-2)]
        valid_positions = [
            pos for pos in available_positions if get_valid_moves(maze, pos)]
        if not valid_positions:
            print("Warning: No valid trap positions with valid moves")
            return []
        pos = random.choice(valid_positions)
        return [pos]

    # Hàm DFS đơn giản cho mummy
    def dfs_mummy(maze, start, goal):
        stack = [(start, [start])]
        visited = set()

        while stack:
            (x, y), path = stack.pop()
            if (x, y) == goal:
                return path[1:] if len(path) > 1 else path

            if (x, y) not in visited:
                visited.add((x, y))
                for next_pos in get_valid_moves(maze, (x, y)):
                    if next_pos not in visited:
                        stack.append((next_pos, path + [next_pos]))
        return []

    # Hàm điều khiển AI cho mummy
    def mummy_ai(maze, mummy_pos, player_pos, algorithm):
        if isinstance(mummy_pos, np.ndarray):
            mummy_pos = tuple(mummy_pos)
        elif not isinstance(mummy_pos, tuple):
            mummy_pos = tuple(mummy_pos)
        if isinstance(player_pos, np.ndarray):
            player_pos = tuple(player_pos)
        elif not isinstance(player_pos, tuple):
            player_pos = tuple(player_pos)

        try:
            if algorithm == "A*":
                path = a_star(maze, mummy_pos, player_pos)
            elif algorithm == "DFS":
                path = dfs_mummy(maze, mummy_pos, player_pos)
            elif algorithm == "SA":
                path = find_path_sa(maze, mummy_pos, player_pos, [])
            elif algorithm == "Backtracking":
                path = find_path_backtracking(
                    maze, mummy_pos, player_pos, [], shields=[], traps=[])
            else:
                print(f"Unknown mummy algorithm {algorithm}, defaulting to A*")
                path = a_star(maze, mummy_pos, player_pos)
            if not path:
                print(
                    f"No path found for mummy from {mummy_pos} to {player_pos} using {algorithm}")
                valid_moves = get_valid_moves(maze, mummy_pos)
                if not valid_moves:
                    print(f"Mummy stuck at {mummy_pos}, no valid moves")
                    return mummy_pos
                return random.choice(valid_moves)
            print(f"Mummy path from {mummy_pos} to {player_pos}: {path}")
            return path[0]
        except Exception as e:
            print(f"Error in mummy pathfinding with {algorithm}: {e}")
            valid_moves = get_valid_moves(maze, mummy_pos)
            if not valid_moves:
                print(f"Mummy stuck at {mummy_pos}, no valid moves")
                return mummy_pos
            return random.choice(valid_moves)

    # Vẽ mê cung
    def draw_maze(screen, maze, item_data, traps, shield, player, mummy, offset_x, offset_y):
        print("Drawing maze...")
        if maze is None or item_data is None or player is None or mummy is None:
            print("Error: Invalid game state for drawing maze")
            return
        wall_image_path = os.path.join(base_dir, "..", "images", "wall.png")
        if os.path.exists(wall_image_path):
            try:
                wall_image = pygame.image.load(wall_image_path)
                wall_image = pygame.transform.scale(
                    wall_image, (cell_size, cell_size))
            except pygame.error as e:
                print(f"Error loading {wall_image_path}: {e}")
                wall_image = pygame.Surface((cell_size, cell_size))
                wall_image.fill((128, 128, 128))
        else:
            print(f"Warning: File {wall_image_path} not found")
            wall_image = pygame.Surface((cell_size, cell_size))
            wall_image.fill((128, 128, 128))

        temp_surface = pygame.Surface(
            (width_maze * cell_size, height_maze * cell_size))
        temp_surface.fill((0, 0, 0))

        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if maze[i, j] == 0:
                    temp_surface.blit(
                        wall_image, (j * cell_size, i * cell_size))

        for (item_x, item_y), image in item_data:
            temp_surface.blit(image, (item_y * cell_size, item_x * cell_size))

        goal_image_path = os.path.join(base_dir, "..", "images", "goal.png")
        if os.path.exists(goal_image_path):
            try:
                goal_img = pygame.image.load(goal_image_path)
                goal_img = pygame.transform.scale(
                    goal_img, (cell_size, cell_size))
            except pygame.error as e:
                print(f"Error loading {goal_image_path}: {e}")
                goal_img = pygame.Surface((cell_size, cell_size))
                goal_img.fill((255, 255, 0))
        else:
            print(f"Warning: File {goal_image_path} not found")
            goal_img = pygame.Surface((cell_size, cell_size))
            goal_img.fill((255, 255, 0))
        temp_surface.blit(goal_img, ((width_maze-2) *
                          cell_size, (height_maze-2) * cell_size))

        trap_image_path = os.path.join(base_dir, "..", "images", "trap.png")
        if os.path.exists(trap_image_path):
            try:
                trap_image = pygame.image.load(trap_image_path)
                trap_image = pygame.transform.scale(
                    trap_image, (cell_size, cell_size))
            except pygame.error as e:
                print(f"Error loading {trap_image_path}: {e}")
                trap_image = pygame.Surface((cell_size, cell_size))
                trap_image.fill((255, 0, 255))
        else:
            print(f"Warning: File {trap_image_path} not found")
            trap_image = pygame.Surface((cell_size, cell_size))
            trap_image.fill((255, 0, 255))

        for trap_x, trap_y in traps:
            temp_surface.blit(
                trap_image, (trap_y * cell_size, trap_x * cell_size))

        if shield:
            shield_x, shield_y = shield[0]
            temp_surface.blit(
                shield_image, (shield_y * cell_size, shield_x * cell_size))

        screen.blit(temp_surface, (offset_x, offset_y))

        mask = light_mask.copy()
        pygame.draw.circle(mask, (0, 0, 0, 0), (int(player.pixel_y + offset_x +
                           cell_size//2), int(player.pixel_x + offset_y + cell_size//2)), LIGHT_RADIUS)
        screen.blit(mask, (offset_x, offset_y))
        print("Maze drawn.")

    # Vẽ thông tin màn hình
    def draw_screen_info(items_collected, start_ticks, shield_status, mode, human_algorithm, mummy_algorithm, difficulty, map_name, player, offset_x):
        print("Drawing screen info...")
        screen.blit(image_info, (offset_x + 1050, offset_y))
        font = pygame.font.SysFont("consolas", 20, bold=True)
        seconds = (pygame.time.get_ticks() - start_ticks) // 1000
        time_text = font.render(f'Time: {seconds}s', True, (224, 205, 145))
        screen.blit(time_text, (offset_x + 1150, offset_y + 100))

        mode_text = font.render(f'Game mode: {mode}', True, (224, 205, 145))
        screen.blit(mode_text, (offset_x + 1150, offset_y + 150))
        human_text = font.render(
            f'Algorithm for human: {"Player" if mode == "Player vs AI" else human_algorithm}', True, (224, 205, 145))
        screen.blit(human_text, (offset_x + 1150, offset_y + 200))
        mummy_text = font.render(
            f'Algorithm for mummy: {mummy_algorithm}', True, (224, 205, 145))
        screen.blit(mummy_text, (offset_x + 1150, offset_y + 250))
        difficulty_text = font.render(
            f'Difficulty: {difficulty}', True, (224, 205, 145))
        screen.blit(difficulty_text, (offset_x + 1150, offset_y + 300))
        map_text = font.render(f'Map game: {map_name}', True, (224, 205, 145))
        screen.blit(map_text, (offset_x + 1150, offset_y + 350))

        text = font.render(
            f'Items: {items_collected}/5', True, (224, 205, 145))
        screen.blit(text, (offset_x + 1150, offset_y + 500))

        shield_text = font.render(
            f'Shield: {"Active" if shield_status else "Lost"}', True, (224, 205, 145))
        screen.blit(shield_text, (offset_x + 1150, offset_y + 450))

        rect = pygame.Rect(offset_x + 1150, offset_y + 550, 300, 100)
        base_color = (180, 180, 180)
        highlight = (255, 255, 255)
        shadow = (100, 100, 100)
        pygame.draw.rect(screen, base_color, rect)

        pygame.draw.line(screen, shadow, rect.topleft, rect.topright, 3)
        pygame.draw.line(screen, shadow, rect.topleft, rect.bottomleft, 3)
        pygame.draw.line(screen, highlight, rect.bottomleft,
                         rect.bottomright, 3)
        pygame.draw.line(screen, highlight, rect.topright, rect.bottomright, 3)

        y_offset = 5
        for notification in player.score_notifications:
            score_text = score_font.render(notification, True, (255, 255, 255))
            if y_offset + score_text.get_height() <= rect.height:
                screen.blit(score_text, (rect.x + 5, rect.y + 5 + y_offset))
                y_offset += 15
        print("Screen info drawn.")

    # Vẽ màn hình GAME OVER
    def draw_game_over_screen(screen):
        screen.fill((0, 0, 0))
        game_over_text = game_over_font.render(
            "GAME OVER", True, (255, 165, 0))
        screen.blit(game_over_text, (width // 2 -
                    game_over_text.get_width() // 2, 200))

        continue_text = menu_font.render("CONTINUE?", True, (255, 255, 255))
        screen.blit(continue_text, (width // 2 -
                    continue_text.get_width() // 2, 300))

        yes_button = Button("YES", width // 2 - 100, 400, 80, 40, restart_game)
        no_button = Button("NO", width // 2 + 20, 400, 80, 40, back_to_menu)
        yes_button.draw(screen)
        no_button.draw(screen)
        return yes_button, no_button

    # Vẽ thông báo nhặt trúng bẫy
    def draw_trap_notification(screen):
        if trap_notification_timer > 0:
            notification_text = notification_font.render(
                "DANGER! MUMMY ACTIVATED!", True, (255, 0, 0))
            screen.blit(notification_text, (width // 2 - notification_text.get_width() //
                        2, height // 2 - notification_text.get_height() // 2))

    # Vẽ "YOU WIN" với hiệu ứng sao giữa màn hình
    def draw_win_text(screen):
        print("Drawing YOU WIN text")
        win_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        text = "YOU WIN!"
        center_x = width // 2
        center_y = height // 2
        words = text.split()
        total_width = sum(win_font.size(
            word)[0] for word in words) + win_font.size(" ")[0] * (len(words) - 1)
        x_offset = center_x - total_width // 2
        for word in words:
            word_surface = win_font.render(word, True, (255, 215, 0))
            win_surface.blit(
                word_surface, (x_offset, center_y - word_surface.get_height() // 2))
            x_offset += word_surface.get_width() + win_font.size(" ")[0]
        for _ in range(20):
            star_x = center_x + random.randint(-200, 200)
            star_y = center_y + random.randint(-150, 150)
            star_size = random.randint(5, 10)
            pygame.draw.circle(win_surface, (255, 215, 0),
                               (int(star_x), int(star_y)), star_size // 2)
        screen.blit(win_surface, (0, 0))
        print("YOU WIN text drawn")

    # Hàm điều khiển nhạc
    def turn_on_music():
        nonlocal music_on
        if not music_on and os.path.exists(music_path):
            try:
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.play(-1)
                music_on = True
            except pygame.error as e:
                print(f"Error turning on music: {e}")

    def turn_off_music():
        nonlocal music_on
        if music_on:
            try:
                pygame.mixer.music.stop()
                music_on = False
            except pygame.error as e:
                print(f"Error turning off music: {e}")

    # Hàm xử lý menu
    def toggle_menu():
        nonlocal show_menu
        show_menu = not show_menu

    def back_to_menu():
        nonlocal running
        running = False
        try:
            state = save_game_state()
            print("Returning to menu with saved state:", {
                  k: type(v).__name__ for k, v in state.items()})
            return ("menu", state)
        except Exception as e:
            print(f"Error in back_to_menu: {e}")
            return ("menu", {})

    def pause_game():
        nonlocal paused
        paused = True

    def resume_game():
        nonlocal paused
        paused = False

    # Hàm khởi động lại game trên cùng map
    def restart_game():
        nonlocal player, mummy, item_data, trap, shield, path, current_step, mummy_active, mummy_timer, start_ticks, paused, show_menu, game_won, game_over, win_timer, trap_notification_timer
        print("Restarting game on the same map...")

        # Đặt lại trạng thái người chơi
        player_start = (1, 1)
        player = Player(player_start[0], player_start[1], running_frames)
        player.items_collected = 0
        player.has_shield = False
        player.score_notifications = []

        # Đặt lại trạng thái mummy
        mummy = Mummy(goal[0], goal[1], mummy_frames)

        # Đặt lại vật phẩm, bẫy, khiên
        item_positions = place_items(maze, num_items=5)
        item_data = list(zip(item_positions, random.sample(
            item_images, len(item_positions))))
        trap = place_trap(maze)
        shield = place_shield(maze)

        # Đặt lại đường đi cho AI nếu ở chế độ AI vs AI
        path = []
        current_step = 0
        if mode == "AI vs AI":
            algorithm = human_algorithm
            try:
                if algorithm == "A*":
                    path = find_path_astar(
                        maze, player_start, goal, item_positions)
                    print(f"A* path: {path}")
                elif algorithm == "DFS":
                    path = find_path_dfs(
                        maze, player_start, goal, item_positions)
                    print(f"DFS path: {path}")
                elif algorithm == "SA":
                    path = find_path_sa(maze, player_start,
                                        goal, item_positions)
                    print(f"SA path: {path}")
                elif algorithm == "Q-Learning":
                    print("Computing Q-Learning path...")
                    path = find_path_qlearning(
                        maze, player_start, goal, item_positions)
                    print("Q-Learning path computed:",
                          path if path else "None")
                    if not path:
                        print("Q-Learning failed, falling back to A*")
                        path = find_path_astar(
                            maze, player_start, goal, item_positions)
                elif algorithm == "Backtracking":
                    print("Computing Backtracking path...")
                    print(
                        f"Input for Backtracking: start={player_start}, goal={goal}, items={item_positions}, shields={shield}, traps={trap}")
                    path = find_path_backtracking(
                        maze, player_start, goal, item_positions, shields=shield, traps=trap)
                    print("Backtracking path computed:",
                          path if path else "None")
                    if not path:
                        print("Backtracking failed, falling back to A*")
                        path = find_path_astar(
                            maze, player_start, goal, item_positions)
                elif algorithm == "BFS-NoObs":
                    print("Computing BFS-NoObs path...")

                    class TempPlayer:
                        def __init__(self, x, y):
                            self.grid_x = x
                            self.grid_y = y
                    temp_player = TempPlayer(player_start[0], player_start[1])
                    path = search_no_observation_bfs_maze(
                        maze, [temp_player], goal)
                    current_pos = (temp_player.grid_x, temp_player.grid_y)
                    path_coords = []
                    for action in path:
                        next_pos = apply_action_maze(current_pos, action)
                        if (0 <= next_pos[0] < maze.shape[0] and 0 <= next_pos[1] < maze.shape[1] and maze[next_pos[0], next_pos[1]] == 1):
                            path_coords.append(next_pos)
                            current_pos = next_pos
                        else:
                            print(
                                f"Invalid action {action} at {current_pos}, stopping path conversion")
                            break
                    path = path_coords
                    print("BFS-NoObs path computed:", path if path else "None")
                    if not path:
                        print("BFS-NoObs failed, falling back to A*")
                        path = find_path_astar(
                            maze, player_start, goal, item_positions)
                else:
                    print(f"Unknown algorithm {algorithm}, defaulting to A*")
                    path = find_path_astar(
                        maze, player_start, goal, item_positions)
                if path and path[-1] != goal:
                    print(
                        f"Warning: Path does not end at goal {goal}, last position: {path[-1]}")
                    path = find_path_astar(
                        maze, player_start, goal, item_positions)
                    print(f"Fallback to A* path: {path}")
            except Exception as e:
                print(f"Error in pathfinding with {algorithm}: {e}")
                print("Falling back to A*")
                path = find_path_astar(
                    maze, player_start, goal, item_positions)

            if path is None or not path:
                print("No valid path found, using A* as final fallback")
                path = find_path_astar(
                    maze, player_start, goal, item_positions)

        # Đặt lại các biến trạng thái
        mummy_active = False
        mummy_timer = 0
        start_ticks = pygame.time.get_ticks()
        paused = False
        show_menu = False
        game_won = False
        game_over = False
        win_timer = 0
        trap_notification_timer = 0

        print("Game restarted with same map.")
        return None

    # Hàm lưu trạng thái game
    def save_game_state():
        item_positions = [(x, y) for (x, y), _ in item_data]
        state = {
            "maze": maze.copy() if maze is not None else generate_maze(width_maze, height_maze, extra_paths=30),
            "player": player.save_state() if player is not None else Player(1, 1, running_frames).save_state(),
            "mummy": mummy.save_state() if mummy is not None else Mummy(goal[0], goal[1], mummy_frames).save_state(),
            "item_positions": item_positions,
            "trap": trap.copy() if trap is not None else [],
            "shield": shield.copy() if shield is not None else [],
            "path": path[current_step:] if mode == "AI vs AI" and path is not None else [],
            "current_step": current_step if mode == "AI vs AI" else 0,
            "mummy_timer": mummy_timer if mummy_timer is not None else 0,
            "mummy_active": mummy_active,
            "start_ticks": start_ticks if start_ticks is not None else pygame.time.get_ticks(),
            "paused": paused,
            "show_menu": show_menu,
            "music_on": music_on,
            "goal": goal if goal is not None else (height_maze - 2, width_maze - 2),
            "width_maze": width_maze,
            "height_maze": height_maze,
            "cell_size": cell_size,
            "config": config if config is not None else {"mode": "AI vs AI", "difficulty": "Easy", "map": "New Map", "mummy_algorithm": "A*"},
            "game_won": game_won,
            "game_over": game_over,
            "win_timer": win_timer,
            "trap_notification_timer": trap_notification_timer
        }
        missing_keys = [k for k in state if state[k] is None]
        if missing_keys:
            print(
                f"Warning: Missing or None values in saved state: {missing_keys}")
        return state

    # Khởi tạo các biến trạng thái game
    game_won = False
    game_over = False
    win_timer = 0
    trap_notification_timer = 0

    # Danh sách các khóa bắt buộc trong saved_state
    required_state_keys = [
        "maze", "player", "mummy", "item_positions", "trap", "shield",
        "path", "current_step", "mummy_timer", "mummy_active", "start_ticks",
        "paused", "show_menu", "music_on", "goal", "width_maze", "height_maze",
        "cell_size", "config", "game_won", "game_over", "win_timer",
        "trap_notification_timer"
    ]

    # Khởi tạo game
    if saved_state:
        print("Loading saved state...")
        try:
            missing_keys = [
                key for key in required_state_keys if key not in saved_state]
            if missing_keys:
                raise KeyError(f"Missing keys in saved_state: {missing_keys}")

            maze = saved_state["maze"]
            if not isinstance(maze, np.ndarray):
                raise ValueError(
                    "Invalid maze in saved state: not a numpy array")
            width_maze = saved_state["width_maze"]
            height_maze = saved_state["height_maze"]
            cell_size = saved_state["cell_size"]
            player = Player(1, 1, running_frames)
            player.load_state(saved_state["player"])
            goal = saved_state["goal"]
            if not isinstance(goal, tuple) or len(goal) != 2:
                raise ValueError("Invalid goal in saved_state")
            mummy = Mummy(goal[0], goal[1], mummy_frames)
            mummy.load_state(saved_state["mummy"])
            item_positions = saved_state["item_positions"]
            if not all(isinstance(pos, tuple) and len(pos) == 2 for pos in item_positions):
                raise ValueError("Invalid item_positions in saved_state")
            item_data = list(zip(item_positions, random.sample(
                item_images, len(item_positions))))
            trap = saved_state["trap"]
            shield = saved_state["shield"]
            path = saved_state["path"]
            current_step = saved_state["current_step"]
            mummy_timer = saved_state["mummy_timer"]
            mummy_active = saved_state["mummy_active"]
            start_ticks = saved_state["start_ticks"]
            paused = saved_state["paused"]
            show_menu = saved_state["show_menu"]
            music_on = saved_state["music_on"]
            game_won = saved_state["game_won"]
            game_over = saved_state["game_over"]
            win_timer = saved_state["win_timer"]
            trap_notification_timer = saved_state["trap_notification_timer"]
            print(
                f"Loaded saved state: player at ({player.grid_x}, {player.grid_y}), mummy at ({mummy.grid_x}, {mummy.grid_y}), items: {len(item_positions)}, trap_notification_timer: {trap_notification_timer}")
        except (KeyError, ValueError) as e:
            print(f"Error loading saved state: {e}. Starting new game...")
            saved_state = None

    if not saved_state:
        print("Initializing new game...")
        # Ưu tiên sử dụng Old Map nếu được chọn
        if map_name == "Old Map" and old_map is not None:
            if isinstance(old_map, np.ndarray):
                if old_map.shape == (height_maze, width_maze):
                    print("Using Old Map from config with matching size")
                    maze = old_map.copy()
                else:
                    print(
                        f"Warning: Old Map size {old_map.shape} does not match expected {height_maze}x{width_maze}. Resizing or generating new map...")
                    # Thử điều chỉnh kích thước hoặc tạo map mới nếu không khớp
                    maze = generate_maze(
                        width_maze, height_maze, extra_paths=30)
            else:
                print(
                    f"Warning: old_map is not a numpy array, got {type(old_map)}. Generating new maze...")
                maze = generate_maze(width_maze, height_maze, extra_paths=30)
        else:
            print("No Old Map specified or New Map selected. Generating new maze...")
            maze = generate_maze(width_maze, height_maze, extra_paths=30)
        print("Maze initialized:\n", maze)
    else:
        # Khi có saved_state, sử dụng maze từ saved_state
        maze = saved_state["maze"]
        print("Using maze from saved state:\n", maze)

    print("Maze generated:\n", maze)
    player_start = (1, 1)
    player = Player(player_start[0], player_start[1], running_frames)
    goal = (height_maze - 2, width_maze - 2)
    mummy = Mummy(goal[0], goal[1], mummy_frames)

    item_positions = place_items(maze, num_items=5)
    item_data = list(zip(item_positions, random.sample(
        item_images, len(item_positions))))
    trap = place_trap(maze)
    shield = place_shield(maze)

    path = []
    current_step = 0
    if mode == "AI vs AI":
        algorithm = human_algorithm
        try:
            if algorithm == "A*":
                path = find_path_astar(
                    maze, player_start, goal, item_positions)
                print(f"A* path: {path}")
            elif algorithm == "DFS":
                path = find_path_dfs(maze, player_start, goal, item_positions)
                print(f"DFS path: {path}")
            elif algorithm == "SA":
                path = find_path_sa(maze, player_start, goal, item_positions)
                print(f"SA path: {path}")
            elif algorithm == "Q-Learning":
                print("Computing Q-Learning path...")
                path = find_path_qlearning(
                    maze, player_start, goal, item_positions)
                print("Q-Learning path computed:", path if path else "None")
                if not path:
                    print("Q-Learning failed, falling back to A*")
                    path = find_path_astar(
                        maze, player_start, goal, item_positions)
            elif algorithm == "Backtracking":
                print("Computing Backtracking path...")
                print(
                    f"Input for Backtracking: start={player_start}, goal={goal}, items={item_positions}, shields={shield}, traps={trap}")
                path = find_path_backtracking(
                    maze, player_start, goal, item_positions, shields=shield, traps=trap)
                print("Backtracking path computed:", path if path else "None")
                if not path:
                    print("Backtracking failed, falling back to A*")
                    path = find_path_astar(
                        maze, player_start, goal, item_positions)
            elif algorithm == "BFS-NoObs":
                print("Computing BFS-NoObs path...")

                class TempPlayer:
                    def __init__(self, x, y):
                        self.grid_x = x
                        self.grid_y = y
                temp_player = TempPlayer(player_start[0], player_start[1])
                path = search_no_observation_bfs_maze(
                    maze, [temp_player], goal)
                current_pos = (temp_player.grid_x, temp_player.grid_y)
                path_coords = []
                for action in path:
                    next_pos = apply_action_maze(current_pos, action)
                    if (0 <= next_pos[0] < maze.shape[0] and 0 <= next_pos[1] < maze.shape[1] and maze[next_pos[0], next_pos[1]] == 1):
                        path_coords.append(next_pos)
                        current_pos = next_pos
                    else:
                        print(
                            f"Invalid action {action} at {current_pos}, stopping path conversion")
                        break
                path = path_coords
                print("BFS-NoObs path computed:", path if path else "None")
                if not path:
                    print("BFS-NoObs failed, falling back to A*")
                    path = find_path_astar(
                        maze, player_start, goal, item_positions)
            else:
                print(f"Unknown algorithm {algorithm}, defaulting to A*")
                path = find_path_astar(
                    maze, player_start, goal, item_positions)
            if path and path[-1] != goal:
                print(
                    f"Warning: Path does not end at goal {goal}, last position: {path[-1]}")
                path = find_path_astar(
                    maze, player_start, goal, item_positions)
                print(f"Fallback to A* path: {path}")
        except Exception as e:
            print(f"Error in pathfinding with {algorithm}: {e}")
            print("Falling back to A*")
            path = find_path_astar(maze, player_start, goal, item_positions)

        if path is None or not path:
            print("No valid path found, generating new maze")
            if map_name == "Old Map" and old_map is not None and isinstance(old_map, np.ndarray) and old_map.shape == (height_maze, width_maze):
                print("Reusing Old Map from config despite path failure")
                maze = old_map.copy()
            else:
                maze = generate_maze(width_maze, height_maze, extra_paths=30)
            item_positions = place_items(maze, num_items=5)
            item_data = list(zip(item_positions, random.sample(
                item_images, len(item_positions))))
            trap = place_trap(maze)
            shield = place_shield(maze)
            try:
                if algorithm == "A*":
                    path = find_path_astar(
                        maze, player_start, goal, item_positions)
                    print(f"Retry A* path: {path}")
                elif algorithm == "DFS":
                    path = find_path_dfs(
                        maze, player_start, goal, item_positions)
                    print(f"Retry DFS path: {path}")
                elif algorithm == "SA":
                    path = find_path_sa(maze, player_start,
                                        goal, item_positions)
                    print(f"Retry SA path: {path}")
                elif algorithm == "Q-Learning":
                    print("Retrying Q-Learning path...")
                    path = find_path_qlearning(
                        maze, player_start, goal, item_positions)
                    print("Q-Learning retry path:", path if path else "None")
                    if not path:
                        print("Q-Learning retry failed, falling back to A*")
                        path = find_path_astar(
                            maze, player_start, goal, item_positions)
                elif algorithm == "Backtracking":
                    print("Retrying Backtracking path...")
                    print(
                        f"Retry input for Backtracking: start={player_start}, goal={goal}, items={item_positions}, shields={shield}, traps={trap}")
                    path = find_path_backtracking(
                        maze, player_start, goal, item_positions, shields=shield, traps=trap)
                    print("Backtracking retry path:", path if path else "None")
                    if not path:
                        print("Backtracking retry failed, falling back to A*")
                        path = find_path_astar(
                            maze, player_start, goal, item_positions)
                elif algorithm == "BFS-NoObs":
                    print("Retrying BFS-NoObs path...")

                    class TempPlayer:
                        def __init__(self, x, y):
                            self.grid_x = x
                            self.grid_y = y
                    temp_player = TempPlayer(player_start[0], player_start[1])
                    path = search_no_observation_bfs_maze(
                        maze, [temp_player], goal)
                    current_pos = (temp_player.grid_x, temp_player.grid_y)
                    path_coords = []
                    for action in path:
                        next_pos = apply_action_maze(current_pos, action)
                        if (0 <= next_pos[0] < maze.shape[0] and 0 <= next_pos[1] < maze.shape[1] and maze[next_pos[0], next_pos[1]] == 1):
                            path_coords.append(next_pos)
                            current_pos = next_pos
                        else:
                            print(
                                f"Invalid action {action} at {current_pos}, stopping path conversion")
                            break
                    path = path_coords
                    print("BFS-NoObs retry path:", path if path else "None")
                    if not path:
                        print("BFS-NoObs retry failed, falling back to A*")
                        path = find_path_astar(
                            maze, player_start, goal, item_positions)
                else:
                    path = find_path_astar(
                        maze, player_start, goal, item_positions)
            except Exception as e:
                print(f"Error in retry pathfinding with {algorithm}: {e}")
                print("Falling back to A*")
                path = find_path_astar(
                    maze, player_start, goal, item_positions)

            if path is None or not path:
                print("Retry failed, using A* as final fallback")
                path = find_path_astar(
                    maze, player_start, goal, item_positions)

    print("Player path:", path)
    mummy_active = False
    mummy_timer = 0
    start_ticks = pygame.time.get_ticks()
    paused = False
    show_menu = False
    trap_notification_timer = 0

    maze_width = width_maze * cell_size
    maze_height = height_maze * cell_size
    total_height = maze_height
    offset_x = (width - maze_width - 500) // 2
    offset_y = 0

    clock = pygame.time.Clock()
    running = True
    mummy_move_delay = 0.5
    menu_button = pygame.Rect(width - 100 + offset_x, height - 90, 60, 45)

    menu_rect_x = (width - 250) // 2
    menu_rect_y = (height - 320) // 2
    menu_buttons = [
        Button("Turn On Music", menu_rect_x + 25,
               menu_rect_y + 20, 200, 40, turn_on_music),
        Button("Turn Off Music", menu_rect_x + 25,
               menu_rect_y + 80, 200, 40, turn_off_music),
        Button("Back to Menu", menu_rect_x + 25,
               menu_rect_y + 140, 200, 40, back_to_menu),
        Button("Pause", menu_rect_x + 25,
               menu_rect_y + 200, 200, 40, pause_game),
        Button("Resume", menu_rect_x + 25,
               menu_rect_y + 260, 200, 40, resume_game)
    ]

    return_value = None
    while running:
        delta_time = clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return_value = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if menu_button.collidepoint(event.pos):
                    toggle_menu()
                if show_menu:
                    for button in menu_buttons:
                        result = button.handle_event(event)
                        if result and isinstance(result, tuple) and result[0] == "menu":
                            running = False
                            return_value = result
                if game_over:
                    yes_button, no_button = draw_game_over_screen(screen)
                    result = None
                    if yes_button.rect.collidepoint(event.pos):
                        print("Restarting game with same map...")
                        if click_sound:
                            click_sound.play()
                        result = yes_button.handle_event(event)
                    elif no_button.rect.collidepoint(event.pos):
                        print("Returning to menu...")
                        if click_sound:
                            click_sound.play()
                        result = no_button.handle_event(event)
                    if result and isinstance(result, tuple):
                        running = False
                        return_value = result
            elif event.type == pygame.KEYDOWN and mode == "Player vs AI" and not paused and not show_menu and not game_won and not game_over:
                next_pos = (player.grid_x, player.grid_y)
                if event.key in (pygame.K_w, pygame.K_UP):
                    next_pos = (player.grid_x - 1, player.grid_y)
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    next_pos = (player.grid_x + 1, player.grid_y)
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    next_pos = (player.grid_x, player.grid_y - 1)
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    next_pos = (player.grid_x, player.grid_y + 1)

                if (0 <= next_pos[0] < maze.shape[0] and 0 <= next_pos[1] < maze.shape[1] and maze[next_pos[0], next_pos[1]] == 1 and player.target_pixel is None):
                    player.move_to(next_pos[0], next_pos[1])
                    print(f"Player moved to {next_pos}")

        if not paused and not show_menu and not game_won and not game_over:
            if mode == "AI vs AI":
                if current_step < len(path) and player.target_pixel is None:
                    next_pos = path[current_step]
                    if heuristic((player.grid_x, player.grid_y), goal) < 3:
                        print(
                            f"Player near goal at step {current_step}: {next_pos}, goal: {goal}")
                    player.move_to(next_pos[0], next_pos[1])
                    current_step += 1
                else:
                    print(
                        f"Player cannot move: current_step={current_step}, len(path)={len(path)}, target_pixel={player.target_pixel}")

            for i, ((item_x, item_y), _) in enumerate(item_data[:]):
                if (player.grid_x, player.grid_y) == (item_x, item_y):
                    player.items_collected += 1
                    current_time = (pygame.time.get_ticks() -
                                    start_ticks) // 1000
                    notification = f"Picked item at ({player.grid_x}, {player.grid_y}) - Time: {current_time}s"
                    player.score_notifications.append(notification)
                    item_data.pop(i)
                    if click_sound:
                        click_sound.play()
                    print(
                        f"Item collected at {player.grid_x, player.grid_y}, total: {player.items_collected}/5")

            if (player.grid_x, player.grid_y) in shield:
                shield = []
                player.has_shield = True
                print(
                    f"Shield collected at ({player.grid_x}, {player.grid_y}), shield active")

            if (player.grid_x, player.grid_y) in trap:
                print("Trap detected!")
                trap.remove((player.grid_x, player.grid_y))
                if player.has_shield:
                    print("Shield consumed")
                    player.has_shield = False
                    print(
                        f"Trap triggered at ({player.grid_x}, {player.grid_y}), shield lost!")
                else:
                    print("Activating mummy")
                    mummy_active = True
                    mummy_timer = 0
                    mummy.move_cooldown = 0
                    mummy.move_to(goal[0], goal[1])
                    trap_notification_timer = 3
                    print(
                        f"Trap triggered at ({player.grid_x}, {player.grid_y}), mummy activated from {goal}, mummy_active={mummy_active}")

            player.update(delta_time)

            print(
                f"Player position: ({player.grid_x}, {player.grid_y}), Goal: {goal}, Items: {player.items_collected}, Target pixel: {player.target_pixel}, Mode: {mode}")
            if mode == "Player vs AI" and (player.grid_x, player.grid_y) == goal:
                game_won = True
                win_timer = 5.0
                print(f"Game won! Player at goal {goal} in {mode} mode")
            elif mode == "AI vs AI" and (player.grid_x, player.grid_y) == goal and player.items_collected >= 5:
                game_won = True
                win_timer = 5.0
                print(
                    f"Game won! Player at goal {goal} with {player.items_collected} items in {mode} mode")
            elif (player.grid_x, player.grid_y) == goal:
                print(
                    f"Player at goal {goal} but items collected: {player.items_collected}/5 in {mode} mode")

            mummy_timer += delta_time
            if mummy_active and not player.has_shield:
                print(
                    f"Mummy active, timer={mummy_timer:.2f}, target_pixel={mummy.target_pixel}, position=({mummy.grid_x}, {mummy.grid_y}), cooldown={mummy.move_cooldown:.2f}")
                if mummy_timer >= mummy_move_delay and mummy.move_cooldown <= 0 and mummy.target_pixel is None:
                    next_pos = mummy_ai(
                        maze, (mummy.grid_x, mummy.grid_y), (player.grid_x, player.grid_y), mummy_algorithm)
                    if (0 <= next_pos[0] < maze.shape[0] and 0 <= next_pos[1] < maze.shape[1] and maze[next_pos[0], next_pos[1]] == 1):
                        mummy.move_to(next_pos[0], next_pos[1])
                        mummy.move_cooldown = mummy_move_delay
                        print(
                            f"Mummy moved to {next_pos} using {mummy_algorithm}")
                    else:
                        print(
                            f"Invalid mummy move to {next_pos}, staying at ({mummy.grid_x}, {mummy.grid_y})")

            mummy.update(delta_time)

            if mummy_active and not player.has_shield:
                if (player.grid_x, player.grid_y) == (mummy.grid_x, mummy.grid_y):
                    game_over = True
                    print(
                        f"Collision detected at ({player.grid_x}, {player.grid_y}), game over!")

        if trap_notification_timer > 0:
            trap_notification_timer -= delta_time
            print(f"Trap notification timer: {trap_notification_timer:.2f}")

        if game_won and win_timer > 0:
            win_timer -= delta_time
            print(f"Win timer: {win_timer:.2f}")
            if win_timer <= 0:
                running = False
                return_value = ("menu", {})
                print("Win timer expired, returning to menu")

        screen.fill((0, 0, 0))
        if game_over:
            yes_button, no_button = draw_game_over_screen(screen)
        elif game_won:
            draw_maze(screen, maze, item_data, trap, shield,
                      player, mummy, offset_x, offset_y)
            player.draw(screen, offset_x, offset_y)
            mummy.draw(screen, offset_x, offset_y)
            draw_screen_info(player.items_collected, start_ticks, player.has_shield, mode,
                             human_algorithm, mummy_algorithm, difficulty, map_name, player, offset_x)
            draw_trap_notification(screen)
            draw_win_text(screen)
        else:
            draw_maze(screen, maze, item_data, trap, shield,
                      player, mummy, offset_x, offset_y)
            player.draw(screen, offset_x, offset_y)
            mummy.draw(screen, offset_x, offset_y)
            draw_screen_info(player.items_collected, start_ticks, player.has_shield, mode,
                             human_algorithm, mummy_algorithm, difficulty, map_name, player, offset_x)
            draw_trap_notification(screen)

        if show_menu:
            menu_rect = pygame.Rect(menu_rect_x, menu_rect_y, 250, 320)
            pygame.draw.rect(screen, (50, 40, 20), menu_rect, border_radius=10)
            pygame.draw.rect(screen, (255, 255, 255),
                             menu_rect, 2, border_radius=10)
            for button in menu_buttons:
                button.draw(screen)

        if not game_won and not game_over:
            mouse_pos = pygame.mouse.get_pos()
            menu_button_color = (240, 200, 100) if menu_button.collidepoint(
                mouse_pos) else (200, 180, 100)
            pygame.draw.rect(screen, menu_button_color,
                             menu_button, border_radius=5)
            pygame.draw.rect(screen, (255, 255, 255),
                             menu_button, 2, border_radius=5)
            for i in range(3):
                pygame.draw.line(screen, (255, 255, 255), (menu_button.x + 10, menu_button.y + 12 + i * 10),
                                 (menu_button.x + 50, menu_button.y + 12 + i * 10), 3)

        pygame.display.flip()

    try:
        pygame.mixer.music.stop()
    except pygame.error as e:
        print(f"Error stopping game music: {e}")
    pygame.quit()
    print("Game loop ended, return_value:", return_value)
    return return_value
