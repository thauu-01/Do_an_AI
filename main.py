import pygame
import sys
from scenes.loading import LoadingScene
from scenes.introduce import IntroduceScene
from scenes.menu import MenuScene
from scenes.main_game import maingame
from config import WIDTH, HEIGHT


def main():
    pygame.init()
    try:
        pygame.font.init()
        print("Pygame font module initialized")
    except pygame.error as e:
        print(f"Error initializing pygame font module: {e}")
        sys.exit(1)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Egypt Maze Game")

    saved_state = None
    has_saved_game = False

    # Loading scene
    if pygame.get_init():
        loading = LoadingScene(screen)
        while not loading.is_finished():
            loading.update()
    else:
        print("Error: Pygame not initialized for LoadingScene")
        pygame.quit()
        sys.exit(1)

    # Introduce scene
    if pygame.get_init():
        introduce = IntroduceScene(screen)
        while not introduce.is_finished():
            introduce.update()
    else:
        print("Error: Pygame not initialized for IntroduceScene")
        pygame.quit()
        sys.exit(1)

    while True:
        # Menu scene
        if not pygame.get_init():
            print("Error: Pygame not initialized, reinitializing")
            pygame.init()
            pygame.font.init()
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Egypt Maze Game")

        menu = MenuScene(screen, has_saved_game=has_saved_game,
                         saved_state=saved_state)
        result = menu.run()

        if result is None:
            print("Exiting game via menu quit")
            break

        if isinstance(result, tuple) and result[0] == "continue":
            print("Resuming game with saved state")
            if result[1] and isinstance(result[1], dict) and "config" in result[1]:
                game_result = maingame(
                    result[1]["config"], saved_state=result[1])
            else:
                print("Error: Invalid saved state for continue, missing 'config'")
                # Hiển thị thông báo lỗi trong menu
                font = pygame.font.SysFont("Papyrus", 28)
                error_text = font.render(
                    "Cannot continue: Invalid saved game!", True, (255, 0, 0))
                screen.fill((0, 0, 0))
                screen.blit(error_text, (WIDTH // 2 - error_text.get_width() // 2,
                                         HEIGHT // 2 - error_text.get_height() // 2))
                pygame.display.flip()
                pygame.time.wait(2000)  # Chờ 2 giây
                continue
        elif isinstance(result, dict):
            required_keys = ["mode", "difficulty", "map", "mummy_algorithm"]
            if all(k in result for k in required_keys) and result["mode"] in ["Player vs AI", "AI vs AI"]:
                print("Starting new game with config:", result)
                game_result = maingame(result)
            else:
                print("Error: Invalid config from menu:", result)
                continue
        else:
            print("Error: Unexpected menu result:", result)
            continue

        if game_result and isinstance(game_result, tuple) and game_result[0] == "menu":
            print("Returned to menu with saved state")
            saved_state = game_result[1]
            # Chỉ đặt True nếu saved_state không rỗng
            has_saved_game = bool(saved_state)
        else:
            print("Game ended, clearing saved state")
            saved_state = None
            has_saved_game = False

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
