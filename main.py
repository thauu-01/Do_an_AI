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
    except pygame.error as e:
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
        pygame.quit()
        sys.exit(1)

    # Introduce scene
    if pygame.get_init():
        introduce = IntroduceScene(screen)
        while not introduce.is_finished():
            introduce.update()
    else:
        pygame.quit()
        sys.exit(1)

    while True:
        # Menu scene
        if not pygame.get_init():
            pygame.init()
            pygame.font.init()
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Egypt Maze Game")

        menu = MenuScene(screen, has_saved_game=has_saved_game,
                         saved_state=saved_state)
        result = menu.run()

        if result is None:
            break

        if isinstance(result, tuple) and result[0] == "continue":
            if result[1] and isinstance(result[1], dict):
                config = result[1].get("config", {
                                       "mode": "Player vs AI", "difficulty": "Easy", "map": "New Map", "mummy_algorithm": "A*"})
                game_result = maingame(config, saved_state=result[1])
            else:
                font = pygame.font.SysFont("Papyrus", 28)
                error_text = font.render(
                    "Cannot continue: No saved game!", True, (255, 0, 0))
                screen.fill((0, 0, 0))
                screen.blit(error_text, (WIDTH // 2 - error_text.get_width() // 2,
                                         HEIGHT // 2 - error_text.get_height() // 2))
                pygame.display.flip()
                pygame.time.wait(2000)  # Wait 2 seconds
                continue
        elif isinstance(result, tuple) and result[0] == "start_game":
            config = result[1]
            if isinstance(config, dict):
                required_keys = ["mode", "difficulty",
                                 "map", "mummy_algorithm"]
                if all(k in config for k in required_keys) and config["mode"] in ["Player vs AI", "AI vs AI"]:
                    game_result = maingame(config)
                else:
                    font = pygame.font.SysFont("Papyrus", 28)
                    error_text = font.render(
                        "Invalid game configuration!", True, (255, 0, 0))
                    screen.fill((0, 0, 0))
                    screen.blit(error_text, (WIDTH // 2 - error_text.get_width() // 2,
                                             HEIGHT // 2 - error_text.get_height() // 2))
                    pygame.display.flip()
                    pygame.time.wait(2000)  # Wait 2 seconds
                    continue
            else:
                continue
        else:
            continue

        if game_result and isinstance(game_result, tuple) and game_result[0] == "menu":
            saved_state = game_result[1]
            # Validate saved_state to set has_saved_game
            has_saved_game = bool(saved_state and isinstance(saved_state, dict) and
                                  "maze" in saved_state and "players" in saved_state and
                                  "mummy" in saved_state and "item_data" in saved_state)
        else:
            saved_state = None
            has_saved_game = False

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
