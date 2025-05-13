import heapq
import random
import numpy as np
import math
import copy
from collections import deque


def heuristic(a, b):
    """Tính heuristic (khoảng cách Manhattan) giữa hai điểm."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_valid_moves(maze, current_pos):
    """Lấy danh sách các ô hợp lệ lân cận từ vị trí hiện tại."""
    if isinstance(current_pos, np.ndarray):
        current_pos = tuple(current_pos)
    elif not isinstance(current_pos, tuple):
        current_pos = tuple(current_pos)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    valid_moves = []
    for dx, dy in directions:
        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
        if (0 <= next_pos[0] < maze.shape[0] and
            0 <= next_pos[1] < maze.shape[1] and
                maze[next_pos[0], next_pos[1]] == 1):
            valid_moves.append(next_pos)
    return valid_moves


def a_star(maze, start, goal):
    """Thuật toán A* tìm đường từ start đến goal."""
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {}
    cost_so_far = {start: 0}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for dx, dy in directions:
            next_pos = (current[0] + dx, current[1] + dy)
            if (0 <= next_pos[0] < maze.shape[0] and
                0 <= next_pos[1] < maze.shape[1] and
                    maze[next_pos[0], next_pos[1]] == 1):
                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

    path = []
    current = goal
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path[1:] if path else []


def dfs(maze, start, goal, visited=None):
    """Thuật toán DFS tìm đường từ start đến goal."""
    if visited is None:
        visited = set()
    if start == goal:
        return [start]
    visited.add(start)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        next_pos = (start[0] + dx, start[1] + dy)
        if (0 <= next_pos[0] < maze.shape[0] and
            0 <= next_pos[1] < maze.shape[1] and
            maze[next_pos[0], next_pos[1]] == 1 and
                next_pos not in visited):
            path = dfs(maze, next_pos, goal, visited)
            if path:
                return [start] + path
    return []


def find_path_dfs(maze, start, goal, items):
    """Tìm đường bằng DFS qua tất cả items rồi đến goal."""
    path = []
    current_pos = start

    for item in items:
        item_path = dfs(maze, current_pos, item)
        if not item_path:
            print(f"Không tìm được đường từ {current_pos} đến item {item}")
            return []
        path.extend(item_path[1:])
        current_pos = item

    final_path = dfs(maze, current_pos, goal)
    if not final_path:
        print(f"Không tìm được đường từ {current_pos} đến goal {goal}")
        return []
    path.extend(final_path[1:])

    return path


def find_path_astar(maze, start, goal, items):
    """Tìm đường bằng A* qua tất cả items rồi đến goal."""
    path = []
    current_pos = start

    for item in items:
        item_path = a_star(maze, current_pos, item)
        if not item_path:
            print(f"Không tìm được đường từ {current_pos} đến item {item}")
            return []
        path.extend(item_path)
        current_pos = item

    final_path = a_star(maze, current_pos, goal)
    if not final_path:
        print(f"Không tìm được đường từ {current_pos} đến goal {goal}")
        return []
    path.extend(final_path)

    return path


def initial_solution(start, goal, maze):
    """Tạo giải pháp ban đầu ngẫu nhiên với giới hạn lặp lại."""
    path = [start]
    current = start
    max_attempts = 5000
    attempts = 0
    visited = set([start])
    while current != goal and attempts < max_attempts:
        valid_moves = get_valid_moves(maze, current)
        valid_moves = [
            move for move in valid_moves if move not in visited or move == goal]
        if not valid_moves:
            print(f"Không có nước đi hợp lệ từ {current}")
            return []
        next_pos = random.choice(valid_moves)
        path.append(next_pos)
        visited.add(next_pos)
        current = next_pos
        attempts += 1
    if current != goal:
        print(f"Không đến được {goal} từ {start}")
        return []
    return path


def cost(path):
    """Tính chi phí của đường đi (độ dài)."""
    return len(path) if path else float('inf')


def neighbor(path, maze):
    """Tạo đường đi lân cận ngẫu nhiên."""
    if not path or len(path) < 3:
        return path
    new_path = copy.deepcopy(path)

    idx = random.randint(1, len(new_path)-2)
    valid_moves = get_valid_moves(maze, new_path[idx])

    if valid_moves:
        new_path[idx] = random.choice(valid_moves)
        for i in range(idx, len(new_path)-1):
            valid_next = get_valid_moves(maze, new_path[i])
            if new_path[i+1] not in valid_next:
                if valid_next:
                    new_path[i+1] = random.choice(valid_next)
                else:
                    return path
    return new_path


def acceptance_probability(old_cost, new_cost, temperature):
    """Xác suất chấp nhận giải pháp mới trong Simulated Annealing."""
    if new_cost < old_cost:
        return 1.0
    return math.exp((old_cost - new_cost) / temperature)


def simulated_annealing(maze, start, goal):
    """Thuật toán Simulated Annealing tìm đường."""
    sol = initial_solution(start, goal, maze)
    if not sol:
        print(f"Không tìm được đường từ {start} đến {goal}")
        return []

    old_cost = cost(sol)
    T = 1.0
    T_min = 0.00001
    alpha = 0.95
    max_iterations = 1000

    while T > T_min:
        i = 1
        while i <= max_iterations:
            new_sol = neighbor(sol, maze)
            new_cost = cost(new_sol)
            ap = acceptance_probability(old_cost, new_cost, T)
            if ap > random.random():
                sol = new_sol
                old_cost = new_cost
            i += 1
        T = T * alpha

    if not sol or sol[-1] != goal:
        print(f"Giải pháp không hợp lệ: {sol}")
        return []

    return sol[1:]


def find_path_sa(maze, start, goal, items):
    """Tìm đường bằng SA qua tất cả items rồi đến goal."""
    path = []
    current_pos = start

    for item in items:
        item_path = simulated_annealing(maze, current_pos, item)
        if not item_path:
            print(f"Không tìm được đường từ {current_pos} đến item {item}")
            return []
        path.extend(item_path)
        current_pos = item

    final_path = simulated_annealing(maze, current_pos, goal)
    if not final_path:
        print(f"Không tìm được đường từ {current_pos} đến goal {goal}")
        return []
    path.extend(final_path)

    return path


def backtracking_search(maze, start, goal, items, shields=None, traps=None):
    """
    Thuật toán Backtracking cải tiến tìm đường từ start đến goal, đi qua items, shields, và traps.

    Parameters:
    - maze: Mê cung (numpy array)
    - start: Vị trí bắt đầu (x, y)
    - goal: Vị trí đích (x, y)
    - items: Danh sách vật phẩm cần thu thập [(x1, y1), (x2, y2), ...]
    - shields: Danh sách vị trí khiên (tùy chọn) [(x1, y1), ...]
    - traps: Danh sách vị trí bẫy (tùy chọn) [(x1, y1), ...]

    Returns:
    - path: Đường đi từ start qua tất cả items, shields, traps, đến goal
    """
    shields = shields if shields is not None else []
    traps = traps if traps is not None else []

    # Tập hợp tất cả các mục tiêu cần đi qua
    all_targets = set(items).union(shields).union(traps).union([goal])
    print(f"All targets to visit: {all_targets}")

    def is_consistent(value, var_index, assignment):
        """Kiểm tra tính nhất quán của giá trị với ràng buộc."""
        if var_index == 0 and value != start:
            return False
        if var_index > 0:
            prev_pos = assignment[var_index - 1]
            if prev_pos is None or value not in get_valid_moves(maze, prev_pos):
                return False
        return True

    def select_unassigned_variable(assignment, domains, remaining_targets):
        """Chọn biến chưa gán dựa trên MRV và heuristic đến mục tiêu gần nhất."""
        unassigned = [i for i in range(
            len(assignment)) if assignment[i] is None]
        if not unassigned:
            return None
        # MRV kết hợp với ưu tiên mục tiêu gần nhất
        return min(unassigned, key=lambda i: (len(domains[i]) if domains[i] else float('inf'),
                                              min(heuristic(start if i == 0 else assignment[i-1], t) for t in remaining_targets) if remaining_targets else 0))

    def order_domain_values(var_index, assignment, domains, remaining_targets):
        """Sắp xếp miền giá trị dựa trên heuristic đến mục tiêu gần nhất."""
        if var_index == 0:
            return [start]
        prev_pos = assignment[var_index - 1]
        if prev_pos is None:
            return []
        valid_moves = get_valid_moves(maze, prev_pos)
        # Loại bỏ các ô đã gán, trừ khi là goal
        valid_moves = [
            pos for pos in valid_moves if pos not in assignment[:var_index] or pos == goal]
        if remaining_targets:
            # Ưu tiên ô gần mục tiêu còn lại
            return sorted(valid_moves, key=lambda pos: min(heuristic(pos, t) for t in remaining_targets))
        return sorted(valid_moves, key=lambda pos: heuristic(pos, goal))

    def backtrack(assignment, domains, remaining_targets, depth_limit=1000):
        """Thuật toán Backtracking với giới hạn độ sâu."""
        if depth_limit <= 0:
            return None
        var = select_unassigned_variable(
            assignment, domains, remaining_targets)
        if var is None:
            # Kiểm tra xem đã đi qua tất cả mục tiêu và đến goal
            if not remaining_targets and assignment[-1] == goal:
                return assignment
            return None

        for value in order_domain_values(var, assignment, domains, remaining_targets):
            if is_consistent(value, var, assignment):
                assignment[var] = value
                new_remaining = remaining_targets - \
                    {value} if value in remaining_targets else remaining_targets
                assignment_copy = assignment.copy()
                domains_copy = [d.copy() for d in domains]
                result = backtrack(assignment_copy, domains_copy,
                                   new_remaining, depth_limit - 1)
                if result is not None:
                    print(
                        f"Visited position: {value}, Remaining targets: {new_remaining}")
                    return result
                assignment[var] = None

        return None

    # Ước lượng độ dài đường đi tối đa
    max_steps = maze.shape[0] * maze.shape[1] * 2
    assignment = [None] * (max_steps + len(all_targets) + 1)
    # Khởi tạo domains
    assignment[0] = start  # Đặt vị trí bắt đầu
    domains = [[] for _ in range(max_steps + len(all_targets) + 1)]
    domains[0] = [start]
    for i in range(1, len(domains)):
        if assignment[i-1] is not None:
            domains[i] = get_valid_moves(maze, assignment[i-1])
        else:
            domains[i] = get_valid_moves(maze, start)

    result = backtrack(assignment, domains, all_targets)
    if result is None:
        print(f"Backtracking failed, falling back to A* or initial solution")
        # Thử A* làm giải pháp dự phòng
        path = find_path_astar(maze, start, goal, items)
        if not path:
            path = initial_solution(start, goal, maze)
        return path or []

    # Xây dựng path từ assignment
    path = [pos for pos in result if pos is not None]
    if not path or path[-1] != goal:
        print(f"Path không hợp lệ: {path}, falling back to A*")
        path = find_path_astar(maze, start, goal, items)
        if not path:
            path = initial_solution(start, goal, maze)
        return path or []

    print(f"Final Backtracking path: {path}")
    return path[1:] if path else []


def find_path_backtracking(maze, start, goal, items, shields=None, traps=None):
    return backtracking_search(maze, start, goal, items, shields, traps)


ACTIONS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}


def is_valid_move_maze(maze, pos):
    x, y = pos
    rows = len(maze)
    cols = len(maze[0])
    return 0 <= x < rows and 0 <= y < cols and maze[x][y] == 1  # 1 là đường


def apply_action_maze(pos, action):
    dx, dy = ACTIONS[action]
    return (pos[0] + dx, pos[1] + dy)


def apply_action_to_player_states(player_states, action, maze, goal):
    new_player_states = []
    path_entry = action  # Mặc định là hành động bình thường
    reached_goal = False
    reached_player_id = None

    for idx, (pos, has_reached) in enumerate(player_states):
        if has_reached:
            # Người chơi đã đến đích, giữ nguyên trạng thái
            new_player_states.append((pos, True))
        else:
            # Thử di chuyển người chơi
            new_pos = apply_action_maze(pos, action)
            if is_valid_move_maze(maze, new_pos):
                new_player_states.append((new_pos, new_pos == goal))
                if new_pos == goal:
                    reached_goal = True
                    reached_player_id = idx + 1  # ID bắt đầu từ 1
            else:
                # Không thể di chuyển, giữ nguyên
                new_player_states.append((pos, False))

    # Nếu có người chơi đến đích, cập nhật path_entry
    if reached_goal:
        path_entry = f"{action}"

    return tuple(new_player_states), path_entry


def all_players_at_goal(player_states, goal):
    return all(pos == goal for pos, _ in player_states)


def search_no_observation_bfs_maze(maze, players, goal):
    # Khởi tạo trạng thái cho từng người chơi: (vị trí, đã đến đích chưa)
    initial_player_states = tuple(
        ((p.grid_x, p.grid_y), (p.grid_x, p.grid_y) == goal) for p in players
    )
    queue = deque()
    visited = set()
    queue.append((initial_player_states, []))

    while queue:
        player_states, path = queue.popleft()

        if player_states in visited:
            continue
        visited.add(player_states)

        # Nếu tất cả người chơi đã đến đích, trả về path
        if all_players_at_goal(player_states, goal):
            return path

        for action in ACTIONS:
            new_player_states, path_entry = apply_action_to_player_states(
                player_states, action, maze, goal)
            if new_player_states not in visited:
                queue.append((new_player_states, path + [path_entry]))

    return []


def get_valid_actions(maze, state):
    """Return valid actions (moves) from the current state for Q-Learning."""
    row, col = state
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    actions = []
    for i, (dx, dy) in enumerate(directions):
        next_pos = (row + dx, col + dy)
        if (0 <= next_pos[0] < maze.shape[0] and
                0 <= next_pos[1] < maze.shape[1] and
                maze[next_pos[0], next_pos[1]] == 1):
            actions.append(i)
    return actions


def step(maze, state, action):
    """Execute an action from the current state, return next state and reward."""
    row, col = state
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    dx, dy = directions[action]
    next_state = (row + dx, col + dy)

    # Reward: -1 for each step to encourage shorter paths
    reward = -1

    # Check if the move is valid
    if (0 <= next_state[0] < maze.shape[0] and
            0 <= next_state[1] < maze.shape[1] and
            maze[next_state[0], next_state[1]] == 1):
        return next_state, reward
    else:
        # Invalid move: stay in the same state, negative reward
        return state, -10


def epsilon_greedy_policy(state, Q, maze, epsilon):
    """Choose an action using ε-greedy policy."""
    valid_actions = get_valid_actions(maze, state)
    if not valid_actions:
        return None
    if random.random() < epsilon:
        # Explore: choose a random valid action
        return random.choice(valid_actions)
    else:
        # Exploit: choose the action with the highest Q-value
        Q_values = [Q[state][a] if a in valid_actions else -
                    float('inf') for a in range(4)]
        return np.argmax(Q_values)


def q_learning(maze, start, goal, episodes=10000, alpha0=0.05, decay=0.005, gamma=0.9, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.999):
    """Q-Learning algorithm to find a path from start to goal."""
    # Initialize Q-table: (rows, cols, actions)
    Q = np.zeros((maze.shape[0], maze.shape[1], 4))

    for episode in range(episodes):
        state = start
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
        visited = set()  # Prevent infinite loops in single episode
        max_steps_per_episode = maze.shape[0] * maze.shape[1] * 2

        step_count = 0
        while state != goal and step_count < max_steps_per_episode:
            # Choose action using ε-greedy policy
            action = epsilon_greedy_policy(state, Q, maze, epsilon)
            if action is None:
                break  # No valid actions, terminate episode

            # Take action, observe next state and reward
            next_state, reward = step(maze, state, action)

            # Update Q-value
            next_Q = np.max(Q[next_state]) if get_valid_actions(
                maze, next_state) else 0
            alpha = alpha0 / (1 + episode * decay)
            Q[state][action] = (1 - alpha) * Q[state][action] + \
                alpha * (reward + gamma * next_Q)

            state = next_state
            visited.add(state)
            step_count += 1

        # Optional: Add a large reward when reaching the goal
        if state == goal:
            Q[state] = 0  # No further actions from goal

    # Extract the optimal path using the learned Q-values
    path = []
    state = start
    max_steps = maze.shape[0] * maze.shape[1] * 2  # Prevent infinite loops
    step_count = 0
    visited = set()

    while state != goal and step_count < max_steps:
        if state in visited:
            print(f"Đường đi bị kẹt trong vòng lặp tại {state}")
            return []
        visited.add(state)
        path.append(state)
        valid_actions = get_valid_actions(maze, state)
        if not valid_actions:
            print(f"Không tìm được đường từ {state} đến goal {goal}")
            return []
        # Choose the action with the highest Q-value
        Q_values = [Q[state][a] if a in valid_actions else -
                    float('inf') for a in range(4)]
        action = np.argmax(Q_values)
        next_state, _ = step(maze, state, action)
        if next_state == state:  # Stuck due to invalid move
            print(f"Đường đi bị kẹt tại {state}")
            return []
        state = next_state
        step_count += 1

    if state == goal:
        path.append(goal)
    else:
        print(f"Không đến được goal {goal} trong giới hạn bước")
        return []

    return path[1:] if path else []


def validate_inputs(maze, start, goal, items):
    """Validate maze, start, goal, and items."""
    if not isinstance(maze, np.ndarray) or maze.size == 0:
        raise ValueError("Maze must be a non-empty NumPy array.")
    if not (0 <= start[0] < maze.shape[0] and 0 <= start[1] < maze.shape[1] and maze[start] == 1):
        raise ValueError("Invalid start position.")
    if not (0 <= goal[0] < maze.shape[0] and 0 <= goal[1] < maze.shape[1] and maze[goal] == 1):
        raise ValueError("Invalid goal position.")
    for item in items:
        if not (0 <= item[0] < maze.shape[0] and 0 <= item[1] < maze.shape[1] and maze[item] == 1):
            raise ValueError(f"Invalid item position: {item}")


def find_path_qlearning(maze, start, goal, items):
    """Find a path from start to goal via items using Q-Learning."""
    validate_inputs(maze, start, goal, items)
    path = []
    current_pos = start

    for item in items:
        item_path = q_learning(maze, current_pos, item)
        if not item_path:
            print(f"Không tìm được đường từ {current_pos} đến item {item}")
            return []
        path.extend(item_path)
        current_pos = item

    final_path = q_learning(maze, current_pos, goal)
    if not final_path:
        print(f"Không tìm được đường từ {current_pos} đến goal {goal}")
        return []
    path.extend(final_path)

    return path
