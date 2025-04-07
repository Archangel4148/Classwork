########################################
# Submission by Simon Edmunds
# Student ID: 12590227
########################################

import random
import sys
from collections import deque

from toad_game import GameState, transition

MOVE_DISTANCES = {"A": -2, "S": -1, "D": 0, "F": 1, "G": 2}
HEALTH_COSTS = {"A": 3, "S": 1, "D": 0, "F": 1, "G": 3}

STARTING_GRID = [
    "#     #",
    "#     #",
    "#     #",
    "#     #",
    "#  T  #",
]


def get_valid_moves(current_state: GameState):
    try:
        toad_pos = current_state.grid[-1].index("T")
    except ValueError:
        return []
    valid_moves = [
        letter for letter, travel in MOVE_DISTANCES.items()
        if 1 <= toad_pos + travel <= 5 and current_state.health - HEALTH_COSTS[letter] > 0
    ]
    return valid_moves


def goal(state: GameState) -> bool:
    return "T" in state.grid[-1] and state.health > 0


def random_plan(initial_state: GameState) -> str:
    """
    Generate a random plan (valid string of moves) based on the game state
    """
    toad_pos = initial_state.grid[-1].index("T")
    health = initial_state.health
    plan = ""

    for i in range(len(initial_state.rolls)):
        valid_moves = get_valid_moves(initial_state)
        # Randomly select a valid move
        letter = random.choice(valid_moves)
        travel_distance = MOVE_DISTANCES[letter]
        # Update the toad's calculated position/health, and update the plan
        toad_pos += travel_distance
        plan += letter
        health -= HEALTH_COSTS[letter]
    return plan


def bfs_find_best_plan(initial_state: GameState, goal_func, game_length: int):
    frontier = deque([""])
    already_visited = set()

    while frontier:
        p = frontier.popleft()
        sk = transition(initial_state, p)
        sk_hashable = sk.to_hashable()
        if len(p) >= game_length and goal_func(sk):
            return p

        # Don't repeat for already visited stuff
        if sk_hashable in already_visited:
            continue
        already_visited.add(sk_hashable)

        # Add "neighbors" to the frontier
        for valid_move in get_valid_moves(sk):
            frontier.append(p + valid_move)

    return None


def bounded_dfs(initial_state: GameState, goal_func, game_length, depth_limit: int):
    frontier = [""]
    limit_hit = False
    visited_states = set()

    while frontier:
        p = frontier.pop()
        sk = transition(initial_state, p)
        state_hash = sk.to_hashable()

        # Don't revisit states (speeds stuff up quite a bit)
        if state_hash in visited_states:
            continue
        visited_states.add(state_hash)

        if len(p) == depth_limit:
            # If the path is the correct length, and the toad is alive, it's a winner!
            if len(p) >= game_length and goal_func(sk):
                return p
            if get_valid_moves(sk):
                limit_hit = True
        else:
            for valid_move in get_valid_moves(sk):
                frontier.append(p + valid_move)
    return limit_hit


def iterative_deepening_find_best_plan(initial_state: GameState, goal_func, game_length: int):
    depth = 0
    while True:
        result = bounded_dfs(initial_state, goal_func, game_length, depth)
        if isinstance(result, str):
            # Found a path!
            return result
        if result is False:
            # No valid paths
            return False
        depth += 1


def run_game_from_file(input_path: str, output_path: str, max_length: int, plan_func: callable,
                       show_steps: bool = False):
    # Read the rolls from the input file
    with open(input_path, "r") as f:
        input_data = [line for line in f.readlines() if line.strip()]
    num_rounds = int(input_data[0].strip())
    rolls = [int(d.strip()) for d in input_data[1:]]

    # Create a starting state
    initial_state = GameState(
        grid=STARTING_GRID,
        health=20,
        rolls=rolls,
    )

    # The game will end either after 15 rounds, or when we run out of rolls
    game_length = min(max_length, num_rounds)

    # Create a move plan using Iterative Deepening DFS
    moves = plan_func(initial_state, goal, game_length)

    # Run the game and print the results
    final_state = transition(initial_state, moves, print_steps=show_steps)

    print("\nMoves:", moves)
    print("Final HP:", final_state.health)
    print("Final State:\n" + str(final_state))

    # Output the properly formatted game results to the output file
    final_state.export(output_path, moves)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        # Default paths (none provided)
        input_file = "inputs/input1.txt"
        output_file = "outputs/output.txt"
        print(f"No filenames provided, using \'{input_file}\' and \'{output_file}\'...")

    test_cases = [

    ]

    # for i in range(1, 7):
    #     input_file = f"inputs/input{i}.txt"
    #     output_file = f"outputs/output{i}.txt"
    #
    #     print(f"\n=== Running {input_file} ===")
    #     try:
    #         run_game_from_file(
    #             input_path=input_file,
    #             output_path=output_file,
    #             max_length=50,
    #             plan_func=iterative_deepening_find_best_plan,
    #             show_steps=False
    #         )
    #     except Exception as e:
    #         print(f"Error running {input_file}: {e}")

    # Run the game
    filenum = 6
    max_length = 25
    input_file = f"inputs/input{filenum}.txt"
    output_file = f"outputs/output{filenum}.txt"
    run_game_from_file(input_file, output_file, max_length, iterative_deepening_find_best_plan, show_steps=True)
