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


def random_plan(initial_state: GameState) -> str:
    """
    Generate a random plan (valid string of moves) based on the game state
    """
    toad_pos = initial_state.grid[-1].index("T")
    health = initial_state.health
    plan = ""

    for i in range(len(initial_state.rolls)):
        # Get a list of valid moves based on each move's health cost and travel distance
        valid_moves = [
            letter for letter, travel in MOVE_DISTANCES.items()
            if 1 <= toad_pos + travel <= 5 and health - HEALTH_COSTS[letter] > 0
        ]
        # Randomly select a valid move
        letter = random.choice(valid_moves)
        travel_distance = MOVE_DISTANCES[letter]
        # Update the toad's calculated position/health, and update the plan
        toad_pos += travel_distance
        plan += letter
        health -= HEALTH_COSTS[letter]
    return plan


def bfs_find_best_plan(max_length, starting_health, starting_position=3):
    # States look like: (toad_position, health, move_sequence, turn)
    queue = deque([(starting_position, starting_health, "", 0)])
    visited = set()  # Visited states

    while queue:
        pos, health, moves, turn = queue.popleft()

        # If we reach the last turn successfully, return the move sequence
        if turn == max_length:
            return moves

        if (pos, health, turn) in visited:
            continue
        visited.add((pos, health, turn))

        # Try all valid moves
        for move, travel in MOVE_DISTANCES.items():
            new_pos = pos + travel
            new_health = health - HEALTH_COSTS[move]

            # If move is valid, add it
            if 1 <= new_pos <= 5 and new_health > 0:
                new_moves = moves + move
                queue.append((new_pos, new_health, new_moves, turn + 1))

    return ""  # No valid path found


if __name__ == '__main__':
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        # Default paths (none provided)
        print("No filenames provided, using \'input.txt\' and \'output.txt\'...")
        input_file = "input.txt"
        output_file = "output.txt"

    # Read the rolls from the input file
    with open(input_file, "r") as f:
        input_data = f.readlines()
    num_rounds = int(input_data[0].strip())
    rolls = [int(d.strip()) for d in input_data[1:]]

    # Create a starting state
    initial_state = GameState(
        grid=STARTING_GRID,
        health=20,
        rolls=rolls,
    )

    # Create a move plan
    moves = random_plan(initial_state)

    # Run the game and print the results
    final_state = transition(initial_state, moves)

    print("\nFinal HP:", final_state.health)
    print("Final State:\n" + str(final_state))

    # Output the properly formatted game results to the output file
    final_state.export(output_file, moves)
