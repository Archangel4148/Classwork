import random
import sys


class ToadGame:
    def __init__(self, rolls, health: int, starting_position: int = 3):
        self.rolls = rolls
        self.health = health

        # Set up initial grid
        self.grid = ["#     #"] * 5
        self.grid[-1] = self.grid[-1][:starting_position] + "T" + self.grid[-1][(starting_position + 1):]

        # Parse states into grid rows
        self.states = [self.state_from_roll(roll) for roll in rolls]

    def print_grid(self):
        for grid in self.grid:
            print(grid)
        print("\n")

    @staticmethod
    def state_from_roll(roll: int) -> str:
        if roll == 32:
            return "F     #"
        elif roll == 33:
            return "#     F"
        elif 0 <= roll <= 31:
            return f"#{str(bin(roll))[2:].zfill(5).replace('0', ' ').replace('1', 'S')}#"
        else:
            return "INVALID"

    def move_toad(self, input: str):
        toad_pos = self.grid[-1].index("T")
        if input == "A":
            toad_pos -= 2
            self.health -= 3
        elif input == "S":
            toad_pos -= 1
            self.health -= 1
        elif input == "F":
            toad_pos += 1
            self.health -= 1
        elif input == "G":
            toad_pos += 2
            self.health -= 3
        if not 1 <= toad_pos <= 5:
            raise ValueError(f"Invalid toad position: {toad_pos}")

        self.grid[-1] = f"#{' ' * (toad_pos - 1)}T{' ' * (5 - toad_pos)}#"

    def shift_grid(self, new_row: str) -> bool:
        toad_pos = self.grid[-1].index("T")
        toad_symbol = "T"
        death = False
        self.grid.pop(-1)
        self.grid.insert(0, new_row)
        if self.grid[-1][toad_pos] not in (" ", "T"):
            death = True
            toad_symbol = "X"
        # Detect if a fly moved next to the toad
        if not death and (self.grid[-1][-1] == "F" and toad_pos == 5 or self.grid[-1][0] == "F" and toad_pos == 1):
            self.health += 5
        self.grid[-1] = f"#{' ' * (toad_pos - 1)}{toad_symbol}{' ' * (5 - toad_pos)}#"

        return death

    def play_game(self, max_length: int, moves: str, result_filename: str = "output.txt"):
        round = 0
        while self.health > 0 and round < max_length:
            # Phase 1 - Toad moves
            print("MOVE:", moves[round])
            self.move_toad(moves[round])
            # Phase 2/3/4 - Spawn new row (also moves snakes/flies)
            snake_death = self.shift_grid(self.states[round])
            if snake_death:
                print("A snake ate the toad on turn", round + 1)
                return
            # Phase 5 - Check HP
            if self.health <= 0:
                print("You ran out of HP...")
                return
            # Print the grid
            self.print_grid()
            round += 1
        print("You made it all", round, "rounds!")

        self.output_to_file(result_filename, moves)

    def output_to_file(self, file_name: str, moves: str):
        with open(file_name, "w") as f:
            f.write(moves + "\n")
            f.write(str(self.health) + "\n")
            for row in self.grid:
                f.write(row + "\n")


def generate_plan(states, plan_length: int, starting_health: int, starting_position: int = 3) -> str:
    """
    Generate a plan (string of moves) based on the game state
    (For now, this just generates a random series of moves that don't kill the toad or take it out of bounds)
    """
    toad_pos = starting_position
    health = starting_health

    # Cost/distance info for accurate "planning"
    move_distances = {"A": -2, "S": -1, "D": 0, "F": 1, "G": 2}
    health_costs = {"A": 3, "S": 1, "D": 0, "F": 1, "G": 3}
    plan = ""

    for i in range(plan_length):
        # Get a list of valid moves based on each move's health cost and travel distance
        valid_moves = [
            letter for letter, travel in move_distances.items()
            if 1 <= toad_pos + travel <= 5 and health - health_costs[letter] > 0
        ]
        # Randomly select a valid move
        letter = random.choice(valid_moves)
        travel_distance = move_distances[letter]
        # Update the toad's calculated position/health, and update the plan
        toad_pos += travel_distance
        plan += letter
        health -= health_costs[letter]
    return plan


if __name__ == '__main__':
    print("SYS ARGS:", sys.argv)
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        # Default paths (none provided)
        print("No filenames provided, using \'input.txt\' and \'output.txt\'...")
        input_file = "input.txt"
        output_file = "output.txt"

    # Read the game state stuff from the input file
    with open("input.txt", "r") as f:
        input_data = f.readlines()
    num_rounds = int(input_data[0].strip())
    rolls = [int(d.strip()) for d in input_data[1:]]

    # Create a move plan
    states = [ToadGame.state_from_roll(roll) for roll in rolls]
    moves = generate_plan(states, num_rounds, 20)

    # Create a toad game object
    game = ToadGame(rolls, 20)

    # Run the game and print the results
    game.play_game(num_rounds, moves)
    print("\nFinal HP:", game.health)
    print("Final State:")
    game.print_grid()

    # Output the properly formatted game results to the output file
    game.output_to_file("output.txt", moves)
