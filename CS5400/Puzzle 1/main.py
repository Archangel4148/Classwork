class ToadGame:
    def __init__(self, rolls: list[int], moves: str, health: int, starting_position: int = 3):
        self.rolls = rolls
        self.moves = moves
        self.health = health

        # Set up initial grid
        self.grid = ["#     #"] * 5
        self.grid[-1] = self.grid[-1][:starting_position] + "T" + self.grid[4][(starting_position + 1):]

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
        self.grid[-1] = f"#{' ' * (toad_pos - 1)}{toad_symbol}{' ' * (5 - toad_pos)}#"

        return death

    def play_game(self, max_length: int):
        round = 0
        while self.health > 0 and round < max_length:
            # Phase 1 - Toad moves
            print("MOVE:", self.moves[round])
            self.move_toad(self.moves[round])
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

    def output_to_file(self, file_name: str):
        with open(file_name, "w") as f:
            f.write(self.moves + "\n")
            f.write(str(self.health) + "\n")
            for row in self.grid:
                f.write(row + "\n")


if __name__ == '__main__':
    # Read the game state information from the input file
    with open("input.txt", "r") as f:
        input_data = f.readlines()
    num_rounds = int(input_data[0].strip())
    rolls = [int(d.strip()) for d in input_data[1:]]

    # Give the move plan
    moves = "FFSSSSFSFF"

    # Create a toad game object
    game = ToadGame(rolls, moves, 20)

    # Run the game and print the results
    game.play_game(num_rounds)
    print("\nFinal HP:", game.health)
    print("Final State:")
    game.print_grid()

    # Output the properly formatted game results to the output file
    game.output_to_file("output.txt")
