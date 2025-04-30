from dataclasses import dataclass


@dataclass
class GameState:
    grid: list
    health: int
    rolls: list

    def __str__(self):
        return "\n".join(self.grid) + "\n"

    def export(self, filename: str, moves: str):
        with open(filename, "w") as f:
            f.write(moves + "\n")
            f.write(str(self.health) + "\n")
            for row in self.grid:
                f.write(row + "\n")

    def to_hashable(self):
        return tuple(self.grid), self.health, tuple(self.rolls)


def state_from_roll(roll: int) -> str:
    if roll == 32:
        return "F     #"
    elif roll == 33:
        return "#     F"
    elif 0 <= roll <= 31:
        return f"#{str(bin(roll))[2:].zfill(5).replace('0', ' ').replace('1', 'S')}#"
    else:
        return "INVALID"


def transition(state: GameState, moves: str, print_steps: bool = False) -> GameState:
    move_distances = {"A": -2, "S": -1, "D": 0, "F": 1, "G": 2}
    health_costs = {"A": 3, "S": 1, "D": 0, "F": 1, "G": 3}
    grid = state.grid.copy()
    health = state.health
    position = grid[-1].index("T")
    rolls = state.rolls

    for i, move in enumerate(moves):
        if print_steps:
            print("HP:", health)
            print("Move:", move)
            print("\n".join(grid), "\n")
        position += move_distances[move]
        health -= health_costs[move]
        if not (1 <= position <= 5):
            raise ValueError(f"Invalid toad position: {position}")
        if i < len(rolls):
            new_row = state_from_roll(rolls[i])
            grid.pop()
            grid.insert(0, new_row)
            if grid[-1][position] not in (" ", "T"):
                # Toad was eaten by a snake!
                grid[-1] = f"#{' ' * (position - 1)}X{' ' * (5 - position)}#"
                return GameState(grid=grid, health=health, rolls=rolls)
            if (grid[-1][-1] == "F" and position == 5) or (grid[-1][0] == "F" and position == 1):
                # Toad ate a fly!
                health += 5
        grid[-1] = f"#{' ' * (position - 1)}T{' ' * (5 - position)}#"
    return GameState(grid=grid, health=health, rolls=rolls)


if __name__ == '__main__':
    starting_position = 3
    empty_grid = [
        "#     #",
        "#     #",
        "#     #",
        "#     #",
        "#  T  #",
    ]

    with open("inputs/input1.txt", "r") as f:
        rolls = [int(r) for r in f.readlines()[1:]]

    initial_state = GameState(
        grid=empty_grid,
        health=20,
        rolls=rolls,
    )
    moves = "FFSSSSFSFF"
    final_state = transition(initial_state, moves)

    print("Input State:")
    print(initial_state)
    print("\nFinal State")
    print(final_state)
    print("HP:", final_state.health)

    final_state.export("output.txt", moves)
