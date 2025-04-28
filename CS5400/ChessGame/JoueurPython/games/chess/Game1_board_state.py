from games.chess.game import Game


class BoardState:
    def __init__(self, game: Game):
        # Get the starting FEN
        self.fen = game.fen

        # Storage variables for all the FEN info
        self.board = [None] * 64
        self.white_to_move = None
        self.castling_rights = {'K': False, 'Q': False, 'k': False, 'q': False}
        self.en_passant_target = None
        self.halfmove_clock = 0
        self.fullmove_number = 1

        # Update values from starting FEN
        self.parse_fen(self.fen)

        print("BoardState initialized")

    def parse_fen(self, fen: str):
        sections = fen.split(" ")

        # Clear the board
        self.board = [None] * 64

        # Parse the board
        for i, char in enumerate(sections[0]):
            if char == "/":
                continue
            elif char.isnumeric():
                # Skip the empty spaces
                i += int(char)
            else:
                self.board[i] = char
