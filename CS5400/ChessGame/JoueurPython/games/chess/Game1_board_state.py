from games.chess.game import Game


class BoardState:
    def __init__(self, game: Game):
        self.fen = game.fen
        print("BoardState initialized")
