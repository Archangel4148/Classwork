#######################################
#  Name: Simon Edmunds
#  Student ID: 12590227
#  Date: 5/5/2025
#  Assignment: CS5400 Game 3
#######################################
import math
import random

from games.chess.game import Game

# Move directions for each "normal" piece
MOVING_RULES = {
    "r": ([(-1, 0), (1, 0), (0, -1), (0, 1)], True),
    "n": ([(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)], False),
    "b": ([(-1, -1), (-1, 1), (1, -1), (1, 1)], True),
    "q": ([(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)], True),
    "k": ([(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)], False)
}

PIECE_VALUES = {
    "p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0
}


class BoardState:
    def __init__(self, game: Game):
        # Get the starting FEN
        self.fen = game.fen

        # Storage variables for all the FEN stuff
        self.board = [None] * 64
        self.white_to_move = None
        self.castling_rights = {'K': False, 'Q': False, 'k': False, 'q': False}
        self.en_passant_target = None
        self.en_passant_index = None
        self.halfmove_clock = 0
        self.fullmove_number = 1

        # Update values from FEN
        self.parse_fen(self.fen)

        # Tracking for heuristic checks
        self.is_white_player = self.white_to_move

    def parse_fen(self, fen: str):
        # Split the FEN into sections
        sections = fen.split(" ")

        # Parse the board
        self.board = []
        for row in sections[0].split("/"):
            for char in row:
                if char.isdigit():
                    self.board.extend([None] * int(char))
                else:
                    self.board.append(char)

        # Get the turn value
        self.white_to_move = sections[1] == "w"

        # Get the castling rights
        for char in sections[2]:
            if char == "K":
                self.castling_rights["K"] = True
            elif char == "Q":
                self.castling_rights["Q"] = True
            elif char == "k":
                self.castling_rights["k"] = True
            elif char == "q":
                self.castling_rights["q"] = True

        # Get the en passant target
        if sections[3] != "-":
            self.en_passant_target = sections[3]
            self.en_passant_index = square_to_index(self.en_passant_target)

        # Get the halfmove clock
        self.halfmove_clock = int(sections[4])

        # Get the fullmove number
        self.fullmove_number = int(sections[5])

    def get_piece_valid_moves(self, position: int):
        # Get all the valid moves for a piece
        piece = self.board[position]
        moves = []
        if piece is None:
            return []
        lower = piece.lower()
        if lower in MOVING_RULES:
            # Normal moving pieces
            moves = get_simple_piece_moves(self, position)
            # Kings can also castle
            if lower == "k":
                moves += get_castling_moves(self, piece.isupper())
        elif lower == "p":
            # Pawns
            moves = get_pawn_valid_moves(self, position)

        # Return move data
        return [(position, move[0], move[1]) for move in moves]

    def get_all_valid_moves(self):
        # Get all valid moves for the current player
        moves = []
        for i in range(64):
            # Only get moves for the current player
            if self.board[i] is not None and self.board[i].isupper() == self.white_to_move:
                moves.extend(self.get_piece_valid_moves(i))
        return moves

    def get_best_move(self):
        # Get the best move using minimax
        return depth_limited_alpha_beta_id_minimax(self)

    def copy(self):
        # Make a deep copy of the board
        new_board = BoardState.__new__(BoardState)
        new_board.board = self.board.copy()
        new_board.white_to_move = self.white_to_move
        new_board.castling_rights = self.castling_rights.copy()
        new_board.en_passant_target = self.en_passant_target
        new_board.en_passant_index = self.en_passant_index
        new_board.halfmove_clock = self.halfmove_clock
        new_board.fullmove_number = self.fullmove_number
        new_board.is_white_player = self.is_white_player
        return new_board

    def get_eval(self):
        # Get the evaluation of the current state
        return evaluate_heuristic(self)


# ========== PIECE MOVEMENT ==========

def get_simple_piece_moves(board_state: BoardState, position: int):
    board = board_state.board
    piece = board[position]
    moving_directions, can_slide = MOVING_RULES[piece.lower()]
    valid_moves = []

    # Get all the valid moves in each direction
    for dx, dy in moving_directions:
        cur_row, cur_col = position // 8, position % 8
        step = 1
        while True:
            # Keep going in this direction until we have to stop
            new_row, new_col = cur_row + step * dx, cur_col + step * dy
            if not (0 <= new_row < 8 and 0 <= new_col < 8):
                break  # off the board

            new_position = new_row * 8 + new_col
            target_piece = board[new_position]

            if target_piece is None:
                # Empty square, can keep going
                valid_moves.append((new_position, None))
            elif is_enemy(piece, target_piece):
                # Can take enemy piece, but can't keep going after
                valid_moves.append((new_position, None))
                break
            else:
                # Can't take our own piece
                break

            if not can_slide:
                break  # Can't slide, so just do one move
            step += 1

    return valid_moves


def get_pawn_valid_moves(board_state: BoardState, position: int):
    # Get the piece and its initial state
    board = board_state.board
    piece = board[position]
    is_white = piece.isupper()
    direction = -1 if is_white else 1
    start_row = 6 if is_white else 1
    curr_row, curr_col = position // 8, position % 8
    valid_moves = []

    # Forward moves
    one_forward = position + direction * 8
    new_row = one_forward // 8
    if is_on_board(one_forward) and board[one_forward] is None:
        # Clear square ahead
        if new_row == 0 or new_row == 7:
            # The pawn is moving to a back rank (promotion)
            for promotion in "qrbn":
                # Add all possible promotions
                valid_moves.append((one_forward, promotion))
        else:
            # Not a back rank (normal step forward)
            valid_moves.append((one_forward, None))

        # Double forward move (start row only)
        two_forward = one_forward + direction * 8
        if curr_row == start_row and board[two_forward] is None:
            # Clear square two ahead
            valid_moves.append((two_forward, None))

    # Diagonal moves (captures)
    for dx in [-1, 1]:
        # Try left and right
        new_col = curr_col + dx
        if 0 <= new_col < 8:
            diagonal = one_forward + dx
            new_row = diagonal // 8
            if is_on_board(diagonal):
                # Check for a target piece
                target_piece = board[diagonal]
                # Can only take enemies
                if target_piece and is_enemy(piece, target_piece):
                    # Check for promotions
                    if new_row == 0 or new_row == 7:
                        # The pawn is moving to a back rank (promotion)
                        for promotion in "qrbn":
                            # Add all possible promotions
                            valid_moves.append((diagonal, promotion))
                    else:
                        # Not a back rank (normal step forward)
                        valid_moves.append((diagonal, None))

    # En passant
    ep_position = board_state.en_passant_index
    if ep_position:
        ep_col = ep_position % 8
        # Can only take en passant if it's diagonally one space away (forward)
        if is_same_rank(one_forward, ep_position) and abs(ep_col - curr_col) == 1:
            valid_moves.append((ep_position, None))

    return valid_moves


def get_castling_moves(board_state: BoardState, is_white: bool):
    rights = board_state.castling_rights
    board = board_state.board
    row = 7 if is_white else 0
    valid_moves = []

    # Kingside
    if (rights["K"] if is_white else rights["k"]):
        f = row * 8 + 5
        g = row * 8 + 6
        # Middle squares are empty
        if board[f] is None and board[g] is None:
            valid_moves.append((g, None))  # King can move to g

    # Queenside
    if (rights["Q"] if is_white else rights["q"]):
        b = row * 8 + 1
        c = row * 8 + 2
        d = row * 8 + 3
        # Middle squares are empty
        if board[b] is None and board[c] is None and board[d] is None:
            valid_moves.append((c, None))  # King can move to c

    return valid_moves


# ========== MINIMAX/EVALUATION ==========

def is_terminal(state: BoardState) -> bool:
    # Since we only have pseudo-legal moves, just check for a stalemate or 50-move rule
    return len(state.get_all_valid_moves()) == 0 or state.halfmove_clock >= 100


def apply_move(state: BoardState, move):
    # Unpack the move
    start_idx, end_idx, promotion = move
    board = state.board
    moving_piece = board[start_idx]
    lower = moving_piece.lower()
    is_white = moving_piece.isupper()
    start_row, start_col = start_idx // 8, start_idx % 8
    end_row, end_col = end_idx // 8, end_idx % 8

    # Handle promotion
    if promotion:
        board[end_idx] = promotion.upper() if is_white else promotion.lower()
    else:
        board[end_idx] = moving_piece

    board[start_idx] = None

    if lower == "p":
        # Handle en passant
        if end_idx == state.en_passant_index:
            # Clear the pawn that was captured
            capture_idx = end_idx + (8 if is_white else -8)
            board[capture_idx] = None

        # Reset en passant then check for double pawn move
        state.en_passant_target = None
        state.en_passant_index = None
        if abs(end_row - start_row) == 2:
            # Double pawn move
            ep_row = (start_row + end_row) // 2
            ep_idx = ep_row * 8 + start_col
            state.en_passant_index = ep_idx
            state.en_passant_target = index_to_square(ep_idx)

    # Handle castling
    if lower == "k":
        if start_col == 4:
            # Kingside
            if end_col == 6:
                # Move the rook left two
                rook_from = start_idx + 3
                rook_to = start_idx + 1
                board[rook_to] = board[rook_from]
                board[rook_from] = None
            # Queenside
            elif end_col == 2:
                # Move the rook right three
                rook_from = start_idx - 4
                rook_to = start_idx - 1
                board[rook_to] = board[rook_from]
                board[rook_from] = None

    # Update castling rights
    # Check if a king moved
    if lower == "k":
        if is_white:
            state.castling_rights['K'] = False
            state.castling_rights['Q'] = False
        else:
            state.castling_rights['k'] = False
            state.castling_rights['q'] = False
    # Check if a rook moved
    elif moving_piece == 'R':
        if start_idx == 63:  # h1
            state.castling_rights['K'] = False
        elif start_idx == 56:  # a1
            state.castling_rights['Q'] = False
    elif moving_piece == 'r':
        if start_idx == 7:  # h8
            state.castling_rights['k'] = False
        elif start_idx == 0:  # a8
            state.castling_rights['q'] = False

    # Update turn
    state.white_to_move = not state.white_to_move

    # Increment fullmoves after black move
    if not state.white_to_move:
        state.fullmove_number += 1

    # Update halfmoves
    if lower == "p" or board[end_idx] is not None:
        state.halfmove_clock = 0
    else:
        state.halfmove_clock += 1


def evaluate_heuristic(state: BoardState) -> float:
    playing_white = state.is_white_player
    evaluation = 0

    # Helper function
    def is_mine(piece: str | None) -> bool:
        return (piece.isupper() and playing_white) or (piece.islower() and not playing_white)

    # Material (difference between white and black)
    for square in state.board:
        if square is not None:
            if is_mine(square):
                evaluation += PIECE_VALUES[square.lower()]
            else:
                evaluation -= PIECE_VALUES[square.lower()]

    # Center control
    center_squares = [27, 35, 36, 28]  # (d4, d5, e4, e5)
    for square in center_squares:
        piece = state.board[square]
        if piece is not None:
            if is_mine(piece):
                evaluation += 0.3
            else:
                evaluation -= 0.3

    for i, piece in enumerate(state.board):
        if piece is None or not is_mine(piece):
            continue
        row = i // 8

        # Reward development
        if piece.lower() in ["n", "b"]:
            # Reward knights and bishops for leaving the back rank
            if (playing_white and row <= 5) or (not playing_white and row >= 2):
                evaluation += 0.2

    # 6. Tempo bonus (slight bonus for turn player to make the bot more aggressive)
    evaluation += 0.05

    return round(evaluation, 6)  # Fix floating point error


def get_next_state(state: BoardState, move: str) -> BoardState:
    next_state = state.copy()
    apply_move(next_state, move)
    return next_state


def is_test_end(state: BoardState, depth: int) -> bool:
    return is_terminal(state) or depth == 0


def dynamic_depth(state: BoardState) -> int:
    num_moves = len(state.get_all_valid_moves())
    if num_moves <= 5:
        return 4
    else:
        return 3


def max_value(state: BoardState, alpha: float, beta: float, depth: int) -> float:
    if is_test_end(state, depth):
        return evaluate_heuristic(state)
    v = -math.inf
    for move in state.get_all_valid_moves():
        v = max(v, min_value(get_next_state(state, move), alpha, beta, depth - 1))
        alpha = max(alpha, v)
        if v >= beta:
            break
    return v


def min_value(state: BoardState, alpha: float, beta: float, depth: int) -> float:
    if is_test_end(state, depth):
        return evaluate_heuristic(state)
    v = math.inf
    for move in state.get_all_valid_moves():
        v = min(v, max_value(get_next_state(state, move), alpha, beta, depth - 1))
        beta = min(beta, v)
        if v <= alpha:
            break
    return v


def depth_limited_alpha_beta_id_minimax(state: BoardState) -> str:
    depth = 1
    target_depth = dynamic_depth(state)
    best_move = None
    # print("Starting minimax...")
    # Iterative deepening
    while depth <= target_depth:
        # print("Search depth:", depth)
        best_score = -math.inf
        best_moves = []
        # Find the move with the best score
        for move in state.get_all_valid_moves():
            score = min_value(get_next_state(state, move), -math.inf, math.inf, depth - 1)
            # print("Move:", move, "Score:", score)
            if score > best_score:
                best_score = score
                best_moves = [move]
                # print("New best move at depth", depth, "is", move, "with score", score)
            elif score == best_score:
                best_moves.append(move)
        # print("Best moves at depth", depth, "are", best_moves)
        best_move = random.choice(best_moves)
        depth += 1
    return best_move


# ========== HELPERS ==========

def is_on_board(position: int):
    return 0 <= position < 64


def is_enemy(piece_1: str, piece_2: str):
    if piece_1 is None or piece_2 is None:
        return False
    return piece_1.islower() != piece_2.islower()


def is_same_rank(position_1: int, position_2: int):
    return position_1 // 8 == position_2 // 8


def square_to_index(square: str):
    col = ord(square[0]) - ord('a')
    row = int(square[1])
    return (8 - row) * 8 + col


def index_to_square(index: int):
    row = 8 - (index // 8)
    col = index % 8
    return chr(ord('a') + col) + str(row)


def move_to_uci(move):
    # Convert start and end indices to UCI (including promotion)
    start, end, promotion = move
    uci = index_to_square(start) + index_to_square(end)
    if promotion is not None:
        uci += promotion
    return uci


if __name__ == '__main__':
    # Testing code
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    test_board = BoardState(test_fen)
    moves = test_board.get_all_valid_moves()
    print(len(moves), "moves -", moves)
