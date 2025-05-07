import torch
import chess
import random


class MyChessEngine:
    def __init__(self, model_path="saved/model.pt"):
        self.model = torch.load(model_path)
        self.model.eval()

    def play(self, board, time_limit=0.001):
        board_fen = board.fen()
        board_tensor = self.fen_to_tensor(board_fen)

        with torch.no_grad():
            prediction = self.model(board_tensor)

        best_move = self.tensor_to_move(prediction, board)

        return best_move

    def fen_to_tensor(self, fen):
        tensor_rep = torch.zeros(773)
        for i, char in enumerate(fen.split(' ')[0].replace('/', '').lower()):
            tensor_rep[i] = self.char_to_tensor_value(char)

        return tensor_rep

    def tensor_to_move(self, tensor, board):
        best_move_idx = torch.argmax(tensor)
        possible_moves = list(board.legal_moves)

        best_move = possible_moves[best_move_idx.item()]

        return best_move

    def char_to_tensor_value(self, char):
        if char == 'p':
            return -1
        elif char == 'P':
            return 1
        elif char == 'r':
            return -5
        elif char == 'R':
            return 5
        elif char == 'n':
            return -3
        elif char == 'N':
            return 3
        elif char == 'b':
            return -3
        elif char == 'B':
            return 3
        elif char == 'q':
            return -9
        elif char == 'Q':
            return 9
        elif char == 'k':
            return -1000
        elif char == 'K':
            return 1000
        else:
            return 0

    def make_move(self, board, move):
        if move in board.legal_moves:
            board.push(move)
            return True
        else:
            print(f"Nielegalny ruch: {move}")
            return False
