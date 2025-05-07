import chess
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import time


def load_dataset_from_csv(csv_path, elo_threshold=1400, min_time=300):
    df = pd.read_csv(csv_path)

    df['white_elo'] = pd.to_numeric(df['white_elo'], errors='coerce')
    df['black_elo'] = pd.to_numeric(df['black_elo'], errors='coerce')
    df = df.dropna(subset=['white_elo', 'black_elo'])
    df = df[(df['white_elo'] >= elo_threshold) | (df['black_elo'] >= elo_threshold)]

    df['base_time'] = df['time_control'].apply(lambda x: int(str(x).split('+')[0]) if '+' in str(x) else 0)
    df = df[df['base_time'] >= min_time]
    df = df[['fen', 'score']].dropna()
    print(f"Pozostało pozycji po filtrowaniu: {len(df)}")
    X = []
    y = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            board = chess.Board(row['fen'])
            input_tensor = board_to_input(board)
            score = float(row['score'])
            X.append(input_tensor.numpy())
            y.append(score)
        except Exception as e:
            continue

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (n, 1)

    return X_tensor, y_tensor


def train_model(model, X, y, epochs=5, batch_size=64, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(X.size(0))
        X = X[perm]
        y = y[perm]

        for i in range(0, X.size(0), batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


class FakeNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 128)  # wejście z planszy
        self.fc2 = nn.Linear(128, 32)   # ukryta warstwa
        self.fc3 = nn.Linear(32, 1)     # wyjście, czyli ocena (plus/minus wartość)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def board_to_input(board):
    input_vector = torch.zeros(768)  # 64 pola * 6 figur (384 dla białych, 384 dla czarnych)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == chess.WHITE else 384
            piece_offset = PIECE_TO_INDEX[piece.piece_type] * 64
            index = color_offset + piece_offset + square
            input_vector[index] = 1.0
    return input_vector


def evaluate_position(board, model):
    input_vec = board_to_input(board)
    score = model(input_vec.unsqueeze(0)).item()
    return score


def minimax(board, model, depth, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_position(board, model)

    if maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, model, depth - 1, False)
            max_eval = max(max_eval, eval)
            board.pop()
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, model, depth - 1, True)
            min_eval = min(min_eval, eval)
            board.pop()
        return min_eval


def pick_best_move(board, model, depth=2):
    best_move = None
    best_value = -float('inf') if board.turn == chess.WHITE else float('inf')

    for move in board.legal_moves:
        board.push(move)
        board_value = minimax(board, model, depth, board.turn == chess.WHITE)
        board.pop()

        if (board.turn == chess.WHITE and board_value > best_value) or (board.turn == chess.BLACK and board_value < best_value):
            best_value = board_value
            best_move = move

    return best_move


def pick_best_move_with_time(board, model, time_limit=1.0):
    best_move = None
    start_time = time.time()
    depth = 1

    while True:
        move_this_depth = None
        best_value = -float('inf') if board.turn == chess.WHITE else float('inf')

        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, model, depth - 1, board.turn == chess.WHITE)
            board.pop()

            if board.turn == chess.WHITE:
                if eval > best_value:
                    best_value = eval
                    move_this_depth = move
            else:
                if eval < best_value:
                    best_value = eval
                    move_this_depth = move

            if time.time() - start_time > time_limit:
                break

        if time.time() - start_time > time_limit:
            break

        if move_this_depth is not None:
            best_move = move_this_depth

        depth += 1

    return best_move if best_move is not None else random.choice(list(board.legal_moves))


def train_and_save_model(filename, epoch=10, out_file='saved/fakennue_trained.pt'):
    model = FakeNNUE()
    X, y = load_dataset_from_csv(filename)
    train_model(model, X, y, epochs=epoch)
    torch.save(model.state_dict(), out_file)


if __name__ == "__main__":
    train_and_save_model("positions.csv")
