import subprocess
import time
from datetime import datetime
import chess
import chess.engine
import random
import chess.pgn
import chess.polyglot
from nnue import FakeNNUE, pick_best_move_with_time
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--games', type=int, required=True, help='Liczba gier do rozegrania')
parser.add_argument('--time-control', type=float, required=True, help='Czas na ruch')
parser.add_argument('--job-id', type=int, required=True, help='ID zadania')
parser.add_argument('--offset', type=int, default=0, help='Indeks początkowy partii')
args = parser.parse_args()



def choose_opening_move(board):
    with chess.polyglot.open_reader(book_path) as reader:
        moves = list(reader.find_all(board))
        if moves:
            return random.choice(moves).move
    return None


def nnue_move(board):
    return pick_best_move_with_time(board, nnue_model, time_limit=args.time_control)


start = time.time()
stockfish_path = "engines/stockfish.exe"
lczero_path = "engines/lc0/lc0.exe"
# stockfish_path = 'C:\\Users\\szatk\\AppData\\Roaming\\org.encroissant.app\\engines\\stockfish\\stockfish-windows-x86-64-avx2.exe'
# lczero_path = 'D:\\Pobrane\\lc0\\lc0.exe'
# stockfish_path = 'Y:\\chessengines\\stockfish\\stockfish-windows-x86-64-avx2.exe'
# lczero_path = 'Y:\\chessengines\\lc0\\lc0.exe'
book_path = "./Komodo.bin"
stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path, stderr=subprocess.DEVNULL)
lczero = chess.engine.SimpleEngine.popen_uci(lczero_path, stderr=subprocess.DEVNULL)
nnue_model = FakeNNUE()
nnue_model.load_state_dict(torch.load("./saved/fakennue_trained.pt"))
nnue_model.eval()

end = time.time()
print(f'Creating engines: {end - start}s')
results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

# num_games = 10
# time_for_move = 10
depth = 20
moves_sum = 0
start = time.time()
for i in range(args.offset, args.offset + args.games):
    for white, black in [("Stockfish", "Lc0"), ("Lc0", "Stockfish"), ("Stockfish", "NNUE"), ("NNUE", "Stockfish"), ("Lc0", "NNUE"), ("NNUE", "Lc0")]:
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["White"] = white
        game.headers["Black"] = black
        game.headers["Event"] = "Praca dyplomowa"
        game.headers["Round"] = str(i)
        game.headers["Date"] = str(datetime.now())
        game.headers["Site"] = "placeholder"
        game.headers["Time"] = str(args.time_control)
        node = game
        moves = 0
        while not board.is_game_over():
            move = choose_opening_move(board)

            if move is None:
                if (white == "NNUE" and board.turn == chess.WHITE) or (black == "NNUE" and board.turn == chess.BLACK):
                    move = nnue_move(board)
                else:
                    engine = stockfish if (white == "Stockfish" and board.turn == chess.WHITE) or (
                                black == "Stockfish" and board.turn == chess.BLACK) else lczero
                    result = engine.play(board, chess.engine.Limit(time=args.time_control))
                    move = result.move

            board.push(move)
            node = node.add_variation(move)
            moves += 1

        game.headers["Result"] = board.result()
        with open(f"games_nnue_{args.job_id}.pgn", "a") as f:
            print(game, file=f)
            f.write("\n")

        moves_sum += moves
        print(f"Moves in game {i}: {moves}")

        results[board.result()] += 1
end = time.time()
print(f'Avg moves: {moves_sum/(args.games*12)}')
print(f'10 games avg: {(end - start)/args.games}s')
print(f"Wyniki: {results}")

stockfish.quit()
lczero.quit()

