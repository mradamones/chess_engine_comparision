import subprocess
import time
from datetime import datetime
import chess
import chess.engine
import random
import chess.pgn
import chess.polyglot
from nnue import FakeNNUE, pick_best_move_with_time, pick_best_move, eval_cache
import torch
import argparse


def start_engine(path, name=None, config=None):
    engine = chess.engine.SimpleEngine.popen_uci(path, stderr=subprocess.DEVNULL)
    if config:
        try:
            engine.configure(config)
        except Exception:
            pass
    return engine


def ensure_engine(engine, path, config=None):
    if hasattr(engine, 'process') and engine.process.returncode is not None:
        return start_engine(path, config=config)
    return engine


def play_with_restart(engine, path, board, limit, config=None):
    try:
        engine = ensure_engine(engine, path, config=config)
        result = engine.play(board, limit)
        return engine, result.move
    except chess.engine.EngineTerminatedError:
        try:
            engine.quit()
        except Exception:
            pass
        engine = start_engine(path, config=config)
        result = engine.play(board, limit)
        return engine, result.move


parser = argparse.ArgumentParser()
parser.add_argument('--games', type=int, required=True, help='Liczba gier do rozegrania')
parser.add_argument('--depth', type=int, required=True, help='Głębokość')
parser.add_argument('--job-id', type=int, required=True, help='ID zadania')
parser.add_argument('--offset', type=int, default=0, help='Indeks początkowy partii')
args = parser.parse_args()

stockfish_path = "engines/stockfish.exe"
lczero_path = "engines/lc0/lc0.exe"
book_path = "./Komodo.bin"

stockfish_config = {"Threads": 1}
lczero_config = {"backend": "cudnn", "gpu_threads": 1}

stockfish = start_engine(stockfish_path, config=stockfish_config)
lczero = start_engine(lczero_path,    config=lczero_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nnue_model = FakeNNUE().to(device)
nnue_model.load_state_dict(torch.load("./saved/fakennue_trained.pt"))
nnue_model.eval()

def choose_opening_move(board):
    if len(board.move_stack) >= 20:
        return None
    with chess.polyglot.open_reader(book_path) as reader:
        moves = [entry.move for entry in reader.find_all(board)]
        legal = [m for m in moves if m in board.legal_moves]
        return random.choice(legal) if legal else None


def nnue_move(board):
    return pick_best_move_with_time(board, nnue_model, time_limit=time_for_move)


def nnue_move_depth(board):
    return pick_best_move(board, nnue_model, depth=args.depth)

results   = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

num_games = 100
time_for_move = 10
moves_sum = 0
start = time.time()
limit = chess.engine.Limit(depth=args.depth)

for i in range(args.offset, args.offset + args.games):
    for white, black in [("Stockfish", "Lc0"), ("Lc0", "Stockfish"), ("Stockfish", "NNUE"), ("NNUE", "Stockfish"), ("Lc0", "NNUE"), ("NNUE", "Lc0")]:
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers.update({
            "White": white,
            "Black": black,
            "Event": "Praca dyplomowa",
            "Round": str(i),
            "Date": datetime.now().strftime("%Y.%m.%d %H:%M:%S"),
            "Site": "GitHub Actions",
            "Depth": str(args.depth)
        })

        node = game
        moves = 0
        while True:
            mv = choose_opening_move(board)
            if not mv:
                break
            board.push(mv)
            node = node.add_variation(mv)
            moves += 1

        while not board.is_game_over():
            if (white == "NNUE" and board.turn == chess.WHITE) or \
               (black == "NNUE" and board.turn == chess.BLACK):
                mv = nnue_move_depth(board)
            else:
                if (white == "Stockfish" and board.turn == chess.WHITE) or \
                   (black == "Stockfish" and board.turn == chess.BLACK):
                    engine, path, conf = stockfish, stockfish_path, stockfish_config
                else:
                    engine, path, conf = lczero, lczero_path, lczero_config
                engine, mv = play_with_restart(engine, path, board, limit, config=conf)
                if path == stockfish_path:
                    stockfish = engine
                else:
                    lczero = engine

            board.push(mv)
            node = node.add_variation(mv)
            moves += 1

        game.headers["Result"] = board.result()
        # with open(f"games_nnue_{args.job_id}.pgn", "a") as f:
        with open(f"depth_{args.job_id}.pgn", "a") as f:
            print(game, file=f)
            f.write("\n")

        moves_sum += moves
        print(f"Moves in game {i}: {moves} ({white}, {black})")

        results[board.result()] += 1
        eval_cache.clear()
end = time.time()
print(f'Avg moves: {moves_sum/(args.games*12)}')
print(f'10 games avg: {(end - start)/args.games}s')
print(f"Wyniki: {results}")

stockfish.quit()
lczero.quit()

