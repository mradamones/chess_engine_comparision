import subprocess
import time
from datetime import datetime
import chess
import chess.engine
import random
import chess.pgn
import chess.polyglot
import csv
import os
from nnue import FakeNNUE, pick_best_move_with_time, pick_best_move, eval_cache
import torch

stockfish_path = "engines/stockfish.exe"
lczero_path = "engines/lc0/lc0.exe"
book_path = "./Komodo.bin"
output_csv = "results.csv"

stockfish_config = {"Threads": 1}
lczero_config = {"backend": "cudnn", "gpu_threads": 1}

max_depth = 10
games_per_matchup = 2  # 1 jako białe, 1 jako czarne
max_avg_time = 10

skip_engine = {
    "Stockfish": False,
    "Lc0": False,
    "NNUE": False
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nnue_model = FakeNNUE().to(device)
nnue_model.load_state_dict(torch.load("./saved/fakennue_trained.pt"))
nnue_model.eval()


def start_engine(path, name=None, config=None):
    engine = chess.engine.SimpleEngine.popen_uci(path, stderr=subprocess.DEVNULL)
    engine.name = name
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
    name = engine.name
    try:
        engine = ensure_engine(engine, path, config)
        result = engine.play(board, limit)
        return engine, result.move
    except chess.engine.EngineTerminatedError:
        try:
            engine.quit()
        except Exception:
            pass
        engine = start_engine(path, name, config)
        result = engine.play(board, limit)
        return engine, result.move


def choose_opening_move(board):
    if len(board.move_stack) >= 20:
        return None
    with chess.polyglot.open_reader(book_path) as reader:
        moves = [entry.move for entry in reader.find_all(board)]
        legal = [m for m in moves if m in board.legal_moves]
        return random.choice(legal) if legal else None


def nnue_move_depth(board, depth):
    return pick_best_move(board, nnue_model, depth=depth)


def get_move(engine_name, board, depth, limit, white_player, black_player, engines):
    if (engine_name == "NNUE"):
        return nnue_move_depth(board, depth)
    engine, path, conf = engines[engine_name]
    engine, mv = play_with_restart(engine, path, board, limit, config=conf)
    engines[engine_name] = (engine, path, conf)
    return mv


def main():
    engines = {
        "Stockfish": (start_engine(stockfish_path, "Stockfish", stockfish_config), stockfish_path, stockfish_config),
        "Lc0": (start_engine(lczero_path, "Lc0", lczero_config), lczero_path, lczero_config),
    }

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["depth", "engine", "avg_time", "color"])

        for depth in range(1, max_depth + 1):
            limit = chess.engine.Limit(depth=depth)
            print(f"Testing depth {depth}")
            avg_times = {"Stockfish": {"white": [], "black": []},
                         "Lc0": {"white": [], "black": []},
                         "NNUE": {"white": [], "black": []}}

            matchups = [("Stockfish", "Lc0"), ("Stockfish", "NNUE"),
                        ("Lc0", "NNUE")]

            for white, black in matchups:
                for color_flip in range(2):
                    white_player = white if color_flip == 0 else black
                    black_player = black if color_flip == 0 else white

                    if skip_engine[white_player] or skip_engine[black_player]:
                        continue

                    board = chess.Board()
                    moves = 0
                    times = {"white": [], "black": []}

                    while not board.is_game_over():
                        mv = choose_opening_move(board)
                        if mv:
                            board.push(mv)
                            continue

                        start_time = time.time()
                        current = white_player if board.turn == chess.WHITE else black_player
                        mv = get_move(current, board, depth, limit, white_player, black_player, engines)
                        elapsed = time.time() - start_time
                        board.push(mv)

                        color = "white" if board.turn == chess.BLACK else "black"
                        times[color].append(elapsed)
                        avg_times[current][color].append(elapsed)
                        eval_cache.clear()

            for engine_name in ["Stockfish", "Lc0", "NNUE"]:
                for color in ["white", "black"]:
                    times = avg_times[engine_name][color]
                    if times:
                        avg = sum(times) / len(times)
                        writer.writerow([depth, engine_name, round(avg, 3), color])
                        print(f"{engine_name} ({color}) avg time @ depth {depth}: {avg:.2f}s")
                        if avg > max_avg_time:
                            skip_engine[engine_name] = True

    for engine, _, _ in engines.values():
        engine.quit()


if __name__ == "__main__":
    main()
