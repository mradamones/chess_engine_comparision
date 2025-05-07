import subprocess
import time
from datetime import datetime
from multiprocessing import Pool
import chess
import chess.engine
import chess.pgn
import chess.polyglot
import random

# stockfish_path = 'C:\\Users\\szatk\\AppData\\Roaming\\org.encroissant.app\\engines\\stockfish\\stockfish-windows-x86-64-avx2.exe'
# lczero_path = 'D:\\Pobrane\\lc0\\lc0.exe'
stockfish_path = 'Y:\\chessengines\\stockfish\\stockfish-windows-x86-64-avx2.exe'
lczero_path = 'Y:\\chessengines\\lc0\\lc0.exe'
book_path = "./Komodo.bin"

# Funkcja do wyboru ruchu z książki debiutowej
def choose_opening_move(board):
    with chess.polyglot.open_reader(book_path) as reader:
        moves = list(reader.find_all(board))
        if moves:
            return random.choice(moves).move
    return None

# Funkcja rozgrywania jednej partii
def play_game(game_id, white_engine_path, black_engine_path, time_for_move=0.0001):
    # Uruchomienie silników
    white_engine = chess.engine.SimpleEngine.popen_uci(white_engine_path, stderr=subprocess.DEVNULL)
    black_engine = chess.engine.SimpleEngine.popen_uci(black_engine_path, stderr=subprocess.DEVNULL)

    engines = {"White": white_engine, "Black": black_engine}
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "Stockfish" if white_engine_path == stockfish_path else "Lc0"
    game.headers["Black"] = "Stockfish" if black_engine_path == stockfish_path else "Lc0"
    game.headers["Event"] = "Praca dyplomowa"
    game.headers["Round"] = str(game_id)
    game.headers["Date"] = str(datetime.now())
    game.headers["Site"] = "XD"
    game.headers["Time"] = str(time_for_move)
    node = game

    while not board.is_game_over():
        move = choose_opening_move(board)
        if move is None:
            engine = engines["White"] if board.turn == chess.WHITE else engines["Black"]
            result = engine.play(board, chess.engine.Limit(time=time_for_move))
            move = result.move
        board.push(move)
        node = node.add_variation(move)

    game.headers["Result"] = board.result()

    # Zapis partii do pliku PGN
    with open("games.pgn", "a") as f:
        print(game, file=f)

    # Zamykanie silników
    white_engine.quit()
    black_engine.quit()

    return board.result()

# Główna funkcja do uruchamiania wielu partii równolegle
def main(num_games=100, time_for_move=0.0001, max_workers=4):
    args = []
    for i in range(num_games):
        args.append((i, stockfish_path, lczero_path, time_for_move))  # Stockfish vs Lc0
        args.append((i, lczero_path, stockfish_path, time_for_move))  # Lc0 vs Stockfish

    # Uruchomienie równoległe
    with Pool(max_workers) as pool:
        results = pool.starmap(play_game, args)

    # Podsumowanie wyników
    summary = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for result in results:
        summary[result] += 1

    print("Wyniki:", summary)

if __name__ == "__main__":
    start = time.time()
    main(num_games=100, time_for_move=0.001, max_workers=4)
    end = time.time()
    print(end - start)
