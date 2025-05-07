import chess.pgn


initial_elo = 1000
K = 32
stockfish_elo = initial_elo
lc0_elo = initial_elo
nnue_elo = initial_elo


def calculate_elo(r_a, r_b, result, k=32):
    e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    new_r_a = r_a + k * (result - e_a)
    return new_r_a


pgn_path = "games_nnue.pgn"
with open(pgn_path) as pgn_file:
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break

        white = game.headers["White"]
        black = game.headers["Black"]
        result = game.headers["Result"]

        if result == "1-0":
            white_score, black_score = 1, 0
        elif result == "0-1":
            white_score, black_score = 0, 1
        else:
            white_score, black_score = 0.5, 0.5


        if white == "Stockfish" and black == "Lc0":
            stockfish_elo = calculate_elo(stockfish_elo, lc0_elo, white_score, K)
            lc0_elo = calculate_elo(lc0_elo, stockfish_elo, black_score, K)
        elif white == "Lc0" and black == "Stockfish":
            lc0_elo = calculate_elo(lc0_elo, stockfish_elo, white_score, K)
            stockfish_elo = calculate_elo(stockfish_elo, lc0_elo, black_score, K)

        elif white == "Stockfish" and black == "NNUE":
            stockfish_elo = calculate_elo(stockfish_elo, nnue_elo, white_score, K)
            nnue_elo = calculate_elo(nnue_elo, stockfish_elo, black_score, K)
        elif white == "NNUE" and black == "Stockfish":
            nnue_elo = calculate_elo(nnue_elo, stockfish_elo, white_score, K)
            stockfish_elo = calculate_elo(stockfish_elo, nnue_elo, black_score, K)

        elif white == "Lc0" and black == "NNUE":
            lc0_elo = calculate_elo(lc0_elo, nnue_elo, white_score, K)
            nnue_elo = calculate_elo(nnue_elo, lc0_elo, black_score, K)
        elif white == "NNUE" and black == "Lc0":
            nnue_elo = calculate_elo(nnue_elo, lc0_elo, white_score, K)
            lc0_elo = calculate_elo(lc0_elo, nnue_elo, black_score, K)

        else:
            print(f"Pominięto partię {white} vs {black}")

print(f"Stockfish Elo: {stockfish_elo:.2f}")
print(f"Lc0 Elo: {lc0_elo:.2f}")
print(f"NNUE Elo: {nnue_elo:.2f}")
