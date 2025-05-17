import os
import random
import chess.pgn
import pandas as pd
import numpy as np

INITIAL_ELO = 1000
K = 32
FOLDS = 10

PGN_DIR = "games/merged"
pgn_files = [f for f in os.listdir(PGN_DIR) if f.endswith(".pgn")]


def calculate_elo(r_a, r_b, result, k=K):
    e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    new_r_a = r_a + k * (result - e_a)
    return new_r_a


all_games_per_limit = {}

for filename in pgn_files:
    limit = filename.split("merged_")[1].split(".pgn")[0]
    games = []

    with open(os.path.join(PGN_DIR, filename)) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            white = game.headers.get("White")
            black = game.headers.get("Black")
            result = game.headers.get("Result")

            if result == "1-0":
                white_score, black_score = 1, 0
            elif result == "0-1":
                white_score, black_score = 0, 1
            elif result == "1/2-1/2":
                white_score, black_score = 0.5, 0.5
            else:
                continue

            games.append((white, black, white_score, black_score))

    all_games_per_limit[limit] = games


elo_results = {}


for fold in range(FOLDS):
    print(f"Przetwarzanie fold {fold + 1}/{FOLDS}...")

    results_per_limit = {}
    for limit, games in all_games_per_limit.items():

        shuffled_games = games.copy()
        random.shuffle(shuffled_games)
        results_per_limit[limit] = shuffled_games

    all_games = []
    for games in results_per_limit.values():
        all_games.extend(games)

    random.shuffle(all_games)
    results_per_limit["all"] = all_games

    fold_results = {}
    for limit, games in results_per_limit.items():
        elo = {"Stockfish": INITIAL_ELO, "Lc0": INITIAL_ELO, "NNUE": INITIAL_ELO}

        for white, black, white_score, black_score in games:
            if white not in elo or black not in elo:
                continue

            white_elo_before = elo[white]
            black_elo_before = elo[black]

            elo[white] = calculate_elo(white_elo_before, black_elo_before, white_score)
            elo[black] = calculate_elo(black_elo_before, white_elo_before, black_score)

        for engine in ["Stockfish", "Lc0", "NNUE"]:
            if limit not in fold_results:
                fold_results[limit] = {}
            if engine not in fold_results[limit]:
                fold_results[limit][engine] = []

            fold_results[limit][engine].append(elo[engine])

    for limit in fold_results:
        if limit not in elo_results:
            elo_results[limit] = {}
        for engine in fold_results[limit]:
            if engine not in elo_results[limit]:
                elo_results[limit][engine] = []
            elo_results[limit][engine].append(fold_results[limit][engine][0])


results_rows = []
for limit in sorted(elo_results.keys()):
    for engine in ["Stockfish", "Lc0", "NNUE"]:
        if engine in elo_results[limit]:
            elo_values = elo_results[limit][engine]
            mean_elo = np.mean(elo_values)
            std_elo = np.std(elo_values)

            results_rows.append({
                "Engine": engine,
                "Limit": limit,
                "Mean_Elo": round(mean_elo, 2),
                "Std_Elo": round(std_elo, 2),
                "Min_Elo": round(min(elo_values), 2),
                "Max_Elo": round(max(elo_values), 2)
            })


final_df = pd.DataFrame(results_rows)
final_df = final_df.sort_values(by=["Limit", "Mean_Elo"], ascending=[True, False])
final_df.reset_index(drop=True, inplace=True)


final_df.to_csv("results_elo_summary.csv", index=False)
