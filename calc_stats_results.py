import os
import chess.pgn
import pandas as pd

games_data = []
results_data = []
for file in os.listdir("games/merged"):
    if file.startswith("merged_") and file.endswith(".pgn"):
        file_path = os.path.join("games/merged", file)
        with open(file_path, encoding="utf-8") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                white = game.headers.get("White", "Unknown")
                black = game.headers.get("Black", "Unknown")
                result = game.headers.get("Result", "*")

                time = game.headers.get("Time")
                depth = game.headers.get("Depth")
                limit = time + "s" if time else ("d" + depth if depth else "?")

                games_data.append({
                    "White": white,
                    "Black": black,
                    "Result": result,
                    "Limit": limit
                })

                if result == "1-0":
                    results_data.append({"Engine": white, "Score": 1.0, "Limit": limit})
                    results_data.append({"Engine": black, "Score": 0.0, "Limit": limit})
                elif result == "0-1":
                    results_data.append({"Engine": white, "Score": 0.0, "Limit": limit})
                    results_data.append({"Engine": black, "Score": 1.0, "Limit": limit})
                elif result == "1/2-1/2":
                    results_data.append({"Engine": white, "Score": 0.5, "Limit": limit})
                    results_data.append({"Engine": black, "Score": 0.5, "Limit": limit})
                else:
                    continue

df_games = pd.DataFrame(games_data)
df_results = pd.DataFrame(results_data)
df_games.to_csv("results_raw_summary.csv", index=False)
df_results.to_csv("results_engine_summary.csv", index=False)
