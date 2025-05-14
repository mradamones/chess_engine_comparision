import chess.pgn
import chess.engine
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--pgn", type=str, required=True, help="PGN file to analyze")
parser.add_argument("--job-id", type=int, required=True)
parser.add_argument("--total-jobs", type=int, required=True)
parser.add_argument("--total-games", type=int, required=True)
parser.add_argument("--depth", type=int, required=True)
args = parser.parse_args()

ENGINE_PATH = "engines/ethereal/Ethereal"

games = []
with open(f'merged/{args.pgn}') as f:
    for _ in range(args.total_games):
        game = chess.pgn.read_game(f)
        if game is None:
            break
        games.append(game)

total = len(games)
games_per_job = total // args.total_jobs
start = args.job_id * games_per_job
end = start + games_per_job if args.job_id < args.total_jobs - 1 else total
my_games = games[start:end]

output_file = f"acpl_job_{args.job_id}.csv"

with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine, open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["GameIndex", "White_ACPL", "Black_ACPL"])

    for idx, game in enumerate(my_games):
        board = game.board()
        white_losses = []
        black_losses = []

        for move in game.mainline_moves():
            try:
                if board.is_game_over():
                    break
                info_before = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                eval_before = info_before["score"].relative.score(mate_score=10000)
            except:
                break
            board.push(move)
            try:
                if board.is_game_over():
                    break
                info_after = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                eval_after = info_after["score"].relative.score(mate_score=10000)
            except:
                break

            if eval_before is None or eval_after is None:
                continue

            loss = abs(eval_after - eval_before)
            mover = board.turn

            if mover == chess.BLACK:
                white_losses.append(loss)
            else:
                black_losses.append(loss)

        white_acpl = round(sum(white_losses) / len(white_losses), 2) if white_losses else "-"
        black_acpl = round(sum(black_losses) / len(black_losses), 2) if black_losses else "-"
        writer.writerow([start + idx, white_acpl, black_acpl])
