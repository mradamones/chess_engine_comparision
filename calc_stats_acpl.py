import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def prepare_df(df: pd.DataFrame, eng_name: str) -> pd.DataFrame:
    engine = df[(df['White'] == eng_name) | (df['Black'] == eng_name)].copy()
    engine.loc[df['White'] == eng_name, 'color'] = 'White'
    engine.loc[df['Black'] == eng_name, 'color'] = 'Black'

    engine['acpl'] = engine['White_ACPL']
    engine.loc[engine['color'] == 'Black', 'acpl'] = engine['Black_ACPL']
    engine['acpl'] = pd.to_numeric(engine['acpl'], errors='coerce')
    engine = engine[['color', 'acpl']]
    return engine


def calculate_stats(df: pd.DataFrame):
    df = df.dropna()
    return {
        "mean": round(np.mean(df['acpl']), 2),
        "min": round(np.min(df['acpl']), 2),
        "max": round(np.max(df['acpl']), 2),
        "median": round(np.median(df['acpl']), 2),
        "std": round(np.std(df['acpl']), 2),
        "025": round(np.quantile(df['acpl'], 0.25), 2),
        "075": round(np.quantile(df['acpl'], 0.75), 2),
    }


def summarize_engine(df, name, limit):
    if name not in df['White'].values and name not in df['Black'].values:
        return []

    summary = []
    df_eng = prepare_df(df, name)

    for color in ['total', 'White', 'Black']:
        if color == 'total':
            sub_df = df_eng
            color_label = 'total'
        else:
            sub_df = df_eng[df_eng['color'] == color]
            color_label = color.lower()

        stats = calculate_stats(sub_df)
        stats.update({
            "engine": name,
            "color": color_label,
            "limit": limit
        })
        summary.append(stats)

    return summary


all_stats = []
for file in os.listdir("acpl"):
    if file.startswith("acpl_") and file.endswith(".csv"):
        file_path = os.path.join("acpl", file)
        main_limit = file.removeprefix("acpl_").removesuffix(".csv")

        try:
            main_df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Błąd przy wczytywaniu {file_path}: {e}")
            continue

        all_stats += summarize_engine(main_df, "Lc0", main_limit)
        all_stats += summarize_engine(main_df, "Stockfish", main_limit)
        all_stats += summarize_engine(main_df, "NNUE", main_limit)


summary_df = pd.DataFrame(all_stats)
summary_df.to_csv("acpl_summary.csv", index=False)
print("Zapisano podsumowanie do: acpl_summary.csv")
