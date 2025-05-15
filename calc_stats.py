import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def prepare_df(df: pd.DataFrame, eng_name: str) -> pd.DataFrame:
    engine = df[(df['White'] == eng_name) | (df['Black'] == eng_name)]
    engine.loc[df['White'] == eng_name, 'color'] = 'White'
    engine.loc[df['Black'] == eng_name, 'color'] = 'Black'

    # engine.loc[df['color'] == 'White', 'acpl'] = df['White_ACPL']
    # engine.loc[df['color'] == 'Black', 'acpl'] = df['Black_ACPL']
    engine['acpl'] = engine['White_ACPL']
    engine['acpl'][engine['color'] == 'Black'] = engine['Black_ACPL']
    engine = engine[['color', 'acpl']]
    return engine


def calculate_stats(df: pd.DataFrame):
    df_mean = np.round(np.mean(df['acpl']), 2)
    df_min = np.round(np.min(df['acpl']), 2)
    df_max = np.round(np.max(df['acpl']), 2)
    df_median = np.round(np.median(df['acpl']), 2)
    df_std = np.round(np.std(df['acpl']), 2)
    df_025 = np.round(np.quantile(df['acpl'], 0.25), 2)
    df_075 = np.round(np.quantile(df['acpl'], 0.75), 2)
    return df_mean, df_min, df_max, df_median, df_std, df_025, df_075


main_df = pd.read_csv('acpl/acpl_001s.csv')

lc0 = prepare_df(main_df, 'Lc0')
stockfish = prepare_df(main_df, 'Stockfish')
own_engine = prepare_df(main_df, 'NNUE')

lc0_mean, lc0_min, lc0_max, lc0_median, lc0_std, lc0_025, lc0_075 = calculate_stats(lc0)
stockfish_mean, stockfish_min, stockfish_max, stockfish_median, stockfish_std, stockfish_025, stockfish_075 = calculate_stats(stockfish)
own_engine_mean, own_engine_min, own_engine_max, own_engine_median, own_engine_std, own_engine_025, own_engine_075 = calculate_stats(own_engine)

print(lc0_mean, lc0_min, lc0_max, lc0_median, lc0_std, lc0_025, lc0_075)
print(stockfish_mean, stockfish_min, stockfish_max, stockfish_median, stockfish_std, stockfish_025, stockfish_075)
print(own_engine_mean, own_engine_max, own_engine_median, own_engine_std, own_engine_025, own_engine_075)