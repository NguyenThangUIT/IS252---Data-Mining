import pandas as pd

def min_max_normalize(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Chuẩn hóa min-max cho các cột được chỉ định."""
    df_norm = df.copy()
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.0  # Nếu giá trị không đổi
    return df_norm