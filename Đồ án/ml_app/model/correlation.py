import pandas as pd

def calculate_correlation(df: pd.DataFrame, attr1: str, attr2: str) -> float:
    """Tính hệ số tương quan Pearson giữa hai thuộc tính."""
    return df[attr1].corr(df[attr2])