import pandas as pd
from itertools import combinations

def lower_approximation(df: pd.DataFrame, attributes: list, target_class: str) -> set:
    """Tính tập xấp xỉ dưới của lớp mục tiêu"""
    grouped = df.groupby(attributes)
    lower_approx = []
    for _, group in grouped:
        if set(group["Lớp"]) == {target_class}:
            lower_approx.extend(group["O"].tolist())
    return set(lower_approx)

def upper_approximation(df: pd.DataFrame, attributes: list, target_class: str) -> set:
    """Tính tập xấp xỉ trên của lớp mục tiêu"""
    grouped = df.groupby(attributes)
    upper_approx = []
    for _, group in grouped:
        if target_class in set(group["Lớp"]):
            upper_approx.extend(group["O"].tolist())
    return set(upper_approx)

def accuracy(df: pd.DataFrame, attributes: list, target_class: str) -> float:
    """Tính độ chính xác của xấp xỉ"""
    lower = lower_approximation(df, attributes, target_class)
    upper = upper_approximation(df, attributes, target_class)
    if not upper:
        return 0.0
    return len(lower) / len(upper)

def dependency_coefficient(df: pd.DataFrame, attributes: list, decision_attr: str) -> float:
    """Tính hệ số phụ thuộc"""
    grouped = df.groupby(attributes)
    pos_region_size = sum(len(group) for _, group in grouped if len(group[decision_attr].unique()) == 1)
    return pos_region_size / len(df)

def find_reducts(df: pd.DataFrame, all_attributes: list, decision_attr: str) -> list:
    """Tìm các tập rút gọn (reducts)"""
    best_reducts = set()
    max_dep = dependency_coefficient(df, all_attributes, decision_attr)

    for r in range(1, len(all_attributes) + 1):
        for subset in combinations(all_attributes, r):
            subset = list(subset)
            if dependency_coefficient(df, subset, decision_attr) == max_dep:
                is_minimal = True
                for smaller in combinations(subset, r - 1):
                    if dependency_coefficient(df, list(smaller), decision_attr) == max_dep:
                        is_minimal = False
                        break
                if is_minimal:
                    best_reducts.add(frozenset(subset))

    return [set(r) for r in best_reducts]