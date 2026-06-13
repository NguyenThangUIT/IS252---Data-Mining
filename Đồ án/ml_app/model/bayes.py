import numpy as np

def calc_prior(df, target_col):
    """Tính xác suất tiên nghiệm P(class) cho từng lớp."""
    class_counts = df[target_col].value_counts()
    total = len(df)
    priors = {cls: count / total for cls, count in class_counts.items()}
    return priors

def calc_likelihood(df, feature, value, target_col, target_value):
    """Tính xác suất có điều kiện P(feature=value | class=target_value) (không làm trơn)."""
    subset = df[df[target_col] == target_value]
    count = np.sum(subset[feature] == value)
    total = len(subset)
    if total == 0:
        return 0
    return count / total

def calc_likelihood_laplace(df, feature, value, target_col, target_value):
    """Tính xác suất có điều kiện P(feature=value | class=target_value) với làm trơn Laplace."""
    subset = df[df[target_col] == target_value]
    count = np.sum(subset[feature] == value)
    total = len(subset)
    n_values = df[feature].nunique()
    return (count + 1) / (total + n_values)

def predict_bayes(df, sample, target_col):
    """
    Dự đoán lớp cho một mẫu bằng Bayes không làm trơn.
    Trả về: (nhãn dự đoán, dict xác suất hậu nghiệm từng nhãn)
    """
    priors = calc_prior(df, target_col)
    posteriors = {}
    for cls in priors:
        prob = priors[cls]
        for feature, value in sample.items():
            prob *= calc_likelihood(df, feature, value, target_col, cls)
        posteriors[cls] = prob
    return max(posteriors, key=lambda cls: posteriors[cls]), posteriors

def predict_bayes_laplace(df, sample, target_col):
    """
    Dự đoán lớp cho một mẫu bằng Bayes có làm trơn Laplace.
    Trả về: (nhãn dự đoán, dict xác suất hậu nghiệm từng nhãn)
    """
    priors = calc_prior(df, target_col)
    posteriors = {}
    for cls in priors:
        prob = priors[cls]
        for feature, value in sample.items():
            prob *= calc_likelihood_laplace(df, feature, value, target_col, cls)
        posteriors[cls] = prob
    return max(posteriors, key=lambda cls: posteriors[cls]), posteriors
