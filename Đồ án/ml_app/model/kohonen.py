import numpy as np
import math
import random

def euclidean_distance_vec(v1: np.ndarray, v2: np.ndarray) -> float:
    """Tính khoảng cách Euclidean giữa hai vector."""
    return np.sqrt(np.sum((v1 - v2) ** 2))

def get_bmu(sample: np.ndarray, weights: np.ndarray) -> tuple:
    """
    Tìm nơ-ron BMU (Best Matching Unit) cho một mẫu.
    Args:
        sample: vector dữ liệu (n_features,)
        weights: mảng trọng số lưới kích thước (grid_height, grid_width, n_features)
    Returns:
        Tuple (i, j) là chỉ số hàng và cột của BMU trên lưới.
    """
    grid_height, grid_width, _ = weights.shape
    bmu_idx = (0, 0)
    min_dist = float('inf')
    for i in range(grid_height):
        for j in range(grid_width):
            dist = euclidean_distance_vec(sample, weights[i, j])
            if dist < min_dist:
                min_dist = dist
                bmu_idx = (i, j)
    return bmu_idx

def decay_parameter(initial_value: float, t: int, max_iters: int) -> float:
    """Hàm giảm tham số (learning rate hoặc sigma) theo thời gian theo hàm mũ."""
    return initial_value * math.exp(-t / max_iters)

def neighborhood_function(bmu_idx: tuple, neuron_idx: tuple, sigma: float) -> float:
    """
    Hàm láng giềng Gaussian giữa BMU và nơ-ron khác trên lưới.
    Args:
        bmu_idx: (i, j) của BMU
        neuron_idx: (i, j) của nơ-ron cần tính láng giềng
        sigma: bán kính ảnh hưởng
    Returns:
        Giá trị láng giềng (Gaussian).
    """
    dist_sq = (bmu_idx[0] - neuron_idx[0])**2 + (bmu_idx[1] - neuron_idx[1])**2
    return math.exp(-dist_sq / (2 * (sigma ** 2)))

def kohonen_som(data: np.ndarray,
                grid_shape: tuple = (3, 3),
                n_iterations: int = 100,
                initial_lr: float = 0.5,
                initial_sigma: float = 1.0,
                random_seed: int = 42) -> np.ndarray:
    """
    Huấn luyện mô hình SOM (Kohonen) cho dữ liệu đầu vào.

    Args:
        data: mảng dữ liệu đầu vào kích thước (n_samples, n_features)
        grid_shape: tuple (height, width) kích thước lưới
        n_iterations: số vòng lặp huấn luyện
        initial_lr: learning rate ban đầu
        initial_sigma: bán kính ảnh hưởng ban đầu
        random_seed: seed cho random

    Returns:
        weights cuối cùng mảng kích thước (grid_height, grid_width, n_features)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    data = np.array(data, dtype=float)
    grid_height, grid_width = grid_shape
    n_features = data.shape[1]

    # Khởi tạo trọng số ngẫu nhiên trong phạm vi dữ liệu
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    weights = np.random.rand(grid_height, grid_width, n_features)
    weights = data_min + weights * (data_max - data_min)

    for t in range(n_iterations):
        lr = decay_parameter(initial_lr, t, n_iterations)
        sigma = decay_parameter(initial_sigma, t, n_iterations)

        # Có thể xáo trộn data
        for sample in data:
            bmu = get_bmu(sample, weights)
            for i in range(grid_height):
                for j in range(grid_width):
                    h = neighborhood_function(bmu, (i, j), sigma)
                    weights[i, j] += lr * h * (sample - weights[i, j])

    return weights


def map_samples_to_bmu(data: np.ndarray, weights: np.ndarray) -> list:
    """
    Gán mỗi mẫu về vị trí BMU tương ứng.

    Args:
        data: mảng dữ liệu (n_samples, n_features)
        weights: mảng trọng số (grid_height, grid_width, n_features)

    Returns:
        List BMU indices [(i,j), ...] cho mỗi mẫu.
    """
    return [get_bmu(sample, weights) for sample in np.array(data, dtype=float)]