import random
import numpy as np

def euclidean_distance(p1, p2):
    """Tính khoảng cách Euclidean giữa hai điểm."""
    return np.sqrt(np.sum((p1 - p2) ** 2))

def k_means_custom(data, k, max_iters=100, initial_labels=None):
    """
    Hàm gom cụm K-means tùy chỉnh.

    Tham số:
        data (ndarray): Dữ liệu đầu vào (dạng mảng 2 chiều).
        k (int): Số cụm cần phân chia.
        max_iters (int): Số lần lặp tối đa.
        initial_labels (list): Nhãn khởi tạo ban đầu (tùy chọn).

    Trả về:
        labels (ndarray): Nhãn cụm tương ứng với từng điểm (bắt đầu từ 1).
        centroids (ndarray): Tọa độ các tâm cụm cuối cùng.
    """
    data = np.array(data)
    n_samples, _ = data.shape

    if initial_labels is not None:
        labels = np.array(initial_labels)
        centroids = np.array([
            np.mean(data[labels == i], axis=0)
            for i in range(k)
        ])
    else:
        labels = np.zeros(n_samples, dtype=int)  # Khởi tạo nhãn bằng zeros
        initial_indices = random.sample(range(n_samples), k)
        centroids = data[initial_indices]

    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        new_labels = []

        for sample in data:
            distances = [euclidean_distance(sample, centroid) for centroid in centroids]
            closest_idx = np.argmin(distances)
            clusters[closest_idx].append(sample)
            new_labels.append(closest_idx)

        new_centroids = np.array([
            np.mean(cluster, axis=0) if cluster else centroids[idx]
            for idx, cluster in enumerate(clusters)
        ])

        # So sánh nhãn mới với nhãn hiện tại
        if np.array_equal(new_labels, labels):
            break

        centroids = new_centroids
        labels = np.array(new_labels)  # Chuyển new_labels thành mảng NumPy

    return np.array(labels) + 1, centroids