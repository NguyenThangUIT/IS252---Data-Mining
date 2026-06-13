import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

def calc_prior(df, target_col):
    """Tính xác suất tiên nghiệm P(class) cho từng lớp."""
    class_counts = df[target_col].value_counts()
    total = len(df)
    priors = {cls: count / total for cls, count in class_counts.items()}
    return priors

def calc_likelihood(df, feature, value, target_col, target_value):
    """Tính xác suất có điều kiện P(feature=value | class=target_value)."""
    subset = df[df[target_col] == target_value]
    count = np.sum(subset[feature] == value)
    total = len(subset)
    if total == 0:
        return 0
    return count / total

def predict_bayes(df, sample, target_col):
    """Dự đoán lớp cho một mẫu bằng Bayes (không làm trơn Laplace)."""
    priors = calc_prior(df, target_col)
    classes = priors.keys()
    posteriors = {}
    for cls in classes:
        prob = priors[cls]
        for feature, value in sample.items():
            prob *= calc_likelihood(df, feature, value, target_col, cls)
        posteriors[cls] = prob
    return max(posteriors, key=posteriors.get), posteriors

def calc_likelihood_laplace(df, feature, value, target_col, target_value):
    """Tính xác suất có điều kiện P(feature=value | class=target_value) với làm trơn Laplace."""
    subset = df[df[target_col] == target_value]
    count = np.sum(subset[feature] == value)
    total = len(subset)
    n_values = df[feature].nunique()
    return (count + 1) / (total + n_values)

def predict_bayes_laplace(df, sample, target_col):
    """Dự đoán lớp cho một mẫu bằng Bayes với làm trơn Laplace."""
    priors = calc_prior(df, target_col)
    classes = priors.keys()
    posteriors = {}
    for cls in classes:
        prob = priors[cls]
        for feature, value in sample.items():
            prob *= calc_likelihood_laplace(df, feature, value, target_col, cls)
        posteriors[cls] = prob
    return max(posteriors, key=posteriors.get), posteriors

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def k_means_custom(data, k, max_iters=100, initial_labels=None):
    data = np.array(data)
    n_samples, n_features = data.shape

    if initial_labels is not None:
        labels = np.array(initial_labels)
        centroids = np.array([
            np.mean(data[labels == i], axis=0)
            for i in range(k)
        ])
    else:
        initial_indices = random.sample(range(n_samples), k)
        centroids = data[initial_indices]

    for iteration in range(max_iters):
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

        if np.array_equal(new_labels, initial_labels if initial_labels is not None else labels):
            break

        centroids = new_centroids
        labels = new_labels
        initial_labels = new_labels

    return np.array(labels) + 1, centroids

def euclidean_distance_vec(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def get_bmu(sample, weights):
    grid_height, grid_width, _ = weights.shape
    bmu_idx = (0, 0)
    min_dist = float('inf')
    for i in range(grid_height):
        for j in range(grid_width):
            w = weights[i, j]
            dist = euclidean_distance_vec(sample, w)
            if dist < min_dist:
                min_dist = dist
                bmu_idx = (i, j)
    return bmu_idx

def decay_parameter(initial_value, t, max_iters):
    return initial_value * math.exp(-t / max_iters)

def neighborhood_function(bmu_idx, neuron_idx, sigma):
    dist_sq = (bmu_idx[0] - neuron_idx[0])**2 + (bmu_idx[1] - neuron_idx[1])**2
    return math.exp(-dist_sq / (2 * (sigma**2)))

def kohonen_som(data, grid_shape=(3, 3), n_iterations=100, initial_lr=0.5, initial_sigma=1.0, random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    data = np.array(data)
    n_samples, n_features = data.shape
    
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    
    grid_height, grid_width = grid_shape
    weights = np.random.rand(grid_height, grid_width, n_features)
    weights = data_min + weights * (data_max - data_min)
    
    for t in range(n_iterations):
        lr = decay_parameter(initial_lr, t, n_iterations)
        sigma = decay_parameter(initial_sigma, t, n_iterations)
        
        for sample in data:
            bmu_idx = get_bmu(sample, weights)
            for i in range(grid_height):
                for j in range(grid_width):
                    h = neighborhood_function(bmu_idx, (i, j), sigma)
                    weights[i, j] += lr * h * (sample - weights[i, j])
    
    return weights

def visualize_mapping(data, weights):
    grid_height, grid_width, _ = weights.shape
    mapping = {}

    for idx, sample in enumerate(data):
        bmu_idx = get_bmu(sample, weights)
        if bmu_idx in mapping:
            mapping[bmu_idx].append(idx + 1)
        else:
            mapping[bmu_idx] = [idx + 1]

    plt.figure(figsize=(6, 6))
    for i in range(grid_height):
        for j in range(grid_width):
            pos = (i, j)
            y = grid_height - i - 1

            if pos in mapping:
                plt.scatter(j, y, s=300, c='skyblue', edgecolors='black')
                text = ','.join(str(x) for x in mapping[pos])
                plt.text(j + 0.05, y, text, fontsize=10, ha='left', va='center', color='black')
            else:
                plt.scatter(j, y, s=300, c='white', edgecolors='black')

    plt.title("Biểu đồ phân cụm Kohonen (SOM)")
    plt.xticks(range(grid_width))
    plt.yticks(range(grid_height))
    plt.grid(True)
    plt.tight_layout()

class ClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Phân cụm dữ liệu")
        self.root.geometry("800x600")
        
        # Tạo notebook để chứa các tab
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, expand=True, fill='both')
        
        # Tạo các tab
        self.kmeans_tab = ttk.Frame(self.notebook)
        self.kohonen_tab = ttk.Frame(self.notebook)
        self.bayes_tab = ttk.Frame(self.notebook)
        self.bayes_laplace_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.kmeans_tab, text='K-means')
        self.notebook.add(self.kohonen_tab, text='Kohonen SOM')
        self.notebook.add(self.bayes_tab, text='Bayes')
        self.notebook.add(self.bayes_laplace_tab, text='Bayes Laplace')
        
        # Khởi tạo các tab
        self.setup_kmeans_tab()
        self.setup_kohonen_tab()
        self.setup_bayes_tab()
        self.setup_bayes_laplace_tab()
        
        self.df = None
        
    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            # Update column selection comboboxes
            if hasattr(self, 'kmeans_column_var'):
                self.kmeans_column_combo['values'] = list(self.df.columns)
                self.kmeans_column_combo.set(self.df.columns[0])
            if hasattr(self, 'kohonen_column_var'):
                self.kohonen_column_combo['values'] = list(self.df.columns)
                self.kohonen_column_combo.set(self.df.columns[0])
            if hasattr(self, 'bayes_target_var'):
                self.bayes_target_combo['values'] = list(self.df.columns)
                self.bayes_target_combo.set(self.df.columns[-1])
            if hasattr(self, 'bayes_laplace_target_var'):
                self.bayes_laplace_target_combo['values'] = list(self.df.columns)
                self.bayes_laplace_target_combo.set(self.df.columns[-1])
            messagebox.showinfo("Thành công", "Đã tải dữ liệu thành công!")
            return True
        return False
        
    def setup_kmeans_tab(self):
        # Frame cho việc load data
        load_frame = ttk.LabelFrame(self.kmeans_tab, text="Tải dữ liệu")
        load_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(load_frame, text="Chọn file dữ liệu", command=self.load_data).pack(pady=5)
        
        # Frame cho input
        input_frame = ttk.LabelFrame(self.kmeans_tab, text="Tham số đầu vào")
        input_frame.pack(fill="x", padx=5, pady=5)
        
        # Chọn cột đầu tiên
        ttk.Label(input_frame, text="Chọn cột đầu tiên:").pack()
        self.kmeans_column_var = tk.StringVar()
        self.kmeans_column_combo = ttk.Combobox(input_frame, textvariable=self.kmeans_column_var, state="readonly")
        self.kmeans_column_combo.pack(pady=5)
        
        # Số cụm
        ttk.Label(input_frame, text="Số cụm (k):").pack()
        self.k_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.k_var).pack(pady=5)
        
        # Nhãn ban đầu
        ttk.Label(input_frame, text="Nhãn ban đầu (phân cách bằng dấu phẩy):").pack()
        self.initial_labels_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.initial_labels_var).pack(pady=5)
        
        ttk.Button(input_frame, text="Thực hiện phân cụm", command=self.perform_kmeans).pack(pady=5)
        
        # Frame cho kết quả
        result_frame = ttk.LabelFrame(self.kmeans_tab, text="Kết quả")
        result_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.kmeans_result = ttk.Label(result_frame, text="")
        self.kmeans_result.pack(pady=5)
        
    def setup_kohonen_tab(self):
        # Frame cho việc load data
        load_frame = ttk.LabelFrame(self.kohonen_tab, text="Tải dữ liệu")
        load_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(load_frame, text="Chọn file dữ liệu", command=self.load_data).pack(pady=5)
        
        # Frame cho input
        input_frame = ttk.LabelFrame(self.kohonen_tab, text="Tham số đầu vào")
        input_frame.pack(fill="x", padx=5, pady=5)
        
        # Chọn cột đầu tiên
        ttk.Label(input_frame, text="Chọn cột đầu tiên:").pack()
        self.kohonen_column_var = tk.StringVar()
        self.kohonen_column_combo = ttk.Combobox(input_frame, textvariable=self.kohonen_column_var, state="readonly")
        self.kohonen_column_combo.pack(pady=5)
        
        # Kích thước lưới
        ttk.Label(input_frame, text="Kích thước lưới (hàng,cột):").pack()
        self.grid_shape_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.grid_shape_var).pack(pady=5)
        
        # Số vòng lặp
        ttk.Label(input_frame, text="Số vòng lặp:").pack()
        self.n_iter_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.n_iter_var).pack(pady=5)
        
        # Tốc độ học ban đầu
        ttk.Label(input_frame, text="Tốc độ học ban đầu:").pack()
        self.lr_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.lr_var).pack(pady=5)
        
        # Sigma ban đầu
        ttk.Label(input_frame, text="Sigma ban đầu:").pack()
        self.sigma_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.sigma_var).pack(pady=5)
        
        ttk.Button(input_frame, text="Huấn luyện SOM", command=self.train_som).pack(pady=5)
        
        # Frame cho kết quả
        result_frame = ttk.LabelFrame(self.kohonen_tab, text="Kết quả")
        result_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.kohonen_result = ttk.Label(result_frame, text="")
        self.kohonen_result.pack(pady=5)
        
    def setup_bayes_tab(self):
        # Frame cho việc load data
        load_frame = ttk.LabelFrame(self.bayes_tab, text="Tải dữ liệu")
        load_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(load_frame, text="Chọn file dữ liệu", command=self.load_data).pack(pady=5)
        
        # Frame cho input
        input_frame = ttk.LabelFrame(self.bayes_tab, text="Tham số đầu vào")
        input_frame.pack(fill="x", padx=5, pady=5)
        
        # Chọn cột mục tiêu
        ttk.Label(input_frame, text="Chọn cột mục tiêu:").pack()
        self.bayes_target_var = tk.StringVar()
        self.bayes_target_combo = ttk.Combobox(input_frame, textvariable=self.bayes_target_var, state="readonly")
        self.bayes_target_combo.pack(pady=5)
        
        # Frame cho nhập dữ liệu mẫu
        sample_frame = ttk.LabelFrame(self.bayes_tab, text="Nhập dữ liệu mẫu")
        sample_frame.pack(fill="x", padx=5, pady=5)
        
        self.sample_entries = {}
        self.sample_frame = ttk.Frame(sample_frame)
        self.sample_frame.pack(pady=5)
        
        ttk.Button(sample_frame, text="Cập nhật trường nhập liệu", command=self.update_sample_entries).pack(pady=5)
        ttk.Button(sample_frame, text="Dự đoán", command=self.perform_bayes).pack(pady=5)
        
        # Frame cho kết quả
        result_frame = ttk.LabelFrame(self.bayes_tab, text="Kết quả")
        result_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.bayes_result = ttk.Label(result_frame, text="")
        self.bayes_result.pack(pady=5)
        
    def setup_bayes_laplace_tab(self):
        # Frame cho việc load data
        load_frame = ttk.LabelFrame(self.bayes_laplace_tab, text="Tải dữ liệu")
        load_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(load_frame, text="Chọn file dữ liệu", command=self.load_data).pack(pady=5)
        
        # Frame cho input
        input_frame = ttk.LabelFrame(self.bayes_laplace_tab, text="Tham số đầu vào")
        input_frame.pack(fill="x", padx=5, pady=5)
        
        # Chọn cột mục tiêu
        ttk.Label(input_frame, text="Chọn cột mục tiêu:").pack()
        self.bayes_laplace_target_var = tk.StringVar()
        self.bayes_laplace_target_combo = ttk.Combobox(input_frame, textvariable=self.bayes_laplace_target_var, state="readonly")
        self.bayes_laplace_target_combo.pack(pady=5)
        
        # Frame cho nhập dữ liệu mẫu
        sample_frame = ttk.LabelFrame(self.bayes_laplace_tab, text="Nhập dữ liệu mẫu")
        sample_frame.pack(fill="x", padx=5, pady=5)
        
        self.sample_entries_laplace = {}
        self.sample_frame_laplace = ttk.Frame(sample_frame)
        self.sample_frame_laplace.pack(pady=5)
        
        ttk.Button(sample_frame, text="Cập nhật trường nhập liệu", command=self.update_sample_entries_laplace).pack(pady=5)
        ttk.Button(sample_frame, text="Dự đoán", command=self.perform_bayes_laplace).pack(pady=5)
        
        # Frame cho kết quả
        result_frame = ttk.LabelFrame(self.bayes_laplace_tab, text="Kết quả")
        result_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.bayes_laplace_result = ttk.Label(result_frame, text="")
        self.bayes_laplace_result.pack(pady=5)
        
    def perform_kmeans(self):
        if self.df is None:
            messagebox.showerror("Lỗi", "Vui lòng tải dữ liệu trước!")
            return
            
        try:
            k = int(self.k_var.get())
            initial_labels = [int(x.strip()) for x in self.initial_labels_var.get().split(",")]
            
            # Get selected column and reorder data
            selected_col = self.kmeans_column_var.get()
            if not selected_col:
                messagebox.showerror("Lỗi", "Vui lòng chọn cột!")
                return
                
            # Reorder columns to put selected column first
            cols = [selected_col] + [col for col in self.df.columns if col != selected_col]
            data = self.df[cols].apply(pd.to_numeric, errors='coerce').values
            
            if np.isnan(data).any():
                messagebox.showerror("Lỗi", "Dữ liệu chứa giá trị không phải số!")
                return
                
            labels, centroids = k_means_custom(data, k, initial_labels=initial_labels)
            
            result_text = "Kết quả phân cụm:\n"
            for i in range(k):
                cluster_points = np.where(labels == i + 1)[0]
                result_text += f"\nCụm {i + 1}:\nĐiểm: {cluster_points + 1}\nTâm cụm: {centroids[i]}\n"
                
            self.kmeans_result.config(text=result_text)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {str(e)}")
        
    def train_som(self):
        if self.df is None:
            messagebox.showerror("Lỗi", "Vui lòng tải dữ liệu trước!")
            return
            
        try:
            # Get selected column and reorder data
            selected_col = self.kohonen_column_var.get()
            if not selected_col:
                messagebox.showerror("Lỗi", "Vui lòng chọn cột!")
                return
                
            # Reorder columns to put selected column first
            cols = [selected_col] + [col for col in self.df.columns if col != selected_col]
            data = self.df[cols].apply(pd.to_numeric, errors='coerce').values
            
            if np.isnan(data).any():
                messagebox.showerror("Lỗi", "Dữ liệu chứa giá trị không phải số!")
                return
                
            grid_shape = tuple(map(int, self.grid_shape_var.get().split(",")))
            n_iterations = int(self.n_iter_var.get())
            initial_lr = float(self.lr_var.get())
            initial_sigma = float(self.sigma_var.get())
            
            weights = kohonen_som(data, grid_shape, n_iterations, initial_lr, initial_sigma)
            
            # Tạo cửa sổ mới để hiển thị biểu đồ
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Biểu đồ phân cụm Kohonen")
            
            # Tạo figure và canvas
            fig = plt.figure(figsize=(8, 6))
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Vẽ biểu đồ
            visualize_mapping(data, weights)
            plt.title("Biểu đồ phân cụm Kohonen (SOM)")
            plt.xticks(range(grid_shape[1]))
            plt.yticks(range(grid_shape[0]))
            plt.grid(True)
            plt.tight_layout()
            
            # Hiển thị kết quả
            result_text = "Kết quả huấn luyện SOM:\n"
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    result_text += f"\nNơ-ron tại ({i},{j}): {weights[i, j]}\n"
            
            self.kohonen_result.config(text=result_text)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {str(e)}")

    def update_sample_entries(self):
        if self.df is None:
            messagebox.showerror("Lỗi", "Vui lòng tải dữ liệu trước!")
            return
            
        # Xóa các entry cũ
        for widget in self.sample_frame.winfo_children():
            widget.destroy()
        self.sample_entries.clear()
        
        # Tạo entry mới cho mỗi cột (trừ cột mục tiêu)
        target_col = self.bayes_target_var.get()
        for col in self.df.columns:
            if col != target_col:
                ttk.Label(self.sample_frame, text=f"{col}:").pack()
                entry = ttk.Entry(self.sample_frame)
                entry.pack(pady=2)
                self.sample_entries[col] = entry
                
    def update_sample_entries_laplace(self):
        if self.df is None:
            messagebox.showerror("Lỗi", "Vui lòng tải dữ liệu trước!")
            return
            
        # Xóa các entry cũ
        for widget in self.sample_frame_laplace.winfo_children():
            widget.destroy()
        self.sample_entries_laplace.clear()
        
        # Tạo entry mới cho mỗi cột (trừ cột mục tiêu)
        target_col = self.bayes_laplace_target_var.get()
        for col in self.df.columns:
            if col != target_col:
                ttk.Label(self.sample_frame_laplace, text=f"{col}:").pack()
                entry = ttk.Entry(self.sample_frame_laplace)
                entry.pack(pady=2)
                self.sample_entries_laplace[col] = entry
                
    def perform_bayes(self):
        if self.df is None:
            messagebox.showerror("Lỗi", "Vui lòng tải dữ liệu trước!")
            return
            
        try:
            target_col = self.bayes_target_var.get()
            if not target_col:
                messagebox.showerror("Lỗi", "Vui lòng chọn cột mục tiêu!")
                return
                
            # Tạo mẫu từ các entry
            sample = {}
            for col, entry in self.sample_entries.items():
                value = entry.get().strip()
                # Chuyển đổi giá trị boolean
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                sample[col] = value
                
            # Thực hiện dự đoán
            predicted_class, posteriors = predict_bayes(self.df, sample, target_col)
            
            # Hiển thị kết quả
            result_text = "Kết quả dự đoán Bayes:\n\n"
            result_text += f"Dự đoán lớp: {predicted_class}\n"
            result_text += "Xác suất hậu nghiệm:\n"
            for cls, prob in posteriors.items():
                result_text += f"  {cls}: {prob:.4f}\n"
                
            self.bayes_result.config(text=result_text)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {str(e)}")

    def perform_bayes_laplace(self):
        if self.df is None:
            messagebox.showerror("Lỗi", "Vui lòng tải dữ liệu trước!")
            return
            
        try:
            target_col = self.bayes_laplace_target_var.get()
            if not target_col:
                messagebox.showerror("Lỗi", "Vui lòng chọn cột mục tiêu!")
                return
                
            # Tạo mẫu từ các entry
            sample = {}
            for col, entry in self.sample_entries_laplace.items():
                value = entry.get().strip()
                # Chuyển đổi giá trị boolean
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                sample[col] = value
                
            # Thực hiện dự đoán
            predicted_class, posteriors = predict_bayes_laplace(self.df, sample, target_col)
            
            # Hiển thị kết quả
            result_text = "Kết quả dự đoán Bayes (Laplace):\n\n"
            result_text += f"Dự đoán lớp: {predicted_class}\n"
            result_text += "Xác suất hậu nghiệm:\n"
            for cls, prob in posteriors.items():
                result_text += f"  {cls}: {prob:.4f}\n"
                
            self.bayes_laplace_result.config(text=result_text)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusteringApp(root)
    root.mainloop()

