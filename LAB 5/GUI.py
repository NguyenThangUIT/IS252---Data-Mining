import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
from Ung_dung_lab_05 import predict_bayes, predict_bayes_laplace, k_means_custom, kohonen_som, visualize_mapping

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng học máy - Lab 05")
        self.root.geometry("600x400")

        self.df = None
        self.label_col = None

        # Giao diện chính
        tk.Button(root, text="Tải file CSV", command=self.load_file).pack(pady=10)
        self.status_label = tk.Label(root, text="Chưa có dữ liệu.", fg="blue")
        self.status_label.pack()

        # Nút chọn label
        tk.Button(root, text="Chọn cột nhãn (label)", command=self.select_label_column).pack(pady=5)

        # Bayes
        tk.Button(root, text="Phân lớp Bayes", command=self.run_bayes).pack(pady=5)
        tk.Button(root, text="Phân lớp Bayes + Laplace", command=self.run_bayes_laplace).pack(pady=5)

        # KMeans
        tk.Button(root, text="Phân cụm KMeans", command=self.run_kmeans).pack(pady=5)

        # Kohonen SOM
        tk.Button(root, text="Phân cụm Kohonen SOM", command=self.run_kohonen).pack(pady=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.status_label.config(text=f"Đã tải: {file_path}")
                messagebox.showinfo("Thành công", f"Đã tải dữ liệu: {self.df.shape[0]} dòng, {self.df.shape[1]} cột")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đọc file: {e}")

    def select_label_column(self):
        if self.df is None:
            messagebox.showwarning("Chưa có dữ liệu", "Vui lòng tải file CSV trước.")
            return

        col_names = self.df.columns.tolist()
        self.label_col = simpledialog.askstring("Nhập tên cột nhãn", f"Chọn một trong các cột sau:\n{col_names}")
        if self.label_col not in col_names:
            messagebox.showerror("Lỗi", "Cột không hợp lệ.")
        else:
            messagebox.showinfo("Thành công", f"Đã chọn cột nhãn: {self.label_col}")

    def run_bayes(self):
        if self.df is None or self.label_col is None:
            messagebox.showerror("Thiếu dữ liệu", "Vui lòng tải dữ liệu và chọn cột nhãn.")
            return

        sample = self.get_sample_input()
        if sample:
            pred, probs = predict_bayes(self.df, sample, self.label_col)
            messagebox.showinfo("Kết quả Bayes", f"Dự đoán: {pred}\nXác suất: {probs}")

    def run_bayes_laplace(self):
        if self.df is None or self.label_col is None:
            messagebox.showerror("Thiếu dữ liệu", "Vui lòng tải dữ liệu và chọn cột nhãn.")
            return

        sample = self.get_sample_input()
        if sample:
            pred, probs = predict_bayes_laplace(self.df, sample, self.label_col)
            messagebox.showinfo("Kết quả Bayes (Laplace)", f"Dự đoán: {pred}\nXác suất: {probs}")

    def get_sample_input(self):
        sample = {}
        feature_cols = [col for col in self.df.columns if col != self.label_col]
        for col in feature_cols:
            val = simpledialog.askstring("Nhập mẫu", f"Giá trị cho '{col}':")
            if val is None:
                return None
            sample[col] = val
        return sample

    def run_kmeans(self):
        if self.df is None:
            messagebox.showwarning("Thiếu dữ liệu", "Vui lòng tải dữ liệu trước.")
            return
        try:
            k = simpledialog.askinteger("Nhập số cụm", "Số cụm K:", minvalue=2, maxvalue=10)
            df_numeric = self.df.select_dtypes(include=[np.number])
            labels, centroids = k_means_custom(df_numeric, k)
            messagebox.showinfo("Kết quả KMeans", f"Phân cụm thành công!\nNhãn cụm: {labels.tolist()}")
        except Exception as e:
            messagebox.showerror("Lỗi KMeans", str(e))

    def run_kohonen(self):
        if self.df is None:
            messagebox.showwarning("Thiếu dữ liệu", "Vui lòng tải dữ liệu trước.")
            return
        try:
            df_numeric = self.df.select_dtypes(include=[np.number])
            weights = kohonen_som(df_numeric.values, grid_shape=(3, 3), n_iterations=100)
            visualize_mapping(df_numeric.values, weights)
        except Exception as e:
            messagebox.showerror("Lỗi Kohonen", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()