import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('tkagg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import ast
from controller.data_controller import DataController

class DataView:
    def __init__(self, root):
        self.root = root
        self.root.title("Khai thác dữ liệu")
        self.root.configure(bg='#ffffff')
        self.controller = DataController(self)
        self.columns = []
        self.create_widgets()

    def update_columns(self, columns):
        self.columns = columns
        for combo in [self.corr_col1_combo, self.corr_col2_combo, self.bayes_target_combo, self.apriori_col_combo]:
            combo['values'] = columns
        for combo in self.rough_set_attr_combos + self.tree_combo:
            if combo:
                combo['values'] = columns
        self.rough_set_decision_combo['values'] = columns
        # Cập nhật danh sách cột cho K-means và Kohonen
        for listbox in [self.kmeans_cols_listbox, self.kohonen_cols_listbox]:
            listbox.delete(0, tk.END)
            for col in columns:
                listbox.insert(tk.END, col)

    def display_result(self, result):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result, 'black')

    def display_plot(self, fig):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        if fig:
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
        else:
            tk.Label(self.plot_frame, text="Không có biểu đồ", fg='black', bg='#ffffff').pack()

    def create_widgets(self):
        style = ttk.Style()
        style.configure('TNotebook', background='#ffffff', foreground='black')
        style.configure('TLabel', background='#ffffff', foreground='black')
        style.configure('TButton', background='#ffffff', foreground='black')
        style.configure('TCombobox', background='#ffffff', foreground='black', fieldbackground='#ffffff')
        style.configure('TEntry', background='#ffffff', foreground='black', fieldbackground='#ffffff')

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky='nsew')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, columnspan=2, sticky='nsew')

        tabs = {
            'Correlation': self.create_correlation_tab,
            'Normalization': self.create_normalization_tab,
            'Apriori & Association Rules': self.create_apriori_tab,
            'Rough Set': self.create_rough_set_tab,
            'Decision Tree': self.create_decision_tree_tab,
            'Naive Bayes': self.create_bayes_tab,
            'K-means': self.create_kmeans_tab,
            'Kohonen SOM': self.create_kohonen_tab
        }
        for name, create_func in tabs.items():
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=name)
            create_func(tab)

        self.result_frame = ttk.Frame(main_frame)
        self.result_frame.grid(row=1, column=0, sticky='nsew', pady=10)
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.grid(row=1, column=1, sticky='nsew', pady=10)

        self.result_text = tk.Text(self.result_frame, height=10, width=50, bg='#ffffff', fg='black')
        self.result_text.grid(row=0, column=0, sticky='nsew')
        self.result_text.tag_configure('black', foreground='black')
        result_scroll = ttk.Scrollbar(self.result_frame, orient='vertical', command=self.result_text.yview)
        result_scroll.grid(row=0, column=1, sticky='ns')
        self.result_text['yscrollcommand'] = result_scroll.set

        load_button = ttk.Button(main_frame, text="Tải dữ liệu", command=self.load_data)
        load_button.grid(row=2, column=1, sticky='e', pady=10)

        clear_button = ttk.Button(main_frame, text="Xóa kết quả", command=self.clear_result_and_plot)
        clear_button.grid(row=2, column=0, sticky='w', pady=10)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=2)
        main_frame.rowconfigure(1, weight=1)

    def clear_result_and_plot(self):
        self.result_text.delete(1.0, tk.END)
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.controller.load_data(file_path)
            except AttributeError as e:
                messagebox.showerror("Lỗi", f"Controller thiếu phương thức load_data: {str(e)}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi tải dữ liệu: {str(e)}")

    def create_correlation_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        frame.grid(sticky='nsew')

        ttk.Label(frame, text="Cột 1:").grid(row=0, column=0, padx=5, pady=5)
        self.corr_col1_combo = ttk.Combobox(frame, state='readonly')
        self.corr_col1_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Cột 2:").grid(row=1, column=0, padx=5, pady=5)
        self.corr_col2_combo = ttk.Combobox(frame, state='readonly')
        self.corr_col2_combo.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Tính tương quan", 
                   command=self.run_correlation).grid(row=2, column=0, columnspan=2, pady=10)

    def run_correlation(self):
        try:
            col1 = self.corr_col1_combo.get()
            col2 = self.corr_col2_combo.get()
            if not col1 or not col2:
                messagebox.showerror("Lỗi", "Vui lòng chọn cả hai cột!")
                return
            self.controller.calculate_correlation(col1, col2)
        except AttributeError as e:
            messagebox.showerror("Lỗi", f"Controller thiếu phương thức calculate_correlation: {str(e)}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi tính tương quan: {str(e)}")

    def create_normalization_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        frame.grid(sticky='nsew')

        ttk.Label(frame, text="Giá trị cần chuyển đổi:").grid(row=0, column=0, padx=5, pady=5)
        self.value_entry = ttk.Entry(frame)
        self.value_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Giá trị nhỏ nhất (thang cũ):").grid(row=1, column=0, padx=5, pady=5)
        self.old_min_entry = ttk.Entry(frame)
        self.old_min_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Giá trị lớn nhất (thang cũ):").grid(row=2, column=0, padx=5, pady=5)
        self.old_max_entry = ttk.Entry(frame)
        self.old_max_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Giá trị nhỏ nhất (thang mới):").grid(row=3, column=0, padx=5, pady=5)
        self.new_min_entry = ttk.Entry(frame)
        self.new_min_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Giá trị lớn nhất (thang mới):").grid(row=4, column=0, padx=5, pady=5)
        self.new_max_entry = ttk.Entry(frame)
        self.new_max_entry.grid(row=4, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Chuẩn hóa Min-Max", 
                   command=self.run_manual_min_max_normalize).grid(row=5, column=0, columnspan=2, pady=10)

    def run_manual_min_max_normalize(self):
        try:
            value = float(self.value_entry.get())
            old_min = float(self.old_min_entry.get())
            old_max = float(self.old_max_entry.get())
            new_min = float(self.new_min_entry.get())
            new_max = float(self.new_max_entry.get())
            self.controller.manual_min_max_normalize(value, old_min, old_max, new_min, new_max)
        except AttributeError as e:
            messagebox.showerror("Lỗi", f"Controller thiếu phương thức manual_min_max_normalize: {str(e)}")
        except ValueError:
            messagebox.showerror("Lỗi", "Tất cả giá trị phải là số thực")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi chuẩn hóa Min-Max: {str(e)}")

    def create_apriori_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        frame.grid(sticky='nsew')

        ttk.Label(frame, text="Min Support:").grid(row=0, column=0, padx=5, pady=5)
        self.min_support_entry = ttk.Entry(frame)
        self.min_support_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Min Confidence:").grid(row=1, column=0, padx=5, pady=5)
        self.min_confidence_entry = ttk.Entry(frame)
        self.min_confidence_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Cột (tùy chọn):").grid(row=2, column=0, padx=5, pady=5)
        self.apriori_col_combo = ttk.Combobox(frame, state='readonly')
        self.apriori_col_combo.grid(row=2, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Chạy Apriori & Association Rules", 
                   command=self.run_apriori).grid(row=3, column=0, columnspan=2, pady=10)

    def run_apriori(self):
        try:
            min_support = float(self.min_support_entry.get())
            min_confidence = float(self.min_confidence_entry.get())
            col = self.apriori_col_combo.get() or None
            if min_support < 0 or min_support > 1 or min_confidence < 0 or min_confidence > 1:
                messagebox.showerror("Lỗi", "Min Support và Min Confidence phải từ 0 đến 1!")
                return
            self.controller.apriori(min_support, min_confidence, col)
        except AttributeError as e:
            messagebox.showerror("Lỗi", f"Controller thiếu phương thức apriori: {str(e)}")
        except ValueError:
            messagebox.showerror("Lỗi", "Min Support và Min Confidence phải là số thực")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi chạy Apriori: {str(e)}")

    def create_rough_set_tab(self, parent):
        self.rough_set_attr_combos = []
        frame = ttk.Frame(parent, padding=10)
        frame.grid(sticky='nsew')

        ttk.Label(frame, text="Số lượng thuộc tính:").grid(row=0, column=0, padx=5, pady=5)
        self.num_attrs_entry = ttk.Entry(frame)
        self.num_attrs_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Cập nhật ô thuộc tính", command=self.update_attr_combos).grid(row=0, column=2, padx=5, pady=5)

        self.attr_frame = ttk.Frame(frame)
        self.attr_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky='nsew')

        ttk.Label(frame, text="Thuộc tính quyết định:").grid(row=2, column=0, padx=5, pady=5)
        self.rough_set_decision_combo = ttk.Combobox(frame, state='readonly')
        self.rough_set_decision_combo.grid(row=2, column=1, padx=5, pady=5, columnspan=2)

        ttk.Label(frame, text="Lớp mục tiêu:").grid(row=3, column=0, padx=5, pady=5)
        self.target_class_entry = ttk.Entry(frame)
        self.target_class_entry.grid(row=3, column=1, padx=5, pady=5, columnspan=2)

        ttk.Button(frame, text="Chạy Rough Set", 
                   command=self.run_rough_set).grid(row=4, column=0, columnspan=3, pady=10)

    def update_attr_combos(self):
        for widget in self.attr_frame.winfo_children():
            widget.destroy()
        self.rough_set_attr_combos.clear()
        try:
            num_attrs = int(self.num_attrs_entry.get())
            if num_attrs < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Lỗi", "Số lượng thuộc tính phải là số nguyên dương!")
            return
        for i in range(num_attrs):
            ttk.Label(self.attr_frame, text=f"Thuộc tính {i+1}:").grid(row=i, column=0, padx=5, pady=5)
            combo = ttk.Combobox(self.attr_frame, state='readonly', values=self.columns)
            combo.grid(row=i, column=1, padx=5, pady=5)
            self.rough_set_attr_combos.append(combo)

    def run_rough_set(self):
        try:
            attributes = [combo.get() for combo in self.rough_set_attr_combos if combo.get()]
            if not attributes:
                messagebox.showerror("Lỗi", "Vui lòng chọn ít nhất một thuộc tính!")
                return
            decision_attr = self.rough_set_decision_combo.get()
            target_class = self.target_class_entry.get()
            if not decision_attr or not target_class:
                messagebox.showerror("Lỗi", "Vui lòng chọn thuộc tính quyết định và nhập lớp mục tiêu!")
                return
            self.controller.rough_set(attributes, decision_attr, target_class)
        except AttributeError as e:
            messagebox.showerror("Lỗi", f"Controller thiếu phương thức rough_set: {str(e)}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi chạy Rough Set: {str(e)}")

    def create_decision_tree_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        frame.grid(sticky='nsew')

        self.tree_combo = []
        for i in range(3):
            ttk.Label(frame, text=f"Thuộc tính {i+1}:").grid(row=i, column=0, padx=5, pady=5)
            combo = ttk.Combobox(frame, state='readonly')
            combo.grid(row=i, column=1, padx=5, pady=5)
            self.tree_combo.append(combo)

        ttk.Label(frame, text="Thuộc tính quyết định:").grid(row=3, column=0, padx=5, pady=5)
        target_attr_combo = ttk.Combobox(frame, state='readonly')
        target_attr_combo.grid(row=3, column=1, padx=5, pady=5)
        self.tree_combo.append(target_attr_combo)

        method = tk.StringVar(value='gain')
        ttk.Radiobutton(frame, text="Gain", value='gain', variable=method).grid(row=4, column=0, padx=5, pady=5)
        ttk.Radiobutton(frame, text="Gini", value='gini', variable=method).grid(row=4, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Xây cây", 
                   command=lambda: self.run_decision_tree(target_attr_combo.get(), method.get())).grid(row=5, column=0, columnspan=2, pady=10)

    def run_decision_tree(self, target_attr, method):
        try:
            if not target_attr:
                messagebox.showerror("Lỗi", "Vui lòng chọn thuộc tính quyết định!")
                return
            self.controller.decision_tree(target_attr, method)
        except AttributeError as e:
            messagebox.showerror("Lỗi", f"Controller thiếu phương thức decision_tree: {str(e)}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi xây cây quyết định: {str(e)}")

    def create_bayes_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        frame.grid(sticky='nsew')

        ttk.Label(frame, text="Cột mục tiêu:").grid(row=0, column=0, padx=5, pady=5)
        self.bayes_target_combo = ttk.Combobox(frame, state='readonly')
        self.bayes_target_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Mẫu (JSON):").grid(row=1, column=0, padx=5, pady=5)
        sample_entry = ttk.Entry(frame)
        sample_entry.grid(row=1, column=1, padx=5, pady=5)

        laplace = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Laplace", variable=laplace).grid(row=2, column=0, columnspan=2, pady=5)

        ttk.Button(frame, text="Dự đoán", 
                   command=lambda: self.run_bayes(sample_entry.get(), laplace.get())).grid(row=3, column=0, columnspan=2, pady=10)

    def run_bayes(self, sample_str, laplace):
        try:
            if not sample_str:
                messagebox.showerror("Lỗi", "Vui lòng nhập mẫu!")
                return
            sample = ast.literal_eval(sample_str)
            if not isinstance(sample, dict):
                messagebox.showerror("Lỗi", "Mẫu phải là dictionary Python hợp lệ!")
                return
            target_col = self.bayes_target_combo.get()
            if not target_col:
                messagebox.showerror("Lỗi", "Vui lòng chọn cột mục tiêu!")
                return
            self.controller.bayes_predict(target_col, sample, laplace)
        except AttributeError as e:
            messagebox.showerror("Lỗi", f"Controller thiếu phương thức bayes_predict: {str(e)}")
        except (SyntaxError, ValueError):
            messagebox.showerror("Lỗi", "Mẫu phải là dictionary Python hợp lệ (VD: {'col1': 'value1', 'col2': 'value2'})")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự đoán Bayes: {str(e)}")

    def create_kmeans_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        frame.grid(sticky='nsew')

        ttk.Label(frame, text="Chọn cột:").grid(row=0, column=0, padx=5, pady=5)
        self.kmeans_cols_listbox = tk.Listbox(frame, selectmode='multiple', height=5)
        self.kmeans_cols_listbox.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(frame, text="Số cụm (k):").grid(row=1, column=0, padx=5, pady=5)
        k_entry = ttk.Entry(frame)
        k_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Nhãn ban đầu (tùy chọn):").grid(row=2, column=0, padx=5, pady=5)
        initial_labels_entry = ttk.Entry(frame)
        initial_labels_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Chạy K-means", 
                   command=lambda: self.run_kmeans(k_entry.get(), initial_labels_entry.get())).grid(row=3, column=0, columnspan=2, pady=10)

    def run_kmeans(self, k_str, initial_labels_str):
        try:
            selected_cols = [self.kmeans_cols_listbox.get(idx) for idx in self.kmeans_cols_listbox.curselection()]
            if not selected_cols:
                messagebox.showerror("Lỗi", "Vui lòng chọn ít nhất một cột!")
                return
            k = int(k_str)
            if k < 1:
                messagebox.showerror("Lỗi", "Số cụm k phải là số nguyên dương!")
                return
            initial_labels = ast.literal_eval(initial_labels_str) if initial_labels_str else None
            self.controller.k_means(selected_cols, k, initial_labels)
        except AttributeError as e:
            messagebox.showerror("Lỗi", f"Controller thiếu phương thức k_means: {str(e)}")
        except ValueError:
            messagebox.showerror("Lỗi", "Số cụm k phải là số nguyên, nhãn ban đầu phải là danh sách (VD: [0, 1, 0])")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi chạy K-means: {str(e)}")

    def create_kohonen_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        frame.grid(sticky='nsew')

        ttk.Label(frame, text="Chọn cột:").grid(row=0, column=0, padx=5, pady=5)
        self.kohonen_cols_listbox = tk.Listbox(frame, selectmode='multiple', height=5)
        self.kohonen_cols_listbox.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Kích thước lưới (height, width):").grid(row=1, column=0, padx=5, pady=5)
        grid_shape_entry = ttk.Entry(frame)
        grid_shape_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Số vòng lặp:").grid(row=2, column=0, padx=5, pady=5)
        n_iterations_entry = ttk.Entry(frame)
        n_iterations_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Tốc độ học ban đầu:").grid(row=3, column=0, padx=5, pady=5)
        initial_lr_entry = ttk.Entry(frame)
        initial_lr_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Sigma ban đầu:").grid(row=4, column=0, padx=5, pady=5)
        initial_sigma_entry = ttk.Entry(frame)
        initial_sigma_entry.grid(row=4, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Chạy Kohonen", 
                   command=lambda: self.run_kohonen(
                       grid_shape_entry.get(), n_iterations_entry.get(), 
                       initial_lr_entry.get(), initial_sigma_entry.get())).grid(row=5, column=0, columnspan=2, pady=10)

    def run_kohonen(self, grid_shape_str, n_iterations_str, initial_lr_str, initial_sigma_str):
        try:
            selected_cols = [self.kohonen_cols_listbox.get(idx) for idx in self.kohonen_cols_listbox.curselection()]
            if not selected_cols:
                messagebox.showerror("Lỗi", "Vui lòng chọn ít nhất một cột!")
                return
            grid_shape = ast.literal_eval(grid_shape_str)
            if not isinstance(grid_shape, tuple) or len(grid_shape) != 2 or not all(isinstance(x, int) and x > 0 for x in grid_shape):
                messagebox.showerror("Lỗi", "Kích thước lưới phải là tuple (VD: (3, 3)) với số nguyên dương!")
                return
            n_iterations = int(n_iterations_str)
            if n_iterations <= 0:
                messagebox.showerror("Lỗi", "Số vòng lặp phải là số nguyên dương!")
                return
            initial_lr = float(initial_lr_str)
            if initial_lr <= 0:
                messagebox.showerror("Lỗi", "Tốc độ học ban đầu phải là số dương!")
                return
            initial_sigma = float(initial_sigma_str)
            if initial_sigma <= 0:
                messagebox.showerror("Lỗi", "Sigma ban đầu phải là số dương!")
                return
            self.controller.kohonen_som(
                selected_cols, grid_shape, n_iterations, initial_lr, initial_sigma)
        except AttributeError as e:
            messagebox.showerror("Lỗi", f"Controller thiếu phương thức kohonen_som: {str(e)}")
        except (SyntaxError, ValueError):
            messagebox.showerror("Lỗi", "Kích thước lưới phải là tuple (VD: (3, 3)), các tham số khác phải là số")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi chạy Kohonen SOM: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    view = DataView(root)
    root.mainloop()