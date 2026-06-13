import itertools
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from itertools import combinations
from PIL import Image, ImageTk

# Tạo dữ liệu từ bảng
data = {
    "O": [1, 2, 3, 4, 5, 6, 7],
    "Kích thước": ["Vừa", "Nhỏ", "Nhỏ", "Lớn", "Lớn", "Lớn", "Lớn"],
    "Màu sắc": ["Xanh", "Đỏ", "Đỏ", "Đỏ", "Lục", "Đỏ", "Lục"],
    "Hình dạng": ["Viên gạch", "Hình nêm", "Hình cầu", "Hình nêm", "Hình trụ", "Hình trụ", "Hình cầu"],
    "Lớp": ["A", "B", "A", "B", "A", "B", "A"]
}

df = pd.DataFrame(data)
# Lưu vào file CSV
csv_file_path = "data.csv"
df.to_csv(csv_file_path, index=False, encoding="utf-8")

def lower_approximation(df, attributes, target_class):
    grouped = df.groupby(attributes)
    lower_approx = []
    
    for _, group in grouped:
        unique_classes = set(group["Lớp"])
        if unique_classes == {target_class}:  
            lower_approx.extend(group["O"].tolist())
    
    return set(lower_approx)

def upper_approximation(df, attributes, target_class):
    grouped = df.groupby(attributes)
    upper_approx = []
    
    for _, group in grouped:
        if target_class in set(group["Lớp"]):  
            upper_approx.extend(group["O"].tolist())
    
    return set(upper_approx)

def accuracy(df, attributes, target_class):
    lower_approx_set = lower_approximation(df, attributes, target_class)
    upper_approx_set = upper_approximation(df, attributes, target_class)
    
    if not upper_approx_set:
        return 0.0
    
    accuracy_value = len(lower_approx_set) / len(upper_approx_set)
    return accuracy_value

def dependency_coefficient(df, attributes, decision_attr):
    grouped = df.groupby(attributes)
    pos_region_size = sum(len(group) for _, group in grouped if len(group[decision_attr].unique()) == 1)
    total_objects = len(df)
    return pos_region_size / total_objects

def find_reducts(df, all_attributes, decision_attr):
    best_reducts = set()
    max_dependency = dependency_coefficient(df, all_attributes, decision_attr)
    
    for r in range(1, len(all_attributes) + 1):
        for subset in combinations(all_attributes, r):
            if dependency_coefficient(df, list(subset), decision_attr) == max_dependency:
                is_reduct = True
                for smaller_subset in combinations(subset, r - 1):
                    if dependency_coefficient(df, list(smaller_subset), decision_attr) == max_dependency:
                        is_reduct = False
                        break
                if is_reduct:
                    best_reducts.add(frozenset(subset))
    
    return [set(reduct) for reduct in best_reducts]

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return
    
    try:
        df = pd.read_csv(file_path)
        text.delete(1.0, tk.END)
        text.insert(tk.END, df.to_string(index=False))
        
        for widget in frame.winfo_children():
            widget.destroy()
        
        attributes = list(df.columns)
        var_list = [tk.StringVar(value=attr) for attr in attributes]
        
        tk.Label(frame, text="Chọn tập thuộc tính từ các thuộc tính của bảng:", bg="#f0f0f0").pack(anchor="w")
        for var in var_list:
            tk.Checkbutton(frame, text=var.get(), variable=var, onvalue=var.get(), offvalue="", bg="#f0f0f0").pack(anchor="w")
        
        decision_var = tk.StringVar(value="<None>")
        tk.Label(frame, text="Thuộc tính quyết định:", bg="#f0f0f0").pack(anchor="w")
        tk.OptionMenu(frame, decision_var, "<None>", *attributes).pack(anchor="w")
        
        def update_target_class_options(*args):
            target_class_menu['menu'].delete(0, 'end')
            unique_classes = df[decision_var.get()].unique().tolist()
            target_class_var.set(unique_classes[0] if unique_classes else "")
            for cls in unique_classes:
                target_class_menu['menu'].add_command(label=cls, command=tk._setit(target_class_var, cls))
        
        decision_var.trace_add("write", update_target_class_options)
        
        target_class_var = tk.StringVar()
        tk.Label(frame, text="Chọn lớp của các đối tượng:", bg="#f0f0f0").pack(anchor="w")
        target_class_menu = tk.OptionMenu(frame, target_class_var, "")
        target_class_menu.pack(anchor="w")
        
        def process():
            selected_attributes = [var.get() for var in var_list if var.get()]
            decision_attr = decision_var.get()
            target_class = target_class_var.get().strip()
            
            if not selected_attributes or not decision_attr or not target_class:
                messagebox.showerror("Lỗi", "Vui lòng chọn thuộc tính, thuộc tính quyết định và lớp mục tiêu.")
                return
            
            try:
                lower_set = lower_approximation(df, selected_attributes, target_class)
                upper_set = upper_approximation(df, selected_attributes, target_class)
                acc = accuracy(df, selected_attributes, target_class)
                dep_coeff = dependency_coefficient(df, selected_attributes, decision_attr)
                reducts = find_reducts(df, selected_attributes, decision_attr)
                
                result = (f"Tập xấp xỉ dưới: {lower_set}\n"
                          f"Tập xấp xỉ trên: {upper_set}\n"
                          f"Độ chính xác của xấp xỉ: {acc:.2f}\n"
                          f"Hệ số phụ thuộc: {dep_coeff:.2f}\n"
                          f"Các tập rút gọn: {reducts}")
                messagebox.showinfo("Kết quả", result)
            except Exception as e:
                messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")
        
        tk.Button(frame, text="Thực hiện", command=process).pack(anchor="w")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể đọc file: {e}")

root = tk.Tk()
root.title("Ứng dụng phân tích tập thô")
root.geometry("1000x706")

# Load ảnh nền với kích thước chuẩn của cửa sổ
image = Image.open("background.jpg")
background_image = ImageTk.PhotoImage(image)

# Dùng Label để hiển thị ảnh nền, đặt xuống dưới cùng
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
background_label.lower()  # Đưa ảnh nền xuống dưới cùng

# Thêm các widget khác sau khi ảnh nền đã đặt đúng vị trí
tk.Button(root, text="Chọn file CSV", command=open_file).pack(pady=5)
text = tk.Text(root, height=10, width=80)
text.pack(pady=5)
frame = tk.Frame(root, bg="#ffffff")  # Thêm màu nền nhẹ cho Frame để dễ nhìn hơn
frame.pack(pady=5)

root.mainloop()