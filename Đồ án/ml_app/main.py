import tkinter as tk
from tkinter import messagebox
from view.data_view import DataView
from controller.data_controller import DataController

def main():
    try:
        # Khởi tạo cửa sổ chính
        root = tk.Tk()
        root.title("Phân tích Dữ liệu - MVC")
        root.geometry("1000x600")
        root.configure(bg='#2d2d2d')

        # Tạo view và controller
        view = DataView(root)
        controller = DataController(view)

        # Chạy vòng lặp chính
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Lỗi khởi động", f"Không thể khởi động ứng dụng: {str(e)}")

if __name__ == "__main__":
    main()