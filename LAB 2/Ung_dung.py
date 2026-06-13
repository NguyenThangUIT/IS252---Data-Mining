import pandas as pd
import itertools
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import defaultdict

transactions = []

# Hàm tính support
def calculate_support(itemset, transactions):
    count = sum(1 for transaction in transactions if set(itemset).issubset(set(transaction)))
    return count / len(transactions)

# Hàm sinh các tập phổ biến
def apriori(transactions, min_support):
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1

    # Tập phổ biến cấp 1
    frequent_itemsets = [[item] for item, count in item_counts.items() if count / len(transactions) >= min_support]

    k = 2
    while True:
        candidate_itemsets = []
        for i in range(len(frequent_itemsets)):
            for j in range(i + 1, len(frequent_itemsets)):
                itemset = list(set(frequent_itemsets[i]) | set(frequent_itemsets[j]))
                if len(itemset) == k:
                    itemset.sort()
                    if itemset not in candidate_itemsets:
                        candidate_itemsets.append(itemset)

        frequent_itemsets_k = []
        for itemset in candidate_itemsets:
            support = calculate_support(itemset, transactions)
            if support >= min_support:
                frequent_itemsets_k.append(itemset)

        if not frequent_itemsets_k:
            break

        frequent_itemsets.extend(frequent_itemsets_k)
        k += 1

    return frequent_itemsets

# Hàm tìm tập phổ biến tối đại
def find_maximal_frequent_itemsets(frequent_itemsets):
    maximal_itemsets = []
    for itemset in frequent_itemsets:
        is_maximal = True
        for other_itemset in frequent_itemsets:
            if set(itemset) < set(other_itemset):
                is_maximal = False
                break
        if is_maximal:
            maximal_itemsets.append(itemset)
    return maximal_itemsets

# Sinh luật từ tập phổ biến tối đại
def generate_rules(maximal_itemsets):
    rules = []
    for itemset in maximal_itemsets:
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                consequent = list(set(itemset) - set(antecedent))
                rules.append((list(antecedent), consequent))
    return rules

# Tính confidence
def calculate_confidence(antecedent, consequent, transactions):
    antecedent_support = calculate_support(antecedent, transactions)
    combined_support = calculate_support(antecedent + consequent, transactions)
    return combined_support / antecedent_support if antecedent_support else 0

# Lọc luật theo min_confidence
def find_satisfied_rules(rules, transactions, min_confidence):
    satisfied_rules = []
    for rule in rules:
        if calculate_confidence(rule[0], rule[1], transactions) >= min_confidence:
            satisfied_rules.append(rule)
    return satisfied_rules

# Hàm tải CSV
def load_csv():
    global transactions
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        transactions = df.groupby('Mã hóa đơn')['Mã hàng'].apply(list).tolist()
        display_data(df)
        messagebox.showinfo("Thông báo", "Tải dữ liệu thành công, hãy nhập min-support và min-confidence rồi nhấn Phân tích!")

# Hàm xử lý phân tích khi nhấn nút
def process():
    try:
        min_support = float(min_support_entry.get())
        min_confidence = float(min_confidence_entry.get())

        if not transactions:
            messagebox.showerror("Lỗi", "Bạn chưa tải dữ liệu CSV.")
            return

        frequent_itemsets = apriori(transactions, min_support)
        maximal_itemsets = find_maximal_frequent_itemsets(frequent_itemsets)
        rules = generate_rules(maximal_itemsets)
        satisfied_rules = find_satisfied_rules(rules, transactions, min_confidence)

        text.delete("1.0", tk.END)
        text.insert("end", "Tất cả các tập phổ biến thỏa min-support:\n")
        for itemset in frequent_itemsets:
            text.insert("end", f"{itemset}\n")

        text.insert("end", "\nTất cả các tập phổ biến tối đại:\n")
        for itemset in maximal_itemsets:
            text.insert("end", f"{itemset}\n")

        text.insert("end", "\nTất cả các luật:\n")
        for rule in rules:
            conf = calculate_confidence(rule[0], rule[1], transactions)
            text.insert("end", f"{rule[0]} -> {rule[1]} | confidence: {conf:.2f}\n")

        text.insert("end", "\nTất cả các luật thỏa min-confidence:\n")
        for rule in satisfied_rules:
            conf = calculate_confidence(rule[0], rule[1], transactions)
            text.insert("end", f"{rule[0]} -> {rule[1]} | confidence: {conf:.2f}\n")

    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng nhập đúng định dạng số cho min-support và min-confidence.")

def display_data(df):
    text.delete("1.0", tk.END)
    text.insert("end", "Dữ liệu CSV:\n")
    text.insert("end", f"{df}\n")

# Giao diện tkinter
root = tk.Tk()
root.title("Apriori & Luật kết hợp")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

load_button = tk.Button(frame, text="Tải CSV", command=load_csv)
load_button.grid(row=0, column=0, padx=5)

tk.Label(frame, text="min-support:").grid(row=0, column=1)
min_support_entry = tk.Entry(frame)
min_support_entry.grid(row=0, column=2, padx=5)

tk.Label(frame, text="min-confidence:").grid(row=0, column=3)
min_confidence_entry = tk.Entry(frame)
min_confidence_entry.grid(row=0, column=4, padx=5)

analyze_button = tk.Button(frame, text="Phân tích", command=process)
analyze_button.grid(row=0, column=5, padx=5)

text = tk.Text(root, height=30, width=100)
text.pack(padx=10, pady=10)

root.mainloop()