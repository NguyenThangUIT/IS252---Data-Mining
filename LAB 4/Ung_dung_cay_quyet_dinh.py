import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import (
    calculate_correlation,
    min_max_normalize,
    z_score_normalize,
    calculate_support,
    apriori,
    find_maximal_frequent_itemsets,
    generate_rules,
    calculate_confidence,
    find_satisfied_rules,
    lower_approximation,
    upper_approximation,
    accuracy,
    dependency_coefficient,
    find_reducts,
    calculate_entropy,
    calculate_gain,
    calculate_gini,
    gain_id3_decision_tree,
    gini_id3_decision_tree,
    extract_rules
)

class DataController:
    def __init__(self, view):
        """Khởi tạo Controller với View và DataFrame rỗng."""
        self.data = None
        self.view = view
        self.view.set_controller(self)

    def load_data(self, file_path):
        """Tải dữ liệu từ file CSV."""
        try:
            self.data = pd.read_csv(file_path)
            self.view.update_columns(self.data.columns.tolist())
            self.view.display_result("Đã tải dữ liệu thành công!")
        except Exception as e:
            self.view.display_result(f"Lỗi khi tải dữ liệu: {str(e)}")

    def calculate_correlation(self, col1, col2):
        """Tính tương quan giữa hai cột và vẽ biểu đồ phân tán."""
        if self.data is None:
            self.view.display_result("Lỗi: Chưa tải dữ liệu!")
            return
        if not col1 or not col2:
            self.view.display_result("Lỗi: Vui lòng chọn hai cột!")
            return
        if col1 not in self.data.columns or col2 not in self.data.columns:
            self.view.display_result("Lỗi: Cột không tồn tại!")
            return
        try:
            if not pd.api.types.is_numeric_dtype(self.data[col1]) or not pd.api.types.is_numeric_dtype(self.data[col2]):
                self.view.display_result("Lỗi: Cột phải là kiểu số!")
                return
            corr = calculate_correlation(self.data, col1, col2)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=self.data, x=col1, y=col2, ax=ax, color='cyan')
            ax.set_title(f"Tương quan: {corr:.4f}", fontsize=12, color='white')
            ax.set_facecolor('#2d2d2d')
            fig.patch.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            plt.tight_layout()
            self.view.display_result(f"Hệ số tương quan giữa {col1} và {col2}: {corr:.4f}")
            self.view.display_plot(fig)
            plt.close(fig)
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")

    def min_max_normalize(self, columns):
        """Chuẩn hóa min-max cho các cột, trả về giá trị chuẩn hóa."""
        if self.data is None:
            self.view.display_result("Lỗi: Chưa tải dữ liệu!")
            return
        if not columns:
            self.view.display_result("Lỗi: Vui lòng chọn ít nhất một cột!")
            return
        if not all(col in self.data.columns for col in columns):
            self.view.display_result("Lỗi: Một hoặc nhiều cột không tồn tại!")
            return
        try:
            if not all(pd.api.types.is_numeric_dtype(self.data[col]) for col in columns):
                self.view.display_result("Lỗi: Các cột phải là kiểu số!")
                return
            df_norm = min_max_normalize(self.data, columns)
            self.data = df_norm
            result = "Dữ liệu sau chuẩn hóa Min-Max:\n"
            result += df_norm[columns].to_string(index=False)
            self.view.display_result(result)
            self.view.display_plot(None)  # Không vẽ biểu đồ
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")

    def z_score_normalize(self, columns):
        """Chuẩn hóa Z-score cho các cột, trả về giá trị chuẩn hóa."""
        if self.data is None:
            self.view.display_result("Lỗi: Chưa tải dữ liệu!")
            return
        if not columns:
            self.view.display_result("Lỗi: Vui lòng chọn ít nhất một cột!")
            return
        if not all(col in self.data.columns for col in columns):
            self.view.display_result("Lỗi: Một hoặc nhiều cột không tồn tại!")
            return
        try:
            if not all(pd.api.types.is_numeric_dtype(self.data[col]) for col in columns):
                self.view.display_result("Lỗi: Các cột phải là kiểu số!")
                return
            df_norm = z_score_normalize(self.data, columns)
            self.data = df_norm
            result = "Dữ liệu sau chuẩn hóa Z-score:\n"
            result += df_norm[columns].to_string(index=False)
            self.view.display_result(result)
            self.view.display_plot(None)  # Không vẽ biểu đồ
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")

    def apriori(self, min_support, column=None):
        """Tìm tập phổ biến và tập tối đa, trả về kết quả văn bản."""
        if self.data is None:
            self.view.display_result("Lỗi: Chưa tải dữ liệu!")
            return
        if not 0 <= min_support <= 1:
            self.view.display_result("Lỗi: min-support phải từ 0 đến 1!")
            return
        try:
            if column and column in self.data.columns:
                transactions = [set(str(item).split()) for item in self.data[column].dropna()]
            else:
                transactions = [set(row.dropna().astype(str)) for _, row in self.data.iterrows()]
            if not transactions:
                self.view.display_result("Lỗi: Không có giao dịch nào hợp lệ!")
                return
            frequent_itemsets = apriori(transactions, min_support)
            maximal_itemsets = find_maximal_frequent_itemsets(frequent_itemsets)
            result = "Tất cả các tập phổ biến:\n"
            for itemset in frequent_itemsets:
                result += f"- {itemset} (support: {calculate_support(itemset, transactions):.4f})\n"
            result += "\nCác tập phổ biến tối đa:\n"
            for itemset in maximal_itemsets:
                result += f"- {itemset}\n"
            self.view.display_result(result)
            self.view.display_plot(None)
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")

    def association_rules(self, min_support, min_confidence, column=None):
        """Tìm luật kết hợp thỏa min-confidence, trả về kết quả văn bản."""
        if self.data is None:
            self.view.display_result("Lỗi: Chưa tải dữ liệu!")
            return
        if not 0 <= min_support <= 1 or not 0 <= min_confidence <= 1:
            self.view.display_result("Lỗi: min-support và min-confidence phải từ 0 đến 1!")
            return
        try:
            if column and column in self.data.columns:
                transactions = [set(str(item).split()) for item in self.data[column].dropna()]
            else:
                transactions = [set(row.dropna().astype(str)) for _, row in self.data.iterrows()]
            if not transactions:
                self.view.display_result("Lỗi: Không có giao dịch nào hợp lệ!")
                return
            frequent_itemsets = apriori(transactions, min_support)
            maximal_itemsets = find_maximal_frequent_itemsets(frequent_itemsets)
            rules = generate_rules(maximal_itemsets)
            satisfied_rules = find_satisfied_rules(rules, transactions, min_confidence)
            result = "Tất cả các luật:\n"
            for ante, cons in rules:
                conf = calculate_confidence(ante, cons, transactions)
                result += f"- {ante} -> {cons} (confidence: {conf:.4f})\n"
            result += "\nLuật thỏa min-confidence:\n"
            if not satisfied_rules:
                result += "- Không có luật nào thỏa min-confidence.\n"
            else:
                for ante, cons in satisfied_rules:
                    conf = calculate_confidence(ante, cons, transactions)
                    result += f"- {ante} -> {cons} (confidence: {conf:.4f})\n"
            self.view.display_result(result)
            self.view.display_plot(None)
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")

    def rough_set(self, attributes, decision_attr, target_class):
        """Tính tập thô: xấp xỉ dưới, xấp xỉ trên, độ chính xác, hệ số phụ thuộc, tập rút gọn."""
        if self.data is None:
            self.view.display_result("Lỗi: Chưa tải dữ liệu!")
            return
        if not attributes or not decision_attr or not target_class:
            self.view.display_result("Lỗi: Vui lòng chọn thuộc tính, thuộc tính quyết định và lớp mục tiêu!")
            return
        if not all(attr in self.data.columns for attr in attributes + [decision_attr]):
            self.view.display_result("Lỗi: Một hoặc nhiều thuộc tính không tồn tại!")
            return
        if target_class not in self.data[decision_attr].unique():
            self.view.display_result(f"Lỗi: Lớp '{target_class}' không tồn tại trong cột '{decision_attr}'!")
            return
        try:
            lower_set = lower_approximation(self.data, attributes, target_class)
            upper_set = upper_approximation(self.data, attributes, target_class)
            acc = accuracy(self.data, attributes, target_class)
            dep_coeff = dependency_coefficient(self.data, attributes, decision_attr)
            reducts = find_reducts(self.data, attributes, decision_attr)
            result = (
                f"Tập xấp xỉ dưới: {lower_set}\n"
                f"Tập xấp xỉ trên: {upper_set}\n"
                f"Độ chính xác của xấp xỉ: {acc:.2f}\n"
                f"Hệ số phụ thuộc: {dep_coeff:.2f}\n"
                f"Các tập rút gọn: {reducts}"
            )
            self.view.display_result(result)
            self.view.display_plot(None)
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")

    def decision_tree(self, target_attribute, method="gain"):
        """Xây dựng cây quyết định ID3 với Gain hoặc Gini, vẽ cây, trả về tính toán và luật."""
        if self.data is None:
            self.view.display_result("Lỗi: Chưa tải dữ liệu!")
            return
        if not target_attribute or target_attribute not in self.data.columns:
            self.view.display_result("Lỗi: Vui lòng chọn thuộc tính quyết định hợp lệ!")
            return
        if method not in ["gain", "gini"]:
            self.view.display_result("Lỗi: Phương pháp phải là 'gain' hoặc 'gini'!")
            return
        try:
            # Xây dựng cây
            if method == "gain":
                tree = gain_id3_decision_tree(self.data, target_attribute)
            else:
                tree = gini_id3_decision_tree(self.data, target_attribute)

            # Tính toán Gain hoặc Gini
            result = f"Cây quyết định ID3 với phương pháp {method.upper()}:\n\n"
            for attribute in self.data.columns.drop(target_attribute):
                if method == "gain":
                    total_entropy = calculate_entropy(self.data, target_attribute)
                    gain = calculate_gain(self.data, attribute, target_attribute)
                    result += f"Thuộc tính: {attribute}\n"
                    result += f"Entropy tổng: {total_entropy:.4f}\n"
                    for value in self.data[attribute].unique():
                        subset = self.data[self.data[attribute] == value]
                        subset_entropy = calculate_entropy(subset, target_attribute)
                        result += f"  Entropy khi {attribute} = {value}: {subset_entropy:.4f}\n"
                    result += f"Gain: {gain:.4f}\n\n"
                else:
                    gini = calculate_gini(self.data, attribute, target_attribute)
                    result += f"Thuộc tính: {attribute}\n"
                    result += f"Hệ số Gini: {gini:.4f}\n\n"

            # Trích xuất luật
            rules = extract_rules(tree, target_attribute)
            result += "Bộ luật quyết định:\n"
            for rule in rules:
                result += f"- {rule}\n"

            # Vẽ cây với theme darkly
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.set_facecolor('#2d2d2d')
            fig.patch.set_facecolor('#2d2d2d')

            def plot_node(tree, parent_pos, current_pos, level, width=2.0):
                if isinstance(tree, dict):
                    attribute = list(tree.keys())[0]
                    plt.plot([parent_pos[0], current_pos[0]], [parent_pos[1], current_pos[1]], 'w-')
                    plt.text(current_pos[0], current_pos[1], attribute, 
                             bbox=dict(facecolor='cyan', edgecolor='white', boxstyle='round,pad=0.5'),
                             ha='center', va='center', color='black')
                    num_children = len(tree[attribute])
                    child_width = width / (2 ** level)
                    for i, (value, subtree) in enumerate(tree[attribute].items()):
                        child_x = current_pos[0] - width/2 + (i+0.5)*child_width/num_children
                        child_y = current_pos[1] - 1
                        child_pos = (child_x, child_y)
                        mid_x = (current_pos[0] + child_x) / 2
                        mid_y = (current_pos[1] + child_y) / 2
                        plt.text(mid_x, mid_y, str(value), ha='center', va='center', color='white')
                        plot_node(subtree, current_pos, child_pos, level+1, width)
                else:
                    plt.plot([parent_pos[0], current_pos[0]], [parent_pos[1], current_pos[1]], 'w-')
                    plt.text(current_pos[0], current_pos[1], str(tree),
                             bbox=dict(facecolor='lightgreen', edgecolor='white', boxstyle='round,pad=0.5'),
                             ha='center', va='center', color='black')

            root_pos = (0, 0)
            plot_node(tree, root_pos, root_pos, 1)
            plt.axis('off')
            plt.title(f"Cây quyết định ID3 ({method.upper()})", color='white', fontsize=14)

            self.view.display_result(result)
            self.view.display_plot(fig)
            plt.close(fig)
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")