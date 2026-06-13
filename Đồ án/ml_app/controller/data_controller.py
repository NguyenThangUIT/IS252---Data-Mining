import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import (
    calculate_correlation,
    min_max_normalize,
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
    extract_rules,
    calc_prior,
    calc_likelihood,
    calc_likelihood_laplace,
    predict_bayes,
    predict_bayes_laplace,
    k_means_custom,
    kohonen_som,
    get_bmu
)

class DataController:
    def __init__(self, view):
        self.data = None
        self.view = view

    def load_data(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
            self.view.update_columns(self.data.columns.tolist())
            self.view.display_result("Đã tải dữ liệu thành công!")
        except Exception as e:
            self.view.display_result(f"Lỗi khi tải dữ liệu: {str(e)}")

    def calculate_correlation(self, col1, col2):
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
            
    def manual_min_max_normalize(self, value, old_min, old_max, new_min, new_max):
        try:
            if old_max == old_min:
                raise ValueError("Giá trị lớn nhất và nhỏ nhất của thang đo cũ không được trùng nhau")
            new_value = new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)
            self.view.display_result(f"Giá trị sau khi chuẩn hóa: {new_value:.4f}\n")
            self.view.display_plot(None)
        except Exception as e:
            self.view.display_result(f"Lỗi khi chuẩn hóa Min-Max: {str(e)}\n")

    def min_max_normalize(self, columns):
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
            self.view.display_plot(None)
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")

    def apriori(self, min_support, min_confidence, column=None):
        if self.data is None:
            self.view.display_result("Lỗi: Chưa tải dữ liệu!")
            return
        if not 0 <= min_support <= 1 or not 0 <= min_confidence <= 1:
            self.view.display_result("Lỗi: min-support và min-confidence phải từ 0 đến 1!")
            return
        try:
            # Kiểm tra dữ liệu có cột "Mã hóa đơn" và "Mã hàng"
            if 'Mã hóa đơn' not in self.data.columns or 'Mã hàng' not in self.data.columns:
                self.view.display_result("Lỗi: Dữ liệu phải có cột 'Mã hóa đơn' và 'Mã hàng'!")
                return

            # Nhóm dữ liệu theo "Mã hóa đơn" để tạo transactions
            transactions = self.data.groupby('Mã hóa đơn')['Mã hàng'].apply(list).tolist()
            if not transactions:
                self.view.display_result("Lỗi: Không có giao dịch nào hợp lệ!")
                return

            # Chạy thuật toán Apriori
            frequent_itemsets = apriori(transactions, min_support)
            maximal_itemsets = find_maximal_frequent_itemsets(frequent_itemsets)
            rules = generate_rules(maximal_itemsets)
            satisfied_rules = find_satisfied_rules(rules, transactions, min_confidence)

            # Hiển thị kết quả
            result = "Tất cả các tập phổ biến thỏa min-support:\n"
            if not frequent_itemsets:
                result += "- Không có tập nào.\n"
            else:
                for itemset in frequent_itemsets:
                    support = calculate_support(itemset, transactions)
                    result += f"- {itemset} (support: {support:.4f})\n"

            result += "\nCác tập phổ biến tối đa:\n"
            if not maximal_itemsets:
                result += "- Không có tập nào.\n"
            else:
                for itemset in maximal_itemsets:
                    result += f"- {itemset}\n"

            result += "\nTất cả các luật:\n"
            if not rules:
                result += "- Không có luật nào.\n"
            else:
                for ante, cons in rules:
                    conf = calculate_confidence(ante, cons, transactions)
                    result += f"- {ante} -> {cons} (confidence: {conf:.4f})\n"

            result += "\nLuật thỏa min-confidence:\n"
            if not satisfied_rules:
                result += "- Không có luật nào thỏa min-confidence.\n"
            else:
                for ante, cons, conf in satisfied_rules:
                    result += f"- {ante} -> {cons} (confidence: {conf:.4f})\n"

            self.view.display_result(result)
            self.view.display_plot(None)
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")

    def association_rules(self, min_support, min_confidence, column=None):
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
                f"Độ chính xác của xấp xỉ: {acc:.4f}\n"
                f"Hệ số phụ thuộc: {dep_coeff:.2f}\n"
                f"Các tập rút gọn: {reducts}"
            )
            self.view.display_result(result)
            self.view.display_plot(None)
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")

    def decision_tree(self, target_attribute, method="gain"):
        if self.data is None:
            self.view.display_result("Lỗi: Chưa tải dữ liệu!")
            return
        if not target_attribute or target_attribute not in self.data.columns:
            self.view.display_result("Lỗi: Vui lòng chọn thuộc tính mục tiêu hợp lệ!")
            return
        if method not in ["gain", "gini"]:
            self.view.display_result("Lỗi: Phương pháp phải là 'gain' hoặc 'gini'!")
            return
        try:
            # Lấy danh sách các thuộc tính được chọn (không bao gồm thuộc tính mục tiêu)
            selected_attrs = [c.get() for c in self.view.tree_combo if c.get() and c.get() != target_attribute]
            if not selected_attrs:
                self.view.display_result("Lỗi: Vui lòng chọn ít nhất một thuộc tính (không bao gồm thuộc tính mục tiêu)!")
                return

            if method == "gain":
                tree = gain_id3_decision_tree(self.data, target_attribute, selected_attrs)
            else:
                tree = gini_id3_decision_tree(self.data, target_attribute, selected_attrs)

            result = f"Cây quyết định ID3 với phương pháp {method.upper()}:\n\n"
            for attribute in selected_attrs:
                if method == "gain":
                    total_entropy = calculate_entropy(self.data, target_attribute)
                    gain = calculate_gain(self.data, attribute, target_attribute)
                    result += f"Thuộc tính: {attribute}\n"
                    result += f"Entropy tổng: {total_entropy:.4f}\n"
                    for value in self.data[attribute].dropna().unique():
                        subset = self.data[self.data[attribute] == value]
                        if not subset.empty:
                            subset_entropy = calculate_entropy(subset, target_attribute)
                            result += f"  Entropy khi {attribute} = {value}: {subset_entropy:.4f}\n"
                    result += f"Gain: {gain:.4f}\n\n"
                else:
                    gini = calculate_gini(self.data, attribute, target_attribute)
                    result += f"Thuộc tính: {attribute}\n"
                    result += f"Hệ số Gini: {gini:.4f}\n\n"
            rules = extract_rules(tree, target_attribute)
            result += "Bộ luật kết quả:\n"
            for rule in rules:
                result += f"- {rule}\n"

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_facecolor('#2d2d2d')
            fig.patch.set_facecolor('#2d2d2d')

            # Đầu tiên, duyệt cây để lấy số lượng lá ở mỗi node (cho việc căn chỉnh)
            def count_leaves(node):
                if not isinstance(node, dict):
                    return 1
                attribute = next(iter(node))
                return sum(count_leaves(subtree) for subtree in node[attribute].values())

            # Đệ quy vẽ cây, căn chỉnh đều các nhánh
            def plot_node(tree, x, y, x_range, level, parent_pos=None, parent_label=None):
                if isinstance(tree, dict):
                    attribute = next(iter(tree))
                    num_leaves = count_leaves(tree)
                    # Vị trí node hiện tại là trung tâm của x_range
                    curr_x = (x_range[0] + x_range[1]) / 2
                    curr_y = y
                    if parent_pos is not None:
                        ax.plot([parent_pos[0], curr_x], [parent_pos[1], curr_y], 'w-', zorder=1)
                        if parent_label is not None:
                            mid_x = (parent_pos[0] + curr_x) / 2
                            mid_y = (parent_pos[1] + curr_y) / 2 + 0.1
                            ax.text(mid_x, mid_y, str(parent_label), ha='center', va='bottom', color='white', fontsize=10)
                    ax.text(curr_x, curr_y, attribute, bbox=dict(facecolor='cyan', edgecolor='white', boxstyle='round,pad=0.5'),
                            ha='center', va='center', color='black', fontsize=12, zorder=2)
                    # Chia x_range cho các nhánh con
                    child_ranges = []
                    leaves_so_far = 0
                    for value, subtree in tree[attribute].items():
                        leaves = count_leaves(subtree)
                        start = x_range[0] + (leaves_so_far / num_leaves) * (x_range[1] - x_range[0])
                        end = start + (leaves / num_leaves) * (x_range[1] - x_range[0])
                        child_ranges.append((value, subtree, (start, end)))
                        leaves_so_far += leaves
                    for value, subtree, child_x_range in child_ranges:
                        plot_node(subtree, 0, curr_y - 1.5, child_x_range, level + 1, (curr_x, curr_y), value)
                else:
                    curr_x = (x_range[0] + x_range[1]) / 2
                    curr_y = y
                    if parent_pos is not None:
                        ax.plot([parent_pos[0], curr_x], [parent_pos[1], curr_y], 'w-', zorder=1)
                        if parent_label is not None:
                            mid_x = (parent_pos[0] + curr_x) / 2
                            mid_y = (parent_pos[1] + curr_y) / 2 + 0.1
                            ax.text(mid_x, mid_y, str(parent_label), ha='center', va='bottom', color='white', fontsize=10)
                    ax.text(curr_x, curr_y, str(tree),
                            bbox=dict(facecolor='lightgreen', edgecolor='white', boxstyle='round,pad=0.5'),
                            ha='center', va='center', color='black', fontsize=12, zorder=2)

            total_leaves = count_leaves(tree)
            plot_node(tree, 0, 0, (0, total_leaves * 2), 1)
            ax.axis('off')
            plt.title(f"Cây quyết định ID3 ({method.upper()})", color='white', fontsize=14)
            plt.tight_layout()
            self.view.display_result(result)
            self.view.display_plot(fig)
            plt.close(fig)
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")

    def bayes_predict(self, target_col, sample, laplace=False):
        if self.data is None:
            self.view.display_result("Lỗi: Chưa tải dữ liệu!")
            return
        if not target_col or target_col not in self.data.columns:
            self.view.display_result("Lỗi: Vui lòng chọn cột mục tiêu hợp lệ!")
            return
        if not sample or not isinstance(sample, dict):
            self.view.display_result("Lỗi: Vui lòng cung cấp mẫu dưới dạng dictionary!")
            return
        if not all(feature in self.data.columns for feature in sample):
            self.view.display_result("Lỗi: Một hoặc nhiều thuộc tính trong mẫu không tồn tại!")
            return
        try:
            for feature, value in sample.items():
                if value not in self.data[feature].values and not laplace:
                    self.view.display_result(f"Lỗi: Giá trị '{value}' không tồn tại trong cột '{feature}' (không dùng Laplace)!")
                    return
            if laplace:
                predicted_class, posteriors = predict_bayes_laplace(self.data, sample, target_col)
                method = "có làm trơn Laplace"
            else:
                predicted_class, posteriors = predict_bayes(self.data, sample, target_col)
                method = "không làm trơn Laplace"
            if all(prob == 0 for prob in posteriors.values()):
                predicted_class = self.data[target_col].mode()[0]
                result = f"Dự đoán Naive Bayes ({method}):\n"
                result += f"Mẫu: {sample}\n"
                result += f"Lớp dự đoán: {predicted_class} (tất cả xác suất hậu nghiệm bằng 0, chọn lớp phổ biến nhất)\n"
                result += "Xác suất hậu nghiệm:\n"
                for cls, prob in posteriors.items():
                    result += f"- P({cls} | mẫu) = {prob:.6f}\n"
            else:
                result = f"Dự đoán Naive Bayes ({method}):\n"
                result += f"Mẫu: {sample}\n"
                result += f"Lớp dự đoán: {predicted_class}\n"
                result += "Xác suất hậu nghiệm:\n"
                for cls, prob in posteriors.items():
                    result += f"- P({cls} | mẫu) = {prob:.6f}\n"
            self.view.display_result(result)
            self.view.display_plot(None)
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")

    def k_means(self, columns, k, initial_labels=None):
        if self.data is None:
            self.view.display_result("Lỗi: Chưa tải dữ liệu!")
            return
        if not columns:
            self.view.display_result("Lỗi: Vui lòng chọn ít nhất một cột!")
            return
        if not all(col in self.data.columns for col in columns):
            self.view.display_result("Lỗi: Một hoặc nhiều cột không tồn tại!")
            return
        if not all(pd.api.types.is_numeric_dtype(self.data[col]) for col in columns):
            self.view.display_result("Lỗi: Các cột phải là kiểu số!")
            return
        if not isinstance(k, int) or k < 1 or k > len(self.data):
            self.view.display_result(f"Lỗi: Số cụm k phải là số nguyên từ 1 đến {len(self.data)}!")
            return
        if initial_labels and (len(initial_labels) != len(self.data) or max(initial_labels) >= k or min(initial_labels) < 0):
            self.view.display_result("Lỗi: Ma trận phân hoạch ban đầu không hợp lệ!")
            return
        try:
            data = self.data[columns].values.astype(float)
            labels, centers = k_means_custom(data, k, initial_labels=initial_labels)
            result = f"Kết quả K-Means (k={k}):\n\n"
            for idx, (point, label) in enumerate(zip(data, labels), 1):
                result += f"Dòng {idx}: {point} → Cụm {label}\n"
            result += "\nTâm cụm cuối cùng:\n"
            for idx, center in enumerate(centers, 1):
                result += f"Cụm {idx}: {center}\n"
            self.view.display_result(result)
            if len(columns) == 2:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='deep', ax=ax)
                sns.scatterplot(x=centers[:, 0], y=centers[:, 1], color='red', marker='X', s=200, label='Tâm cụm', ax=ax)
                ax.set_title(f"K-Means (k={k})", fontsize=12, color='white')
                ax.set_facecolor('#2d2d2d')
                fig.patch.set_facecolor('#2d2d2d')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
                plt.tight_layout()
                self.view.display_plot(fig)
                plt.close(fig)
            else:
                self.view.display_plot(None)
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")

    def kohonen_som(self, columns, grid_shape=(3, 3), n_iterations=100, initial_lr=0.5, initial_sigma=1.0):
        if self.data is None:
            self.view.display_result("Lỗi: Chưa tải dữ liệu!")
            return
        if not columns:
            self.view.display_result("Lỗi: Vui lòng chọn ít nhất một cột!")
            return
        if not all(col in self.data.columns for col in columns):
            self.view.display_result("Lỗi: Một hoặc nhiều cột không tồn tại!")
            return
        if not all(pd.api.types.is_numeric_dtype(self.data[col]) for col in columns):
            self.view.display_result("Lỗi: Các cột phải là kiểu số!")
            return
        if not isinstance(grid_shape, tuple) or len(grid_shape) != 2 or not all(isinstance(x, int) and x > 0 for x in grid_shape):
            self.view.display_result("Lỗi: Kích thước lưới phải là tuple (height, width) với các số nguyên dương!")
            return
        if not isinstance(n_iterations, int) or n_iterations <= 0:
            self.view.display_result("Lỗi: Số vòng lặp phải là số nguyên dương!")
            return
        if initial_lr <= 0 or initial_sigma <= 0:
            self.view.display_result("Lỗi: Tốc độ học và sigma ban đầu phải dương!")
            return
        try:
            data = self.data[columns].values.astype(float)
            weights = kohonen_som(data, grid_shape, n_iterations, initial_lr, initial_sigma)
            grid_height, grid_width, _ = weights.shape
            result = f"Kết quả Kohonen SOM (lưới {grid_shape}, {n_iterations} vòng lặp):\n\n"
            for i in range(grid_height):
                for j in range(grid_width):
                    result += f"Nơ-ron tại ({i+1},{j+1}): {weights[i, j]}\n"
            mapping = {}
            for idx, sample in enumerate(data):
                bmu_idx = get_bmu(sample, weights)
                if bmu_idx in mapping:
                    mapping[bmu_idx].append(idx + 1)
                else:
                    mapping[bmu_idx] = [idx + 1]
            result += "\nÁnh xạ dữ liệu lên lưới SOM:\n"
            for pos in sorted(mapping):
                result += f"Nơ-ron tại ({pos[0]+1},{pos[1]+1}): {mapping[pos]}\n"
            self.view.display_result(result)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_facecolor('#2d2d2d')
            fig.patch.set_facecolor('#2d2d2d')
            for i in range(grid_height):
                for j in range(grid_width):
                    pos = (i, j)
                    y = grid_height - i - 1
                    if pos in mapping:
                        ax.scatter(j, y, s=300, color='skyblue', edgecolors='white')
                        text = ','.join(map(str, mapping[pos]))
                        ax.text(j + 0.1, y, text, fontsize=10, ha='left', va='center', color='white')
                    else:
                        ax.scatter(j, y, s=300, color='black', edgecolors='red')
            ax.set_title(f"Kohonen SOM (lưới {grid_shape})", fontsize=12, color='white')
            ax.set_xticks(range(grid_width))
            ax.set_yticks(range(grid_height))
            ax.tick_params(colors='white')
            ax.grid(True, color='white', linestyle='--', alpha=0.5)
            fig.tight_layout()
            self.view.display_plot(fig)
            plt.close(fig)
        except Exception as e:
            self.view.display_result(f"Lỗi: {str(e)}")