# Ứng dụng Phân Tích Dữ Liệu và Học Máy

Đây là một ứng dụng dựa trên Python triển khai các kỹ thuật phân tích dữ liệu và học máy sử dụng kiến trúc Model-View-Controller (MVC). Ứng dụng cung cấp giao diện đồ họa tương tác (GUI) để người dùng thực hiện phân tích dữ liệu khám phá, phân cụm, phân loại và khai thác quy tắc kết hợp từ các tập dữ liệu.

## Các Tính Năng

- **Phân Tích Tương Quan**: Tính hệ số tương quan Pearson giữa hai cột số và trực quan hóa bằng biểu đồ phân tán.
- **Chuẩn Hóa Min-Max**: Chuẩn hóa một giá trị hoặc tập dữ liệu về một khoảng xác định (mặc định từ 0 đến 1).
- **Apriori & Quy Tắc Kết Hợp**: Phát hiện tập phổ biến và tạo quy tắc kết hợp từ dữ liệu giao dịch.
- **Phân Tích Tập Thô**: Tính tập xấp xỉ dưới và trên, độ chính xác, hệ số phụ thuộc và các tập rút gọn.
- **Cây Quyết Định**: Xây dựng và trực quan hóa cây quyết định sử dụng tiêu chí Gain hoặc Gini, kèm trích xuất quy tắc.
- **Naive Bayes**: Thực hiện phân loại với hoặc không sử dụng làm trơn Laplace.
- **Phân Cụm K-means**: Phân cụm dữ liệu số với số cụm tùy chỉnh và nhãn ban đầu, kèm trực quan hóa 2D.
- **Kohonen SOM**: Thực hiện Self-Organizing Maps để phân cụm và trực quan hóa dữ liệu đa chiều.

## Cấu trúc
main.py: Điểm khởi chạy của ứng dụng.
data_view.py: Triển khai giao diện GUI sử dụng tkinter (lớp View).
data_controller.py: Logic điều phối giữa View và Model (lớp Controller).
model/: Chứa các triển khai của lớp Model:
correlation.py: Tính toán tương quan.
normalization.py: Chuẩn hóa Min-Max.
association.py: Thuật toán Apriori và quy tắc kết hợp.
rough_set.py: Tính toán tập thô.
decision_tree.py: Thuật toán cây quyết định.
bayes.py: Bộ phân loại Naive Bayes.
k_means.py: Thuật toán phân cụm K-means.
kohonen.py: Triển khai Kohonen SOM.

## Cài đặt các thư viện cần thiết
pip install -r requirements.txt

## Chạy ứng dụng
python main.py