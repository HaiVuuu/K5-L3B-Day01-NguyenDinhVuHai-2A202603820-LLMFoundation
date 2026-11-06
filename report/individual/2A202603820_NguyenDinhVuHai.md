# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:2A202603820**
- **Họ và tên:Nguyễn Đình Vũ Hải**
- **SEED** (in ở ô đầu notebook):SEED=6446

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính
| | |f1|f1_SUB|
|1.| BoW + LogReg (tự code)|        0.906|   0.947|
|2.| BoW + MLP (deep learning)|     0.945|   0.947|
|3a.| Encoder (MiniLM) + kNN|       0.927|   0.976|
|3b.| Encoder (MiniLM) + LogReg|    0.942|   1.000|
|3c. |Encoder (MiniLM) + MLP head|  0.959|   1.000|
|4a.| LLM nhỏ zero-shot |           0.909|   0.909|
|4b.| LLM lớn zero-shot |           0.952|   0.952|
|4c.| LLM nhỏ few-shot|             0.905|   0.905|
*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
llm_fewshot: F1 tăng, chí phí chắc tăng gấp 2 đoán. Đoán : F1 sẽ tăng lên khi được cung cấp thêm các ví dụ mẫu (Few-shot). Thực tế : F1 lại giảm (từ 0.930 xuống 0.905), đồng thời thời gian xử lý chậm đi nhiều lần. Vì sao: Đưa thêm một vài ví dụ đôi khi làm giới hạn khả năng phán đoán tự do của LLM. Mô hình có thể bị "lệch tủ", học vẹt một cách máy móc theo văn phong của các ví dụ đó, hoặc bị nhiễu ngữ cảnh thay vì tận dụng kiến thức tổng quát sắc bén vốn có trong chế độ Zero-shot.

tradeoff: Chắc là cách sử dụng LLM hoặc các foudation model để tiếp cận chăng? bên phải vạch SLA thì chắc là LLM vì nó sẽ cần nhiều thời gian để chạy hơn nếu so với các cách khác. Đoán : LLM hoặc các Foundation model sẽ nằm trên đường Pareto (tức là những mô hình tối ưu nhất không bị lấn át). Thực tế : LLM nằm rất xa đường Pareto (bị lấn át hoàn toàn). Đường Pareto thuộc về các mô hình nhỏ gọn: BoW + LogReg (vô địch về tốc độ) và MiniLM + MLP (tối ưu về chất lượng). Vì sao: Vì đã đánh giá quá cao độ thông minh mà quên mất trục "độ trễ" (Latency) của đồ thị. LLM dù thông minh nhưng chạy tốn đến hơn 2000 ms (quá chậm so với vạch SLA 50 ms), khiến nó không thể cạnh tranh được sự tối ưu toàn diện của các mô hình nhỏ.

ml_f1:F1 ~ 0.85, precision cao hơn recall vì số lượng False Negative lớn hơn False Positive. Có thể là vì tỉ lệ các từ là spam có thể bị bỏ sót nhiều hơn so với các từ là spam thật vì nếu không xét theo context thì các từ spam rất dễ bị nhầm lẫn với các từ không phải spam hoặc là do mô hình quá thận trọng. Đoán: Điểm F1 (của BoW) chỉ khoảng 0.85 vì mô hình dễ bỏ sót từ hoặc nhầm lẫn do không xét đến ngữ cảnh (context). Thực tế : F1 của mô hình lên tới 0.947 (rất cao). Vì sao: dự đoán đúng về điểm yếu cốt lõi của BoW (không hiểu ngữ cảnh), nhưng lại đánh giá quá thấp sự "dễ dãi" của tập dữ liệu test thông thường. Trong tập test thường, spam phần lớn dùng các từ khóa chèo kéo rất lộ liễu, nên BoW chỉ cần đếm từ vựng (học vẹt) là đã đủ đạt điểm F1 cực kỳ lạc quan.
## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
Model tụt nhiều nhất: BoW + LogReg (điểm F1 rơi tự do từ 0.947 trên tập test thường xuống chỉ còn 0.286 trên tập 50 case khó).

Ở nhóm case nào: Tụt thê thảm nhất ở nhóm "Obfuscation" (Làm nhiễu chữ / Né từ khóa) và "Natural spam" (Spam giọng tự nhiên).

2 case cụ thể và cơ chế khiến model sai:
Case 1 (Nhóm Obfuscation): "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4r..." (Nhãn thật: spam, nhưng BoW đoán là ham)
Cơ chế sai: Mô hình Bag-of-Words (BoW) dựa hoàn toàn vào việc khớp chính xác chuỗi ký tự (token). Khi kẻ tấn công dùng thủ thuật leetspeak (thay chữ cái bằng số như o thành 0, i thành 1), các từ khóa mang tính quyết định ("Congratulations", "gift card") biến thành các từ vựng hoàn toàn mới chưa từng xuất hiện trong tập huấn luyện (Out-of-Vocabulary). BoW bị "mù", không tìm thấy dấu hiệu từ khóa rác nào nên để lọt.

Case 2 (Nhóm Natural spam): "Hey it's Jess from the gym 😊 we're doing a mem..." (Nhãn thật: spam, nhưng BoW đoán là ham)
Cơ chế sai: Tin nhắn được ngụy trang bằng văn phong giao tiếp cực kỳ tự nhiên, thân thiện và không hề chứa các từ khóa chèo kéo, khẩn cấp kinh điển (như URGENT, FREE, CLICK HERE). Thay vì hiểu được ý đồ ẩn sau câu chữ, BoW chỉ bóc tách ra các từ ngữ cảnh đời thường ("Hey", "gym") vốn có trọng số cao nghiêng về ham, dẫn đến việc mô hình tự tin chấm đây là tin nhắn an toàn.


## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
Đề #3: DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?
ới lớp ẩn 4 nơ-ron:   
Số tham số: 8,009 
F1 test: 0.945   
Thời gian train: 7.2 giây   


ới lớp ẩn 512 nơ-ron:  
Số tham số: 1,025,025 (hơn 1 triệu)  
F1 test: 0.945  
Thời gian train: 46.2 giây  
nên chọn mạng nhỏ (4 nơ-ron) trong bài toán này vì tránh Overfitting nhìn vào biểu đồ loss, mạng 512 nơ-ron bị overfit cực kỳ nặng và rất sớm. Loss trên tập validation (val) chạm đáy ngay tại epoch số 1, sau đó tăng vọt lên liên tục trong khi loss tập train giảm về gần 0. Mạng 4 nơ-ron thì ngược lại, học rất ổn định, đường val loss bám sát train loss và từ từ giảm.   Chất lượng không đổi: Dù nhồi nhét số tham số gấp hơn 120 lần (từ ~8 nghìn lên hơn 1 triệu), điểm số phân loại thực tế trên tập Test vẫn dậm chân tại chỗ (F1 = 0.945). Điều này cho thấy mạng lớn không mang lại giá trị gia tăng nào về chất lượng.Tiết kiệm tài nguyên: Mạng nhỏ train nhanh hơn hơn 6 lần (7.2s so với 46.2s) và tốn ít dung lượng RAM hơn hẳn.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
1.Đạt yêu cầu độ trễ (p95 < 50 ms): Các mô hình học máy gọn nhẹ (BoW + LogReg, BoW + MLP) và mô hình nhúng (MiniLM + LogReg/MLP). Toàn bộ nhóm LLM bị loại vì p95 lên tới 2000-3000 ms. Đạt cả độ trễ lẫn chất lượng: MiniLM + MLP head (hoặc LogReg). Nó xử lý tốt bài toán ngữ nghĩa, chống chịu bẫy ổn hơn BoW rất nhiều, trong khi p95 (~28 ms) vẫn nằm an toàn dưới vạch 50 ms.
Ước tính chi phí LLM/tháng:
1 triệu SMS/ngày tốn $10.50 (từ phép tính tay trước đó: 100M token input = $10, 1M token output = $0.5).
10 triệu SMS/ngày = $105/ngày.
Chi phí 1 tháng (30 ngày) = $3.150/tháng (hơn 78 triệu VNĐ chỉ tiền gọi API, chưa tính việc nó sẽ làm nghẽn toàn bộ hệ thống vì quá chậm).
2.Thiết kế lai (Router)
Đề xuất kiến trúc:Lớp 1 (Vệ sĩ - MiniLM + MLP): Chạy nội bộ, nhận toàn bộ 10 triệu tin nhắn. Nếu xác suất p ~ 0.1 (rất chắc chắn là Ham) hoặc p ~ 0.9 (rất chắc chắn là Spam), chốt luôn kết quả.Lớp 2 (Chuyên gia - LLM): Nếu tin nhắn rơi vào vùng lưỡng lự 0.1 < p < 0.9 (thường là các ca khó như prompt injection, obfuscation), hệ thống mới gọi API lên LLM để đọc hiểu chuyên sâu.
3.Điều dự đoán sai nhất: Tư duy "Cứ to là tốt". Ban đầu, mình nghĩ LLM 120B sẽ đè bẹp mọi thứ để chiếm đỉnh Pareto, và việc thêm ví dụ (Few-shot) chắc chắn làm LLM thông minh hơn. Kết quả thực tế thì khác LLM văng khỏi đường Pareto vì độ trễ thảm họa, và Few-shot lại làm tụt điểm F1 do gây nhiễu ngữ cảnh.
Sự thay đổi hiểu biết: Nhận ra rằng trong kỹ thuật thực tế (Production), điểm số (Accuracy/F1) chỉ là điều kiện cần, Độ trễ (Latency) và Tiền (Cost) mới là điều kiện đủ. Một thuật toán "cổ đại" như đếm từ vựng (BoW) hoàn toàn có thể đánh bại siêu AI tỷ tham số trên bàn cân kỹ thuật nếu nó giải quyết được bài toán thời gian thực của doanh nghiệp.
## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
AI log (bắt buộc nếu bạn dùng AI ở bất kỳ đâu)
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
|Thí nghiệm 4: LLM Zero-shot / Few-shot |Giải thích vì sao Few-shot làm F1 giảm; vì sao cần temperature=0 và max_tokens nhỏ; lợi ích của Zero-shot khi chưa có nhãn. |Giải thích: few-shot dễ gây nhiễu/lệch tủ, temp=0 giúp chống ảo giác, Zero-shot hữu ích nhất cho bài toán "Cold start" (khởi tạo lạnh). |Đọc lại slide bài giảng về temperature và max_tokens để xác nhận logic, đối chiếu với lý thuyết Cold start. |
|Phân tích Pareto (Trade-off)|Đọc hình vẽ đường Pareto để xác định cách nào bị lấn át, cách nào đạt SLA < 50ms, tính mức thay đổi điểm do tập test chỉ có 40 tin.|Xác định BoW+LogReg siêu nhanh, MiniLM+MLP cân bằng tốt nhất. Tính được 1/40 = 2.5%, cho thấy độ tin cậy thấp do nhiễu.|Tự nhìn hình vẽ để xác minh các chấm trên đường Pareto và nhẩm lại công thức 1/40 thấy hợp lý.|
|Phân tích 50 case khó (Bẫy mô hình)|Điểm mù khó nhất của BoW và LLM là gì? Đánh giá prompt phòng thủ có hiệu quả không? Viết 5 case mới (MY_CASES) để lừa model.|BoW sợ từ biến dị (leetspeak), LLM sợ prompt injection. Phân tích prompt phòng thủ chặn được injection nhưng làm hỏng case nhận quà bình thường. Gợi ý 5 case mới.|Chạy thử 5 case AI gợi ý vào đoạn code MY_CASES trên notebook để xem model có thực sự "sập bẫy" hay không|
|Tính toán tay (HAND_CALC)|Tính baseline_accuracy (với 5169 tin, 4516 ham), 0, F1, độ tương tự cosine, và chi phí API cho 1 triệu tin nhắn.|Trả về kết quả làm tròn 2 chữ số thập phân: Baseline 0.87, Sigmoid 0.50, F1 0.73, Cosine 0.60, Chi phí 10.5/ngày.|Đã tự bấm máy tính lại các phép toán (công thức F1, nhân vô hướng vector) và kiểm chứng khớp với đáp án AI.|
|Thiết kế hệ thống thực tế|Chọn cách cho lưu lượng 10 triệu tin/ngày (SLA 50ms). Đề xuất mô hình lai (router) nhiều lớp và rút ra bài học dự đoán sai nhất.|Chọn MiniLM+MLP. Đề xuất Router: MiniLM xử lý 90% tin rõ ràng, LLM chỉ gọi khi tin "lưỡng lự" (0.1 < p < 0.9). Nhận ra Latency và Cost quan trọng hơn chỉ nhìn vào Accuracy.|Tham khảo kiến trúc thiết kế Router trong slide và đánh giá tính thực tiễn về mặt tối ưu chi phí hạ tầng.|
