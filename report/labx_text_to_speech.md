# Nghiên cứu về Text-to-speech

## 1. Bối cảnh

### 1.1. Định nghĩa và động lực

Text-to-Speech (TTS), hay còn gọi là tổng hợp giọng nói từ văn bản, là lĩnh vực chuyển đổi nội dung văn bản thành đầu ra là giọng nói tổng hợp [1]. Thị trường TTS hiện nay đang tăng trưởng rất nhanh, với tốc độ tăng trưởng kép hàng năm (CAGR) dự kiến đạt khoảng 19.5% trong giai đoạn 2024–2029 [2].

Sự phát triển mạnh mẽ này đến từ nhiều yếu tố quan trọng:
* **Nhu cầu về khả năng tiếp cận (Accessibility):** Số lượng người gặp các vấn đề về học tập, thị lực và các rối loạn thần kinh ngày càng tăng, làm cho nhu cầu sử dụng TTS trở nên cấp thiết hơn [2].
* **Tích hợp AI và hệ thống thông minh:** Sự bùng nổ của các ứng dụng AI, trợ lý ảo và chatbot cũng kéo theo nhu cầu rất lớn đối với các giải pháp TTS có khả năng mở rộng tốt [2].
* **Xu hướng công nghệ:** Các xu hướng lớn hiện nay gồm cải tiến công nghệ TTS dựa trên mạng nơ-ron (Neural TTS), tích hợp yếu tố cảm xúc vào giọng nói và tăng cường hỗ trợ đa ngôn ngữ [2].

### 1.2. Khung phân loại công nghệ TTS (3 Level)

Nghiên cứu về TTS thường được phân loại dựa theo mục tiêu tối ưu, từ hiệu suất cơ bản cho tới mức độ cá nhân hóa cao [1]:

| Cấp độ | Tên gọi | Mục tiêu tối ưu | Đặc điểm kỹ thuật và nhược điểm |
|--------|---------|------------------|----------------------------------|
| **Level 1** | **Truyền thống** | Tốc độ xử lý nhanh, tính phổ quát và hỗ trợ đa ngôn ngữ [1] | Dựa trên các luật âm vị, âm tiết cơ bản. Giọng nói tạo ra thường kém tự nhiên, nghe khá “máy móc” [1]. |
| **Level 2** | **Deep Learning cổ điển** | Tăng tính tự nhiên và khả năng thích ứng người nói [1] | Sử dụng các mô hình học sâu để tạo âm thanh tự nhiên hơn, tuy nhiên yêu cầu lượng dữ liệu lớn [1]. Thường áp dụng fine-tuning cho từng người dùng, giúp mô hình tiêu tốn ít tài nguyên hơn so với Level 3 [1]. |
| **Level 3** | **Few-shot/Zero-shot Cloning** | Cá nhân hóa gần như tức thì và giảm tối đa công sức người dùng [1] | Chỉ cần vài giây âm thanh tham chiếu đã có thể sao chép đặc trưng giọng nói. Mô hình phức tạp và tiêu tốn nhiều tài nguyên tính toán hơn [1]. |

Sự khác biệt cốt lõi giữa các cấp độ này nằm ở sự đánh đổi giữa **tốc độ xử lý, tài nguyên tính toán, độ tự nhiên, khả năng đa ngôn ngữ và mức độ can thiệp của người dùng** [1].

## 2. Hướng tiếp cận TTS cấp độ 3: Neural Synthesis và tối ưu hóa kiến trúc

Các hệ thống TTS hiện đại ngày nay (ví dụ như XTTS, CosyVoice2) chủ yếu dựa trên mô hình ngôn ngữ lớn và được huấn luyện từ lượng dữ liệu nói rất lớn. Nhờ vậy, chúng đạt được chất lượng giọng nói rất tự nhiên, đặc biệt là trong bài toán nhân bản giọng nói zero-shot [3].

Hiện tại, các kiến trúc TTS cấp độ 3 thường được chia thành các nhóm chính sau [3]:

1. **Mô hình ngôn ngữ codec thần kinh (Neural Codec Language Model):**  
   Ví dụ điển hình là VALL-E, sử dụng codec đa mã sách (multi-codebook codec) để đảm bảo chất lượng âm thanh. Tuy nhiên, thời gian huấn luyện và suy luận còn khá dài, đồng thời độ ổn định chưa thực sự cao [3].

2. **TTS dựa trên Diffusion/Flow-matching End-to-End:**  
   Ví dụ như F5-TTS. Các mô hình này tạo ra âm thanh chất lượng rất cao và phù hợp cho các tác vụ chỉnh sửa giọng nói. Dù vậy, việc streaming trong thời gian thực vẫn còn gặp nhiều khó khăn [3].

3. **Kiến trúc lai (Hybrid):**  
   Thường sử dụng một mã sách đơn (single codebook) kết hợp với codec có tốc độ bit thấp, sau đó tái tạo âm thanh thông qua bộ giải mã độc lập. Kiến trúc này có sự cân bằng khá tốt giữa hiệu suất và chất lượng, đồng thời độ ổn định cũng cao hơn [3].

## 3. Thách thức đạo đức và giải pháp kỹ thuật

Khả năng nhân bản giọng nói gần như tức thì của các hệ thống Level 3 đang đặt ra nhiều vấn đề đạo đức nghiêm trọng. Đặc biệt là nguy cơ bị lạm dụng để tạo deepfake, giả mạo danh tính hoặc phát tán thông tin sai lệch [2]. Đây là một thách thức lớn và đòi hỏi phải có sự giám sát chặt chẽ từ cả góc độ pháp lý lẫn kỹ thuật [2].

Một trong những giải pháp kỹ thuật bắt buộc nhằm đối phó với rủi ro deepfake là **watermarking**. Việc nhúng watermark vào các sản phẩm giọng nói do AI tạo ra giúp đánh dấu nguồn gốc, qua đó hạn chế phần nào các hành vi lạm dụng công nghệ [1].

---

## Tài liệu

[1] Nội dung pdf được anh trợ giảng cung cấp.

[2] Research and Markets. (2024). *Text-to-Speech Market Report 2024–2029*:https://www.researchandmarkets.com/reports/5951798/text-to-speech-market-report

[3] ArXiv. (2025). *Text-to-Speech Synthesis: A Comprehensive Survey of Modern Architectures*: https://arxiv.org/html/2502.05512v1
