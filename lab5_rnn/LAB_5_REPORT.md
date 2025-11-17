# Báo cáo Thực hành Lab 5 - RNNs
_Hoàng Văn Phú_
_23001546_

## Tổng quan

Lab 5 bao gồm 3 phần chính:
1. **PyTorch Introduction**: Làm quen với PyTorch framework
2. **Part-of-Speech (PoS) Tagging**: Xây dựng mô hình RNN cho gán thẻ từ loại
3. **Text Classification**: Xây dựng các mô hình khác nhau cho phân loại intent

---

## Phần 1: PyTorch Introduction

### Nội dung
Phần này giới thiệu các khái niệm cơ bản của PyTorch:
- **Tensor operations**: Tạo và thao tác với tensor
- **Autograd**: Tự động tính đạo hàm cho neural networks
- **Neural Network Module**: Xây dựng mô hình đầu tiên với `torch.nn`

### Kết quả chính
- Tạo tensor từ list và NumPy array
- Hiểu được cơ chế backward propagation
- Xây dựng được mô hình `MyFirstModel` với các lớp: Embedding, Linear, Activation

### Phân tích
Phần này là nền tảng quan trọng cho các phần sau. Việc hiểu rõ PyTorch giúp triển khai các mô hình RNN phức tạp hơn một cách hiệu quả.

---

## Phần 2: Part-of-Speech Tagging với RNN

### Phương pháp
- **Mô hình**: Simple RNN cho token classification
- **Architecture**: Embedding → RNN → Linear → Softmax
- **Dataset**: Universal Dependencies English EWT corpus
- **Hyperparameters**:
  - Embedding dim: 64
  - Hidden dim: 128
  - Learning rate: 0.001
  - Batch size: 32
  - Epochs: 20 (với early stopping)

### Kết quả định lượng

| Metric | Training | Validation |
|--------|----------|------------|
| Final Accuracy | 94.44% | 87.65% |
| Best Val Accuracy | - | 87.65% |
| Total Parameters | 1,320,211 | - |

### Kết quả F1-score theo tag

| Tag | Precision | Recall | F1-score |
|-----|-----------|--------|----------|
| PUNCT | 0.99 | 0.99 | 0.99 |
| DET | 0.97 | 0.95 | 0.96 |
| PRON | 0.95 | 0.97 | 0.96 |
| AUX | 0.95 | 0.94 | 0.95 |
| VERB | 0.90 | 0.87 | 0.88 |
| NOUN | 0.72 | 0.94 | 0.82 |
| PROPN | 0.85 | 0.55 | 0.67 |

### Phân tích lỗi
Các lỗi phổ biến nhất:
1. **PROPN → NOUN**: Proper noun bị nhầm với common noun
2. **X → PUNCT**: Foreign words bị nhận diện nhầm thành dấu câu
3. **SCONJ → ADP**: Subordinating conjunction bị nhầm với adposition

### Ví dụ phân tích định tính

**Câu ví dụ 1**: "The cat sits on the mat"
```
The          | DET (1.000)
cat          | NOUN (0.998)
sits         | NOUN (0.718)  ← Lỗi:VERB
on           | ADP (0.948)
the          | DET (1.000)
mat          | NOUN (0.879)
```

**Phân tích**: Mô hình gặp khó khăn trong việc phân biệt VERB/NOUN cho các từ có thể đóng vai trò cả hai (như "sits"). Điều này cho thấy RNN đơn giản chưa đủ để nắm bắt ngữ cảnh phức tạp.

---

## Phần 3: Text Classification so sánh nhiều phương pháp

### Các phương pháp so sánh

#### 1. TF-IDF + Logistic Regression
- **Approach**: Bag-of-words với TF-IDF weighting
- **Classifier**: Logistic Regression
- **Features**: 5000 most frequent terms

#### 2. Word2Vec (Average) + Dense Network
- **Approach**: Average Word2Vec vectors
- **Architecture**: Dense layers với BatchNormalization
- **Word2Vec**: Trained on dataset, vector_size=100

#### 3. Pre-trained Embedding + LSTM
- **Approach**: Word2Vec embedding frozen + LSTM
- **Architecture**: Embedding → LSTM → Dense
- **Coverage**: 20.42% words có trong Word2Vec

#### 4. Scratch Embedding + LSTM
- **Approach**: Learn embeddings from scratch
- **Architecture**: Bidirectional LSTM với attention
- **Embedding dim**: 200

### Bảng so sánh kết quả định lượng

| Pipeline | F1-score (Macro) | Test Loss | Test Accuracy |
|----------|------------------|-----------|----------------|
| TF-IDF + Logistic Regression | 0.918 | N/A | 0.94 |
| Word2Vec (Avg) + Dense | 0.303 | 2.539 | 0.31 |
| Embedding (Pre-trained) + LSTM | 0.036 | 3.480 | 0.10 |
| **Embedding (Scratch) + LSTM** | **0.971** | **0.036** | **0.99** |

### Phân tích kết quả

#### 1. TF-IDF + Logistic Regression (F1: 0.918)
**Ưu điểm:**
- Đơn giản, hiệu quả
- Huấn luyện nhanh
- Chiếm ít tài nguyên

**Nhược điểm:**
- Không nắm được thứ tự từ
- Không xử lý được out-of-vocabulary words

#### 2. Word2Vec Average + Dense (F1: 0.303)
**Ưu điểm:**
- Nắm được nghĩa ngữ nghĩa của từ

**Nhược điểm:**
- Mất thông tin thứ tự
- Kết quả trung bình làm mất specificity

**Khó khăn gặp phải:**
- Dataset quá nhỏ để train Word2Vec hiệu quả
- Vocabulary size lớn nhưng tần suất xuất hiện thấp

#### 3. Pre-trained Embedding + LSTM (F1: 0.036)
**Ưu điểm:**
- Lợi thế từ pre-trained knowledge

**Nhược điểm:**
- Coverage thấp (20.42%)
- Embedding bị frozen, không adapt được với domain
- Architecture quá đơn giản

**Khó khăn gặp phải:**
- Pre-trained Word2Vec không match với domain-specific vocabulary
- Low coverage khiến model hoạt động kém

#### 4. Scratch Embedding + LSTM (F1: 0.971) - TỐT NHẤT
**Ưu điểm:**
- Learn embeddings phù hợp với task
- Bidirectional LSTM nắm được context cả hai chiều
- Architecture sâu với regularization hiệu quả

**Khó khăn gặp phải:**
- Huấn luyện lâu
- Cần nhiều dữ liệu

### Phân tích định tính cho các câu ví dụ

#### Câu 1: "show contact of john" (Intent: email_querycontact)
- **TF-IDF + LR**:  Đúng - "contact", "john" là features mạnh
- **Word2Vec + Dense**:  Sai - recommendation_locations
- **LSTM Pre-trained**:  Sai - lists_remove
- **LSTM Scratch**:  Đúng - Nắm được dependency từ xa

**Phân tích**: LSTM scratch hoạt động tốt vì có thể học được patterns như "show [action] of [person]" → email context.

#### Câu 2: "remind me at six thirty pm" (Intent: alarm_set)
- **TF-IDF + LR**:  Sai - calendar_set
- **Word2Vec + Dense**:  Đúng
- **LSTM Pre-trained**:  Đúng
- **LSTM Scratch**:  Đúng

**Phân tích**: "remind", "time expressions" là indicators mạnh cho alarm intent.

#### Câu 3: "turn off my wemo socket" (Intent: iot_wemo_off)
- **TF-IDF + LR**:  Đúng - "wemo", "socket", "turn off" là keywords unique
- **Word2Vec + Dense**:  Đúng
- **LSTM Pre-trained**:  Sai - iot_hue_lightoff
- **LSTM Scratch**:  Đúng

**Phân tích**: Pre-trained model không recognize "wemo" brand name, trong khi scratch embedding học được từ vocabulary.

### Phân tích tại sao LSTM hoạt động tốt hơn

1. **Long-range dependencies**: LSTM có thể nắm được mối quan hệ giữa các từ ở xa nhau
   - Ví dụ: "show" ở đầu câu với "contact" ở cuối câu

2. **Context awareness**: Bidirectional LSTM hiểu context cả trước và sau
   - Ví dụ: "water is essential" - LSTM hiểu "water" là subject

3. **Domain-specific embeddings**: Scratch embeddings học được vocabulary đặc thù
   - Ví dụ: "wemo", "hue" - device names không có trong general Word2Vec

4. **Sequence modeling**: LSTM hiểu được grammar và syntax patterns
   - Ví dụ: "turn [action] [device]" structure

### Nhận xét chung và khó khăn

#### Khó khăn gặp phải:
1. **Data preprocessing**: Tokenization và vocabulary building phức tạp
2. **Class imbalance**: Một số intents có rất few examples
3. **Hyperparameter tuning**: Tìm optimal architecture tốn nhiều thời gian
4. **Computational resources**: LSTM models đòi hỏi GPU cho efficient training

#### Lessons learned:
1. **Domain-specific embeddings** thường tốt hơn pre-trained cho narrow domains
2. **Simple baselines** như TF-IDF+LR vẫn rất competitive
3. **Architecture design** quan trọng hơn model complexity
4. **Early stopping** và regularization là essential để prevent overfitting

#### Recommendations:
1. Start với simple baseline trước khi thử complex models
2. Experiment với different embedding strategies
3. Use data augmentation cho rare classes
4. Consider ensemble methods cho production systems

---

## Tổng kết

Lab 5 đã cung cấp trải nghiệm thực hành toàn diện với RNNs trong NLP, từ:
- **PyTorch fundamentals** → **Token classification** → **Text classification**

Mô hình LSTM với scratch embeddings đạt kết quả tốt nhất (F1: 0.971) vì có khả năng:
- Nắm bắt long-range dependencies
- Học domain-specific representations
- Model sequential nature của language effectively

 Tuy nhiên, simple TF-IDF + Logistic Regression vẫn là một strong baseline đáng cân nhắc cho nhiều practical applications.

Xem code đầy đủ trên [Github](https://github.com/phuhoangg/NLP)