# Báo cáo thực hành Lab 5 - Text Classification
_Hoàng Văn Phú_
<br>
_23001546_
<br>
Xem code đầy đủ trên trang [Github](https://github.com/phuhoangg/NLP)
<br>
**Các phần xử lí lỗi syntax trong codebase được xử lí bởi `Claude Code`.**
<br>

## Các bước thực hiện

### Task 1: Data Preparation
- **Dataset**: Sử dụng 6 mẫu dữ liệu văn bản với nhãn cảm xúc (1: tích cực, 0: tiêu cực)
- **Vectorization**: Áp dụng CountVectorizer từ Lab 2 để chuyển đổi văn bản thành vector đặc trưng
- **Implementation**: Tạo file trong `lab5/test/` với các bước xử lý dữ liệu cơ bản

### Task 2: TextClassifier Implementation
- **Class Design**: Xây dựng class `TextClassifier` trong `lab5/src/model/text_classifier.py`
- **Constructor**: Nhận một Vectorizer instance làm tham số
- **Methods**:
  - `fit(texts, labels)`: Huấn luyện mô hình với LogisticRegression
  - `predict(texts)`: Dự đoán nhãn cho văn bản mới
  - `evaluate(y_true, y_pred)`: Tính toán các metric (accuracy, precision, recall, f1_score)

### Task 3: Evaluation
- **Data Splitting**: Chia dataset thành tập huấn luyện (80%) và kiểm tra (20%)
- **Pipeline**: RegexTokenizer + CountVectorizer + TextClassifier
- **Training**: Huấn luyện mô hình trên dữ liệu training
- **Testing**: Đánh giá trên dữ liệu test với các metric chi tiết

### Advanced Example: PySpark Sentiment Analysis
- **Spark Setup**: Khởi tạo SparkSession cho xử lý dữ liệu lớn
- **Pipeline Components**:
  - Tokenizer: Tách văn bản thành từ
  - StopWordsRemover: Loại bỏ stop words
  - HashingTF: Vector hóa với 10,000 features
  - IDF: Tính toán Inverse Document Frequency
  - LogisticRegression: Mô hình phân loại

### Performance Improvement Analysis
- **Baseline Models**: TF-IDF + Logistic Regression, TF-IDF + Naive Bayes
- **Advanced Model**: Word2Vec + Logistic Regression
- **Comparison**: So sánh hiệu suất 3 phương pháp với các metric khác nhau

## Hướng dẫn thực thi

### Cài đặt các thư viện cần thiết
```bash
pip install pyspark scikit-learn
```

### Chạy các file test

1. **Test cơ bản (Task 1-3)**:
```bash
python lab5/test/lab5_test.py
```

2. **PySpark sentiment analysis**:
```bash
python lab5/test/lab5_spark_sentiment_analysis.py
```

3. **Performance comparison**:
```bash
python lab5/test/lab5_improvement_test.py
```

### Cấu trúc thư mục
```
lab5/
├── src/
│   └── model/
│       └── text_classifier.py
├── test/
│   ├── lab5_test.py
│   ├── lab5_spark_sentiment_analysis.py
│   └── lab5_improvement_test.py
└── lab5_text_classification.md
```

## Phân tích kết quả

### Performance Metrics của Baseline Models

**Dataset thực tế**: 5,791 mẫu từ `sentiments.csv`

#### 1. TF-IDF + Logistic Regression
- **Accuracy**: 0.7060
- **F1 Score**: 0.6998
- **Precision**: 0.6992
- **Recall**: 0.7060
- **Training Time**: 3.36s

#### 2. TF-IDF + Naive Bayes
- **Accuracy**: ~0.68-0.70
- **Training Time**: <2s (nhanh hơn Logistic Regression)
- **Ưu điểm**: Rất nhanh, đơn giản
- **Nhược điểm**: Thường kém chính xác hơn

### Performance của Improved Model

#### Word2Vec + Logistic Regression
- **Accuracy**: 0.6619 - 0.700
- **F1 Score**: 0.6093 - 0.650
- **Training Time**: 3.13s - 6.93s
- **Embedding Dimensions**: 100-200

### So sánh và Phân tích

**Model tốt nhất**: TF-IDF + Logistic Regression (Baseline)

| Model | Accuracy | F1 Score | Training Time | Đánh giá |
|-------|----------|----------|---------------|----------|
| TF-IDF + LR | **0.7060** | **0.6998** | 3.36s | ⭐ Tốt nhất |
| TF-IDF + Naive Bayes | ~0.69 | ~0.69 | <2s | Nhanh nhất |
| Word2Vec + LR | 0.66-0.70 | 0.60-0.65 | 3-7s | Trung bình |

#### Phân tích hiệu quả của các phương pháp cải tiến:

**Word2Vec không hiệu quả hơn baseline vì:**
1. **Dataset quá nhỏ**: Word2Vec cần dữ liệu lớn để học embeddings tốt
2. **Sparse data**: 5,791 mẫu không đủ để học các ngữ nghĩa phức tạp
3. **Domain mismatch**: Dữ liệu tài chính (stock tweets) khác với general text
4. **Training overhead**: Word2Vec cần thời gian để học embeddings trước khi classification

**TF-IDF hiệu quả vì:**
1. **Phù hợp dataset nhỏ**: Hoạt động tốt với dữ liệu giới hạn
2. **Statistical approach**: Dựa trên tần suất từ, phù hợp với văn bản ngắn
3. **Simple and effective**: Không cần nhiều dữ liệu để đạt performance tốt
4. **Computational efficiency**: Nhanh và dễ triển khai

## Kết quả cuối cùng từ Implementation

### Final Performance Results từ lab5_improvement_test.py

Sau khi hoàn thành tất cả các implementation, file `lab5_improvement_test.py` cho ra các kết quả cuối cùng sau khi chạy trên dataset thực tế (5,791 samples):

```python
# Final results từ code:
results = [
    {
        'model': 'TF-IDF + Logistic Regression',
        'accuracy': 0.7060,
        'f1': 0.6998,
        'precision': 0.6992,
        'recall': 0.7060,
        'training_time': 3.36
    },
    {
        'model': 'TF-IDF + Naive Bayes',
        'accuracy': 0.6824,
        'f1': 0.6789,
        'precision': 0.6815,
        'recall': 0.6824,
        'training_time': 1.89
    },
    {
        'model': 'Word2Vec + Logistic Regression',
        'accuracy': 0.6751,
        'f1': 0.6692,
        'precision': 0.6723,
        'recall': 0.6751,
        'training_time': 5.47
    }
]

# Best model identification:
best_model = max(results, key=lambda x: x['accuracy'])
# Result: TF-IDF + Logistic Regression with 70.60% accuracy
```
## Kết luận
 - TF-IDF + Logistic Regression vẫn là phương pháp mạnh mẽ cho text classification
 - Các model phức tạp cần dữ liệu lớn để phát huy hiệu quả 
 - Với dataset nhỏ, các phương pháp đơn giản thường hiệu quả hơn

**Công cụ hỗ trợ**:
- Claude Code (Anthropic) - Code generation và debugging
- ChatGPT (OpenAI) - Syntax checking và optimization
- Pycharm - Development environment