# Xử lý Ngôn ngữ Tự nhiên

## Thông tin sinh viên
- **Họ và tên:** Hoàng Văn Phú
- **Mã sinh viên:** 23001546
- **Học kỳ:** HK1 (2025-2026)
- **Môn học:** Xử lý ngôn ngữ tự nhiên và Học sâu

## Tổng quan
Dự án này bao gồm các bài tập thực hành môn Xử lý ngôn ngữ tự nhiên và Học sâu, bao gồm các labs  từ cơ bản đến nâng cao:

### 📁 Project Structure
```
NLP/
├── lab1/                    # Lab 1: Tokenization
│   ├── src/
│   │   └── preprocessing/
│   │       ├── __init__.py
│   │       ├── simple_tokenizer.py
│   │       └── regex_tokenizer.py
│   └── LAB_1.md
├── lab2/                    # Lab 2: Text Representation
│   ├── src/
│   │   ├── representations/
│   │   │   ├── __init__.py
│   │   │   └── count_vectorizer.py
│   │   └── test/
│   │       └── lab2_test.py
│   └── LAB_2.md
├── lab3/                    # Lab 3: Advanced Text Representation
│   ├── src/
│   │   └── representations/
│   │       └── tfidf_vectorizer.py
├── lab4/                    # Lab 4: Word Embeddings
│   ├── src/
│   └── LAB4.md
├── lab5/                    # Lab 5: Text Classification
│   ├── src/
│   │   └── model/
│   │       └── text_classifier.py
│   ├── test/
│   │   ├── lab5_test.py
│   │   ├── lab5_spark_sentiment_analysis.py
│   │   └── lab5_improvement_test.py
│   └── lab5_text_classification.md
├── data/                    # Datasets
│   └── sentiments.csv
└── README.md
```

## 📚 Chi tiết các Lab

### Lab 1: Tokenization
- **Mục tiêu:** Triển khai các thuật toán tokenization khác nhau
- **Thực hiện:**
  - `SimpleTokenizer`: Sử dụng logic vòng lặp cơ bản
  - `RegexTokenizer`: Sử dụng biểu thức chính quy cho flexibility cao hơn
- **Kết quả:** Thành công tách các văn bản thành tokens, xử lý punctuation và special characters

### Lab 2: Text Representation
- **Mục tiêu:** Chuyển đổi văn bản thành vector đặc trưng
- **Thực hiện:**
  - `CountVectorizer`: Tạo vocabulary và document-term matrix
  - TF-IDF vectorization
- **Kết quả:** Vector hóa thành công với vocabulary size tùy chỉnh

### Lab 3: Advanced Text Representation
- **Mục tiêu:** Implement TF-IDF vectorization nâng cao
- **Thực hiện:**
  - `TfidfVectorizer`: Custom implementation
  - IDF calculation và normalization
- **Kết quả:** Advanced text representation với TF-IDF weights

### Lab 4: Word Embeddings
- **Mục tiêu:** Triển khai và sử dụng word embeddings
- **Thực hiện:**
  - Gensim cho pre-trained models
  - Sentence/document embeddings
  - Visualization với PCA
  - PySpark implementation cho large datasets
- **Kết quả:** Thành công sử dụng word embeddings cho semantic similarity

### Lab 5: Text Classification
- **Mục tiêu:** Xây dựng complete text classification system
- **Thực hiện:**
  - **Task 1:** Data preparation với CountVectorizer
  - **Task 2:** TextClassifier class với Logistic Regression
  - **Task 3:** Evaluation với train/test split
  - **Advanced:** PySpark sentiment analysis cho large datasets
  - **Performance Analysis:** So sánh 3 methods (TF-IDF+LR, TF-IDF+Naive Bayes, Word2Vec+LR)
- **Kết quả:** 70.6% accuracy với TF-IDF + Logistic Regression

## 🛠️ Installation & Setup

### Yêu cầu hệ thống
- Python 3.8+
- Git
- IDE (PyCharm/VS Code recommended)

### Cài đặt dependencies
```bash
# Basic dependencies
pip install numpy pandas scikit-learn

# Advanced dependencies cho Lab 5
pip install pyspark

# Word embeddings cho Lab 4
pip install gensim

# Visualization
pip install matplotlib seaborn
```

### Environment setup
```bash
# Clone repository
git clone <repository-url>
cd NLP

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage Examples

### Chạy Lab 1: Tokenization
```bash
cd lab1/src/preprocessing/
python simple_tokenizer.py
python regex_tokenizer.py
```

### Chạy Lab 2: Text Representation
```bash
cd lab2/src/test/
python lab2_test.py
```

### Chạy Lab 5: Text Classification
```bash
# Test cơ bản
python lab5/test/lab5_test.py

# PySpark analysis
python lab5/test/lab5_spark_sentiment_analysis.py

# Performance comparison
python lab5/test/lab5_improvement_test.py
```

## Development Tools

**Công cụ hỗ trợ:**
- **Claude Code (Anthropic)** - Code generation và debugging
- **ChatGPT (OpenAI)** - Syntax checking và optimization
- **PyCharm/VS Code** - Development environment
- **Git/GitHub** - Version control

## 📄 License

This project is for educational purposes as part of the NLP course.

---

**Note:** Các phần xử lý lỗi syntax và optimization được thực hiện với sự hỗ trợ của AI tools, logic chính và implementation do sinh viên tự thực hiện.