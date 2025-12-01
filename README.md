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
├── lab5_rnn/                # Lab 5: Recurrent Neural Networks
│   ├── part_1/              # PyTorch Introduction
│   │   └── lab5_pytorch_introduction.ipynb
│   ├── part_2/              # RNN Text Classification
│   ├── part_3/              # LSTM Text Classification
│   └── part_4/              # Advanced RNN Techniques
├── lab6/                    # Lab 6: Transformers
│   └── lab6_intro_transformers.ipynb
├── report/                  # All lab reports and documentation
│   ├── lab_1.md            # Lab 1 report
│   ├── lab_2.md            # Lab 2 report
│   ├── lab_4.md            # Lab 4 report
│   ├── lab_4_visualize_embedding.pdf  # Visualization results
│   ├── lab_5_text_classification.md  # Text classification report
│   ├── lab_5_rnns.md       # RNN implementation report
│   └── lab6_intro_transformers.pdf  # Transformers notebook
├── data/                    # Datasets and metadata
│   ├── c4-train.*.json     # Common Crawl dataset for language modeling
│   ├── sentiments.csv      # Sentiment analysis dataset
│   └── UD_English-EWT/     # Universal Dependencies English dataset
│       ├── *.conllu       # Annotated text data
│       ├── *.txt          # Plain text versions
│       ├── LICENSE.txt    # Dataset license
│       └── README.md      # Dataset structure documentation
├── .gitignore             # Git ignore rules for data files
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

### Lab 5: Recurrent Neural Networks (RNNs)
- **Mục tiêu:** Implement RNN architectures cho NLP tasks
- **Thực hiện:**
  - **Part 1:** PyTorch foundation và tensor operations
  - **Part 2:** Basic RNN implementation cho text classification
  - **Part 3:** LSTM và GRU models cho long sequences
  - **Part 4:** Advanced RNN techniques và optimizations
- **Công nghệ:** PyTorch, TensorFlow, Keras, Gensim
- **Applications:** Text classification, sequence modeling, sentiment analysis

### Lab 6: Transformers
- **Mục tiêu:** Introduction to Transformer architecture và pre-trained models
- **Thực hiện:**
  - **Task 1:** Masked Language Modeling với BERT (fill-mask)
  - **Task 2:** Text generation với GPT models
  - **Task 3:** Sentence embeddings và Mean Pooling với BERT
  - **Advanced:** Attention mechanisms và transformer applications
- **Công nghệ:** Hugging Face Transformers, BERT, GPT, DistilBERT
- **Kết quả:** Thành công sử dụng pre-trained models cho các NLP tasks
  - BERT fill-mask: "capital" predicted with 40.33% confidence
  - Sentence embeddings: 768-dimensional vectors from BERT

## 🛠️ Installation & Setup

### Yêu cầu hệ thống
- Python 3.10+ (for Lab 5 RNNs and Lab 6 Transformers)
- GPU (khuyến khích cho PyTorch/TensorFlow training)
- VRAM: Tối thiểu 4GB cho deep learning models
- Git
- IDE (PyCharm/VS Code/Jupyter recommended)

### Cài đặt dependencies
```bash
# Basic dependencies
pip install numpy pandas scikit-learn

# Lab 1-3: Basic NLP
pip install matplotlib seaborn

# Lab 4: Word embeddings
pip install gensim

# Lab 5: PySpark cho large datasets
pip install pyspark

# Lab 5 RNNs: Deep learning frameworks
pip install torch torchvision torchaudio
pip install tensorflow tensorflow-addons

# Lab 6: Transformers
pip install transformers datasets tokenizers

# Jupyter notebooks cho interactive labs
pip install jupyter notebook ipykernel
```

### Environment setup
```bash
# Clone repository
git clone <repository-url>
cd NLP

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies theo từng lab hoặc install tất cả
pip install -r requirements.txt  # If available, hoặc install theo batch ở trên

# Đối với Colab users: Các dependencies đã được cài đặt sẵn
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

### Chạy Lab 5 RNNs:
```bash
# Part 1: PyTorch Introduction
jupyter notebook lab5_rnn/part_1/lab5_pytorch_introduction.ipynb

# Part 2-4: RNN models
# Các notebook khác trong thư mục lab5_rnn/part_2, part_3, part_4
```

### Chạy Lab 6: Transformers
```bash
# Interactive notebook
jupyter notebook lab6/lab6_intro_transformers.ipynb
```

## 📊 Project Summary

### Các thay đổi và cải tiến đã thực hiện
- **File Organization:** Tạo `report/` folder và chuẩn hóa tên file theo quy luật lowercase với underscores
- **Data Management:** Implement `.gitignore` rules để chỉ upload metadata và giữ lại file description
- **Documentation:** Tạo comprehensive data schema documentation với field descriptions và sample data
- **README Updates:** Cập nhật thông tin đầy đủ về Lab 5 RNNs và Lab 6 Transformers

### Dataset Information
Project sử dụng các dataset đa dạng:
- **Sentiment Analysis**: Binary classification với textual data
- **C4 Training**: Large-scale language modeling dataset
- **Universal Dependencies**: Annotated text cho NLP tasks (POS tagging, dependency parsing)
- **Custom datasets**: Generated và processed cho các lab exercises

### Model Performance Highlights
- **Text Classification**: 70.6% accuracy (TF-IDF + Logistic Regression)
- **BERT Fill-Mask**: 40.33% confidence cho "capital" prediction
- **Word Embeddings**: Successfully implemented semantic similarity analysis

## Development Tools

**Công cụ hỗ trợ:**
- **Claude Code (Anthropic)** - Code generation và debugging
- **ChatGPT (OpenAI)** - Syntax checking và optimization
- **PyCharm/VS Code/Jupyter** - Development environment
- **Git/GitHub** - Version control
- **Hugging Face** - Pre-trained model repository
- **Colab/Kaggle** - Cloud GPU environments

## 📄 License

This project is for educational purposes as part of the NLP course.

---

**Note:** Các phần xử lý lỗi syntax và optimization được thực hiện với sự hỗ trợ của AI tools, logic chính và implementation do sinh viên tự thực hiện.