from sklearn.model_selection import train_test_split
from lab1.src.preprocessing.regex_tokenizer import RegexTokenizer
from lab2.src.representations.count_vectorizer import CountVectorizer
from lab5.src.model.text_classifier import TextClassifier

# Task 1: Data Preparation
# Tạo datasets và nhãn
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

print("Task 1: Data Preparation")
print("Texts:")
for i, text in enumerate(texts):
    print(f"{i+1}. {text}")
print(f"\nLabels: {labels}")

# Task 2: TextClassifier - Vectorization
print("\nTask 2: Text Classifier Setup")
tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer)

# Sử dụng class text_classifier đã tạo
X = vectorizer.fit_transform(texts)
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print("Vocabulary:", vectorizer.vocabulary_)
print(f"Feature matrix shape: {len(X)} x {len(X[0])}")

print("\nTask 3: Evaluation")

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print()

tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer)

classifier = TextClassifier(vectorizer)

print("Training the classifier...")
# Huấn luyện mô hình phân loại
classifier.fit(X_train, y_train)
print("Training completed!")
print()

print("Making predictions...")
# Dự đoán trên tập test
y_pred = classifier.predict(X_test)

print("Test texts and predictions:")
for i, (text, true_label, pred_label) in enumerate(zip(X_test, y_test, y_pred)):
    sentiment_true = "Positive" if true_label == 1 else "Negative"
    sentiment_pred = "Positive" if pred_label == 1 else "Negative"
    print(f"Test {i+1}: '{text}'")
    print(f"  True label: {sentiment_true} ({true_label})")
    print(f"  Predicted: {sentiment_pred} ({pred_label})")
    print()

# Tính toán và in ra các thông số thống kê:
print("Evaluation Metrics:")
metrics = classifier.evaluate(y_test, y_pred)
for metric_name, value in metrics.items():
    print(f"{metric_name.capitalize()}: {value:.4f}")
