from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec
import time

spark = SparkSession.builder.appName("ImprovementTest").getOrCreate()

# 1. Tải dữ liệu
df = spark.read.csv("C:/Users/hvphu/Desktop/NLP/data/sentiments.csv", header=True, inferSchema=True)
print("Tai du lieu thanh cong")

# 2. Tiền xử lí nâng cao
df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
df = df.withColumn("clean_text", lower(col("text")))
df = df.withColumn("clean_text", regexp_replace(col("clean_text"), r"[^a-zA-Z0-9\s]", ""))
df = df.withColumn("clean_text", regexp_replace(col("clean_text"), r"\s+", " "))
df = df.dropna(subset=["clean_text", "sentiment"])

print("Dữ liệu sau khi làm sạch:")
df.select("text", "clean_text").show(5, truncate=False)

# 3. Chia dữ liệu
(train_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)
print(f"Training set: {train_data.count()} rows")
print(f"Test set: {test_data.count()} rows")

# 4. Xây dựng các phương pháp nâng cao

def evaluate_model(model_name, pipeline, train_data, test_data):
    start_time = time.time()
    model = pipeline.fit(train_data)
    training_time = time.time() - start_time

    start_time = time.time()
    predictions = model.transform(test_data)
    prediction_time = time.time() - start_time

    evaluator = MulticlassClassificationEvaluator()
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)

    print(f"Thời gian huấn luyện : {training_time:.2f}s")
    print(f"Thơì gian dự đoán: {prediction_time:.2f}s")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return {
        'model': model_name,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'training_time': training_time,
        'prediction_time': prediction_time
    }

# Model 1:TF-IDF + Logistic Regression
print("\n1. TF-IDF + Logistic Regression")
tokenizer1 = Tokenizer(inputCol="clean_text", outputCol="words")
stopwords_remover1 = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashing_tf1 = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
idf1 = IDF(inputCol="raw_features", outputCol="features")
lr1 = LogisticRegression(maxIter=10, regParam=0.01, featuresCol="features", labelCol="label")
pipeline1 = Pipeline(stages=[tokenizer1, stopwords_remover1, hashing_tf1, idf1, lr1])
results1 = evaluate_model("TF-IDF + Logistic Regression", pipeline1, train_data, test_data)


# Model 2: TF-IDF + Naive Bayes
print("\n2. TF-IDF + Naive Bayes")
tokenizer2 = Tokenizer(inputCol="clean_text", outputCol="words")
stopwords_remover2 = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashing_tf2 = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
idf2 = IDF(inputCol="raw_features", outputCol="features")
nb2 = NaiveBayes(featuresCol="features", labelCol="label", smoothing=1.0)
pipeline2 = Pipeline(stages=[tokenizer2, stopwords_remover2, hashing_tf2, idf2, nb2])
results2 = evaluate_model("TF-IDF + Naive Bayes", pipeline2, train_data, test_data)

# Model 3: Word2Vec + Logistic Regression
print("\n3. Word2Vec + Logistic Regression")
tokenizer3 = Tokenizer(inputCol="clean_text", outputCol="words")
stopwords_remover3 = StopWordsRemover(inputCol="words", outputCol="filtered_words")
word2vec3 = Word2Vec(vectorSize=100, minCount=1, inputCol="filtered_words", outputCol="features")
lr3 = LogisticRegression(maxIter=10, regParam=0.01, featuresCol="features", labelCol="label")
pipeline3 = Pipeline(stages=[tokenizer3, stopwords_remover3, word2vec3, lr3])
results3 = evaluate_model("Word2Vec + Logistic Regression", pipeline3, train_data, test_data)

# So sánh tổng hợp 3 models
results = [results1, results2, results3]

print(f"{'Model':<45} {'Accuracy':<10} {'F1 Score':<10} {'Training Time':<15}")
print("-" * 85)
for result in results:
    print(f"{result['model']:<45} {result['accuracy']:<10.4f} {result['f1']:<10.4f} {result['training_time']:<15.2f}s")

# Tìm mô hình tốt nhất
best_model = max(results, key=lambda x: x['accuracy'])
print(f"\nMO HINH TOT NHAT: {best_model['model']}")
print(f"Accuracy: {best_model['accuracy']:.4f}")
print(f"F1 Score: {best_model['f1']:.4f}")
print(f"Training Time: {best_model['training_time']:.2f}s")

spark.stop()

