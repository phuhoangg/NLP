# Phân tích cảm xúc với PySpark
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

# 1. Khởi tạo Spark Session
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# 2. Tải dữ liệu
# Đọc dữ liệu từ file CSV
df = spark.read.csv("C:/Users/hvphu/Desktop/NLP/data/sentiments.csv", header=True, inferSchema=True)
df.show(5)

# 3. Tiền xử lý dữ liệu
# Convert -1/1 labels to 0/1: Normalize sentiment labels
df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
# Drop rows with null sentiment values before processing
initial_row_count = df.count()
df = df.dropna(subset=["sentiment"])

# 4. Xây dựng Pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")

stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=12000)

idf = IDF(inputCol="raw_features", outputCol="features")

lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

# 5. Chia dữ liệu thành tập huấn luyện và kiểm tra
(train_data, test_data) = df.randomSplit([0.8, 0.2], seed=912)

# 6. Huấn luyện mô hình
print("Đang huấn luyện mô hình...")
model = pipeline.fit(train_data)
print("Huấn luyện hoàn thành")

# 7. Dự đoán trên tập kiểm tra
predictions = model.transform(test_data)
predictions.select("text", "sentiment", "label", "prediction").show()

# 8. Đánh giá mô hình
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Độ chính xác: {accuracy}")

evaluator.setMetricName("f1")
f1 = evaluator.evaluate(predictions)
print(f"F1 Score: {f1}")

# 9. Dừng Spark session
spark.stop()



