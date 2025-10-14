from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace, split, trim
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Khởi tạo Spark session
        print("Initializing Spark session...")
        try:
            os.environ["PYSPARK_PYTHON"] = r"C:\Users\hvphu\anaconda3\envs\NLP\python.exe"
            os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\hvphu\anaconda3\envs\NLP\python.exe"
            spark = (SparkSession.builder
                     .getOrCreate())

            spark.sparkContext.setLogLevel("ERROR")
            print("Spark session initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Spark session: {str(e)}")
            print(f"Error: Could not initialize Spark session. Please ensure Spark is properly configured.")
            return

        # Load dataset
        # Dataset C4 có định dạng JSON
        # Chúng ta quan tâm đến trường 'text'
        data_path = "C:/Users/hvphu/Desktop/NLP/data/c4-train.00000-of-01024-30K.json"
        print(f"Loading dataset from {data_path}")

        try:
            dataset = spark.read.json(data_path)
            print(f"Dataset loaded successfully with {dataset.count()} rows")
        except Exception as e:
            logger.error(f"Failed to load dataset from {data_path}: {str(e)}")
            print(f"Error: Could not load dataset from {data_path}")
            spark.stop()
            return

        # Kiểm tra cấu trúc dataset
        try:
            if "text" not in dataset.columns:
                logger.warning("Dataset does not contain 'text' column. Available columns: {dataset.columns}")
                # Thử tìm cột text hoặc tạo một
                text_columns = [column for column in dataset.columns if "text" in column.lower() or "content" in column.lower()]
                if text_columns:
                    logger.info(f"Using column '{text_columns[0]}' as text column")
                    dataset = dataset.withColumnRenamed(text_columns[0], "text")
                else:
                    logger.error("No suitable text column found in dataset")
                    print("Error: Dataset does not contain a text column")
                    spark.stop()
                    return
        except Exception as e:
            logger.error(f"Error checking dataset structure: {str(e)}")
            print(f"Error checking dataset structure: {str(e)}")
            spark.stop()
            return

        # Preprocessing theo yêu cầu trong đề bài
        print("Starting text preprocessing...")
        try:
            # xóa ủl, kí tự đặc biệt, khoảng trắng thừa
            cleaned_data = dataset.withColumn(
                "cleaned_text",
                regexp_replace(
                    trim(
                        regexp_replace(
                            regexp_replace(lower(col("text")), r"https?://\S+|www\.\S+", ""),
                            r"[^\w\s]", " "
                        )
                    ),
                    r"\s+", " "
                )
            )

            # Tokenize cột text
            tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="tokens")
            tokenized_data = tokenizer.transform(cleaned_data)

            # Lọc bỏ các mảng token rỗng
            tokenized_data = tokenized_data.filter(col("tokens").isNotNull())

            print(f"Data preprocessing completed. {tokenized_data.count()} rows remaining.")

        except Exception as e:
            logger.error(f"Error during text preprocessing: {str(e)}")
            print(f"Error during text preprocessing: {str(e)}")
            spark.stop()
            return

        # Cấu hình và train mô hình Word2Vec
        print("Training Word2Vec model...")
        try:
            word2vec = Word2Vec(
                vectorSize=100, # yêu cầu vector 100 chiều
                inputCol="tokens",
                outputCol="features"
            )

            model = word2vec.fit(tokenized_data)
            print("Word2Vec model training completed successfully")

            # Lấy các vector từ
            word_vectors = model.getVectors()
            print(f"Model vocabulary size: {word_vectors.count()}")

            # Hiển thị một số vector từ mẫu
            print("\nSample word vectors:")
            word_vectors.show(10)

        except Exception as e:
            logger.error(f"Error training Word2Vec model: {str(e)}")
            print(f"Error training Word2Vec model: {str(e)}")
            spark.stop()
            return

        # Demo sử dụng mô hình
        # Tìm synonyms cho một từ
        try:
            sample_word = "computer"
            vocabulary_words = word_vectors.select("word").rdd.flatMap(lambda x: x).collect()
            if sample_word in vocabulary_words:
                synonyms = model.findSynonyms(sample_word, 10)
                print(f"\nTop 10 synonyms for '{sample_word}':")
                synonyms.show()
            else:
                logger.warning(f"Word '{sample_word}' not found in vocabulary")
                print(f"Word '{sample_word}' not found in vocabulary")
        except Exception as e:
            logger.error(f"Error finding synonyms: {str(e)}")
            print(f"Error finding synonyms: {str(e)}")

        print("\nSpark Word2Vec demo completed successfully!")

    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print(f"An unexpected error occurred: {str(e)}")

    finally:
        # Dừng Spark session
        try:
            if 'spark' in locals():
                spark.stop()
                print("Spark session stopped")
        except Exception as e:
            logger.error(f"Error stopping Spark session: {str(e)}")


if __name__ == "__main__":
    main()



