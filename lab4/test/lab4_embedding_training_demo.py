import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import logging
import re

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def load_data(filename):
    try:
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string")

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found: {filename}")

        sentences = []
        with open(filename, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue

                    line_sentences = re.split(r'(?<=[.!?])\s+', line)

                    for sent in line_sentences:
                        tokens = simple_preprocess(sent)
                        if tokens:
                            sentences.append(tokens)

                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {str(e)}")
                    continue

        if not sentences:
            raise ValueError("No valid sentences could be extracted from the data file")

        logger.info(f"Successfully loaded {len(sentences)} sentences from {filename}")
        return sentences

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {str(e)}")
        raise


def train_word2vec_model(filename, vector_size=100, window=5, min_count=2, workers=4):
    # Huấn luyện mô hình Word2vec
    try:
        # Validate parameters
        if not isinstance(vector_size, int) or vector_size <= 0:
            raise ValueError("vector_size must be a positive integer")
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer")
        if not isinstance(min_count, int) or min_count < 1:
            raise ValueError("min_count must be a positive integer")
        if not isinstance(workers, int) or workers <= 0:
            raise ValueError("workers must be a positive integer")

        print("Loading and tokenizing data...")
        sentences = load_data(filename)
        print(f"Loaded {len(sentences)} sentences.")

        if len(sentences) < min_count:
            logger.warning(f"Number of sentences ({len(sentences)}) is less than min_count ({min_count})")

        print("Training Word2Vec model...")
        model = Word2Vec(
            sentences=sentences,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            sg=1,  # dùng skip-gram
            epochs=20
        )

        print("Training complete.")
        return model

    except Exception as e:
        logger.error(f"Error training Word2Vec model: {str(e)}")
        raise


def save_model(model, filename):
    # Lưu model vào đường dẫn chỉ định
    try:
        if model is None:
            raise ValueError("Model cannot be None")

        if not isinstance(filename, str):
            raise ValueError("Filename must be a string")

        # Tạo thư mục nếu thư mịc đó không tồn tại
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        model.save(filename)
        print(f"Model saved to: {filename}")
        logger.info(f"Model successfully saved to {filename}")

    except Exception as e:
        logger.error(f"Error saving model to {filename}: {str(e)}")
        raise


def demo(model):
    # Hàm test
    try:
        if model is None:
            raise ValueError("Model cannot be None")

        print("\nDemo Word2Vec Usage")

        # Tìm kiếm từ tương tự với quả táo bằng cóine similarity
        word = "apple"
        if word in model.wv.key_to_index:
            print(f"\nTop 10 most similar words to '{word}':")
            try:
                similar_words = model.wv.most_similar(word, topn=10)
                for w, sim in similar_words:
                    print(f"{w:15s} {sim:.4f}")
            except Exception as e:
                logger.error(f"Error finding similar words to '{word}': {str(e)}")
                print(f"Error finding similar words to '{word}'")
        else:
            print(f"Word '{word}' not found in vocabulary.")

        # Word Analogy Example (ví dụ được lấy từ ChatGPT)
        analogy = ("paris", "france", "japan")
        # Kì vọng của ví dụ này sẽ mong muốn kết quả là "tokyo"
        if all(w in model.wv.key_to_index for w in analogy):
            print(f"\nAnalogy: {analogy[0]} - {analogy[1]} + {analogy[2]} ≈ ?")
            try:
                results = model.wv.most_similar(
                    positive=[analogy[0], analogy[2]],  # paris + japan
                    negative=[analogy[1]],  # - france
                    topn=5
                )
                for w, sim in results:
                    print(f"{w:15s} {sim:.4f}")
            except Exception as e:
                logger.error(f"Error computing analogy: {str(e)}")
                print("Error computing analogy")
        else:
            print("Some words in the analogy are missing from the vocabulary.")

    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        print(f"Error occurred during demo: {str(e)}")


def load_model(filename):
    # Thêm phương thức load model đã train
    try:
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found: {filename}")
        model = Word2Vec.load(filename)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {filename}: {str(e)}")
        raise

def main():
    try:
        data_path = "C:/Users/hvphu/Desktop/NLP/data/UD_English-EWT/en_ewt-ud-train.txt"
        model_path = "model/word2vec_ewt.model"

        # Validate paths
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            print(f"Error: Data file not found at {data_path}")
            return

        # Train model
        try:
            # model = train_word2vec_model(data_path)
            #
            # # Save model
            # save_model(model, model_path)
            #
            # # Demo usage
            # # demo(model)
            #
            # # Test load model
            model = load_model(model_path)

            # Demo sau khi load
            demo(model)

        except Exception as e:
            logger.error(f"Error in model training pipeline: {str(e)}")
            print(f"Error occurred during model training: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()