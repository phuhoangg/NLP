import gensim.downloader as api
from lab1.src.preprocessing.regex_tokenizer import RegexTokenizer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WordEmbedder:
    def __init__(self, model_name):
        #Khởi tạo word_embedder
        try:
            logger.info(f"Loading model: {model_name}")
            self.model = api.load(model_name)
            self.model_name = model_name
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise RuntimeError(f"Could not load model {model_name}: {str(e)}")

    def get_vector(self, word):
        # Lấy ra vector số gắn với từ được đưa vào
        try:
            if not isinstance(word, str):
                logger.warning(f"Word must be a string, got {type(word)}")
                return None

            if word in self.model:
                return self.model[word]
            else:
                logger.warning(f"Word '{word}' not found in vocabulary")
                return None
        except Exception as e:
            logger.error(f"Error getting vector for word '{word}': {str(e)}")
            return None

    def get_similarity(self, word1, word2):
        # Lấy ra cosine similarity của 2 từ
        try:
            vec1 = self.get_vector(word1)
            vec2 = self.get_vector(word2)

            if vec1 is None or vec2 is None:
                logger.warning(f"Cannot compute similarity: one or both words not found in vocabulary")
                return None

            # Chuẩn hóa để tránh việc chia cho số 0
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                logger.warning("Cannot compute similarity: one of the vectors has zero norm")
                return None

            return np.dot(vec1, vec2) / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error computing similarity between '{word1}' and '{word2}': {str(e)}")
            return None

    def get_most_similar(self, word, topn=10):
        # Dùng hàm built in của model để lấy ra top_n từ gần nhất với word
        try:
            if not isinstance(word, str):
                logger.warning(f"Word must be a string, got {type(word)}")
                return None

            if not isinstance(topn, int) or topn <= 0:
                logger.warning(f"topn must be a positive integer, got {topn}")
                return None

            if word in self.model:
                return self.model.most_similar(word, topn=topn)
            else:
                logger.warning(f"Word '{word}' not found in vocabulary")
                return None
        except Exception as e:
            logger.error(f"Error getting most similar words for '{word}': {str(e)}")
            return None

    def embed_document(self, document):
        try:
            if not isinstance(document, str):
                logger.warning(f"Document must be a string, got {type(document)}")
                return None

            tokenizer = RegexTokenizer() # Dùng regex tokenizer ở lab1 để thực hiện tokenizer
            tokens = tokenizer.tokenize(document)
            print(tokens)
            vectors = []

            for token in tokens:
                vec = self.get_vector(token)
                if vec is not None:
                    vectors.append(vec)

            if not vectors:
                logger.warning("No valid tokens found in document")
                return None # Nếu không có từ hợp lệ trả về None

            return np.mean(vectors, axis=0)
        except Exception as e:
            logger.error(f"Error embedding document: {str(e)}")
            return None