from lab4.src.representations.word_embedder import WordEmbedder
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Phần test có sự hỗ trợ của ChatGPT

def main():
    try:
        # Khởi tạo mô hình word embedding
        print("Initializing WordEmbedder with 'glove-wiki-gigaword-50'...")
        try:
            embedder = WordEmbedder('glove-wiki-gigaword-50')
            print("WordEmbedder initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WordEmbedder: {str(e)}")
        print("Word Embedding Tests")

        # Lấy vector cho từ "king"
        print("\n[1] Vector for 'king':")
        try:
            king_vec = embedder.get_vector('king')
            if king_vec is not None:
                print("Dimension:", king_vec.shape[0])
            else:
                print("Word 'king' not found in vocabulary.")
        except Exception as e:
            logger.error(f"Error getting vector for 'king': {str(e)}")
            print("Error occurred while getting vector for 'king'")

        # Tính độ tương đồng giữa các cặp từ
        print("\n[2] Word Similarities:")
        try:
            sim_king_queen = embedder.get_similarity('king', 'queen')
            sim_king_man = embedder.get_similarity('king', 'man')

            if sim_king_queen is not None:
                print(f"Similarity(king, queen) = {sim_king_queen:.4f}")
            else:
                print("Could not compute similarity(king, queen)")

            if sim_king_man is not None:
                print(f"Similarity(king, man)   = {sim_king_man:.4f}")
            else:
                print("Could not compute similarity(king, man)")
        except Exception as e:
            logger.error(f"Error computing similarities: {str(e)}")
            print("Error occurred while computing similarities")

        # 10 từ giống nhất với "computer"
        print("\n[3] Top 10 most similar words to 'computer':")
        try:
            most_similar_words = embedder.get_most_similar('computer', topn=10)
            if most_similar_words is not None:
                for word, score in most_similar_words:
                    print(f"{word:15s}  {score:.4f}")
            else:
                print("Could not find similar words to 'computer'")
        except Exception as e:
            logger.error(f"Error getting most similar words: {str(e)}")
            print("Error occurred while getting most similar words")

        # Biểu diễn câu văn bằng document embedding
        print("\n[4] Document Embedding:")
        try:
            sentence = "The queen rules the country."
            doc_vector = embedder.embed_document(sentence)
            if doc_vector is not None:
                print(f"Sentence: \"{sentence}\"")
                print("Dimension:", doc_vector)
            else:
                print(f"Could not embed document: \"{sentence}\"")
        except Exception as e:
            logger.error(f"Error embedding document: {str(e)}")
            print("Error occurred while embedding document")

        print("\nAll tests completed!")

    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
