from abc import ABC
from lab1.src.core.interface import Tokenizer
from lab2.src.core.interface import Vectorizer


class CountVectorizer(Vectorizer, ABC):
    def __init__(self, tokenizer : Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_: dict[str, int] = {}

    def fit(self, corpus):
        unique_tokens = set()
        for doc in corpus:
            # Chuyển tokens về chữ thường
            unique_tokens.update(token.lower() for token in self.tokenizer.tokenize(doc))
        self.vocabulary_ = { token: idx for idx, token in enumerate(unique_tokens) }
        return self

    def transform(self, corpus):
        vectors = []
        for doc in corpus:
            vector = [0] * len(self.vocabulary_)
            # Tokenize và chuyển về chữ thường
            for token in self.tokenizer.tokenize(doc):
                token_lower = token.lower()
                if token_lower in self.vocabulary_:
                    vector[self.vocabulary_[token_lower]] += 1
            vectors.append(vector)
        return vectors