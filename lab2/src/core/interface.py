from abc import ABC, abstractmethod

class Vectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: list[str]):
        pass
    @abstractmethod
    def transform(self, documents: list[str]) -> list[list[int]]:
        pass

    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        self.fit(corpus)
        return self.transform(corpus)