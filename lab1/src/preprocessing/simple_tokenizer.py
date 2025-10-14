from lab1.src.core.interface import Tokenizer

class SimpleTokenizer(Tokenizer):
    def __init__(self):
        self.punctuations = {".", ",", "?", "!"}

    def tokenize(self, text:str) -> list[str]:
        text = text.lower()
        tokens = []
        for word in text.split():
            current = ""
            for char in word:
                if char in self.punctuations:
                    if current:
                        tokens.append(current)
                        current = ""
                    tokens.append(char)
                else:
                    current += char
            if current:
                tokens.append(current)
        return tokens
