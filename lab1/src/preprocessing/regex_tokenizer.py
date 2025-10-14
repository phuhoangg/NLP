import re

from lab1.src.core.interface import Tokenizer


class RegexTokenizer(Tokenizer):
    def __init__(self):
        self.pattern = re.compile(r"\w+|[^\w\s]")

    def tokenize(self, text:str) -> list[str]:
        return self.pattern.findall(text.lower())
