from lab1.src.preprocessing.regex_tokenizer import RegexTokenizer
from lab2.src.representations.count_vectorizer import CountVectorizer

corpus = [
"I love NLP.",
"I love programming.",
"NLP is a subfield of AI."
]

tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer)

dt_matrix = vectorizer.fit_transform(corpus)

print("Learned Vocabulary:")
print(vectorizer.vocabulary_)

print("\nDocument-Term Matrix:")
for i, vector in enumerate(dt_matrix):
    print(f"Doc {i}:", vector)