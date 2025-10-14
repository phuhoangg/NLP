from lab1.src.preprocessing.regex_tokenizer import RegexTokenizer
from lab1.src.preprocessing.simple_tokenizer import SimpleTokenizer

sentences = [
    "Hello, world! This is a test.",
    "NLP is fascinating... isn't it?",
    "Let's see how it handles 123 numbers and punctuation!"
]

simple_tokenizer = SimpleTokenizer()
regex_tokenizer = RegexTokenizer()

print("Testing SimpleTokenizer:")
for s in sentences:
    print(f"Input: {s}")
    print(f"Tokens: {simple_tokenizer.tokenize(s)}\n")

print("Testing RegexTokenizer:")
for s in sentences:
    print(f"Input: {s}")
    print(f"Tokens: {regex_tokenizer.tokenize(s)}\n")


from lab1.src.core.dataset_loaders import load_raw_text_data
# ... (your tokenizer imports and instantiations) ...
dataset_path = "C:/Users/hvphu/Desktop/NLP/UD_English-EWT/en_ewt-ud-train.txt"
raw_text = load_raw_text_data(dataset_path)
# Take a small portion of the text for demonstration
sample_text = raw_text[:500] # First 500 characters
print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
print(f"Original Sample: {sample_text[:100]}...")
simple_tokens = simple_tokenizer.tokenize(sample_text)
print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
regex_tokens = regex_tokenizer.tokenize(sample_text)
print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")