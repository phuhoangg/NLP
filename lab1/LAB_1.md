# Test Results:

## Testing SimpleTokenizer

### Input: "Hello, world! This is a test."
**Tokens:** `['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']`

### Input: "NLP is fascinating... isn't it?"
**Tokens:** `['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']`

### Input: "Let's see how it handles 123 numbers and punctuation!"
**Tokens:** `["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']`

## Testing RegexTokenizer

### Input: "Hello, world! This is a test."
**Tokens:** `['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.']`

### Input: "NLP is fascinating... isn't it?"
**Tokens:** `['NLP', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']`

### Input: "Let's see how it handles 123 numbers and punctuation!"
**Tokens:** `['Let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']`

---

## Tokenizing Sample Text from UD_English-EWT

**Original Sample:** Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of ...

**SimpleTokenizer Output (first 20 tokens):** `['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',']`

**RegexTokenizer Output (first 20 tokens):** `['Al', '-', 'Zaman', ':', 'American', 'forces', 'killed', 'Shaikh', 'Abdullah', 'al', '-', 'Ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']`



---

**So sánh sự khác biệt:**
 Dùng regex sẽ cho ra kết quả "hợp lí" hơn so với sử dụng logic vòng lặp ở `simple_tokenizer.py`, có thể thấy ở một số trường hợp, dùng phương thức lập trình sẽ không bao quát hết các trường hợp.

**Chú thích dùng LLM:** `dataset_loaders.py`, một số syntax
