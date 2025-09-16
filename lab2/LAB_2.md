# Test Results:

## Testing CountVectorizer

### Corpus:
```
[
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]
```

### Learned Vocabulary:
`{'a': 0, 'NLP': 1, 'programming': 2, 'subfield': 3, 'AI': 4, 'love': 5, '.': 6, 'of': 7, 'I': 8, 'is': 9}`

### Document-Term Matrix:
**Doc 0:** `[0, 0, 0, 0, 0, 0, 1, 0, 1, 0]`
**Doc 1:** `[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]`
**Doc 2:** `[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]`

---

## Phân tích kết quả:

**CountVectorizer** đã chuyển đổi văn bản thành ma trận Document-Term với:
- **Vocabulary size:** 10 tokens
- **Matrix dimensions:** 3 documents × 10 terms

**Chú thích:** Vectorizer sử dụng `RegexTokenizer` từ `lab1` để tách từ trước khi tạo vector.

**Lưu ý:** Kết quả cho thấy tất cả document đều có token '.' (dấu chấm) và 'I' xuất hiện ở tất cả các document, điều này cho thấy cần cân nhắc thêm bước xử lý stop words và punctuation.

**Chú thích dùng LLM:** Claude code được dùng để sinh file .md này, bên cạnh đó có sử dụng ChatGPT free để chỉnh sửa một số lỗi syntax