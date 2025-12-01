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
`{'a': 0, 'of': 1, '.': 2, 'is': 3, 'subfield': 4, 'ai': 5, 'i': 6, 'programming': 7, 'nlp': 8, 'love': 9}`

### Document-Term Matrix:
**Doc 0:** `[0, 0, 1, 0, 0, 0, 1, 0, 1, 1]`
**Doc 1:** `[0, 0, 1, 0, 0, 0, 1, 1, 0, 1]`
**Doc 2:** `[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]`

---

## Phân tích kết quả:

**CountVectorizer** đã chuyển đổi văn bản thành ma trận Document-Term với:
- **Vocabulary size:** 10 tokens
- **Matrix dimensions:** 3 documents × 10 terms

**Cải tiến:** Đã thêm bước chuyển đổi chữ hoa thành chữ thường (lowercase) để đảm bảo tính nhất quán trong việc đếm tần suất xuất hiện của các từ. Điều này giúp các từ giống nhau nhưng khác biệt về chữ hoa/thường (ví dụ: "NLP" và "nlp") được coi là cùng một token.

**Chú thích:** Vectorizer sử dụng `RegexTokenizer` từ `lab1` để tách từ trước khi tạo vector.

**Lưu ý:** Kết quả cho thấy tất cả document đều có token '.' (dấu chấm), điều này cho thấy cần cân nhắc thêm bước xử lý punctuation.

**Chú thích dùng LLM:** Claude code được dùng để sinh file .md này, bên cạnh đó có sử dụng ChatGPT free để chỉnh sửa một số lỗi syntax