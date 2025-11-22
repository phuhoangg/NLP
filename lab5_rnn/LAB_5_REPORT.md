# Báo cáo Thực hành Lab 5 - RNNs

**Họ và tên:** Hoàng Văn Phú
**Mã sinh viên:** 23001546
**Code repository:** [GitHub](https://github.com/phuhoangg/NLP)

**Lưu ý:** Các phần xử lý lỗi trong codebase được thực hiện bởi Claude Code, các logic còn lại do sinh viên tự thực hiện.

---

# Hướng dẫn chạy code

## Cài đặt môi trường

**Phiên bản Python yêu cầu:** 3.10+ hoặc sử dụng môi trường colab.
**GPU:** Khuyến khích sử dụng CUDA cho Phần 3 và 4
**Bộ nhớ VRAM:** Tối thiểu 4GB

## Cài đặt thư viện

```bash
# Các thư viện cơ bản
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install matplotlib seaborn

# Phần 2: Phân loại văn bản
pip install tensorflow tensorflow-addons
pip install gensim

# Phần 3 & 4: Gán nhãn chuỗi
pip install datasets==2.19.1
pip install seqeval
```


# Phần 1: Tìm hiểu về RNNs và Phân loại Token

## Code triển khai

### 1. Toán tử Tensor
```python
import torch
import numpy as np

# Tạo tensor từ list và numpy
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)                    # Từ list
x_np = torch.from_numpy(np.array(data))       # Từ numpy

print(x_data + x_data)  # Cộng tensor theo từng phần tử
print(x_data * 5)       # Nhân với số vô hướng
print(x_data @ x_data.T) # Nhân ma trận
```

### 2. Tính đạo hàm và Autograd
```python
x = torch.ones(1, requires_grad=True)
y = x + 2
z = y * y * 3
z.backward()
print(x.grad)  # Kết quả: tensor([18.])
```

### 3. Xây dựng mạng nơ-ron
```python
class MyFirstModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MyFirstModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, indices):
        embeds = self.embedding(indices)
        hidden = self.activation(self.linear(embeds))
        output = self.output_layer(hidden)
        return output

# Kiểm tra mô hình
model = MyFirstModel(vocab_size=100, embedding_dim=16, hidden_dim=8, output_dim=2)
input_data = torch.LongTensor([[1, 2, 5, 9]])
output_data = model(input_data)
print(f"Kích thước output mô hình: {output_data.shape}")  # torch.Size([1, 4, 2])
```

## Vấn đề đã gặp:

### 1: Sự khác biệt giữa view() và reshape()


```python
x_rand = torch.rand(4,4)
print(f"Tensor gốc kích thước: {x_rand.shape}")  # torch.Size([4, 4])

# Phương pháp 1: Dùng view() - hoạt động hiệu quả
reshaped_view = x_rand.view(16,1)
print(f"Kết quả view: {reshaped_view.shape}")   # torch.Size([16, 1])

# Phương pháp 2: Dùng reshape() - cho kết quả tương tự nhưng có thể khác về bộ nhớ
reshaped_reshape = x_rand.reshape(16,1)
print(f"Kết quả reshape: {reshaped_reshape.shape}")  # torch.Size([16, 1])
```

- `view()` thường hiệu quả hơn vì nó chỉ thay đổi cách nhìn tensor trong bộ nhớ, trong khi `reshape()` có thể tạo bản sao khi cần. Tuy nhiên, `view()` yêu cầu tensor phải có bộ nhớ liên tục, điều này gây ra vấn đề trong một số trường hợp.

### 2: Autograd và việc quản lý đồ thị tính toán

```python
# Từ lab5_pytorch_introduction.ipynb:13-15
x = torch.ones(1, requires_grad=True)
y = x + 2
z = y * y * 3
z.backward()
print(f"Đạo hàm lần đầu: {x.grad}")  # tensor([18.])

# Thử gọi backward lần thứ hai
# z.backward()  # Đây là lúc lỗi xảy ra!
```

- Sau khi gọi `z.backward()`, PyTorch tự động giải phóng đồ thị tính toán để tiết kiệm bộ nhớ. Khi cố gắng gọi `backward()` gây ra lỗi `RuntimeError: Trying to backward through the graph a second time`.

### 3: Hiểu đúng về indexing và broadcasting

```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# Cách hoạt động của các phương thức indexing
print(f"Dòng đầu tiên: {x_data[0,:]}")    # tensor([1, 2])
print(f"Cột thứ hai: {x_data[:,1]}")    # tensor([2, 4])
print(f"Phần tử cụ thể: {x_data[1,1]}")  # tensor(4) - không có mảng!
```

- `x_data[1,1]` trả về tensor vô hướng (scalar) thay vì tensor có shape [1].

---

# Phần 2: RNNs cho Phân loại Văn bản
## Chi tiết triển khai

### Nhiệm vụ 1: TF-IDF + Hồi quy Logistic
```python
tfidf_lr_pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000),
    LogisticRegression(max_iter=1000)
)

tfidf_lr_pipeline.fit(df_train['text'], y_train)
y_pred_tfidf = tfidf_lr_pipeline.predict(df_test['text'])
```

**Kết quả thực tế:** F1-score (Macro) = 0.8226, Độ chính xác = 84%

### Nhiệm vụ 2: Dense Layer + Word2Vec
```python
# Huấn luyện Word2Vec trên dữ liệu miền
sentences = [text.split() for text in df_train['text']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Chuyển câu thành vector trung bình
def sentence_to_avg_vector(text, model):
    tokens = text.split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
```

**Kết quả:** F1-score (Macro) = 0.1872, Test Loss = 3.633

### Nhiệm vụ 3: LSTM với Pre-trained Embedding
```python
# Vấn đề tìm thấy: Chỉ bao phủ 22.60% của embedding huấn luyện trước
embedding_matrix = np.zeros((vocab_size, embedding_dim))
coverage = np.sum(embedding_matrix.sum(axis=1) > 0) / vocab_size
print(f"Tỷ lệ words có trong Word2Vec: {coverage:.2%}")  # 22.60%
```

**Vấn đề:** Độ bao phủ thấp của pretraned-embeddings
**Kết quả:** F1-score (Macro) = 0.0461, Test Accuracy = 11.14%, Test Loss = 3.453

### Nhiệm vụ 4: LSTM Model with End-to-End Training
```python
lstm_model_scratch = Sequential([
    Embedding(input_dim=vocab_size, output_dim=200, input_length=max_len),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

**Kết quả:** F1-score (Macro) = 0.8367, Test Accuracy = 84.87%, Test Loss = 0.603

## Vấn đề:

### Pre-trained embeddings với độ bao phủ thấp

Đây là một trong những phát hiện quan trọng nhất trong quá trình thực hành:

```python
embedding_matrix = np.zeros((vocab_size, embedding_dim))
coverage = np.sum(embedding_matrix.sum(axis=1) > 0) / vocab_size
print(f"Tỷ lệ words có trong Word2Vec: {coverage:.2%}")  # Chỉ 20.42%
```

**Phân tích vấn đề:**
- Pretrained - embeddings chỉ 20.42% coverage dây là con số thấp
- Word2Vec trained trên general text không phù hợp với domain IoT/Home automation
- 80% tokens được khởi tạo ngẫu nhiên, làm hỏng toàn bộ mô hình

**Hệ quả thực tế:** Mô hình LSTM với pre-trained embeddings đạt F1-score chỉ 0.036 - gần như random guessing.

## Bảng so sánh hiệu suất thực tế

| Mô hình | F1-score (Macro) | Test Loss | Test Accuracy | Trạng thái |
|---------|------------------|-----------|---------------|------------|
| TF-IDF + LR | **0.8226** | N/A | 84% | ✅ Strong baseline |
| Word2Vec + Dense | 0.1872 | 3.633 | Low | ❌ Very poor |
| LSTM Pre-trained | 0.0461 | 3.453 | 11% | ❌ Disaster |
| LSTM Scratch | **0.8367** | 0.603 | 85% | 🏆 Winner |

**Phân tích kết quả:**
- **LSTM Scratch wins:** F1-score 0.8367, vượt TF-IDF baseline 1.45%
- **Pre-trained disaster:** 22.6% coverage gây ra performance thảm khốc
- **TF-IDF surprisingly strong:** Baseline đạt 82.26% F1-score
- **Word2Vec Dense failed:** Vector trung bình mất sequence information

## Phân tích lỗi thực tế từ test set

Dựa trên 5 câu random từ test set, đây là các pattern được quan sát:

### Ví dụ 1: "is this my sisters cellphone number"
```
Thực tế: email_querycontact
TF-IDF + LR: ✅ email_querycontact
Word2Vec + Dense: ❌ transport_traffic
LSTM Pre-trained: ❌ qa_factoid
LSTM Scratch: ✅ email_querycontact
```

### Ví dụ 2: "please talk softer"
```
Thực tế: audio_volume_down
TF-IDF + LR: ❌ audio_volume_mute
LSTM Scratch: ✅ audio_volume_down
```

### Ví dụ 3: "reduce brightness"
```
Thực tế: iot_hue_lightdim
TF-IDF + LR: ✅ iot_hue_lightdim
LSTM Scratch: ✅ iot_hue_lightdim
```

### Ví dụ 4: "i want some coffee now"
```
Thực tế: iot_coffee
TF-IDF + LR: ✅ iot_coffee
LSTM Scratch: ✅ iot_coffee
```

## Giải pháp chính đã áp dụng

### Giải pháp 1: Cải thiện kiến trúc
- LSTM hai chiều cho ngữ cảnh
- SpatialDropout1D cho regularization
- BatchNormalization cho sự ổn định
- Nhiều lớp LSTM cho chiều sâu

### Giải pháp 2: Thích ứng với domain
```python
# Huấn luyện embedding trên domain cụ thể
sentences = [text.split() for text in df_train['text']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
```

---

# Phần 3: Part-of-Speech Tagging với RNN

## Chi tiết triển khai

### Nhiệm vụ 1: Tải dữ liệu và xử lý
```python
def load_conllu(file_path):
    sentences = []
    current_sentence = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            parts = line.split('\t')
            if len(parts) >= 4:
                word = parts[1]
                upos_tag = parts[3]
                current_sentence.append((word, upos_tag))
    return sentences
```

### Nhiệm vụ 2: Xây dựng kiến trúc
```python
class SimpleRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size, tag_vocab_size, embedding_dim=64, hidden_dim=128):
        super(SimpleRNNForTokenClassification, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.hidden2tag = nn.Linear(hidden_dim, tag_vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, sentences, lengths=None):
        embeds = self.embedding(sentences)
        embeds = self.dropout(embeds)
        rnn_out, hidden = self.rnn(embeds)
        rnn_out = self.dropout(rnn_out)
        tag_scores = self.hidden2tag(rnn_out)
        return tag_scores
```

### Nhiệm vụ 3: Cấu hình huấn luyện
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Hàm loss quan trọng với xử lý padding
criterion = nn.CrossEntropyLoss(ignore_index=tag_to_ix['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
```

## Vấn đề gặp phải

### Overfitting


```python
# Epoch   1/20: Loss: 1.5131 | Train Acc: 0.5423 | Dev Acc: 0.6653
# Epoch   5/20: Loss: 0.6554 | Train Acc: 0.7915 | Dev Acc: 0.8068
# Epoch  20/20: Loss: 0.2547 | Train Acc: 0.9167 | Dev Acc: 0.8765
```

- Mô hình hội tụ khá nhanh, chỉ trong khoảng 4 epoch đầu tiên đã cho hiệu suất tốt, sau epoch thứ 4 thì mô hình có vẻ không học thêm được gì và loss dần mất ổn định.


## Kết quả hiệu suất

### Tiến trình huấn luyện
```
Epoch   1/20: Loss: 1.5131 | Train Acc: 0.5423 | Dev Acc: 0.6653
Epoch   5/20: Loss: 0.6554 | Train Acc: 0.7915 | Dev Acc: 0.8068
Epoch  10/20: Loss: 0.4234 | Train Acc: 0.8656 | Dev Acc: 0.8513
Epoch  20/20: Loss: 0.2547 | Train Acc: 0.9167 | Dev Acc: 0.8765
```

### Kết quả cuối cùng
| Metric | Huấn luyện | Validation | Trạng thái |
|--------|------------|------------|------------|
| Độ chính xác | 94.44% | 87.65% | ✅ Hiệu suất tốt |
| Loss | N/A | N/A | ✅ Hội tụ |

### Hiệu suất cấp độ tag
| POS Tag | F1-Score | Trạng thái |
|---------|----------|------------|
| PUNCT | 0.99 | ✅ Xuất sắc |
| DET | 0.96 | ✅ Xuất sắc |
| PRON | 0.96 | ✅ Xuất sắc |
| PROPN | 0.67 | ⚠️ Cần cải thiện |
| NOUN | 0.82 | ✅ Tốt |

## Phân tích lỗi

### Các mẫu lỗi hàng đầu
1. PROPN → NOUN: Danh từ riêng bị nhầm với danh từ chung
2. X → PUNCT: Từ nước ngoài bị nhận diện sai thành dấu câu
3. SCONJ → ADP: Liên kết phụ bị nhầm với giới từ
4. ADJ → NOUN: Tính từ bị nhầm với danh từ
5. ADV → ADP: Trạng từ bị nhầm với giới từ

### Ví dụ dự đoán
```python
Câu: "The cat sits on the mat"
Dự đoán: The(DET) cat(NOUN) sits(NOUN) on(ADP) the(DET) mat(NOUN)
Vấn đề: "sits" dự đoán là NOUN thay vì VERB (độ tin cậy: 0.718)

Câu: "Water is essential for life"
Dự đoán: Water(VERB) is(AUX) essential(ADJ) for(ADP) life(NOUN)
Vấn đề: "Water" dự đoán là VERB thay vì NOUN (độ tin cậy: 0.305)
```

## Giải pháp chính đã áp dụng
### Giải pháp: Triển khai Early Stopping
```python
patience = 2
epochs_without_improvement = 0
if avg_dev_accuracy > best_dev_accuracy:
    best_dev_accuracy = avg_dev_accuracy
    epochs_without_improvement = 0
    # Lưu mô hình tốt nhất
else:
    epochs_without_improvement += 1

if epochs_without_improvement >= patience:
    print(f"Dừng sớm sau {epoch+1} epoch")
    break
```

# Phần 4: Named Entity Recognition với RNN


## Chi tiết triển khai
### Nhiệm vụ 1: Tải dataset
```python
# Từ lab5_rnn_for_ner.ipynb:2-3 - Tải dataset HuggingFace
import datasets
dataset = datasets.load_dataset("conll2003")

train_sentences = dataset['train']['tokens']
train_labels = dataset['train']['ner_tags']

# Quan trọng: Ánh xạ chỉ số tag sang tên
id_to_name_map = dataset["train"].features["ner_tags"].feature.names
# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
```

### Nhiệm vụ 2: Tiền xử lý dữ liệu
```python
# Từ lab5_rnn_for_ner.ipynb:9 - Xây dựng từ vựng
combined_train_data = []
for i in range(len(train_sentences)):
    combined_train_data.append(list(zip(train_sentences[i], train_labels_str[i])))

word_to_ix, tag_to_ix = build_vocabularies(combined_train_data)
print(f"Kích thước từ vựng: {len(word_to_ix)}")  # 23,625
print(f"Kích thước tag: {len(tag_to_ix)}")  # 10
```

### Nhiệm vụ 3: Kiến trúc mô hình
```python
# Từ lab5_rnn_for_ner.ipynb:17 - RNN đơn giản cho NER
class NER_RNN_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super(NER_RNN_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word_to_ix['<PAD>'])
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, output_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        rnn_out, _ = self.rnn(embeds)
        tag_space = self.hidden2tag(rnn_out.reshape(-1, rnn_out.shape[2]))
        tag_scores = tag_space.view(sentence.shape[0], sentence.shape[1], -1)
        return tag_scores
```

### Nhiệm vụ 4: Vòng lặp huấn luyện
```python
# Từ lab5_rnn_for_ner.ipynb:21 - Huấn luyện 7 epochs
num_epochs = 7
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=tag_to_ix['<PAD>'])

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for sentences, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(sentences)

        # Reshape cho CrossEntropyLoss
        outputs = outputs.view(-1, output_size)
        labels = labels.view(-1)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
```

## Vấn đề

### Vấn đề 3: Evaluation mismatch - Token vs Entity level metrics

```python
# Token-level accuracy: 92.18% (cao nhưng gây hiểu lầm)
val_loss, val_accuracy = evaluate(model, valid_dataloader, loss_fn, tag_to_ix, output_size)
print(f"Validation Accuracy: {val_accuracy:.4f}")  # 0.9218

# Entity-level F1: 63% (thực tế hơn)
from seqeval.metrics import classification_report
# Overall F1-score: 0.63 (macro avg), 0.63 (micro avg)
```

**Phân tích sâu về metric differences:**
- "O" (non-entity) tokens chiếm tỉ lệ cao→ accuracy cao khiến cho mô hình tập trung vào lớp này mà các lớp khác bị bỏ sót, nếu mô hình chỉ dự đoán 'O' cũng sẽ có độ chính xác cao.

## Kết quả hiệu suất

### Tiến trình huấn luyện
```
Epoch 1/7: Loss: 0.6325
Epoch 2/7: Loss: 0.3690
Epoch 3/7: Loss: 0.2543
Epoch 4/7: Loss: 0.1835
Epoch 5/7: Loss: 0.1351
Epoch 6/7: Loss: 0.1005
Epoch 7/7: Loss: 0.0749
```

### Kết quả đánh giá

| Metric | Giá trị | Diễn giải |
|--------|--------|-----------|
| Độ chính xác Token | 92.18% | Cao nhưng gây hiểu lầm |
| Validation Loss | 0.2676 | Hội tụ tốt |
| Entity F1 (Macro) | 0.63 | Hiệu suất vừa phải |
| Entity F1 (Micro) | 0.63 | Nhất quán trên thực thể |

### Hiệu suất cấp độ thực thể
| Loại | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| LOC | 0.75 | 0.76 | 0.75 |
| PER | 0.69 | 0.69 | 0.69 |
| MISC | 0.58 | 0.64 | 0.61 |
| ORG | 0.35 | 0.71 | 0.47 |

## Phân tích lỗi

### Ví dụ dự đoán
```python
# Mẫu 1: Ví dụ dataset
Input: "SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT ."
Dự đoán: SOCCER(O) JAPAN(B-ORG) GET(I-ORG) LUCKY(I-ORG) WIN(O) CHINA(O) IN(O) SURPRISE(B-ORG) DEFEAT(I-ORG) .
Vấn đề: Sai ranh giới thực thể, "JAPAN" nên là LOC, không phải ORG

# Mẫu 2: Ví dụ tùy chỉnh
Input: "Google was founded by Larry Page and Sergey Brin in California ."
Dự đoán: Google(B-ORG) was(O) founded(O) by(O) Larry(B-PER) Page(I-PER) and(O) Sergey(O) Brin(I-PER) in(O) California(B-LOC) .
Thành công một phần: Một số thực thể đúng, ranh giới cần cải thiện
```

### Các vấn đề chính xác định
- Khó khăn với phân biệt B-* vs I-*
- Hiệu suất thấp nhất (F1: 0.47)
- RNN đơn giản bị giới hạn cho các mẫu phức tạp
- Recall cao nhưng precision thấp cho ORG

## Giải pháp cho các vấn đề

 - Hiện tại đối với việc nhãn 'O' chiếm số lượng lớn thì notebook chưa đưa ra giải pháp, có thể cải thiện bằng cách tìm thêm các mẫu được gán nhãn chính xác hoặc giảm số lượng nhãn 'O' xuống cho độ lệch không lớn.



*Chú thích*: Các phần chèn bảng, chèn code và thông tin kết quả trong code được hỗ trợ bởi Claude Code.