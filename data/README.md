# Data Directory Structure

This directory contains datasets and metadata for the NLP project.

## File Types

### Data Files (Ignored by Git)
- **Training Data**: Large files containing actual datasets
  - `*.json` - JSON format datasets
  - `*.csv` - CSV format datasets
  - `*.conllu` - CoNLL-U format for annotated text
  - `*.txt` - Plain text datasets

### Metadata Files (Tracked by Git)
- **`LICENSE.txt`** - License information for datasets
- **`stats.xml`** - Dataset statistics and schema information
- **`README.md`** - This file with directory structure documentation

## Dataset Information

### UD_English-EWT/
Universal Dependencies English dataset containing:
- Part-of-speech tagging data
- Dependency parsing annotations
- Syntactic tree structures

### Other Datasets
- **c4-train**: Common Crawl dataset for language modeling
- **sentiments.csv**: Sentiment analysis dataset

## Dataset Schemas

### 1. Sentiment Analysis Dataset (`sentiments.csv`)
**Fields:**
- `text` (string): The input text for sentiment analysis
- `sentiment` (integer): Sentiment label (1=positive, 0=negative)

**Sample Data:**
```csv
text,sentiment
"Kickers on my watchlist XIDE TIT SOQ PNK CPW BPZ AJ trade method 1 or method 2",1
"MNTA Over 12.00",1
```

### 2. C4 Training Dataset (`c4-train.*.json`)
**Fields:**
- `text` (string): Long-form text content for language modeling
- `timestamp` (ISO 8601 datetime): When the content was created
- `url` (string): Source URL of the content

**Sample Data:**
```json
{
  "text": "Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at making delicious BBQ?",
  "timestamp": "2019-04-25T12:57:54Z",
  "url": "https://klyq.com/beginners-bbq-class-taking-place-in-missoula/"
}
```

### 3. Universal Dependencies English Dataset (`UD_English-EWT/*.conllu`)
**Format:** CoNLL-U format with tab-separated columns

**Fields:**
- ID: Token index (int)
- FORM: Word form (string)
- LEMMA: Lemma/base form (string)
- UPOS: Universal POS tag (string)
- XPOS: Language-specific POS tag (string)
- FEATS: Morphological features (string)
- HEAD: Head token index (int)
- DEPREL: Dependency relation (string)
- DEPS: Enhanced dependencies (string)
- MISC: Miscellaneous annotations (string)

**Sample Data:**
```
1	From	from	ADP	IN	_	3	case	3:case	_
2	the	the	DET	DT	Definite=Def|PronType=Art	3	det	3:det	_
3	AP	AP	PROPN	NNP	Number=Sing	4	obl	4:obl:from	_
```

## General Data Conventions
- Text files use UTF-8 encoding
- CSV files use comma separators with headers
- JSON files follow standard JSON format
- CoNLL-U files follow Universal Dependencies format
- All datasets are pre-processed and cleaned for NLP tasks