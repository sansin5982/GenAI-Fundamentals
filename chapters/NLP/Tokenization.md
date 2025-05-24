# Tokenization

### ✂️ What is Tokenization?
**Tokenization** is the process of breaking down a **string of text** into **smaller units called tokens**. These tokens can be **words**, **characters**, or **sentences**.

### 🎯 Why is Tokenization Used in Text Processing?
| Purpose                | Reason                                                             |
|:---------------------- |:--------------------------------------------------------------------------- |
| 📦 Structuring Text    | Raw text is messy. Tokenization converts it into structured units.          |
| 📚 Basis for NLP Tasks | Tasks like POS tagging, sentiment analysis, translation all require tokens. |
| 🔍 Enables Analysis    | Enables counting, vectorization, and semantic understanding.                |
| 🧹 First Step          | It's usually the **first step** in any NLP pipeline.                        |
* Without tokenization, machines **cannot understand** where one word ends and another begins.


```python
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required models
#nltk.download('punkt')

# Sample 10-line paragraph (can also be multi-line string)
paragraph = """
Natural Language Processing is a fascinating field of Artificial Intelligence.
It helps computers understand human language.
Tokenization is the first step in NLP tasks.
It breaks large texts into sentences and words.
Without tokenization, deeper analysis like POS tagging is difficult.
NLTK is a powerful Python library for NLP.
It includes tools for tokenizing, tagging, and parsing text.
Sentence tokenization splits the paragraph into sentences.
Word tokenization further splits each sentence into individual words.
Let’s see how it works using NLTK in Python!
"""

```

## sent_tokenize() 
### 📌 What is sent_tokenize()?
**`sent_tokenize()`** is a function from the nltk.tokenize module that:
* Splits a **paragraph or document into individual sentences**.
* Uses a **pre-trained Punkt sentence tokenizer** (unsupervised algorithm).
* Recognizes abbreviations and handles sentence-ending punctuation (like ., !, ?) intelligently.


```python
# Sentence Tokenization
sentences = sent_tokenize(paragraph)
print("Sentence Tokens:")
print(sentences)
```

    Sentence Tokens:
    ['\nNatural Language Processing is a fascinating field of Artificial Intelligence.', 'It helps computers understand human language.', 'Tokenization is the first step in NLP tasks.', 'It breaks large texts into sentences and words.', 'Without tokenization, deeper analysis like POS tagging is difficult.', 'NLTK is a powerful Python library for NLP.', 'It includes tools for tokenizing, tagging, and parsing text.', 'Sentence tokenization splits the paragraph into sentences.', 'Word tokenization further splits each sentence into individual words.', 'Let’s see how it works using NLTK in Python!']
    

## word_tokenize() 
### 📌 What is word_tokenize()?
**`word_tokenize()`** is a tokenizer from the nltk.tokenize module that:
* Uses the Treebank tokenizer under the hood (so it's linguistically smart).
* Splits text into words and punctuation, while keeping common language rules in mind.
* Handles contractions and abbreviations appropriately.
* Keeps punctuation like periods (.) and commas (,) as separate tokens only when necessary.


```python
# Word Tokenization
words = word_tokenize(paragraph)
print("\nWord Tokens:")
print(words)
```

    
    Word Tokens:
    ['Natural', 'Language', 'Processing', 'is', 'a', 'fascinating', 'field', 'of', 'Artificial', 'Intelligence', '.', 'It', 'helps', 'computers', 'understand', 'human', 'language', '.', 'Tokenization', 'is', 'the', 'first', 'step', 'in', 'NLP', 'tasks', '.', 'It', 'breaks', 'large', 'texts', 'into', 'sentences', 'and', 'words', '.', 'Without', 'tokenization', ',', 'deeper', 'analysis', 'like', 'POS', 'tagging', 'is', 'difficult', '.', 'NLTK', 'is', 'a', 'powerful', 'Python', 'library', 'for', 'NLP', '.', 'It', 'includes', 'tools', 'for', 'tokenizing', ',', 'tagging', ',', 'and', 'parsing', 'text', '.', 'Sentence', 'tokenization', 'splits', 'the', 'paragraph', 'into', 'sentences', '.', 'Word', 'tokenization', 'further', 'splits', 'each', 'sentence', 'into', 'individual', 'words', '.', 'Let', '’', 's', 'see', 'how', 'it', 'works', 'using', 'NLTK', 'in', 'Python', '!']
    

## wordpunct_tokenize
### 📌 What is wordpunct_tokenize?
**`wordpunct_tokenize()`** is a tokenizer from the nltk.tokenize module that:
* Splits text by **words and punctuation separately**.
* It **treats punctuation as separate tokens** — unlike word_tokenize which groups some punctuation with words.


```python
# WordPunct tokenization
from nltk.tokenize import wordpunct_tokenize
print("\nwordpunct_tokenize:")
print(wordpunct_tokenize(paragraph))
```

    
    wordpunct_tokenize:
    ['Natural', 'Language', 'Processing', 'is', 'a', 'fascinating', 'field', 'of', 'Artificial', 'Intelligence', '.', 'It', 'helps', 'computers', 'understand', 'human', 'language', '.', 'Tokenization', 'is', 'the', 'first', 'step', 'in', 'NLP', 'tasks', '.', 'It', 'breaks', 'large', 'texts', 'into', 'sentences', 'and', 'words', '.', 'Without', 'tokenization', ',', 'deeper', 'analysis', 'like', 'POS', 'tagging', 'is', 'difficult', '.', 'NLTK', 'is', 'a', 'powerful', 'Python', 'library', 'for', 'NLP', '.', 'It', 'includes', 'tools', 'for', 'tokenizing', ',', 'tagging', ',', 'and', 'parsing', 'text', '.', 'Sentence', 'tokenization', 'splits', 'the', 'paragraph', 'into', 'sentences', '.', 'Word', 'tokenization', 'further', 'splits', 'each', 'sentence', 'into', 'individual', 'words', '.', 'Let', '’', 's', 'see', 'how', 'it', 'works', 'using', 'NLTK', 'in', 'Python', '!']
    

#### Comparison Table
| Feature                    | `word_tokenize()`                 | `wordpunct_tokenize()`           |
| -------------------------- | --------------------------------- | -------------------------------- |
| Handles contractions       | Yes (e.g., “don’t” → “do”, “n’t”) | No (keeps as one word)           |
| Separates punctuation      | Partially                         | Fully (punctuation is separate)  |
| Language-aware             | Yes                               | No (simple regex-based)          |
| Suitable for preprocessing | ✅ Best for general NLP            | ✅ Best for simple text splitting |


### TreebankWordTokenizer
#### 📌 What is TreebankWordTokenizer?
TreebankWordTokenizer is a tokenizer in NLTK that splits text using the Penn Treebank conventions — commonly used in linguistic corpora.

It:
* Handles punctuation smartly
* Splits contractions (e.g., “don’t” → “do” + “n’t”)
* Separates punctuation (e.g., `"."`, `","`, `":"`)
* Keeps consistency with syntactic treebanks


```python
from nltk.tokenize import TreebankWordTokenizer

paragraph = """
Mr. Smith isn't coming today. He's been delayed — possibly by traffic.
However, Dr. Brown, who arrived earlier, said, "Let's begin without him!"
"""

# Initialize tokenizer
treebank_tokenizer = TreebankWordTokenizer()

# Apply tokenizer
tokens = treebank_tokenizer.tokenize(paragraph)

# Display output
print("Treebank Tokens:")
print(tokens)

```

    Treebank Tokens:
    ['Mr.', 'Smith', 'is', "n't", 'coming', 'today.', 'He', "'s", 'been', 'delayed', '—', 'possibly', 'by', 'traffic.', 'However', ',', 'Dr.', 'Brown', ',', 'who', 'arrived', 'earlier', ',', 'said', ',', '``', 'Let', "'s", 'begin', 'without', 'him', '!', "''"]
    
