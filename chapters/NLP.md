# Natural Language Processing (NLP)
### 📌 What is NLP?
**Natural Language Processing (NLP)** is a branch of Artificial Intelligence (AI) that helps computers understand, interpret, and generate human language.

It combines:
* **Linguistics** (rules of language)
* **Computer Science** (data structures, algorithms)
* **Statistics & Machine Learning** (to find patterns)

### 🎯 Purpose of NLP

| Area               | Goal                                   |
| ------------------ | -------------------------------------- |
| Understanding      | Extract meaning from human language    |
| Translation        | Convert from one language to another   |
| Classification     | Categorize text (e.g., spam detection) |
| Generation         | Generate responses (e.g., ChatGPT)     |
| Summarization      | Condense long documents                |
| Sentiment Analysis | Understand emotions in text            |

### 🔖 Applications of NLP
* **Search Engines** (e.g., Google): Better interpretation of queries.
* **Voice Assistants** (e.g., Siri, Alexa): Understand spoken commands.
* **Customer Support**: Chatbots automate replies to common questions.
* **Machine Translation**: Google Translate uses NLP to convert one language into another.
* **Email Filtering**: Identify and filter spam or categorize emails.

**NLTK** and **spaCy** are two very commonly used library for NLP. We will use NLTK for exaplanation and later we will also see how to use spaCy.

### Installing nltk library


```python
!pip install nltk
```

### Important terminologies:

#### Corpus: 
* Just like whole paragraph. A **corpus** is a structured dataset of text or speech that is representative of a particular language or language variety.

#### Documents: 
* Sentences used in corpus.

#### Vocabulary:
* Unique words used in corpus/sentence.

#### Word:
* Word used in a corpus or sentence

#### Tokenization
* Break text into individual words or sentences.

#### Stemming
* Reduces a word to its stem (e.g. **running** --> **run**)

#### Lemmatization
* Lemmatization returns the dictionary form of a word (e.g., "better" → "good").

#### Stop Words
* Common words (e.g., "the", "is") that add little meaning.

#### Parts-of-Speech Tagging
* Assigning parts of speech like noun, verb, etc.

#### Named Entity Recognition
* Identifying names, places, dates, etc., in text.

#### One Hot Encoding
* A way to represent text where each word is a unique binary vector.

#### Bag of Words
* Represents documents as word count vectors.

#### N-Grams
* Sequences of N consecutive words.

#### TF-IDF (Term Frequency-Inverse Document Frequency)
* Highlights important words in a document that are rare in the corpus.

#### Word Embedding
* Dense numerical representation of words capturing semantic meaning.

#### Word2Vec
* Algorithm to learn vector representations of words using neural networks.

#### Average Word2Vec
* Average of all word vectors in a sentence to represent it.
