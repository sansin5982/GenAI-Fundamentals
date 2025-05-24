# Stemming
### 🌿 What is Stemming?
**Stemming** is the process of reducing a word to its **base or root** form by chopping off suffixes and prefixes.
* It doesn't always produce real words, but it helps in grouping similar words.
* Example: `"playing"`, `"played"`, `"plays"` → `"play"`

### 🎯 Why is Stemming Used in Text Processing?

| Purpose                      | Description                                                    |
|:---------------------------- |:-------------------------------------------------------------- |
| 🔍 Normalize Text            | Brings related words to a common root for analysis             |
| 📉 Reduce Dimensionality     | Fewer unique words → better model efficiency                   |
| 🔎 Improve Search & Matching | Helps in matching “compute” with “computing”, “computer”, etc. |
| 🧠 Basis for NLP Models      | Common in search engines, text classification, and clustering  |



```python
paragraph = """
The students were studying various NLP algorithms. They were playing with different models and analyzing the results.
Some were focusing on stemming, while others were interested in lemmatization.
They played multiple times to test consistency.
Understanding the difference between stemming and lemmatization is important.
Researchers analyze, compute, and compare outputs from different tools.
Models like logistic regression and decision trees were applied.
The data was preprocessed and tokenized before training.
They tested the models repeatedly to improve accuracy.
Text normalization was one of the main goals.
They continue improving their pipeline daily.
"""

```

## 🔍 Explanation of Stemmers
| Stemmer| Description| Pros | Cons                                                        |
|:------------------- |:----------------|:----|:-------------------------- |
| **PorterStemmer**   | Classic stemmer developed in 1980. Uses heuristic rules.        | Simple, fast, widely used  | Sometimes too aggressive (e.g., “univers” for “university”) |
| **SnowballStemmer** | Improved version of PorterStemmer. Supports multiple languages. | More accurate & consistent | Slightly slower                                             |
| **RegexpStemmer**   | Uses regular expressions to strip suffixes                      | Highly customizable        | Not intelligent; purely rule-based                          |



```python
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer, RegexpStemmer

# Download required resources
# nltk.download('punkt')

# Tokenize sentences and words
sentences = sent_tokenize(paragraph)
print("Sentences:\n", sentences)

words = word_tokenize(paragraph)
print("\nOriginal Words:\n", words)
```

    Sentences:
     ['\nThe students were studying various NLP algorithms.', 'They were playing with different models and analyzing the results.', 'Some were focusing on stemming, while others were interested in lemmatization.', 'They played multiple times to test consistency.', 'Understanding the difference between stemming and lemmatization is important.', 'Researchers analyze, compute, and compare outputs from different tools.', 'Models like logistic regression and decision trees were applied.', 'The data was preprocessed and tokenized before training.', 'They tested the models repeatedly to improve accuracy.', 'Text normalization was one of the main goals.', 'They continue improving their pipeline daily.']
    
    Original Words:
     ['The', 'students', 'were', 'studying', 'various', 'NLP', 'algorithms', '.', 'They', 'were', 'playing', 'with', 'different', 'models', 'and', 'analyzing', 'the', 'results', '.', 'Some', 'were', 'focusing', 'on', 'stemming', ',', 'while', 'others', 'were', 'interested', 'in', 'lemmatization', '.', 'They', 'played', 'multiple', 'times', 'to', 'test', 'consistency', '.', 'Understanding', 'the', 'difference', 'between', 'stemming', 'and', 'lemmatization', 'is', 'important', '.', 'Researchers', 'analyze', ',', 'compute', ',', 'and', 'compare', 'outputs', 'from', 'different', 'tools', '.', 'Models', 'like', 'logistic', 'regression', 'and', 'decision', 'trees', 'were', 'applied', '.', 'The', 'data', 'was', 'preprocessed', 'and', 'tokenized', 'before', 'training', '.', 'They', 'tested', 'the', 'models', 'repeatedly', 'to', 'improve', 'accuracy', '.', 'Text', 'normalization', 'was', 'one', 'of', 'the', 'main', 'goals', '.', 'They', 'continue', 'improving', 'their', 'pipeline', 'daily', '.']
    


```python
# Initialize PorterStemmer
porter = PorterStemmer()


print("\nPorter Stemmer:")
print([porter.stem(w) for w in words])

```

    
    Porter Stemmer:
    ['the', 'student', 'were', 'studi', 'variou', 'nlp', 'algorithm', '.', 'they', 'were', 'play', 'with', 'differ', 'model', 'and', 'analyz', 'the', 'result', '.', 'some', 'were', 'focus', 'on', 'stem', ',', 'while', 'other', 'were', 'interest', 'in', 'lemmat', '.', 'they', 'play', 'multipl', 'time', 'to', 'test', 'consist', '.', 'understand', 'the', 'differ', 'between', 'stem', 'and', 'lemmat', 'is', 'import', '.', 'research', 'analyz', ',', 'comput', ',', 'and', 'compar', 'output', 'from', 'differ', 'tool', '.', 'model', 'like', 'logist', 'regress', 'and', 'decis', 'tree', 'were', 'appli', '.', 'the', 'data', 'wa', 'preprocess', 'and', 'token', 'befor', 'train', '.', 'they', 'test', 'the', 'model', 'repeatedli', 'to', 'improv', 'accuraci', '.', 'text', 'normal', 'wa', 'one', 'of', 'the', 'main', 'goal', '.', 'they', 'continu', 'improv', 'their', 'pipelin', 'daili', '.']
    


```python
# Initialize SnowballStemmer
snowball = SnowballStemmer("english")
print("\nSnowball Stemmer:")
print([snowball.stem(w) for w in words])

```

    
    Snowball Stemmer:
    ['the', 'student', 'were', 'studi', 'various', 'nlp', 'algorithm', '.', 'they', 'were', 'play', 'with', 'differ', 'model', 'and', 'analyz', 'the', 'result', '.', 'some', 'were', 'focus', 'on', 'stem', ',', 'while', 'other', 'were', 'interest', 'in', 'lemmat', '.', 'they', 'play', 'multipl', 'time', 'to', 'test', 'consist', '.', 'understand', 'the', 'differ', 'between', 'stem', 'and', 'lemmat', 'is', 'import', '.', 'research', 'analyz', ',', 'comput', ',', 'and', 'compar', 'output', 'from', 'differ', 'tool', '.', 'model', 'like', 'logist', 'regress', 'and', 'decis', 'tree', 'were', 'appli', '.', 'the', 'data', 'was', 'preprocess', 'and', 'token', 'befor', 'train', '.', 'they', 'test', 'the', 'model', 'repeat', 'to', 'improv', 'accuraci', '.', 'text', 'normal', 'was', 'one', 'of', 'the', 'main', 'goal', '.', 'they', 'continu', 'improv', 'their', 'pipelin', 'daili', '.']
    

#### Issues


```python
porter.stem("fairly"),porter.stem("sportingly") # still issue
```




    ('fairli', 'sportingli')




```python
snowball.stem("fairly"),snowball.stem("sportingly") # still issue
```




    ('fair', 'sport')




```python
# Some words may still have issues
snowball.stem('goes')
```




    'goe'




```python
porter.stem('goes')
```




    'goe'




```python
# Initialize RegexpStemmer
regexp = RegexpStemmer('ing$|ed$|s$', min=4)  # remove common suffixes
print("\nRegexp Stemmer:")
print([regexp.stem(w) for w in words])
```

    
    Regexp Stemmer:
    ['The', 'student', 'were', 'study', 'variou', 'NLP', 'algorithm', '.', 'They', 'were', 'play', 'with', 'different', 'model', 'and', 'analyz', 'the', 'result', '.', 'Some', 'were', 'focus', 'on', 'stemm', ',', 'while', 'other', 'were', 'interest', 'in', 'lemmatization', '.', 'They', 'play', 'multiple', 'time', 'to', 'test', 'consistency', '.', 'Understand', 'the', 'difference', 'between', 'stemm', 'and', 'lemmatization', 'is', 'important', '.', 'Researcher', 'analyze', ',', 'compute', ',', 'and', 'compare', 'output', 'from', 'different', 'tool', '.', 'Model', 'like', 'logistic', 'regression', 'and', 'decision', 'tree', 'were', 'appli', '.', 'The', 'data', 'was', 'preprocess', 'and', 'tokeniz', 'before', 'train', '.', 'They', 'test', 'the', 'model', 'repeatedly', 'to', 'improve', 'accuracy', '.', 'Text', 'normalization', 'was', 'one', 'of', 'the', 'main', 'goal', '.', 'They', 'continue', 'improv', 'their', 'pipeline', 'daily', '.']
    

### 🧠 Key Points
Stemming is language-agnostic but rule-based.

Helps in converting different forms of a word to a common stem.

Essential for information retrieval, text classification, topic modeling, etc.
