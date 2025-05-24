# Lemmatization
### 🍃 What is Lemmatization?
**Lemmatization** is the process of reducing a word to its **base or dictionary** form (**lemma**). Unlike stemming, lemmatization uses **vocabulary and morphological analysis**, so it always returns a **real word**.

#### 📌 Example:
* "running", "ran" → "run"
* "better" → "good"

### 🎯 Why is Lemmatization Used in Text Processing?
| Purpose                     | Description                                            |
|:--------------------------- |:------------------------------------------------------ |
| 🧠 Linguistic Accuracy      | Considers the context and part-of-speech               |
| 🔎 Search Normalization     | Helps group word variations together                   |
| 🔬 Improves Model Precision | More meaningful preprocessing for ML/NLP tasks         |
| ✅ Real Word Output          | Always returns real dictionary words (unlike stemming) |



```python
paragraph = """
The children were running in the playground. They had played several games already.
One boy was reading while others were talking.
Their teacher encouraged them to read, write, and speak.
Books were shared among the students.
Each child had a favorite book and read it daily.
They spoke fluently and improved every day.
The principal praised their progress.
Writing, reading, and speaking were regular habits.
Even during holidays, they liked studying.
Learning became a part of their lifestyle.
"""

```


```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
```


```python
# downloads required
# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
```


```python
# Initialize tokenizer and lemmatizer
lemmatizer = WordNetLemmatizer()

# Tokenize paragraph into words
words = word_tokenize(paragraph)
```


```python
# Lemmatize each word
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
```


```python
print("\nLemmatized Words (no POS):\n", lemmatized_words)
```

    
    Lemmatized Words (no POS):
     ['The', 'child', 'were', 'running', 'in', 'the', 'playground', '.', 'They', 'had', 'played', 'several', 'game', 'already', '.', 'One', 'boy', 'wa', 'reading', 'while', 'others', 'were', 'talking', '.', 'Their', 'teacher', 'encouraged', 'them', 'to', 'read', ',', 'write', ',', 'and', 'speak', '.', 'Books', 'were', 'shared', 'among', 'the', 'student', '.', 'Each', 'child', 'had', 'a', 'favorite', 'book', 'and', 'read', 'it', 'daily', '.', 'They', 'spoke', 'fluently', 'and', 'improved', 'every', 'day', '.', 'The', 'principal', 'praised', 'their', 'progress', '.', 'Writing', ',', 'reading', ',', 'and', 'speaking', 'were', 'regular', 'habit', '.', 'Even', 'during', 'holiday', ',', 'they', 'liked', 'studying', '.', 'Learning', 'became', 'a', 'part', 'of', 'their', 'lifestyle', '.']
    


```python
'''
POS- Noun-n
verb-v
adjective-a
adverb-r
'''
```




    '\nPOS- Noun-n\nverb-v\nadjective-a\nadverb-r\n'




```python
# Lemmatize each word
lemmatized_words = [lemmatizer.lemmatize(word, pos="v") for word in words]
```


```python
for word in words:
    lemmatized_words = lemmatizer.lemmatize(word, pos="v")
    print(lemmatized_words)
```

    The
    children
    be
    run
    in
    the
    playground
    .
    They
    have
    play
    several
    game
    already
    .
    One
    boy
    be
    read
    while
    others
    be
    talk
    .
    Their
    teacher
    encourage
    them
    to
    read
    ,
    write
    ,
    and
    speak
    .
    Books
    be
    share
    among
    the
    students
    .
    Each
    child
    have
    a
    favorite
    book
    and
    read
    it
    daily
    .
    They
    speak
    fluently
    and
    improve
    every
    day
    .
    The
    principal
    praise
    their
    progress
    .
    Writing
    ,
    read
    ,
    and
    speak
    be
    regular
    habit
    .
    Even
    during
    holiday
    ,
    they
    like
    study
    .
    Learning
    become
    a
    part
    of
    their
    lifestyle
    .
    


```python
words=["running","ran","played","playing","plays","programming","programs","history","finally","finalized"]
```


```python
for word in words:
    print(word+"---->"+lemmatizer.lemmatize(word,pos='v'))
```

    running---->run
    ran---->run
    played---->play
    playing---->play
    plays---->play
    programming---->program
    programs---->program
    history---->history
    finally---->finally
    finalized---->finalize
    
