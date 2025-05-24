# Named Entity Recognition

### 🧠 What is Named Entity Recognition (NER)?
**Named Entity Recognition (NER)** is an NLP task that identifies and categorizes entities in text into predefined categories such as:
* **Person** (e.g., "Sandeep")
* **Organization** (e.g., "Odin")
* **Location** (e.g., "India", "Lucknow")
* **Date**, **Time**, **Money**, **Percent**, etc.

### 🎯 Why is NER Used in Text Processing?

| Purpose                   | Explanation                                              |
|:------------------------- |:-------------------------------------------------------- |
| 🔍 Information Extraction | Helps extract names, places, events from large documents |
| 📂 Text Structuring       | Adds semantic meaning to raw text                        |
| 🤖 Dialogue Systems       | Enables chatbots to recognize real-world entities        |
| 🧠 Downstream NLP Tasks   | Useful for question answering, summarization, and more   |



```python
paragraph = """
Barack Obama was born in Hawaii and served as the 44th President of the United States.
He studied at Columbia University and later attended Harvard Law School.
Obama joined the Illinois State Senate in 1997.
Microsoft and Google have invested in artificial intelligence technologies.
Elon Musk leads companies like Tesla and SpaceX based in California.
The COVID-19 pandemic started in 2019 and affected many countries.
Amazon's headquarters is located in Seattle.
Apple Inc. unveiled the iPhone 13 in September 2021.
India and the United Kingdom held a diplomatic summit last year.
The World Health Organization declared COVID-19 a global emergency.
"""

```


```python
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
```


```python
# Download necessary resources
#nltk.download('punkt')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('averaged_perceptron_tagger')
```


```python
# Tokenize into sentences
sentences = nltk.sent_tokenize(paragraph)

print("Named Entities:\n")
```

    Named Entities:
    
    


```python
# Process each sentence
for sent in sentences:
    words = word_tokenize(sent)
    tagged = pos_tag(words)                  # Step 1: POS tagging
    named_ent = ne_chunk(tagged)            # Step 2: Named Entity Recognition
    print(named_ent)
```

    (S
      (PERSON Barack/NNP)
      (PERSON Obama/NNP)
      was/VBD
      born/VBN
      in/IN
      (GPE Hawaii/NNP)
      and/CC
      served/VBD
      as/IN
      the/DT
      44th/CD
      President/NNP
      of/IN
      the/DT
      (GPE United/NNP States/NNPS)
      ./.)
    (S
      He/PRP
      studied/VBD
      at/IN
      (ORGANIZATION Columbia/NNP University/NNP)
      and/CC
      later/RB
      attended/VBD
      (PERSON Harvard/NNP Law/NNP School/NNP)
      ./.)
    (S
      (PERSON Obama/NNP)
      joined/VBD
      the/DT
      (ORGANIZATION Illinois/NNP)
      State/NNP
      Senate/NNP
      in/IN
      1997/CD
      ./.)
    (S
      (PERSON Microsoft/NNP)
      and/CC
      (GPE Google/NNP)
      have/VBP
      invested/VBN
      in/IN
      artificial/JJ
      intelligence/NN
      technologies/NNS
      ./.)
    (S
      (PERSON Elon/NNP)
      (ORGANIZATION Musk/NNP)
      leads/VBZ
      companies/NNS
      like/IN
      (PERSON Tesla/NNP)
      and/CC
      (ORGANIZATION SpaceX/NNP)
      based/VBN
      in/IN
      (GPE California/NNP)
      ./.)
    (S
      The/DT
      COVID-19/NNP
      pandemic/NN
      started/VBD
      in/IN
      2019/CD
      and/CC
      affected/VBD
      many/JJ
      countries/NNS
      ./.)
    (S
      (GPE Amazon/NNP)
      's/POS
      headquarters/NN
      is/VBZ
      located/VBN
      in/IN
      (GPE Seattle/NNP)
      ./.)
    (S
      (PERSON Apple/NNP)
      (ORGANIZATION Inc./NNP)
      unveiled/VBD
      the/DT
      (ORGANIZATION iPhone/NN)
      13/CD
      in/IN
      September/NNP
      2021/CD
      ./.)
    (S
      (GPE India/NNP)
      and/CC
      the/DT
      (ORGANIZATION United/NNP Kingdom/NNP)
      held/VBD
      a/DT
      diplomatic/JJ
      summit/NN
      last/JJ
      year/NN
      ./.)
    (S
      The/DT
      (ORGANIZATION World/NNP)
      Health/NNP
      Organization/NNP
      declared/VBD
      COVID-19/NNP
      a/DT
      global/JJ
      emergency/NN
      ./.)
    


```python
# creates interesting graph
words=nltk.word_tokenize(paragraph)
tag_elements=nltk.pos_tag(words)
# creates interesting graph
nltk.ne_chunk(tag_elements).draw()
```

### 🔍 What Happens Internally?

| Step              | Tool/Method                                                | Description |
|:----------------- |:---------------------------------------------------------- |:----------- |
| `word_tokenize()` | Tokenizes sentence into words                              |             |
| `pos_tag()`       | Tags each word with part-of-speech                         |             |
| `ne_chunk()`      | Performs NER and builds a parse tree with labeled entities |             |


### 🧠 Common NER Tags in NLTK

| Tag                                | Meaning                           |
|:---------------------------------- |:--------------------------------- |
| `PERSON`                           | Names of people                   |
| `ORGANIZATION`                     | Companies, agencies, institutions |
| `GPE`                              | Countries, cities, states         |
| `LOCATION`                         | Geographical locations            |
| `FACILITY`                         | Buildings, airports, highways     |
| `DATE`, `TIME`, `MONEY`, `PERCENT` | Self-explanatory                  |

### ✅ Summary

| Feature                 | Why Important                                        |
|:----------------------- |:---------------------------------------------------- |
| **NER**                 | Extracts structured information from raw text        |
| **NLTK’s `ne_chunk()`** | A fast, rule-based way to identify entities          |
| **Real-world use**      | Search engines, news analysis, chatbots, data mining |

