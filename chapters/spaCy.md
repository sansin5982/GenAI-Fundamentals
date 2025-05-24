# NLP with spaCy
### 📦 What is spaCy?
**spaCy** is an advanced open-source NLP library in Python designed for:
    * Efficiency
    * Robust pipelines
    * Industrial-strength NLP

It supports:
* Tokenization
* POS tagging
* Lemmatization
* Named Entity Recognition (NER)
* Dependency parsing
* Word vectors (embeddings)



## Installation and Imports


```python
#!pip install spacy
#!python -m spacy download en_core_web_sm
```

## Load spaCy English Model


```python
import spacy
nlp = spacy.load("en_core_web_sm")
```


```python
text = """
Apple is looking at buying U.K. startup for $1 billion. Tim Cook, the CEO of Apple, visited London in June 2023.
"""

```

## Sentence Segmentation
Breaking down text into individual sentences.


```python
doc = nlp(text)
[sent.text for sent in doc.sents]
```




    ['\nApple is looking at buying U.K. startup for $1 billion.',
     'Tim Cook, the CEO of Apple, visited London in June 2023.\n']



## Tokenization
* Breaking a sentence into words or punctuation marks.
* Splits the text into meaningful units (tokens).
* Forms the base for all other processing.
* Identifies words, numbers, punctuation, etc.


```python
[token.text for token in doc]
```




    ['\n',
     'Apple',
     'is',
     'looking',
     'at',
     'buying',
     'U.K.',
     'startup',
     'for',
     '$',
     '1',
     'billion',
     '.',
     'Tim',
     'Cook',
     ',',
     'the',
     'CEO',
     'of',
     'Apple',
     ',',
     'visited',
     'London',
     'in',
     'June',
     '2023',
     '.',
     '\n']




```python
for token in doc:
    print(token.text)
```

    
    
    Apple
    is
    looking
    at
    buying
    U.K.
    startup
    for
    $
    1
    billion
    .
    Tim
    Cook
    ,
    the
    CEO
    of
    Apple
    ,
    visited
    London
    in
    June
    2023
    .
    
    
    

* Spacy does not support stemming

## Lemmatization

Returning the base form of each word (lemma).


```python
print("\nLemmatization:")
for token in doc:
    print(f"{token.text:15} {token.pos_:10} {token.lemma_}")
```

    
    Lemmatization:
    
                   SPACE      
    
    Apple           PROPN      Apple
    is              AUX        be
    looking         VERB       look
    at              ADP        at
    buying          VERB       buy
    U.K.            PROPN      U.K.
    startup         VERB       startup
    for             ADP        for
    $               SYM        $
    1               NUM        1
    billion         NUM        billion
    .               PUNCT      .
    Tim             PROPN      Tim
    Cook            PROPN      Cook
    ,               PUNCT      ,
    the             DET        the
    CEO             NOUN       ceo
    of              ADP        of
    Apple           PROPN      Apple
    ,               PUNCT      ,
    visited         VERB       visit
    London          PROPN      London
    in              ADP        in
    June            PROPN      June
    2023            NUM        2023
    .               PUNCT      .
    
                   SPACE      
    
    

## Stop Words
Filtering common words that add little meaning.


```python
print(nlp.Defaults.stop_words)
```

    {'becomes', 'just', 'third', 'same', 'several', 'this', 'always', 'besides', 'where', 'will', 'somewhere', 'are', 'in', 'out', 'quite', 'above', 'four', 'his', 'by', 'your', 'due', 'none', '’ll', 'must', 'if', 'since', 'whereupon', 'name', 'whenever', 'such', 'fifty', 'indeed', 'nevertheless', 'onto', 'not', 'the', 'both', 'else', 'from', 'along', "'m", 'should', 'though', 'please', 'behind', 'less', 'go', 'whoever', 'or', 'toward', 'more', 'everything', 'all', 'hereafter', "'ll", 'at', 'even', 'see', 'n‘t', 'empty', 'am', 'how', 'five', 'front', 'those', 'twelve', 'namely', 'still', 'anything', 'moreover', 'move', 'once', 'take', 'throughout', 'together', 'seems', 'very', 'while', 'someone', 'between', 'what', 'full', 'rather', 'ten', '‘d', 'therefore', 'thus', 'does', 'had', 'might', 'somehow', 'itself', 'therein', 'unless', 'others', 'among', 'beside', 'least', 'yourselves', '’m', 'also', 'under', 'being', 'become', 'often', 'her', 'except', 'can', 'wherever', 'whole', 'sometimes', 'per', 'bottom', 'thereupon', 'two', 'and', 'amongst', 'anyhow', 'almost', 'again', 'before', 'hers', 'herein', 'part', 'next', 'there', 'why', 'wherein', 'fifteen', 'them', 'never', 'thereafter', 'eight', 'until', 'although', 'really', 'noone', 'when', 'during', 'made', 'whom', 'did', 'over', 'give', 'anywhere', 'seeming', 'around', 'seem', 'than', 'formerly', 'side', 'most', 'twenty', 'you', 'some', 'into', 'keep', 'meanwhile', 'of', 'because', 'latter', 'down', 'three', 'us', 'however', 'show', '’s', 'hereby', 'hundred', 'through', 'been', 'latterly', 'forty', 'upon', 'ourselves', 'regarding', 'as', 'much', 'various', 'to', 'neither', 'our', 'doing', 'otherwise', 'about', 'me', 'has', 'nine', "'d", 'top', 'own', 'get', 'its', '‘s', 'other', 'myself', 'who', '‘ve', '’re', 'everywhere', 'hereupon', "n't", 'on', 'last', 'could', 'perhaps', 'hence', 'further', 'nor', 'whence', 'many', 'former', 'anyone', 'within', 'beforehand', 'elsewhere', 'sometime', 'thence', 'is', 'either', 'one', 'something', 'whatever', 'yours', 'up', 'may', 'an', 'now', 'via', 'were', 'yet', 'everyone', 'enough', 'themselves', '’d', '‘ll', 'their', 'say', 'would', 'ever', 'each', 'towards', "'s", 'afterwards', 'i', 'becoming', 'already', "'re", 'any', 'nobody', 'have', 'below', 'him', 'serious', 'do', 'used', 'these', 'too', 'few', 'became', 'thru', 'we', 'using', 'so', 'amount', 'whereby', '’ve', 'himself', 'put', 'done', 'sixty', 'call', 'well', 'he', 'across', "'ve", 'she', 'thereby', 'but', 'only', 'whereafter', 'after', 'seemed', 'against', 'without', 'a', 'every', 'was', 'which', 'be', 'my', 'mine', 'whether', 'n’t', 'whose', 'anyway', 'yourself', 're', 'whither', 'that', 'cannot', 'ours', '‘m', 'another', '‘re', 'whereas', 'no', 'nowhere', 'off', 'then', 'first', 'alone', 'eleven', 'here', 'back', 'nothing', 'beyond', 'make', 'it', 'they', 'six', 'mostly', 'ca', 'herself', 'for', 'with'}
    


```python
# Check if a Word is a Stop Word
doc1 = nlp("This is an example sentence with some common words.")

for token in doc1:
    print(f"{token.text:10} → Stop Word: {token.is_stop}")

```

    This       → Stop Word: True
    is         → Stop Word: True
    an         → Stop Word: True
    example    → Stop Word: False
    sentence   → Stop Word: False
    with       → Stop Word: True
    some       → Stop Word: True
    common     → Stop Word: False
    words      → Stop Word: False
    .          → Stop Word: False
    


```python
# Remove Stop Words from Text
filtered_tokens = [token.text for token in doc1 if not token.is_stop and token.is_alpha]
print(filtered_tokens)
```

    ['example', 'sentence', 'common', 'words']
    


```python
# Add a Custom Stop Word
nlp.vocab["customword"].is_stop = True
```


```python
# Remove a Word from Stop List
nlp.vocab["customword"].is_stop = False

```

## Part-of-Speech Tagging
Labeling each word with its grammatical role.


```python
print("\nPOS Tags:")
for token in doc:
    print(f"{token.text:15} {token.pos_:10} {token.tag_}")
```

    
    POS Tags:
    
                   SPACE      _SP
    Apple           PROPN      NNP
    is              AUX        VBZ
    looking         VERB       VBG
    at              ADP        IN
    buying          VERB       VBG
    U.K.            PROPN      NNP
    startup         VERB       VBD
    for             ADP        IN
    $               SYM        $
    1               NUM        CD
    billion         NUM        CD
    .               PUNCT      .
    Tim             PROPN      NNP
    Cook            PROPN      NNP
    ,               PUNCT      ,
    the             DET        DT
    CEO             NOUN       NN
    of              ADP        IN
    Apple           PROPN      NNP
    ,               PUNCT      ,
    visited         VERB       VBD
    London          PROPN      NNP
    in              ADP        IN
    June            PROPN      NNP
    2023            NUM        CD
    .               PUNCT      .
    
                   SPACE      _SP
    

## Named Entity Recognition
Identifying named entities such as people, locations, and organizations.


```python
print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text:30} → {ent.label_}")
```

    
    Named Entities:
    Apple                          → ORG
    U.K.                           → GPE
    $1 billion                     → MONEY
    Tim Cook                       → PERSON
    Apple                          → ORG
    London                         → GPE
    June 2023                      → DATE
    

## Dependency Parsing
Shows how words are related in a sentence (grammar tree).
Dependency parsing is the process of finding how words relate to each other in a sentence.

It answers:

❓ What is the main verb?

❓ What is the subject of the sentence?

❓ What is the object the verb acts on?

❓ Which word is describing or modifying another?

Think of it like a grammar tree where every word is connected to another word — kind of like a family tree for words.

#### Layman Analogy
Imagine a sentence is a company:

The main verb is the CEO.

The subject (who is doing the action) is a Manager reporting to the CEO.

The object (what the action is done to) is an Employee reporting to the manager.

Other words (modifiers, prepositions) are assistants or tools used.


```python
sentence = "John bought a new phone in New York last week."
doc2 = nlp(sentence)

for token in doc2:
    print(f"{token.text:10} → {token.dep_:15} → Head: {token.head.text}")
```

    John       → nsubj           → Head: bought
    bought     → ROOT            → Head: bought
    a          → det             → Head: phone
    new        → amod            → Head: phone
    phone      → dobj            → Head: bought
    in         → prep            → Head: bought
    New        → compound        → Head: York
    York       → pobj            → Head: in
    last       → amod            → Head: week
    week       → npadvmod        → Head: bought
    .          → punct           → Head: bought
    

### Layman Explanation of Key Terms

| Term       | Meaning (Layman)                                     |
| ---------- | ---------------------------------------------------- |
| `ROOT`     | Main verb or action of the sentence                  |
| `nsubj`    | **Nominal subject** — the doer of the action         |
| `dobj`     | **Direct object** — the thing being acted on         |
| `prep`     | Preposition (like "in", "on")                        |
| `pobj`     | **Prepositional object** — object of the preposition |
| `amod`     | **Adjective modifier** (e.g., “new” phone)           |
| `det`      | Determiner (e.g., “a”, “the”)                        |
| `npadvmod` | Noun phrase adverbial modifier (e.g., “last week”)   |
| `compound` | Compound noun part (e.g., "New" in "New York")       |
| `punct`    | Punctuation mark                                     |

#### What This Tells Us About the Sentence
* Action: bought (ROOT)

* Who did it: John (subject → nsubj)

* What did he buy: phone (object → dobj)

* Where: in New York (prepositional phrase)

* When: last week (adverbial phrase)


```python
from spacy import displacy
displacy.render(doc, style="dep")
```


<span class="tex2jax_ignore"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="en" id="849d573739714a2da384289ccf130f45-0" class="displacy" width="4250" height="399.5" direction="ltr" style="max-width: none; height: 399.5px; color: #000000; background: #ffffff; font-family: Arial; direction: ltr">
<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="50">
</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">SPACE</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="225">Apple</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="225">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="400">is</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="400">AUX</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="575">looking</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="575">VERB</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="750">at</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="750">ADP</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="925">buying</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="925">VERB</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="1100">U.K.</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1100">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="1275">startup</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1275">VERB</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="1450">for</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1450">ADP</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="1625">$</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1625">SYM</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="1800">1</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1800">NUM</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="1975">billion.</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1975">NUM</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="2150">Tim</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="2150">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="2325">Cook,</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="2325">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="2500">the</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="2500">DET</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="2675">CEO</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="2675">NOUN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="2850">of</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="2850">ADP</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="3025">Apple,</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="3025">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="3200">visited</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="3200">VERB</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="3375">London</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="3375">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="3550">in</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="3550">ADP</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="3725">June</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="3725">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="3900">2023.</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="3900">PUNCT</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="309.5">
    <tspan class="displacy-word" fill="currentColor" x="4075">
</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="4075">SPACE</tspan>
</text>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-0" stroke-width="2px" d="M70,264.5 C70,177.0 215.0,177.0 215.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-0" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">dep</textPath>
    </text>
    <path class="displacy-arrowhead" d="M70,266.5 L62,254.5 78,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-1" stroke-width="2px" d="M245,264.5 C245,89.5 570.0,89.5 570.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-1" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">nsubj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M245,266.5 L237,254.5 253,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-2" stroke-width="2px" d="M420,264.5 C420,177.0 565.0,177.0 565.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-2" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">aux</textPath>
    </text>
    <path class="displacy-arrowhead" d="M420,266.5 L412,254.5 428,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-3" stroke-width="2px" d="M595,264.5 C595,177.0 740.0,177.0 740.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-3" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">prep</textPath>
    </text>
    <path class="displacy-arrowhead" d="M740.0,266.5 L748.0,254.5 732.0,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-4" stroke-width="2px" d="M770,264.5 C770,177.0 915.0,177.0 915.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-4" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">pcomp</textPath>
    </text>
    <path class="displacy-arrowhead" d="M915.0,266.5 L923.0,254.5 907.0,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-5" stroke-width="2px" d="M1120,264.5 C1120,177.0 1265.0,177.0 1265.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-5" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">nsubj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1120,266.5 L1112,254.5 1128,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-6" stroke-width="2px" d="M945,264.5 C945,89.5 1270.0,89.5 1270.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-6" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">ccomp</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1270.0,266.5 L1278.0,254.5 1262.0,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-7" stroke-width="2px" d="M1295,264.5 C1295,177.0 1440.0,177.0 1440.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-7" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">prep</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1440.0,266.5 L1448.0,254.5 1432.0,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-8" stroke-width="2px" d="M1645,264.5 C1645,89.5 1970.0,89.5 1970.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-8" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">quantmod</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1645,266.5 L1637,254.5 1653,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-9" stroke-width="2px" d="M1820,264.5 C1820,177.0 1965.0,177.0 1965.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-9" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">compound</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1820,266.5 L1812,254.5 1828,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-10" stroke-width="2px" d="M1470,264.5 C1470,2.0 1975.0,2.0 1975.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-10" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">pobj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1975.0,266.5 L1983.0,254.5 1967.0,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-11" stroke-width="2px" d="M2170,264.5 C2170,177.0 2315.0,177.0 2315.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-11" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">compound</textPath>
    </text>
    <path class="displacy-arrowhead" d="M2170,266.5 L2162,254.5 2178,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-12" stroke-width="2px" d="M2345,264.5 C2345,2.0 3200.0,2.0 3200.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-12" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">nsubj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M2345,266.5 L2337,254.5 2353,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-13" stroke-width="2px" d="M2520,264.5 C2520,177.0 2665.0,177.0 2665.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-13" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">det</textPath>
    </text>
    <path class="displacy-arrowhead" d="M2520,266.5 L2512,254.5 2528,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-14" stroke-width="2px" d="M2345,264.5 C2345,89.5 2670.0,89.5 2670.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-14" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">appos</textPath>
    </text>
    <path class="displacy-arrowhead" d="M2670.0,266.5 L2678.0,254.5 2662.0,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-15" stroke-width="2px" d="M2695,264.5 C2695,177.0 2840.0,177.0 2840.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-15" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">prep</textPath>
    </text>
    <path class="displacy-arrowhead" d="M2840.0,266.5 L2848.0,254.5 2832.0,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-16" stroke-width="2px" d="M2870,264.5 C2870,177.0 3015.0,177.0 3015.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-16" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">pobj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M3015.0,266.5 L3023.0,254.5 3007.0,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-17" stroke-width="2px" d="M3220,264.5 C3220,177.0 3365.0,177.0 3365.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-17" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">dobj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M3365.0,266.5 L3373.0,254.5 3357.0,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-18" stroke-width="2px" d="M3220,264.5 C3220,89.5 3545.0,89.5 3545.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-18" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">prep</textPath>
    </text>
    <path class="displacy-arrowhead" d="M3545.0,266.5 L3553.0,254.5 3537.0,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-19" stroke-width="2px" d="M3570,264.5 C3570,177.0 3715.0,177.0 3715.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-19" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">pobj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M3715.0,266.5 L3723.0,254.5 3707.0,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-20" stroke-width="2px" d="M3220,264.5 C3220,2.0 3900.0,2.0 3900.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-20" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">punct</textPath>
    </text>
    <path class="displacy-arrowhead" d="M3900.0,266.5 L3908.0,254.5 3892.0,254.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-849d573739714a2da384289ccf130f45-0-21" stroke-width="2px" d="M3920,264.5 C3920,177.0 4065.0,177.0 4065.0,264.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-849d573739714a2da384289ccf130f45-0-21" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">dep</textPath>
    </text>
    <path class="displacy-arrowhead" d="M4065.0,266.5 L4073.0,254.5 4057.0,254.5" fill="currentColor"/>
</g>
</svg></span>


## Word Embeddings (Vector Representation)
spaCy’s en_core_web_sm model does not include pre-trained word vectors. Use en_core_web_md or en_core_web_lg instead:


```python
# Load model with vectors
nlp = spacy.load("en_core_web_md")
doc = nlp("Apple is looking at buying a startup.")

# Access vector for a token
print(doc[0].text, "→", doc[0].vector[:5])  # Show first 5 dimensions
```

    Apple → [-0.6334   0.18981 -0.53544 -0.52658 -0.30001]
    

### What it does:
* doc[0] is the word "Apple"
* .vector gives its word embedding (typically 300-dimensional)
* [:5] just shows the first 5 numbers.

### What Is a Dimension?
* Each word is turned into a vector (a list of numbers).
* A 300-dimensional vector = list of 300 numbers.
* These dimensions encode semantic properties.


```python
# Similarity between words
print("Similarity between Apple and startup:", doc[0].similarity(doc[-2]))
```

    Similarity between Apple and startup: 0.10326027125120163
    

"Apple is looking at buying a startup."

* When `nlp()` processes this, it creates a `Doc` object — a sequence of `Token` objects, like a list:

print("Document similarity:", doc1.similarity(doc2))

| Index | Token   |
| ----- | ------- |
| 0     | Apple   |
| 1     | is      |
| 2     | looking |
| 3     | at      |
| 4     | buying  |
| 5     | a       |
| 6     | startup |
| 7     | .       |



```python
doc1 = nlp("Apple is a technology company.")
doc2 = nlp("Microsoft develops software.")
```

#### What it does:
* Measures how similar the meanings of doc1 and doc2 are.
* Internally: It averages the word vectors of each document and calculates cosine similarity between the two.

#### Layman Explanation:
* Imagine each document is a cloud of meaning in a semantic space:
* "Apple is a technology company" 🟦
* "Microsoft develops software" 🟥
* Since both talk about tech companies, their clouds are close together, hence high similarity (usually around ~0.7).


```python
token1 = nlp("king")[0]
token2 = nlp("queen")[0]
print("Word similarity:", token1.similarity(token2))
```

    Word similarity: 0.38253092765808105
    

### What it does:
* Retrieves pre-trained vectors for king and queen
* Computes cosine similarity between them

##  Cosine Similarity (Simple Math Behind It)
Similarity = **cos(θ)** between two word vectors
If:

* cos(0°) = 1 (same direction)

* cos(90°) = 0 (orthogonal = unrelated)

* cos(180°) = -1 (opposite meaning)


```python

```
