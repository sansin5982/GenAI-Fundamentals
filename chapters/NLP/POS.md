# Part-of-Speech tagging (POS)
### 🏷️ What are POS Tags?
**POS tagging** (Part-of-Speech tagging) is the process of **labeling each word** in a sentence with its appropriate grammatical role — like noun, verb, adjective, etc.

#### 📌 Examples of POS Tags:
* `NN` → Noun, `VB` → Verb, `JJ` → Adjective, `RB` → Adverb
* `NNS` → Plural Noun, `VBD` → Past Tense Verb

### 🎯 Why Are POS Tags Used in Text Processing?
| Purpose              | Explanation                           |
|:---------------------------- |:--------------------------------------------- |
| 🔎 Grammatical Understanding | Helps understand the role of words in context                       |
| 🧠 Better Lemmatization      | Correct POS improves lemmatization accuracy                         |
| 📊 NLP Tasks                 | Crucial for Named Entity Recognition, Dependency Parsing, and more  |
| ✅ Disambiguation             | Differentiates between similar words (e.g., "book" as noun vs verb) |


CC coordinating conjunction 
CD cardinal digit 
DT determiner 
EX existential there (like: “there is” … think of it like “there exists”) 
FW foreign word 
IN preposition/subordinating conjunction 
JJ adjective – ‘big’ 
JJR adjective, comparative – ‘bigger’ 
JJS adjective, superlative – ‘biggest’ 
LS list marker 1) 
MD modal – could, will 
NN noun, singular ‘- desk’ 
NNS noun plural – ‘desks’ 
NNP proper noun, singular – ‘Harrison’ 
NNPS proper noun, plural – ‘Americans’ 
PDT predeterminer – ‘all the kids’ 
POS possessive ending parent’s 
PRP personal pronoun –  I, he, she 
PRP$ possessive pronoun – my, his, hers 
RB adverb – very, silently, 
RBR adverb, comparative – better 
RBS adverb, superlative – best 
RP particle – give up 
TO – to go ‘to’ the store. 
UH interjection – errrrrrrrm 
VB verb, base form – take 
VBD verb, past tense – took 
VBG verb, gerund/present participle – taking 
VBN verb, past participle – taken 
VBP verb, sing. present, non-3d – take 
VBZ verb, 3rd person sing. present – takes 
WDT wh-determiner – which 
WP wh-pronoun – who, what 
WP$ possessive wh-pronoun, eg- whose 
WRB wh-adverb, eg- where, when


```python
paragraph = """The researcher collected various text samples for analysis. 
She was designing a new NLP pipeline for preprocessing tasks.
The system tokenized, normalized, and vectorized the text.
Her goal was to improve accuracy and reduce noise.
By tagging words with their parts of speech, she made better decisions.
The verbs were processed differently from nouns.
Adjectives and adverbs were handled carefully for contextual understanding.
Later, she applied stemming to reduce word forms.
She also used lemmatization for linguistically correct root forms.
These techniques helped in building a robust language model.
"""

```


```python
import nltk
from nltk.corpus import stopwords
sentences=nltk.sent_tokenize(paragraph)
```


```python
sentences
```




    ['The researcher collected various text samples for analysis.',
     'She was designing a new NLP pipeline for preprocessing tasks.',
     'The system tokenized, normalized, and vectorized the text.',
     'Her goal was to improve accuracy and reduce noise.',
     'By tagging words with their parts of speech, she made better decisions.',
     'The verbs were processed differently from nouns.',
     'Adjectives and adverbs were handled carefully for contextual understanding.',
     'Later, she applied stemming to reduce word forms.',
     'She also used lemmatization for linguistically correct root forms.',
     'These techniques helped in building a robust language model.']




```python
# nltk.download('averaged_perceptron_tagger')
```


```python
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
```


```python
# Functions
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default
```


```python
# Initialize
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
```


```python
# Tokenize and tag
words = word_tokenize(paragraph)
print("Words:\n", words)
```

    Words:
     ['The', 'researcher', 'collected', 'various', 'text', 'samples', 'for', 'analysis', '.', 'She', 'was', 'designing', 'a', 'new', 'NLP', 'pipeline', 'for', 'preprocessing', 'tasks', '.', 'The', 'system', 'tokenized', ',', 'normalized', ',', 'and', 'vectorized', 'the', 'text', '.', 'Her', 'goal', 'was', 'to', 'improve', 'accuracy', 'and', 'reduce', 'noise', '.', 'By', 'tagging', 'words', 'with', 'their', 'parts', 'of', 'speech', ',', 'she', 'made', 'better', 'decisions', '.', 'The', 'verbs', 'were', 'processed', 'differently', 'from', 'nouns', '.', 'Adjectives', 'and', 'adverbs', 'were', 'handled', 'carefully', 'for', 'contextual', 'understanding', '.', 'Later', ',', 'she', 'applied', 'stemming', 'to', 'reduce', 'word', 'forms', '.', 'She', 'also', 'used', 'lemmatization', 'for', 'linguistically', 'correct', 'root', 'forms', '.', 'These', 'techniques', 'helped', 'in', 'building', 'a', 'robust', 'language', 'model', '.']
    


```python
pos_tags = nltk.pos_tag(words)
print("\nPOS Tags:\n", pos_tags)
```

    
    POS Tags:
     [('The', 'DT'), ('researcher', 'NN'), ('collected', 'VBD'), ('various', 'JJ'), ('text', 'NN'), ('samples', 'NNS'), ('for', 'IN'), ('analysis', 'NN'), ('.', '.'), ('She', 'PRP'), ('was', 'VBD'), ('designing', 'VBG'), ('a', 'DT'), ('new', 'JJ'), ('NLP', 'NNP'), ('pipeline', 'NN'), ('for', 'IN'), ('preprocessing', 'VBG'), ('tasks', 'NNS'), ('.', '.'), ('The', 'DT'), ('system', 'NN'), ('tokenized', 'VBD'), (',', ','), ('normalized', 'VBN'), (',', ','), ('and', 'CC'), ('vectorized', 'VBD'), ('the', 'DT'), ('text', 'NN'), ('.', '.'), ('Her', 'PRP$'), ('goal', 'NN'), ('was', 'VBD'), ('to', 'TO'), ('improve', 'VB'), ('accuracy', 'NN'), ('and', 'CC'), ('reduce', 'VB'), ('noise', 'NN'), ('.', '.'), ('By', 'IN'), ('tagging', 'VBG'), ('words', 'NNS'), ('with', 'IN'), ('their', 'PRP$'), ('parts', 'NNS'), ('of', 'IN'), ('speech', 'NN'), (',', ','), ('she', 'PRP'), ('made', 'VBD'), ('better', 'JJR'), ('decisions', 'NNS'), ('.', '.'), ('The', 'DT'), ('verbs', 'NN'), ('were', 'VBD'), ('processed', 'VBN'), ('differently', 'RB'), ('from', 'IN'), ('nouns', 'NNS'), ('.', '.'), ('Adjectives', 'NNS'), ('and', 'CC'), ('adverbs', 'NNS'), ('were', 'VBD'), ('handled', 'VBN'), ('carefully', 'RB'), ('for', 'IN'), ('contextual', 'JJ'), ('understanding', 'NN'), ('.', '.'), ('Later', 'RB'), (',', ','), ('she', 'PRP'), ('applied', 'VBD'), ('stemming', 'VBG'), ('to', 'TO'), ('reduce', 'VB'), ('word', 'NN'), ('forms', 'NNS'), ('.', '.'), ('She', 'PRP'), ('also', 'RB'), ('used', 'VBD'), ('lemmatization', 'NN'), ('for', 'IN'), ('linguistically', 'RB'), ('correct', 'JJ'), ('root', 'NN'), ('forms', 'NNS'), ('.', '.'), ('These', 'DT'), ('techniques', 'NNS'), ('helped', 'VBD'), ('in', 'IN'), ('building', 'VBG'), ('a', 'DT'), ('robust', 'JJ'), ('language', 'NN'), ('model', 'NN'), ('.', '.')]
    


```python
# Stemming and Lemmatization
stemmed = [stemmer.stem(word) for word in words]
print("\nStemmed Words:\n", stemmed)
```

    
    Stemmed Words:
     ['the', 'research', 'collect', 'variou', 'text', 'sampl', 'for', 'analysi', '.', 'she', 'wa', 'design', 'a', 'new', 'nlp', 'pipelin', 'for', 'preprocess', 'task', '.', 'the', 'system', 'token', ',', 'normal', ',', 'and', 'vector', 'the', 'text', '.', 'her', 'goal', 'wa', 'to', 'improv', 'accuraci', 'and', 'reduc', 'nois', '.', 'by', 'tag', 'word', 'with', 'their', 'part', 'of', 'speech', ',', 'she', 'made', 'better', 'decis', '.', 'the', 'verb', 'were', 'process', 'differ', 'from', 'noun', '.', 'adject', 'and', 'adverb', 'were', 'handl', 'care', 'for', 'contextu', 'understand', '.', 'later', ',', 'she', 'appli', 'stem', 'to', 'reduc', 'word', 'form', '.', 'she', 'also', 'use', 'lemmat', 'for', 'linguist', 'correct', 'root', 'form', '.', 'these', 'techniqu', 'help', 'in', 'build', 'a', 'robust', 'languag', 'model', '.']
    


```python
# Print lemmatized words alongside their POS tags
for (word, tag) in pos_tags: # Loops over each word and its corresponding POS tag
    wn_tag = get_wordnet_pos(tag)
    lemma = lemmatizer.lemmatize(word, wn_tag) # Lemmatizes the word based on its POS (accurate for verbs, nouns, adjectives, etc.)
    print(f"Word: {word:15} POS: {tag:5} → Lemmatized: {lemma}")
```

    Word: The             POS: DT    → Lemmatized: The
    Word: researcher      POS: NN    → Lemmatized: researcher
    Word: collected       POS: VBD   → Lemmatized: collect
    Word: various         POS: JJ    → Lemmatized: various
    Word: text            POS: NN    → Lemmatized: text
    Word: samples         POS: NNS   → Lemmatized: sample
    Word: for             POS: IN    → Lemmatized: for
    Word: analysis        POS: NN    → Lemmatized: analysis
    Word: .               POS: .     → Lemmatized: .
    Word: She             POS: PRP   → Lemmatized: She
    Word: was             POS: VBD   → Lemmatized: be
    Word: designing       POS: VBG   → Lemmatized: design
    Word: a               POS: DT    → Lemmatized: a
    Word: new             POS: JJ    → Lemmatized: new
    Word: NLP             POS: NNP   → Lemmatized: NLP
    Word: pipeline        POS: NN    → Lemmatized: pipeline
    Word: for             POS: IN    → Lemmatized: for
    Word: preprocessing   POS: VBG   → Lemmatized: preprocessing
    Word: tasks           POS: NNS   → Lemmatized: task
    Word: .               POS: .     → Lemmatized: .
    Word: The             POS: DT    → Lemmatized: The
    Word: system          POS: NN    → Lemmatized: system
    Word: tokenized       POS: VBD   → Lemmatized: tokenized
    Word: ,               POS: ,     → Lemmatized: ,
    Word: normalized      POS: VBN   → Lemmatized: normalize
    Word: ,               POS: ,     → Lemmatized: ,
    Word: and             POS: CC    → Lemmatized: and
    Word: vectorized      POS: VBD   → Lemmatized: vectorized
    Word: the             POS: DT    → Lemmatized: the
    Word: text            POS: NN    → Lemmatized: text
    Word: .               POS: .     → Lemmatized: .
    Word: Her             POS: PRP$  → Lemmatized: Her
    Word: goal            POS: NN    → Lemmatized: goal
    Word: was             POS: VBD   → Lemmatized: be
    Word: to              POS: TO    → Lemmatized: to
    Word: improve         POS: VB    → Lemmatized: improve
    Word: accuracy        POS: NN    → Lemmatized: accuracy
    Word: and             POS: CC    → Lemmatized: and
    Word: reduce          POS: VB    → Lemmatized: reduce
    Word: noise           POS: NN    → Lemmatized: noise
    Word: .               POS: .     → Lemmatized: .
    Word: By              POS: IN    → Lemmatized: By
    Word: tagging         POS: VBG   → Lemmatized: tag
    Word: words           POS: NNS   → Lemmatized: word
    Word: with            POS: IN    → Lemmatized: with
    Word: their           POS: PRP$  → Lemmatized: their
    Word: parts           POS: NNS   → Lemmatized: part
    Word: of              POS: IN    → Lemmatized: of
    Word: speech          POS: NN    → Lemmatized: speech
    Word: ,               POS: ,     → Lemmatized: ,
    Word: she             POS: PRP   → Lemmatized: she
    Word: made            POS: VBD   → Lemmatized: make
    Word: better          POS: JJR   → Lemmatized: good
    Word: decisions       POS: NNS   → Lemmatized: decision
    Word: .               POS: .     → Lemmatized: .
    Word: The             POS: DT    → Lemmatized: The
    Word: verbs           POS: NN    → Lemmatized: verb
    Word: were            POS: VBD   → Lemmatized: be
    Word: processed       POS: VBN   → Lemmatized: process
    Word: differently     POS: RB    → Lemmatized: differently
    Word: from            POS: IN    → Lemmatized: from
    Word: nouns           POS: NNS   → Lemmatized: noun
    Word: .               POS: .     → Lemmatized: .
    Word: Adjectives      POS: NNS   → Lemmatized: Adjectives
    Word: and             POS: CC    → Lemmatized: and
    Word: adverbs         POS: NNS   → Lemmatized: adverb
    Word: were            POS: VBD   → Lemmatized: be
    Word: handled         POS: VBN   → Lemmatized: handle
    Word: carefully       POS: RB    → Lemmatized: carefully
    Word: for             POS: IN    → Lemmatized: for
    Word: contextual      POS: JJ    → Lemmatized: contextual
    Word: understanding   POS: NN    → Lemmatized: understanding
    Word: .               POS: .     → Lemmatized: .
    Word: Later           POS: RB    → Lemmatized: Later
    Word: ,               POS: ,     → Lemmatized: ,
    Word: she             POS: PRP   → Lemmatized: she
    Word: applied         POS: VBD   → Lemmatized: apply
    Word: stemming        POS: VBG   → Lemmatized: stem
    Word: to              POS: TO    → Lemmatized: to
    Word: reduce          POS: VB    → Lemmatized: reduce
    Word: word            POS: NN    → Lemmatized: word
    Word: forms           POS: NNS   → Lemmatized: form
    Word: .               POS: .     → Lemmatized: .
    Word: She             POS: PRP   → Lemmatized: She
    Word: also            POS: RB    → Lemmatized: also
    Word: used            POS: VBD   → Lemmatized: use
    Word: lemmatization   POS: NN    → Lemmatized: lemmatization
    Word: for             POS: IN    → Lemmatized: for
    Word: linguistically  POS: RB    → Lemmatized: linguistically
    Word: correct         POS: JJ    → Lemmatized: correct
    Word: root            POS: NN    → Lemmatized: root
    Word: forms           POS: NNS   → Lemmatized: form
    Word: .               POS: .     → Lemmatized: .
    Word: These           POS: DT    → Lemmatized: These
    Word: techniques      POS: NNS   → Lemmatized: technique
    Word: helped          POS: VBD   → Lemmatized: help
    Word: in              POS: IN    → Lemmatized: in
    Word: building        POS: VBG   → Lemmatized: build
    Word: a               POS: DT    → Lemmatized: a
    Word: robust          POS: JJ    → Lemmatized: robust
    Word: language        POS: NN    → Lemmatized: language
    Word: model           POS: NN    → Lemmatized: model
    Word: .               POS: .     → Lemmatized: .
    


```python
"Taj Mahal is a beautiful Monument".split()
```




    ['Taj', 'Mahal', 'is', 'a', 'beautiful', 'Monument']




```python
# it needs a list of words
print(nltk.pos_tag("Taj Mahal is a beautiful Monument".split()))
```

    [('Taj', 'NNP'), ('Mahal', 'NNP'), ('is', 'VBZ'), ('a', 'DT'), ('beautiful', 'JJ'), ('Monument', 'NN')]
    
