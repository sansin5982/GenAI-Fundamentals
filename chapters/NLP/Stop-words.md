# Stop Words
### 🚫 What are Stop Words?
**Stop words** are **commonly used words** in a language that **carry little meaning** on their own in the context of text analysis.

Examples in English:

`“is”`, `“the”`, `“a”`, `“of”`, `“to”`, `“in”`, `“and”`, `“on”`, `“it”`, `“with”`

These words are usually removed during preprocessing because they **don’t contribute meaningful information** for NLP tasks like classification, clustering, or search.

### 🎯 Why are Stop Words Used in Text Processing?

| Purpose                      | Explanation                                          |
| ---------------------------- | ---------------------------------------------------- |
| 📉 Reduces Dimensionality    | Fewer unique words → faster processing               |
| 🧹 Cleans the Text           | Removes irrelevant or filler content                 |
| 💡 Focus on Keywords         | Helps algorithms concentrate on **important tokens** |
| ✅ Improves Model Performance | Especially in Bag-of-Words or TF-IDF models          |



```python
paragraph = """
The student was working on an interesting NLP project. It involved text preprocessing and analysis.
She was studying how to clean and prepare raw data.
Many common words in the English language were not adding much value.
So, she removed these stop words to simplify the input.
Then she used stemming to reduce words like "running", "played", and "studies".
In contrast, she also explored lemmatization to get more meaningful base words.
This helped in improving the machine learning model’s accuracy.
The pipeline included tokenization, normalization, and vectorization.
She tested both methods and found lemmatization better for her use case.
The final model gave excellent results on real-world text data.
"""

```


```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


```


```python
# Download required resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
```


```python
# Initialize
stop_words = set(stopwords.words("english"))

stop_words
```




    {'a',
     'about',
     'above',
     'after',
     'again',
     'against',
     'ain',
     'all',
     'am',
     'an',
     'and',
     'any',
     'are',
     'aren',
     "aren't",
     'as',
     'at',
     'be',
     'because',
     'been',
     'before',
     'being',
     'below',
     'between',
     'both',
     'but',
     'by',
     'can',
     'couldn',
     "couldn't",
     'd',
     'did',
     'didn',
     "didn't",
     'do',
     'does',
     'doesn',
     "doesn't",
     'doing',
     'don',
     "don't",
     'down',
     'during',
     'each',
     'few',
     'for',
     'from',
     'further',
     'had',
     'hadn',
     "hadn't",
     'has',
     'hasn',
     "hasn't",
     'have',
     'haven',
     "haven't",
     'having',
     'he',
     "he'd",
     "he'll",
     "he's",
     'her',
     'here',
     'hers',
     'herself',
     'him',
     'himself',
     'his',
     'how',
     'i',
     "i'd",
     "i'll",
     "i'm",
     "i've",
     'if',
     'in',
     'into',
     'is',
     'isn',
     "isn't",
     'it',
     "it'd",
     "it'll",
     "it's",
     'its',
     'itself',
     'just',
     'll',
     'm',
     'ma',
     'me',
     'mightn',
     "mightn't",
     'more',
     'most',
     'mustn',
     "mustn't",
     'my',
     'myself',
     'needn',
     "needn't",
     'no',
     'nor',
     'not',
     'now',
     'o',
     'of',
     'off',
     'on',
     'once',
     'only',
     'or',
     'other',
     'our',
     'ours',
     'ourselves',
     'out',
     'over',
     'own',
     're',
     's',
     'same',
     'shan',
     "shan't",
     'she',
     "she'd",
     "she'll",
     "she's",
     'should',
     "should've",
     'shouldn',
     "shouldn't",
     'so',
     'some',
     'such',
     't',
     'than',
     'that',
     "that'll",
     'the',
     'their',
     'theirs',
     'them',
     'themselves',
     'then',
     'there',
     'these',
     'they',
     "they'd",
     "they'll",
     "they're",
     "they've",
     'this',
     'those',
     'through',
     'to',
     'too',
     'under',
     'until',
     'up',
     've',
     'very',
     'was',
     'wasn',
     "wasn't",
     'we',
     "we'd",
     "we'll",
     "we're",
     "we've",
     'were',
     'weren',
     "weren't",
     'what',
     'when',
     'where',
     'which',
     'while',
     'who',
     'whom',
     'why',
     'will',
     'with',
     'won',
     "won't",
     'wouldn',
     "wouldn't",
     'y',
     'you',
     "you'd",
     "you'll",
     "you're",
     "you've",
     'your',
     'yours',
     'yourself',
     'yourselves'}




```python
stopwords.words("arabic")
```




    ['إذ',
     'إذا',
     'إذما',
     'إذن',
     'أف',
     'أقل',
     'أكثر',
     'ألا',
     'إلا',
     'التي',
     'الذي',
     'الذين',
     'اللاتي',
     'اللائي',
     'اللتان',
     'اللتيا',
     'اللتين',
     'اللذان',
     'اللذين',
     'اللواتي',
     'إلى',
     'إليك',
     'إليكم',
     'إليكما',
     'إليكن',
     'أم',
     'أما',
     'أما',
     'إما',
     'أن',
     'إن',
     'إنا',
     'أنا',
     'أنت',
     'أنتم',
     'أنتما',
     'أنتن',
     'إنما',
     'إنه',
     'أنى',
     'أنى',
     'آه',
     'آها',
     'أو',
     'أولاء',
     'أولئك',
     'أوه',
     'آي',
     'أي',
     'أيها',
     'إي',
     'أين',
     'أين',
     'أينما',
     'إيه',
     'بخ',
     'بس',
     'بعد',
     'بعض',
     'بك',
     'بكم',
     'بكم',
     'بكما',
     'بكن',
     'بل',
     'بلى',
     'بما',
     'بماذا',
     'بمن',
     'بنا',
     'به',
     'بها',
     'بهم',
     'بهما',
     'بهن',
     'بي',
     'بين',
     'بيد',
     'تلك',
     'تلكم',
     'تلكما',
     'ته',
     'تي',
     'تين',
     'تينك',
     'ثم',
     'ثمة',
     'حاشا',
     'حبذا',
     'حتى',
     'حيث',
     'حيثما',
     'حين',
     'خلا',
     'دون',
     'ذا',
     'ذات',
     'ذاك',
     'ذان',
     'ذانك',
     'ذلك',
     'ذلكم',
     'ذلكما',
     'ذلكن',
     'ذه',
     'ذو',
     'ذوا',
     'ذواتا',
     'ذواتي',
     'ذي',
     'ذين',
     'ذينك',
     'ريث',
     'سوف',
     'سوى',
     'شتان',
     'عدا',
     'عسى',
     'عل',
     'على',
     'عليك',
     'عليه',
     'عما',
     'عن',
     'عند',
     'غير',
     'فإذا',
     'فإن',
     'فلا',
     'فمن',
     'في',
     'فيم',
     'فيما',
     'فيه',
     'فيها',
     'قد',
     'كأن',
     'كأنما',
     'كأي',
     'كأين',
     'كذا',
     'كذلك',
     'كل',
     'كلا',
     'كلاهما',
     'كلتا',
     'كلما',
     'كليكما',
     'كليهما',
     'كم',
     'كم',
     'كما',
     'كي',
     'كيت',
     'كيف',
     'كيفما',
     'لا',
     'لاسيما',
     'لدى',
     'لست',
     'لستم',
     'لستما',
     'لستن',
     'لسن',
     'لسنا',
     'لعل',
     'لك',
     'لكم',
     'لكما',
     'لكن',
     'لكنما',
     'لكي',
     'لكيلا',
     'لم',
     'لما',
     'لن',
     'لنا',
     'له',
     'لها',
     'لهم',
     'لهما',
     'لهن',
     'لو',
     'لولا',
     'لوما',
     'لي',
     'لئن',
     'ليت',
     'ليس',
     'ليسا',
     'ليست',
     'ليستا',
     'ليسوا',
     'ما',
     'ماذا',
     'متى',
     'مذ',
     'مع',
     'مما',
     'ممن',
     'من',
     'منه',
     'منها',
     'منذ',
     'مه',
     'مهما',
     'نحن',
     'نحو',
     'نعم',
     'ها',
     'هاتان',
     'هاته',
     'هاتي',
     'هاتين',
     'هاك',
     'هاهنا',
     'هذا',
     'هذان',
     'هذه',
     'هذي',
     'هذين',
     'هكذا',
     'هل',
     'هلا',
     'هم',
     'هما',
     'هن',
     'هنا',
     'هناك',
     'هنالك',
     'هو',
     'هؤلاء',
     'هي',
     'هيا',
     'هيت',
     'هيهات',
     'والذي',
     'والذين',
     'وإذ',
     'وإذا',
     'وإن',
     'ولا',
     'ولكن',
     'ولو',
     'وما',
     'ومن',
     'وهو',
     'يا',
     'أبٌ',
     'أخٌ',
     'حمٌ',
     'فو',
     'أنتِ',
     'يناير',
     'فبراير',
     'مارس',
     'أبريل',
     'مايو',
     'يونيو',
     'يوليو',
     'أغسطس',
     'سبتمبر',
     'أكتوبر',
     'نوفمبر',
     'ديسمبر',
     'جانفي',
     'فيفري',
     'مارس',
     'أفريل',
     'ماي',
     'جوان',
     'جويلية',
     'أوت',
     'كانون',
     'شباط',
     'آذار',
     'نيسان',
     'أيار',
     'حزيران',
     'تموز',
     'آب',
     'أيلول',
     'تشرين',
     'دولار',
     'دينار',
     'ريال',
     'درهم',
     'ليرة',
     'جنيه',
     'قرش',
     'مليم',
     'فلس',
     'هللة',
     'سنتيم',
     'يورو',
     'ين',
     'يوان',
     'شيكل',
     'واحد',
     'اثنان',
     'ثلاثة',
     'أربعة',
     'خمسة',
     'ستة',
     'سبعة',
     'ثمانية',
     'تسعة',
     'عشرة',
     'أحد',
     'اثنا',
     'اثني',
     'إحدى',
     'ثلاث',
     'أربع',
     'خمس',
     'ست',
     'سبع',
     'ثماني',
     'تسع',
     'عشر',
     'ثمان',
     'سبت',
     'أحد',
     'اثنين',
     'ثلاثاء',
     'أربعاء',
     'خميس',
     'جمعة',
     'أول',
     'ثان',
     'ثاني',
     'ثالث',
     'رابع',
     'خامس',
     'سادس',
     'سابع',
     'ثامن',
     'تاسع',
     'عاشر',
     'حادي',
     'أ',
     'ب',
     'ت',
     'ث',
     'ج',
     'ح',
     'خ',
     'د',
     'ذ',
     'ر',
     'ز',
     'س',
     'ش',
     'ص',
     'ض',
     'ط',
     'ظ',
     'ع',
     'غ',
     'ف',
     'ق',
     'ك',
     'ل',
     'م',
     'ن',
     'ه',
     'و',
     'ي',
     'ء',
     'ى',
     'آ',
     'ؤ',
     'ئ',
     'أ',
     'ة',
     'ألف',
     'باء',
     'تاء',
     'ثاء',
     'جيم',
     'حاء',
     'خاء',
     'دال',
     'ذال',
     'راء',
     'زاي',
     'سين',
     'شين',
     'صاد',
     'ضاد',
     'طاء',
     'ظاء',
     'عين',
     'غين',
     'فاء',
     'قاف',
     'كاف',
     'لام',
     'ميم',
     'نون',
     'هاء',
     'واو',
     'ياء',
     'همزة',
     'ي',
     'نا',
     'ك',
     'كن',
     'ه',
     'إياه',
     'إياها',
     'إياهما',
     'إياهم',
     'إياهن',
     'إياك',
     'إياكما',
     'إياكم',
     'إياك',
     'إياكن',
     'إياي',
     'إيانا',
     'أولالك',
     'تانِ',
     'تانِك',
     'تِه',
     'تِي',
     'تَيْنِ',
     'ثمّ',
     'ثمّة',
     'ذانِ',
     'ذِه',
     'ذِي',
     'ذَيْنِ',
     'هَؤلاء',
     'هَاتانِ',
     'هَاتِه',
     'هَاتِي',
     'هَاتَيْنِ',
     'هَذا',
     'هَذانِ',
     'هَذِه',
     'هَذِي',
     'هَذَيْنِ',
     'الألى',
     'الألاء',
     'أل',
     'أنّى',
     'أيّ',
     'ّأيّان',
     'أنّى',
     'أيّ',
     'ّأيّان',
     'ذيت',
     'كأيّ',
     'كأيّن',
     'بضع',
     'فلان',
     'وا',
     'آمينَ',
     'آهِ',
     'آهٍ',
     'آهاً',
     'أُفٍّ',
     'أُفٍّ',
     'أفٍّ',
     'أمامك',
     'أمامكَ',
     'أوّهْ',
     'إلَيْكَ',
     'إلَيْكَ',
     'إليكَ',
     'إليكنّ',
     'إيهٍ',
     'بخٍ',
     'بسّ',
     'بَسْ',
     'بطآن',
     'بَلْهَ',
     'حاي',
     'حَذارِ',
     'حيَّ',
     'حيَّ',
     'دونك',
     'رويدك',
     'سرعان',
     'شتانَ',
     'شَتَّانَ',
     'صهْ',
     'صهٍ',
     'طاق',
     'طَق',
     'عَدَسْ',
     'كِخ',
     'مكانَك',
     'مكانَك',
     'مكانَك',
     'مكانكم',
     'مكانكما',
     'مكانكنّ',
     'نَخْ',
     'هاكَ',
     'هَجْ',
     'هلم',
     'هيّا',
     'هَيْهات',
     'وا',
     'واهاً',
     'وراءَك',
     'وُشْكَانَ',
     'وَيْ',
     'يفعلان',
     'تفعلان',
     'يفعلون',
     'تفعلون',
     'تفعلين',
     'اتخذ',
     'ألفى',
     'تخذ',
     'ترك',
     'تعلَّم',
     'جعل',
     'حجا',
     'حبيب',
     'خال',
     'حسب',
     'خال',
     'درى',
     'رأى',
     'زعم',
     'صبر',
     'ظنَّ',
     'عدَّ',
     'علم',
     'غادر',
     'ذهب',
     'وجد',
     'ورد',
     'وهب',
     'أسكن',
     'أطعم',
     'أعطى',
     'رزق',
     'زود',
     'سقى',
     'كسا',
     'أخبر',
     'أرى',
     'أعلم',
     'أنبأ',
     'حدَث',
     'خبَّر',
     'نبَّا',
     'أفعل به',
     'ما أفعله',
     'بئس',
     'ساء',
     'طالما',
     'قلما',
     'لات',
     'لكنَّ',
     'ءَ',
     'أجل',
     'إذاً',
     'أمّا',
     'إمّا',
     'إنَّ',
     'أنًّ',
     'أى',
     'إى',
     'أيا',
     'ب',
     'ثمَّ',
     'جلل',
     'جير',
     'رُبَّ',
     'س',
     'علًّ',
     'ف',
     'كأنّ',
     'كلَّا',
     'كى',
     'ل',
     'لات',
     'لعلَّ',
     'لكنَّ',
     'لكنَّ',
     'م',
     'نَّ',
     'هلّا',
     'وا',
     'أل',
     'إلّا',
     'ت',
     'ك',
     'لمّا',
     'ن',
     'ه',
     'و',
     'ا',
     'ي',
     'تجاه',
     'تلقاء',
     'جميع',
     'حسب',
     'سبحان',
     'شبه',
     'لعمر',
     'مثل',
     'معاذ',
     'أبو',
     'أخو',
     'حمو',
     'فو',
     'مئة',
     'مئتان',
     'ثلاثمئة',
     'أربعمئة',
     'خمسمئة',
     'ستمئة',
     'سبعمئة',
     'ثمنمئة',
     'تسعمئة',
     'مائة',
     'ثلاثمائة',
     'أربعمائة',
     'خمسمائة',
     'ستمائة',
     'سبعمائة',
     'ثمانمئة',
     'تسعمائة',
     'عشرون',
     'ثلاثون',
     'اربعون',
     'خمسون',
     'ستون',
     'سبعون',
     'ثمانون',
     'تسعون',
     'عشرين',
     'ثلاثين',
     'اربعين',
     'خمسين',
     'ستين',
     'سبعين',
     'ثمانين',
     'تسعين',
     'بضع',
     'نيف',
     'أجمع',
     'جميع',
     'عامة',
     'عين',
     'نفس',
     'لا سيما',
     'أصلا',
     'أهلا',
     'أيضا',
     'بؤسا',
     'بعدا',
     'بغتة',
     'تعسا',
     'حقا',
     'حمدا',
     'خلافا',
     'خاصة',
     'دواليك',
     'سحقا',
     'سرا',
     'سمعا',
     'صبرا',
     'صدقا',
     'صراحة',
     'طرا',
     'عجبا',
     'عيانا',
     'غالبا',
     'فرادى',
     'فضلا',
     'قاطبة',
     'كثيرا',
     'لبيك',
     'معاذ',
     'أبدا',
     'إزاء',
     'أصلا',
     'الآن',
     'أمد',
     'أمس',
     'آنفا',
     'آناء',
     'أنّى',
     'أول',
     'أيّان',
     'تارة',
     'ثمّ',
     'ثمّة',
     'حقا',
     'صباح',
     'مساء',
     'ضحوة',
     'عوض',
     'غدا',
     'غداة',
     'قطّ',
     'كلّما',
     'لدن',
     'لمّا',
     'مرّة',
     'قبل',
     'خلف',
     'أمام',
     'فوق',
     'تحت',
     'يمين',
     'شمال',
     'ارتدّ',
     'استحال',
     'أصبح',
     'أضحى',
     'آض',
     'أمسى',
     'انقلب',
     'بات',
     'تبدّل',
     'تحوّل',
     'حار',
     'رجع',
     'راح',
     'صار',
     'ظلّ',
     'عاد',
     'غدا',
     'كان',
     'ما انفك',
     'ما برح',
     'مادام',
     'مازال',
     'مافتئ',
     'ابتدأ',
     'أخذ',
     'اخلولق',
     'أقبل',
     'انبرى',
     'أنشأ',
     'أوشك',
     'جعل',
     'حرى',
     'شرع',
     'طفق',
     'علق',
     'قام',
     'كرب',
     'كاد',
     'هبّ']




```python
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

```


```python
# Tokenize paragraph into words
words = word_tokenize(paragraph)
```


```python
# Output
print("Original Words:\n", words)
```

    Original Words:
     ['The', 'student', 'was', 'working', 'on', 'an', 'interesting', 'NLP', 'project', '.', 'It', 'involved', 'text', 'preprocessing', 'and', 'analysis', '.', 'She', 'was', 'studying', 'how', 'to', 'clean', 'and', 'prepare', 'raw', 'data', '.', 'Many', 'common', 'words', 'in', 'the', 'English', 'language', 'were', 'not', 'adding', 'much', 'value', '.', 'So', ',', 'she', 'removed', 'these', 'stop', 'words', 'to', 'simplify', 'the', 'input', '.', 'Then', 'she', 'used', 'stemming', 'to', 'reduce', 'words', 'like', '``', 'running', "''", ',', '``', 'played', "''", ',', 'and', '``', 'studies', "''", '.', 'In', 'contrast', ',', 'she', 'also', 'explored', 'lemmatization', 'to', 'get', 'more', 'meaningful', 'base', 'words', '.', 'This', 'helped', 'in', 'improving', 'the', 'machine', 'learning', 'model', '’', 's', 'accuracy', '.', 'The', 'pipeline', 'included', 'tokenization', ',', 'normalization', ',', 'and', 'vectorization', '.', 'She', 'tested', 'both', 'methods', 'and', 'found', 'lemmatization', 'better', 'for', 'her', 'use', 'case', '.', 'The', 'final', 'model', 'gave', 'excellent', 'results', 'on', 'real-world', 'text', 'data', '.']
    


```python
# Remove stop words
filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
print("\nFiltered Words (Stop Words Removed):\n", filtered_words)
```

    
    Filtered Words (Stop Words Removed):
     ['student', 'working', 'interesting', 'NLP', 'project', 'involved', 'text', 'preprocessing', 'analysis', 'studying', 'clean', 'prepare', 'raw', 'data', 'Many', 'common', 'words', 'English', 'language', 'adding', 'much', 'value', 'removed', 'stop', 'words', 'simplify', 'input', 'used', 'stemming', 'reduce', 'words', 'like', 'running', 'played', 'studies', 'contrast', 'also', 'explored', 'lemmatization', 'get', 'meaningful', 'base', 'words', 'helped', 'improving', 'machine', 'learning', 'model', 'accuracy', 'pipeline', 'included', 'tokenization', 'normalization', 'vectorization', 'tested', 'methods', 'found', 'lemmatization', 'better', 'use', 'case', 'final', 'model', 'gave', 'excellent', 'results', 'text', 'data']
    


```python
stemmed = [stemmer.stem(word) for word in filtered_words]
print("\nStemmed Words:\n", stemmed)
```

    
    Stemmed Words:
     ['student', 'work', 'interest', 'nlp', 'project', 'involv', 'text', 'preprocess', 'analysi', 'studi', 'clean', 'prepar', 'raw', 'data', 'mani', 'common', 'word', 'english', 'languag', 'ad', 'much', 'valu', 'remov', 'stop', 'word', 'simplifi', 'input', 'use', 'stem', 'reduc', 'word', 'like', 'run', 'play', 'studi', 'contrast', 'also', 'explor', 'lemmat', 'get', 'meaning', 'base', 'word', 'help', 'improv', 'machin', 'learn', 'model', 'accuraci', 'pipelin', 'includ', 'token', 'normal', 'vector', 'test', 'method', 'found', 'lemmat', 'better', 'use', 'case', 'final', 'model', 'gave', 'excel', 'result', 'text', 'data']
    


```python
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_words]
print("\nLemmatized Words:\n", lemmatized)
```

    
    Lemmatized Words:
     ['student', 'working', 'interesting', 'NLP', 'project', 'involved', 'text', 'preprocessing', 'analysis', 'studying', 'clean', 'prepare', 'raw', 'data', 'Many', 'common', 'word', 'English', 'language', 'adding', 'much', 'value', 'removed', 'stop', 'word', 'simplify', 'input', 'used', 'stemming', 'reduce', 'word', 'like', 'running', 'played', 'study', 'contrast', 'also', 'explored', 'lemmatization', 'get', 'meaningful', 'base', 'word', 'helped', 'improving', 'machine', 'learning', 'model', 'accuracy', 'pipeline', 'included', 'tokenization', 'normalization', 'vectorization', 'tested', 'method', 'found', 'lemmatization', 'better', 'use', 'case', 'final', 'model', 'gave', 'excellent', 'result', 'text', 'data']
    
