# %% [markdown]
# ## Text cleaning and Preprocessing

# %%
pip install spacy

# %%
import pandas as pd
import numpy as np
import spacy

# %%
from spacy.lang.en.stop_words import STOP_WORDS as stopwords

# %%
df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/twitter-data/master/twitter4000.csv',encoding= 'latin-1')

# %%
df

# %%
df['sentiment'].value_counts()

# %% [markdown]
# ### Word Counts

# %%
df['word_counts'] = df['twitts'].apply(lambda x: len(str(x).split()))

# %%
df.sample(5)

# %%
df['word_counts'].max()

# %%
df['word_counts'].min()

# %%
df['word_counts'] == 1

# %%
df[df['word_counts']==1]

# %% [markdown]
# ### Character Counts

# %%
def char_counts(x):
    s = x.split()
    x = ''.join(s)
    return len(x)

# %%
char_counts('I am wide awake')

# %%
df['char_counts'] = df['twitts'].apply(lambda x: char_counts(str(x)))

# %%
df.sample(5)

# %% [markdown]
# ### Average Word Length

# %%
df['avg_word_len'] = df['char_counts']/df['word_counts']

# %%
df.sample(5)

# %% [markdown]
# ### Stop Words Count

# %%
print(stopwords)

# %%
len(stopwords)

# %%
df['stop_words_len'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t in stopwords]))

# %% [markdown]
# ### #Hashtags and @Mentions Count

# %%
# [t for t in x.split() if t.starstwith('#')]

# %%
df['hashtags_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('#')]))

# %%
df['mentions_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))

# %%
df.sample(5)

# %% [markdown]
# ### Numeric Digits Count

# %%
x = 'this is 1 and 2'
x.split()

# %%
x.split()[4].isdigit()

# %%
[t for t in x.split() if t.isdigit()]

# %%
df['numeric_count'] = df['twitts'].apply(lambda x : len([t for t in x.split() if t.isdigit()]))

# %%
df.sample(5)

# %% [markdown]
# ### Upper Case Words Count

# %%
x = "I GOT THE JOB"
y = "I got the job"

# %%
[t for t in x.split() if t.isupper()]

# %%
[t for t in y.split() if t.isupper()]

# %%
df['upper_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.isupper()]))

# %%
df.sample(5)

# %%
df.iloc[483]['twitts']

# %% [markdown]
# ### Lower Case Conversion

# %%
df['twitts'] = df['twitts'].apply(lambda x: str(x).lower())

# %%
df.sample(5)

# %% [markdown]
# ## Contraction to Expansion

# %%
# x = "don't shouldn't, i'll "  # do not should not i will

# %%
pip install contractions

# %%
import contractions

# %%
custom_contractions = {
"a’ight": "alright",
"ain’t" : "i am not",
"amn’t": "am not",
"arencha" : "are not you",
"aren’t": "are not",
"’bout": "about",
"boy's": "boy has",
"can’t": "cannot",
"cap’n": "captain",
"’cause" : "because",
"cuz": "because",
"’cept": "except",
"could’ve": "could have",
"couldn’t": "could not",
"couldn’t’ve" :	"could not have",
"cuppa" : "cup of",
"daren’t" : "dare not",
"daresn’t" : "dare not",
"dasn’t" : "dare not",
"didn’t": "did not",
"doesn't": "does not",
"don’t": "do not",
"dunno"	: "do not know",
"d’ye" : "do you",
"d’ya": "did you",
"e’en":	"even",
"e’er": "ever",
"’em": "them",
"everybody’s": "everybody is",
"everyone’s": "everyone is",
"everything's": "everything is",
"finna": "fixing to",
"fo’c’sle": "forecastle",
"’gainst" : "against",
"g’day":"good day",
"gimme": "give me",
"girl's": "girl is",
"giv’n": "given",
"gi’z": "give us",
"gonna":"going to",
"gon’t": "go not",
"gotta": "got to",
"guy's": "guy is",
"hadn’t" :	"had not",
"had’ve": "had have",
"hasn’t": "has not",
"haven’t":	"have not",
"he’d": "he had",
"he’d": "he would",
"he'll": "he will",
"helluva": "hell of a",
"yes'nt":" yes not",
"he’s ": "he is",
"here’s": "here is",
"how’d": "how did",
"howdy": "how do you do",
"how’ll": "how shall",
"how’re": "how are",
"how’s": "how is",
"i’d":	"I would",
"i’d’ve": "I would have",
"i’d’nt": "I would not",
"i’d’nt’ve": "I would not have",
"if’n": "If and when",
"i’ll": "I will",
"i’m": "I am",
"imma": "I am going to",
"i’mo": "I am going to",
"innit": "isn’t it",
"ion": "I do not",
"i’ve":	"I have",
"isn’t":" is not",
"it’d": "it would",
"it’ll": "it will",
"it’s": "it is",
"idunno": "I don’t know",
"kinda": "kind of",
"let’s": "let us",
"loven’t": "love not",
"ma’am": "madam",
"mayn’t": "may not",
"may’ve": "may have",
"methinks": "I think",
"mightn’t": "might not",
"might’ve": "might have",
"mine’s": "mine is",
"mustn’t": "must not",
"mustn’t’ve": "must not have",
"must’ve": "must have",
"’neath": "beneath",
"needn’t": "need not",
"nal": "and all",
"ne’er": "never",
"o’er":	"over",
"ol’": "old",
"ought’ve": "ought have",
"oughtn’t": "ought not",
"oughtn’t’ve": "ought not have",
"’round": "around",
"’s": "is",
"shan’t": "shall not",
"she’d": "she would",
"she’ll": "she will",
"she’s": "she is",
"should’ve": "should have",
"shouldn’t": "should not",
"shouldn’t’ve": "should not have",
"somebody’s" : "somebody has",
"someone’s": "someone is",
"something’s": "something is",
"so’re": "so are",
"so’s": "so is",
"so’ve": "so have",
"that’ll": "that will",
"that’re": "that are",
"that’s": "that is",
"that’d": "that would",
"there’d": "there would",
"there’ll": "there will",
"there’re": "there are",
"there’s": "there is",
"these’re": "these are",
"these’ve": "these have",
"they’d": "they would",
"they’d've": "they would have",
"they’ll": "they will",
"they’re": "they are",
"they’ve": "they have",
"this’s": "this is",
"those’re": "those are",
"those’ve": "those have",
"’thout": "without",
"’til": "until",
"’tis": "it is",
"to’ve": "to have",
"tryna": "trying to",
"’twas": "it was",
"’tween": "between",
"’twere": "it were",
"w’all": "we all",
"w’at": "we at",
"ur": "your",
"wanna": "want to",
"wasn’t": "was not",
"we’d": "we would",
"we’d’ve": "we would have",
"we’ll": "we will",
"we’re": "we are",
"we’ve": "we have",
"weren’t": "were not",
"whatcha": 	"what are you",
"what’d": "what did",
"what’ll": "what will",
"what’re": 	"what are",
"what’s": "what is",
"what’ve": "what have",
"when’s": "when is",
"where’d": "where did",
"where’ll": "where will",
"where’re": "where are",
"where’s": 	"where is",
"where’ve": "where have",
"which’d": "which would",
"which’ll": "which will",
"which’re": "which are",
"which’s": 	"which is",
"which’ve": "which have",
"who’d": "who would",
"who’d’ve": "who would have",
"who’ll": "who will",
"who’re": "who are",
"who’s": "who is",
"who’ve": "who have",
"why’d": "why did",
"why’re": "why are",
"why’s": "why is",
"willn’t": 	"will not",
"won’t": "will not",
"wonnot": "will not",
"would’ve": "would have",
"wouldn’t": "would not",
"wouldn’t’ve": "would not have",
"y’ain’t ":	"you are not",
"y’all": "you all",
"y’all’d’ve": "you all would have",
"y’all’dn't’ve":"you all would not have",
"y’all’re": "you all are",
"y’all’ren’t": 	"you all are not",
"y’at": "you at",
"yes’m": "yes madam",
"yever": "have you ever?",
"y’know": "you know",
"yessir": "yes sir",
"you’d": "you would",
"you’ll": "you will",
"you’re": "you are",
"you’ve": "you have",
"when’d": "when did"
}

# %%

import contractions

x = "y'all know I wouldn't forget ur birthday y'know"

def cont_to_exp(x):
    return contractions.fix(x)

result = cont_to_exp(x)
print(result)

# %%
%%timeit
df['twitts'] = df['twitts'].apply(lambda x: cont_to_exp(x))

# %%
df.sample(5)

# %% [markdown]
# ### Count and Remove Emails

# %%
df[df['twitts'].str.contains('hotmail.com')]

# %%
df.iloc[3713]['twitts']

# %%
import re

# %%
x = '@securerecs arghh me please  markbradbury_16@hotmail.com'

# %%
re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', x)

# %%
df['emails'] = df['twitts'].apply(lambda x: re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', x))

# %%
df['emails_count'] = df['emails'].apply(lambda x: len(x))

# %%
df[df['emails_count']>0]

# %% [markdown]
# Remove the emails

# %%
re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)

# %%
df['twitts'] = df['twitts'].apply(lambda x : re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x))

# %% [markdown]
# ### Count and Remove URLS

# %%
x = 'hi, thanks for watching. for more videos, visit https://youtube.com/xaimli or github.com/xxy'

# %%
re.findall(r'(http|https|ftp|ssh|)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)

# %%
df['url_flags'] = df['twitts'].apply(lambda x: len(re.findall(r'(http|https|ftp|ssh|)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))

# %%
df[df['url_flags']>0].sample(5)

# %%
re.sub(r'(http|https|ftp|ssh|)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?','', x)

# %%
df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'(http|https|ftp|ssh|)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?','', x))

# %%
df.sample(10)

# %% [markdown]
# ### Remove RT

# %%
df[df['twitts'].str.contains('rt')]

# %%
x = 'rt @username: hello hi'

# %%
re.sub(r'\brt\b', '', x).strip()

# %%
df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'\brt\b', '', x).strip())

# %% [markdown]
# ### Special Character or Punctuation Removal

# %%
df.sample(3)

# %%
x = '@mayoryoung hey man, i had fun being on your s...'

# %%
re.sub(r'[^\w ]+', '',x)

# %%
df['twitts'] = df['twitts'].apply(lambda x : re.sub(r'[^\w ]+', '',x))

# %% [markdown]
# ### Remove Multiple Spaces
# 

# %%
x = ' hi      how have you been'

# %%
' '.join(x.split())

# %%
df['twitts'] = df['twitts'].apply(lambda x : ' '.join(x.split()))

# %% [markdown]
# ### Remove HTML tags

# %%
!pip install beautifulsoup4

# %%
from bs4 import BeautifulSoup

# %%
x = '<html><h1> thanks for watching </h1><html>'

# %%
# normal method

# x.replace('<html><h1>', '').replace('</h1>/<html', '')

# %%
pip install lxml

# %%
BeautifulSoup(x, 'lxml').get_text().strip()

# %%
%%time
df['twitts'] = df['twitts'].apply(lambda x : BeautifulSoup(x, 'lxml').get_text().strip())

# %% [markdown]
# ### Remove Accented Chars

# %%
x = 'áccénted ímprÓper Úndérstándíng Ñoñe'

# %%
import unicodedata

# %%
def remove_accented_char(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x

# %%
remove_accented_char(x)

# %%
df['twitts'] = df['twitts'].apply(lambda x: remove_accented_char(x) )

# %% [markdown]
# ### Remove Stop Words

# %%
x = 'this is a stop word'

# %%
' '.join([t for t in x.split() if t not in stopwords])

# %%
df['twitts_no_stop'] = df['twitts'].apply(lambda x : ' '.join([t for t in x.split() if t not in stopwords]))

# %%
df.sample(5)

# %% [markdown]
# ### Convert Word into its Root Base or Form(Lemmatization)

# %%
!python -m spacy download en_core_web_sm

# %%
nlp = spacy.load('en_core_web_sm')

# %%
x = 'I am looking out of the window in memory of the times when we bought chocolates for fun'

# %%
def convert_to_base(x):
    x = str(x)
    x_list = []
    doc = nlp(x)

    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma =='be':
            lemma = token.text

        x_list.append(lemma)
    return ' '.join(x_list)

# %%
convert_to_base(x)

# %%
df['twitts'] = df['twitts'].apply(lambda x : convert_to_base(x) )

# %% [markdown]
# ### Common Words Removal( Most Frequent Words)

# %%
x = 'this is okay this bye'

# %%
# join the series

text = ' '.join(df['twitts'])

# %%
len(text)

# %%
# get the number of words
text = text.split()

# %%
len(text)

# %%
# convert text datato Pandas Series
freq_words = pd.Series(text).value_counts()

# %%
# get the top 20 most frequently occuring words
top20 = freq_words[:20]
top20

# %%
# remove the top20 words

df['twitts'] = df['twitts'].apply(lambda x : ' '.join([t for t in x.split() if t not in top20]))

# %%
df.sample(5)

# %% [markdown]
# ### Rare Words Removal

# %%
# the least occuring words
rare20 = freq_words.tail(20)

# %%
df['twitts'] = df['twitts'].apply(lambda x : ' '.join([t for t in x.split() if t not in rare20]))

# %% [markdown]
# ### Remove rows with missing values

# %%
df.dropna(inplace=True)

# %% [markdown]
# ### Word Cloud Visualization

# %%
!pip install wordcloud

# %%
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline

# %%
text = ' '.join(df['twitts'])

# %%
len(text)

# %%
word_cloud = WordCloud(width=800, height=400).generate(text)
plt.imshow(word_cloud)
plt.axis('off')
plt.show()

# %% [markdown]
# ### Spelling Correction Using TextBlob

# %%
!pip install textblob

# %%
!python -m textblob.download_corpora

# %%
from textblob import TextBlob

# %%
x = 'it is a graet movie. I wached it twice. I wil wacth it agian'

# %%
TextBlob(x).correct()

# %%
df['twitts'] = df['twitts'].apply(lambda x : TextBlob(x).correct() )

# %% [markdown]
# ### Tokenization using TextBlob

# %%
x = 'stay#tuned for more episodes. Have a great time'

# %%
TextBlob(x).words

# %%
# tokenization with spacy
doc = nlp(x)
for token in doc:
    print(token)

# %% [markdown]
# ### Nouns Detection from a Text Data

# %%
x = 'We are pleased to announce the acquisition of Twitter by Elon Musk, the CEO of Tesla'

# %%
doc = nlp(x)

# %%
for noun in doc.noun_chunks:
    print(noun)

# %% [markdown]
# ### Language Translation and Detection using TextBlob

# %%
txtblob = TextBlob(x)

# %%
txtblob.detect_language()

# %%
txtblob.translate(to='spanish')

# %% [markdown]
# ### Sentiment Classifier Using TextBlob

# %%
from textblob.sentiments import NaiveBayesAnalyzer

# %%
x = 'she is in a terrible place, emotionally.'

# %%
txb = TextBlob(x, analyzer=NaiveBayesAnalyzer())
txb.sentiment

# %%


# %%
"""
def cont_to_exp(x):
    if type(x) is str:
        for key in custom_contractions:
            value = custom_contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x
"""

# %%
"""
def preprocess_with_custom_dict(text, custom_dict):
    for key, value in custom_dict.items():
        text = text.replace(key, f"{value}_TOKEN")
    return text

def postprocess_with_custom_dict(text, custom_dict):
    for key, value in custom_dict.items():
        text = text.replace(f"{value}_TOKEN", value)
    return text

x = "y'all know I wouldn't forget ur birthday y'know"

# Preprocess: Replace custom contractions with special tokens
x_preprocessed = preprocess_with_custom_dict(x, custom_contractions)

# Use contractions library
x_expanded = contractions.fix(x_preprocessed)

# Postprocess: Replace special tokens with expanded forms
result = postprocess_with_custom_dict(x_expanded, custom_contractions)

print(result)
"""


