#!/usr/bin/env python
# coding: utf-8

# ## Text cleaning and Preprocessing

# In[ ]:


pip install spacy


# In[3]:


import pandas as pd
import numpy as np
import spacy


# In[4]:


from spacy.lang.en.stop_words import STOP_WORDS as stopwords


# In[5]:


df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/twitter-data/master/twitter4000.csv',encoding= 'latin-1')


# In[6]:


df


# In[7]:


df['sentiment'].value_counts()


# ### Word Counts

# In[8]:


df['word_counts'] = df['twitts'].apply(lambda x: len(str(x).split()))


# In[9]:


df.sample(5)


# In[10]:


df['word_counts'].max()


# In[11]:


df['word_counts'].min()


# In[12]:


df['word_counts'] == 1


# In[13]:


df[df['word_counts']==1]


# ### Character Counts

# In[14]:


def char_counts(x):
    s = x.split()
    x = ''.join(s)
    return len(x)


# In[15]:


char_counts('I am wide awake')


# In[16]:


df['char_counts'] = df['twitts'].apply(lambda x: char_counts(str(x)))


# In[17]:


df.sample(5)


# ### Average Word Length

# In[18]:


df['avg_word_len'] = df['char_counts']/df['word_counts']


# In[19]:


df.sample(5)


# ### Stop Words Count

# In[20]:


print(stopwords)


# In[21]:


len(stopwords)


# In[22]:


df['stop_words_len'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t in stopwords]))


# ### #Hashtags and @Mentions Count

# In[23]:


# [t for t in x.split() if t.starstwith('#')]


# In[24]:


df['hashtags_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('#')]))


# In[25]:


df['mentions_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))


# In[26]:


df.sample(5)


# ### Numeric Digits Count

# In[27]:


x = 'this is 1 and 2'
x.split()


# In[28]:


x.split()[4].isdigit()


# In[29]:


[t for t in x.split() if t.isdigit()]


# In[30]:


df['numeric_count'] = df['twitts'].apply(lambda x : len([t for t in x.split() if t.isdigit()]))


# In[31]:


df.sample(5)


# ### Upper Case Words Count

# In[32]:


x = "I GOT THE JOB"
y = "I got the job"


# In[33]:


[t for t in x.split() if t.isupper()]


# In[34]:


[t for t in y.split() if t.isupper()]


# In[35]:


df['upper_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.isupper()]))


# In[36]:


df.sample(5)


# In[37]:


df.iloc[483]['twitts']


# ### Lower Case Conversion

# In[38]:


df['twitts'] = df['twitts'].apply(lambda x: str(x).lower())


# In[39]:


df.sample(5)


# ## Contraction to Expansion

# In[40]:


# x = "don't shouldn't, i'll "  # do not should not i will


# In[41]:


pip install contractions


# In[42]:


import contractions


# In[43]:


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


# In[44]:


import contractions

x = "y'all know I wouldn't forget ur birthday y'know"

def cont_to_exp(x):
    return contractions.fix(x)

result = cont_to_exp(x)
print(result)


# In[45]:


get_ipython().run_cell_magic('timeit', '', "df['twitts'] = df['twitts'].apply(lambda x: cont_to_exp(x))\n")


# In[46]:


df.sample(5)


# ### Count and Remove Emails

# In[47]:


df[df['twitts'].str.contains('hotmail.com')]


# In[48]:


df.iloc[3713]['twitts']


# In[49]:


import re


# In[50]:


x = '@securerecs arghh me please  markbradbury_16@hotmail.com'


# In[51]:


re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', x)


# In[52]:


df['emails'] = df['twitts'].apply(lambda x: re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', x))


# In[53]:


df['emails_count'] = df['emails'].apply(lambda x: len(x))


# In[54]:


df[df['emails_count']>0]


# Remove the emails

# In[55]:


re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)


# In[56]:


df['twitts'] = df['twitts'].apply(lambda x : re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x))


# ### Count and Remove URLS

# In[57]:


x = 'hi, thanks for watching. for more videos, visit https://youtube.com/xaimli or github.com/xxy'


# In[58]:


re.findall(r'(http|https|ftp|ssh|)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)


# In[59]:


df['url_flags'] = df['twitts'].apply(lambda x: len(re.findall(r'(http|https|ftp|ssh|)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))


# In[60]:


df[df['url_flags']>0].sample(5)


# In[61]:


re.sub(r'(http|https|ftp|ssh|)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?','', x)


# In[62]:


df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'(http|https|ftp|ssh|)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?','', x))


# In[63]:


df.sample(10)


# ### Remove RT

# In[64]:


df[df['twitts'].str.contains('rt')]


# In[65]:


x = 'rt @username: hello hi'


# In[66]:


re.sub(r'\brt\b', '', x).strip()


# In[67]:


df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'\brt\b', '', x).strip())


# ### Special Character or Punctuation Removal

# In[68]:


df.sample(3)


# In[69]:


x = '@mayoryoung hey man, i had fun being on your s...'


# In[70]:


re.sub(r'[^\w ]+', '',x)


# In[71]:


df['twitts'] = df['twitts'].apply(lambda x : re.sub(r'[^\w ]+', '',x))


# ### Remove Multiple Spaces
# 

# In[72]:


x = ' hi      how have you been'


# In[73]:


' '.join(x.split())


# In[74]:


df['twitts'] = df['twitts'].apply(lambda x : ' '.join(x.split()))


# ### Remove HTML tags

# In[75]:


get_ipython().system('pip install beautifulsoup4')


# In[76]:


from bs4 import BeautifulSoup


# In[83]:


x = '<html><h1> thanks for watching </h1><html>'


# In[78]:


# normal method

# x.replace('<html><h1>', '').replace('</h1>/<html', '')


# In[ ]:


pip install lxml


# In[84]:


BeautifulSoup(x, 'lxml').get_text().strip()


# In[85]:


get_ipython().run_cell_magic('time', '', "df['twitts'] = df['twitts'].apply(lambda x : BeautifulSoup(x, 'lxml').get_text().strip())\n")


# ### Remove Accented Chars

# In[86]:


x = 'áccénted ímprÓper Úndérstándíng Ñoñe'


# In[87]:


import unicodedata


# In[88]:


def remove_accented_char(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x


# In[89]:


remove_accented_char(x)


# In[90]:


df['twitts'] = df['twitts'].apply(lambda x: remove_accented_char(x) )


# ### Remove Stop Words

# In[93]:


x = 'this is a stop word'


# In[92]:


' '.join([t for t in x.split() if t not in stopwords])


# In[94]:


df['twitts_no_stop'] = df['twitts'].apply(lambda x : ' '.join([t for t in x.split() if t not in stopwords]))


# In[95]:


df.sample(5)


# ### Convert Word into its Root Base or Form(Lemmatization)

# In[ ]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[100]:


nlp = spacy.load('en_core_web_sm')


# In[102]:


x = 'I am looking out of the window in memory of the times when we bought chocolates for fun'


# In[105]:


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


# In[106]:


convert_to_base(x)


# In[107]:


df['twitts'] = df['twitts'].apply(lambda x : convert_to_base(x) )


# ### Common Words Removal( Most Frequent Words)

# In[108]:


x = 'this is okay this bye'


# In[110]:


# join the series

text = ' '.join(df['twitts'])


# In[111]:


len(text)


# In[112]:


# get the number of words
text = text.split()


# In[113]:


len(text)


# In[116]:


# convert text datato Pandas Series
freq_words = pd.Series(text).value_counts()


# In[117]:


# get the top 20 most frequently occuring words
top20 = freq_words[:20]
top20


# In[120]:


# remove the top20 words

df['twitts'] = df['twitts'].apply(lambda x : ' '.join([t for t in x.split() if t not in top20]))


# In[121]:


df.sample(5)


# ### Rare Words Removal

# In[123]:


# the least occuring words
rare20 = freq_words.tail(20)


# In[124]:


df['twitts'] = df['twitts'].apply(lambda x : ' '.join([t for t in x.split() if t not in rare20]))


# ### Remove rows with missing values

# In[158]:


df.dropna(inplace=True)


# ### Word Cloud Visualization

# In[ ]:


get_ipython().system('pip install wordcloud')


# In[126]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[127]:


text = ' '.join(df['twitts'])


# In[128]:


len(text)


# In[129]:


word_cloud = WordCloud(width=800, height=400).generate(text)
plt.imshow(word_cloud)
plt.axis('off')
plt.show()


# ### Spelling Correction Using TextBlob

# In[ ]:


get_ipython().system('pip install textblob')


# In[ ]:


get_ipython().system('python -m textblob.download_corpora')


# In[132]:


from textblob import TextBlob


# In[137]:


x = 'it is a graet movie. I wached it twice. I wil wacth it agian'


# In[138]:


TextBlob(x).correct()


# In[ ]:


df['twitts'] = df['twitts'].apply(lambda x : TextBlob(x).correct() )


# ### Tokenization using TextBlob

# In[140]:


x = 'stay#tuned for more episodes. Have a great time'


# In[141]:


TextBlob(x).words


# In[143]:


# tokenization with spacy
doc = nlp(x)
for token in doc:
    print(token)


# ### Nouns Detection from a Text Data

# In[144]:


x = 'We are pleased to announce the acquisition of Twitter by Elon Musk, the CEO of Tesla'


# In[145]:


doc = nlp(x)


# In[146]:


for noun in doc.noun_chunks:
    print(noun)


# ### Language Translation and Detection using TextBlob

# In[151]:


txtblob = TextBlob(x)


# In[ ]:


txtblob.detect_language()


# In[ ]:


txtblob.translate(to='spanish')


# ### Sentiment Classifier Using TextBlob

# In[153]:


from textblob.sentiments import NaiveBayesAnalyzer


# In[154]:


x = 'she is in a terrible place, emotionally.'


# In[156]:


txb = TextBlob(x, analyzer=NaiveBayesAnalyzer())
txb.sentiment


# In[ ]:





# In[81]:


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


# In[82]:


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

