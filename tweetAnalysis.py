import re
import nltk
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem.porter import *
from wordcloud import WordCloud

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore", category=DeprecationWarning)

train = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')
combine = train.append(test, ignore_index=True)


def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)
    return input_text


# # Removing twitter handles, punctuations, short words and split the remaining words as tokens
combine['tidy_tweet'] = np.vectorize(remove_pattern)(combine['tweet'], "@[\w]*")
combine['tidy_tweet'] = combine['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
combine['tidy_tweet'] = combine['tidy_tweet'].apply(lambda x: ''.join([w for w in x.split() if len(w) > 3]))
token_tweets = combine['tidy_tweet'].apply(lambda x: x.split())
s = PorterStemmer()
token_tweets = token_tweets.apply(lambda x: [s.stem(i) for i in x])
for i in range(len(token_tweets)):
    token_tweets[i] = ' '.join(token_tweets[i])
combine['tidy_tweet'] = token_tweets

# Creating a word cloud to understand the words for each tweet being racist/sexist or not
all_words = ' '.join(text for text in combine['tidy_tweet'])
wordCloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()

