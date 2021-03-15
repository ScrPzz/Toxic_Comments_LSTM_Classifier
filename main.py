from clean_text import clean_text
from stem_and_stopwords import stem_and_stopwords

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns


import nltk  # Python library for NLP
import warnings
import sys

warnings.simplefilter(action="ignore")

sys.setrecursionlimit(3000)


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_subm = pd.read_csv('sample_submission.csv')
nltk.download('stopwords')

train.head()


train['comment_text'] = train['comment_text'].apply(lambda x: clean_text(x))

words = ' '.join([text for text in train["comment_text"]])

word_cloud = WordCloud(
    width=1600,
    height=800,
    # colormap='PuRd',
    margin=0,
    max_words=100,  # Maximum numbers of words we want to see
    min_word_length=3,  # Minimum numbers of letters of each word to be part of the cloud
    max_font_size=150, min_font_size=30,  # Font size range
    background_color="white").generate(words)

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('Comments and their Nature: training set', fontsize=40)
plt.axis("off")
plt.show()


train['comment_text'].head()

# Applying the remove_stopwords on train and test set

train['StemStop'] = train['comment_text'].apply(lambda x: stem_and_stopwords(x))
test['StemStop'] = test['comment_text'].apply(lambda x: stem_and_stopwords(x))

train.head()
test.head()


cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = train[cols].values

train_df = train['comment_text']
test_df = test['comment_text']


val_counts = train[cols].sum()

plt.figure(figsize=(8,5))
ax = sns.barplot(val_counts.index, val_counts.values, alpha=0.8)

plt.title("Comments per Class")
plt.xlabel("Comment label")
plt.ylabel("Count")

rects = ax.patches
labels = val_counts.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+5, label, ha="center", va="bottom")


plt.show()




# Word Cloud for test set

words = ' '.join([text for text in test['comment_text'] ])


word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       #colormap='PuRd',
                       margin=0,
                       max_words=500, # Maximum numbers of words we want to see
                       min_word_length=3, # Minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white").generate(words)

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="bilinear")
plt.title('Comments and their Nature: test set', fontsize = 40)
plt.axis("off")
plt.show()



from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Input,  Activation
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



max_features = 22000

tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(list(train_df))

tokenized_train = tokenizer.texts_to_sequences(train_df)
tokenized_test = tokenizer.texts_to_sequences(test_df)





embed_size = 128
maxlen = 200
max_features = 22000
X_train = pad_sequences(tokenized_train, maxlen = maxlen)
X_test = pad_sequences(tokenized_test, maxlen = maxlen)



inp = Input(shape = (maxlen, ))
x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)


model = Model(inputs=inp, outputs=x)
model.compile(
loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy']
)

model.summary()


batch_size = 64
epochs = 10
model.fit(X_train, targets, batch_size=batch_size, epochs=epochs, validation_split=0.1)

prediction = model.predict(X_test)
prediction

