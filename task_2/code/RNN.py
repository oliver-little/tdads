'''
# Classifying Airline Sentiment using LSTM cells (Recurrent Neural Network)
# Aim is to train the model on 80% of the data provided and then test it with the other 20%
# My tweets_for_RNN.csv file contains the extra datasets on top of the data provided but only the first 14640 rows are used in this code
'''

# Import of libraries
import re
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Import of preprocessing script
from preproc import *

def preprocess(tweets):
    # Utilise preprocessing scripts
    tweets = remove_airline_tags(tweets)
    tweets.text = remove_links(tweets.text)
    tweets.text = lt_gt_conversion(ampersand_conversion(arrow_conversion(tweets.text)))
    tweets.text = with_without_conversion(tweets.text)
    tweets.text = hashtag_to_words(tweets.text)
    tweets = translate_all_emoji(tweets)
    tweets.text = remove_contractions(tweets.text)
    tweets.text = remove_punctuation(tweets.text)
    tweets.text = lemmatize_texts(tweets.text)
    tweets = tweets[['airline_sentiment', 'text']]

    # Convert all text to lowercase and remove symbols
    tweets['text'].apply(lambda x: x.lower())
    tweets['text'] = tweets['text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))

    # Tokenise texts and make sure our text vectors all have the same length
    tokenizer = Tokenizer(num_words=5000, split=" ")
    tokenizer.fit_on_texts(tweets['text'].values)
    X = tokenizer.texts_to_sequences(tweets['text'].values)
    X = pad_sequences(X)

    # Acquire labels
    Y = pd.get_dummies(tweets['airline_sentiment']).values

    return X[:14640], Y[:14640]

def fit(x_train, y_train):
    if (len(sys.argv) == 1):
        model = Sequential()
        model.add(Embedding(5000, 256, input_length=x_train.shape[1]))
        model.add(Dropout(0.3))
        model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
        model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.2))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)
        model.save('sentiment_analysis_model_temp.h5')
    else:
        model = load_model(sys.argv[1])
    return model

def predict(x_test, model):
    return model.predict(x_test)

if __name__ == "__main__":
    # Acquire texts containing sentiment and corresponding labels
    X, Y = preprocess(pd.read_csv('tweets_for_RNN.csv'))

    # Split the dataset into training and testing batches
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Train the model
    model = fit(x_train, y_train)

    # Make predictions
    predictions = predict(x_test, model)

    # Calculate the accuracy of the predictions
    total_correct = 0

    for i in range(len(y_test)):
        if (np.argmax(predictions[i]) == np.argmax(y_test[i])):
            total_correct += 1
    print("Accuracy: " + str(total_correct/len(y_test)))