import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words = []
classes = []
documents = []
remove_punct = ['?', '!']
data_file = open('DL Model 2/intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        word = nltk.word_tokenize(pattern)
        words.extend(word) #extend because append will give output as [1, 2, 3, [4, 5]] and extend will give output as [1, 2, 3, 4, 5]
        documents.append((word, intent['tag'])) #adds documents in the corpus

        if intent['tag'] not in classes: #adds to classes list
            classes.append(intent['tag'])

lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in remove_punct] #lemmatizes lower case non duplicate words
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")

print(len(classes), "classes", classes)

print(len(words), "Lemmatized words", words)

pickle.dump(words,open('DL Model 2/words.pkl','wb'))
pickle.dump(classes,open('DL Model 2/classes.pkl','wb'))

training = []
# Empty list for output
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        # create our bag of words array with 1, if word match found in current pattern
        bag.append(1) if w in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        
        training = list(training)
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training,dtype=object)

    train_x = list(training[:,0])
    train_y = list(training[:,1])
    print("Training Data Created")

# print(training)

model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('DL Model 2/chatbot_model.h5', hist)

print('Model Created')

