# Chatbot
A repository of intents based deep learning chatbots for eventual deployment on website. 


# Project Description
There are currently 2 deep learning based chatbot models. Both use same data file 

In Model 1 the data is seperated into 2 lists containing pattern and tags. Label encoder is used to encode the tags. The training sentences are then tokenised and padded and then passed into model which uses Adam optimizer and Sparse Categorical Crossentropy loss.

In Model 2 using NLTK, the data is first extracted (patterns and tags), tokensised, lemmatized, then the words and tags are converted into pickle files. Then a training file is created which is then passed to a model using SGD optimizer and Categorical Crossentropy loss.

A chatbot gui (created using tkinter) is present in both models for testing purposes.

# Using it:
Run app.py from either models

