from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import os
import pickle
import gradio as gr
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score


loaded_model = pickle.load(open('C:/Users/irfanizam/workspace/FYP-Example/FinalFinalFYP/LongText/lt_svm_model.pkl', 'rb'))
dataframe = pd.read_csv('C:/Users/irfanizam/workspace/FYP-Example/FinalFinalFYP/Data2/ArrangedLt.csv')
X = dataframe['summary']
y = dataframe[['4', '5', '6']]

# Tokenization
dataframe['tokens'] = dataframe['summary'].apply(word_tokenize)
# Train Word2Vec model
word2vec_model = Word2Vec(dataframe['tokens'], vector_size=100, window=5, min_count=1, workers=4)
# Function to average word vectors for a sentence
def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    
    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    
    return feature_vector
# Apply the average_word_vectors function to each row
dataframe['word2vec_features'] = dataframe['tokens'].apply(
    lambda x: average_word_vectors(x, word2vec_model, word2vec_model.wv.index_to_key, 100)
)
# Prepare data for SVM
X = np.array(list(dataframe['word2vec_features']), copy=True)  # Explicitly create a new array
# Convert multi-labels to binary form
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(dataframe[['4', '5', '6']].astype(int))# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# SVM Classifier
svm_classifier = SVC(kernel='linear', C=1, probability=True)
multi_output_classifier = MultiOutputClassifier(svm_classifier, n_jobs=-1)  # n_jobs=-1 uses all available CPU cores
multi_output_classifier.fit(X_train, y_train)
y_pred = multi_output_classifier.predict(X_test)

pickle.dump(multi_output_classifier, open('lt_svm_model.pkl', 'wb'))

loaded_model = pickle.load(open('lt_svm_model.pkl', 'rb'))


def predict(summary):
    input_data = [summary]
    
    # Tokenize the input data
    tokens = word_tokenize(input_data[0])
    
    # Get the Word2Vec embeddings for the tokens
    vectorized_input_data = average_word_vectors(tokens, word2vec_model, word2vec_model.wv.index_to_key, 100)
    
    # Reshape to match the format expected by the model
    vectorized_input_data = vectorized_input_data.reshape(1, -1)

    # Make predictions using the loaded_model
    prediction = loaded_model.predict(vectorized_input_data)
     # Check if any element in the prediction result array is True
    if prediction[0, 3] == 1:
        result = '4 FALLS, SLIPS, TRIPS'
    elif prediction[0, 4] == 1:
        result = '5 EXPOSURE TO HARMFUL SUBSTANCES OR ENVIRONMENTS'
    elif prediction[0, 5] == 1:
        result = '6 CONTACT WITH OBJECTS AND EQUIPMENT'
    else:
        result = 'Out of topic'

    return result

print(predict('on november approximately employee standing hydro mobile scaffolding approximately feet ground employee mortar loading section scaffolding preparing receive full tub mortar employee guardrail supported sliding gate guardrail detached platform sending employee feet death'))

lt_demo = gr.Interface(
    fn = predict,
    inputs = gr.components.Textbox(label = 'Input long text'),
    outputs = gr.components.Label(label = 'Predictions'),
    allow_flagging='never'
)

file_lt_demo = gr.Interface(
    fn = predict,
    inputs = gr.components.File(label="Long Text"),
    outputs = gr.components.Textbox(label="answer", type="text"),
    allow_flagging='never'

)
