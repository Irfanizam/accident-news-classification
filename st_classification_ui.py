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
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import tempfile


root_path = os.path.dirname(os.path.abspath(__file__))
loaded_model = pickle.load(open(os.path.join(root_path, 'ShortText/st_svm_model.pkl'), 'rb'))
dataframe = pd.read_csv(os.path.join(root_path, 'Data1/ArrangedSt.csv'))
X = dataframe['tweet']
y = dataframe[['0', '1', '2']]
print("LOADED")
# Tokenization
dataframe['tokens'] = dataframe['tweet'].apply(word_tokenize)
# Train Word2Vec model
word2vec_model = Word2Vec(dataframe['tokens'], vector_size=100, window=5, min_count=1, workers=4)
print("CREATED WORD2VEC")
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
# y = label_binarizer.fit_transform(dataframe[['0', '1', '2']].astype(int))# Split the data into training and testing sets
# print("FIT BINARY TRANSFORM")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# SVM Classifier
# svm_classifier = SVC(kernel='linear', C=1, probability=True)
# multi_output_classifier = MultiOutputClassifier(svm_classifier, n_jobs=-1)  # n_jobs=-1 uses all available CPU cores
# multi_output_classifier.fit(X_train, y_train)
# y_pred = multi_output_classifier.predict(X_test)
# print("TRAINED MULTIOUTPUTCLASSIFIER")
# pickle.dump(multi_output_classifier, open('st_svm_model.pkl', 'wb'))

loaded_model = pickle.load(open('st_svm_model.pkl', 'rb'))
print("LOADED MODEL")
def predict(tweet):
    print(tweet)
    input_data = [tweet]
    
    # Tokenize the input data
    tokens = word_tokenize(input_data[0])
    
    # Get the Word2Vec embeddings for the tokens
    vectorized_input_data = average_word_vectors(tokens, word2vec_model, word2vec_model.wv.index_to_key, 100)
    
    # Reshape to match the format expected by the model
    vectorized_input_data = vectorized_input_data.reshape(1, -1)

    # Make predictions using the loaded_model
    prediction = loaded_model.predict(vectorized_input_data)

     # Check if any element in the prediction result array is True
    print(prediction)

    if prediction[0, 0] == 1:
        result = '0 - Not Related'
    elif prediction[0, 1] == 1:
        result = '1 - Traffic Incident'
    elif prediction[0, 2] == 1:
        result = '2 - Traffic Info'
    elif prediction[0, 1] == 1 and prediction[0, 2] == 1:
        result = '1 - Traffic Incident and 2 - Traffic Info'
    elif prediction[0, 0] == 1 and prediction[0, 1] == 1:
        result = '0 - Not Related and 1 - Traffic Incident'
    elif prediction[0, 0] == 1 and prediction[0, 2] == 1:
        result = '0 - Not Related and 2 - Traffic Info'
    elif prediction[0, 1] == 1 and prediction[0, 2] == 1 and prediction[0, 0] == 0:
        result = '1 - Traffic Incident and 2 - Traffic Info'
    elif prediction[0, 0] == 0 and prediction[0, 1] == 0 and prediction[0, 2] == 0:
        result = 'No Label'
    else:
        result = 'Out of topic'

    return result


# Global variable to store predictions
global_predictions = None

def predict_file(df: pd.DataFrame):
    global global_predictions
    global_predictions = df['tweet'].apply(predict)
    return global_predictions

# print(predict('breaking traffic there person lying roadway ramp highway north main street we'))

def visualize_data(tweet):
    img_dir = os.path.expanduser('~/visualization_images')
    os.makedirs(img_dir, exist_ok=True)


    # Plot histogram for word lengths
    word_lengths = [len(word) for word in word_tokenize(tweet)]
    plt.figure(figsize=(12, 6))
    plt.hist(word_lengths, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Word Lengths')
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(img_dir, 'word_length_distribution.png'))

    # Plot word cloud for most frequent words
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(tweet)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Most Frequent Words')
    plt.savefig(os.path.join(img_dir, 'word_cloud.png'))

    # Plot bar chart for top N most frequent words
    tokens = word_tokenize(tweet)
    word_freq = pd.Series(tokens).value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=word_freq.values, y=word_freq.index, palette='viridis')
    plt.title('Top 10 Most Frequent Words')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.savefig(os.path.join(img_dir, 'top_10_words_bar_chart.png'))

    # Add more visualizations as needed

    # Save the generated images to the directory
    word_length_img_path = os.path.join(img_dir, 'word_length_distribution.png')
    word_cloud_img_path = os.path.join(img_dir, 'word_cloud.png')
    top_10_words_img_path = os.path.join(img_dir, 'top_10_words_bar_chart.png')
    prediction = predict(tweet)

    return word_length_img_path, word_cloud_img_path, top_10_words_img_path, prediction
# Update your Gradio interface
st_demo = gr.Interface(
    fn=visualize_data,
    inputs=gr.components.Textbox(label='Input short text'),
    outputs=[gr.Image(type="pil", label="Word Length Distribution"),
             gr.Image(type="pil", label="Word Cloud"),
             gr.Image(type="pil", label="Top 10 Words Bar Chart"),
             gr.components.Label(label="Text Predictions")],
    allow_flagging='never'
)

# upload = gr.UploadButton("Click to Upload a File", file_types=["file"])
# inp_file=gr.components.File(label="Short Text")



def download_df(file: pd.DataFrame, predictions: pd.DataFrame):
    # Combine the original text DataFrame (file) with the predictions DataFrame
    result_df = pd.concat([file, predictions], axis=1)
    
    download_path = os.path.join(root_path, "st_predicted_combined.csv")
    result_df.to_csv(download_path)
    print(f"Combined Predictions Downloaded to: {download_path}")

with gr.Blocks() as file_st_demo:
    with gr.Row():
        with gr.Column():
            st_df = gr.components.DataFrame(label="Short Text")
            with gr.Row():
                upload_button = gr.UploadButton("Click to Upload a File", file_types=["file"])
                run_button = gr.Button("Run")
        with gr.Column():
            file_out = gr.DataFrame(visible=False)
            out = gr.components.Textbox(label="Prediction", type="text")
            download_button = gr.Button("Download")

    upload_button.upload(lambda file_path: file_path, inputs=upload_button, outputs=st_df)
    run_button.click(predict_file, inputs=st_df, outputs=out)
    download_button.click(lambda file_df: download_df(file_df, global_predictions), inputs=st_df)


    


# upload = gr.Interface(
#     fn=lambda filename: pd.read_csv(filename),
#     inputs=gr.components.File(label="Short Text"),
#     outputs = st_df,
#     allow_flagging='never'
# )



# file_st_demo = gr.Interface(
#     fn = predict_file,
#     inputs = st_df,
#     outputs = gr.components.Textbox(label="answer", type="text"),
#     allow_flagging='never'

# )