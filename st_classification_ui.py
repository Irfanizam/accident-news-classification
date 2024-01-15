from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
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
import joblib
import gensim.downloader

root_path = os.path.dirname(os.path.abspath(__file__))
label_mapping = {0: 'Not Related', 1: 'Traffic Incident', 2: 'Traffic Info'}

# Load the saved SVM classifier
svm_classifier = joblib.load('ShortText/st_svm_model.h5')
# Load the saved word2vec vectorizer
loaded_vectorizer = joblib.load('ShortText/word2vec_model.model')

# Function to calculate the average Word2Vec vector for a tweet
def average_word_vectors(words, model):
    feature_vector = np.zeros((model.vector_size,), dtype="float64")
    nwords = 0
    for word in words:
        if word in model.wv.key_to_index:
            nwords = nwords + 1
            feature_vector = np.add(feature_vector, model.wv.get_vector(word))
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector


def predict(tweet, loaded_vectorizer, svm_classifier):
    tokenized_tweet = [word.lower() for word in tweet.split()]
    vectorized_input_data = [average_word_vectors(tokenized_tweet, loaded_vectorizer)]
    predictions = svm_classifier.predict(vectorized_input_data)
    print(predictions)

    if predictions[0] == 0:
        result = '0 - Not A Traffic Incident'
    elif predictions[0] == 1:
        result = '1 - Traffic Incident'
    elif predictions[0] == 2:
        result = '2 - Traffic Info'
    else:
        result = 'Out of topic'

    return result


# Global variable to store predictions
global_predictions = None

def predict_file(df: pd.DataFrame, loaded_vectorizer, svm_classifier):
    global global_predictions
    global_predictions = df['tweet'].apply(lambda tweet: predict(tweet, loaded_vectorizer, svm_classifier))
    return global_predictions

# queries = [
#     'rt the entrance ramp dalrymple drive highway west open congestion remains minimal','well said amp i grateful strong efforts expand amp grow american agriculture','cleared traffic congestion manatee highway south beyond exit st east exit highway last updated','nb sb lincoln rd st ave roadway closed due structure fire allegan county randy weits','in weekly address president obama discusses progress made combating climate left','cleared accident eb highway high rise bridge chesapeake'
# ]


# for query in queries:
#     print(query, predict(query, loaded_vectorizer, svm_classifier))
#     print()

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
    prediction = predict(tweet, loaded_vectorizer, svm_classifier)

    return word_length_img_path, word_cloud_img_path, top_10_words_img_path, prediction

# Update your Gradio interface
st_demo = gr.Interface(
    fn=visualize_data,
    inputs=gr.components.Textbox(label='Input Twitter News'),
    outputs=[gr.Image(type="pil", label="Word Length Distribution"),
             gr.Image(type="pil", label="Word Cloud"),
             gr.Image(type="pil", label="Top 10 Words Bar Chart"),
             gr.components.Label(label="Text Predictions")],
    allow_flagging='never',    
)

# upload = gr.UploadButton("Click to Upload a File", file_types=["file"])
# inp_file=gr.components.File(label="Short Text")

def download_df(file: pd.DataFrame, predictions: pd.DataFrame):
    # Combine the original text DataFrame (file) with the predictions DataFrame
    result_df = pd.concat([file, predictions], axis=1)
    
    download_path = os.path.join(root_path, "st_predicted_combined.csv")
    result_df.to_csv(download_path)
    print(f"Combined Predictions Downloaded to: {download_path}")

# Gradio Interface for file upload and predictions
with gr.Blocks(css="#warning {background-color: red} .feedback {font-size: 74px} .gradio-container {max-width: none;") as file_st_demo:
    with gr.Row():
        with gr.Column():
            st_df = gr.components.DataFrame(label="Twitter News")
            upload_button = gr.UploadButton("Click to Upload a File", file_types=["csv"])
            run_button = gr.Button("Run")
        with gr.Column():
            # Replace file_out with st_df
            st_df_out = gr.DataFrame(visible=False)
            out = gr.components.Textbox(label="Prediction", type="text")
            download_button = gr.Button("Download")

    # Adjust the lambda functions to call actual functions
    upload_button.upload(lambda file_path: pd.read_csv(file_path), inputs=upload_button, outputs=st_df)
    run_button.click(lambda file_df: predict_file(file_df, loaded_vectorizer, svm_classifier), inputs=st_df, outputs=out)
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
