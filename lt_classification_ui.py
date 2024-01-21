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
from collections import Counter

root_path = os.path.dirname(os.path.abspath(__file__))
label_mapping = {1: 'Falls,Slips,Trips', 2: 'Expose to Harmful Substance', 3: 'Contact with objects/equipments'}

# Load the saved SVM classifier
svm_classifier = joblib.load('LongText/lt_svm_model.h5')
# Load the saved word2vec vectorizer
loaded_vectorizer = joblib.load('LongText/word2vec_model.model')

# Function to calculate the average Word2Vec vector for a summary
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

def lt_predict(summary, model, classifier):
    tokenized_summary = [word.lower() for word in summary.split()]
    vectorized_input_data = [average_word_vectors(tokenized_summary, model)]
    predictions = classifier.predict(vectorized_input_data)

    if predictions[0] == 0:
        result = '1 Falls,Slips,Trips'
    elif predictions[0] == 1:
        result = '2 Expose to Harmful Substance'
    elif predictions[0] == 2:
        result = '3 Contact with objects/equipments'
    else:
        result = 'Out of topic'
    
    return result

# Global variable to store predictions
global_predictions = None
# Define separate folders for short text and long text interfaces
new_file_folder_lt = "TweetNews_Figures_Lt"

# Ensure the folders exist, create them if not
os.makedirs(new_file_folder_lt, exist_ok=True)

def lt_predict_file(df: pd.DataFrame, loaded_vectorizer, svm_classifier):
    global global_predictions
    global_predictions = df['summary'].apply(lambda summary: lt_predict(summary, loaded_vectorizer, svm_classifier))
    return global_predictions

def lt_visualize_file(file_df):
    img_dir = os.path.expanduser('~/visualization_images_lt')
    os.makedirs(img_dir, exist_ok=True)
    df = pd.read_csv('lt_predicted_combined.csv')
    df.columns = ['summary', 'tag']
    label_counts = df['tag'].value_counts().rename(index=label_mapping)
    formatted_output_lt = '\n'.join([f"{label}: {count}" for label, count in label_counts.items()])
    print(df)
    text_column = 'summary'
    all_text = ' '.join(df[text_column].astype(str))
    words = word_tokenize(all_text)
    word_counts = Counter(words)

    # Find the word with the maximum count
    max_word_lt, max_count_lt = max(word_counts.items(), key=lambda x: x[1])

    # Find the word with the minimum count
    min_word_lt, min_count_lt = min(word_counts.items(), key=lambda x: x[1])

    # Display word cloud
    all_tweets_text = ' '.join(df['summary'])
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(all_tweets_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Word Cloud')
    plt.savefig(os.path.join(img_dir, "wordcloud_lt.png"))
    plt.close()

    # Display tweet length distribution
    tweet_lengths = df['summary'].apply(len)
    plt.figure(figsize=(10, 5))
    plt.hist(tweet_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Tweet Lengths')
    plt.xlabel('Tweet Length')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(img_dir, "wordlength_lt.png"))
    plt.close()

    # Display class distribution
    sns.countplot(x='tag', data=df, hue='tag')
    plt.title('Class Distribution')
    plt.savefig(os.path.join(img_dir, "class_lt.png"))
    plt.close()

    word_length_img_path_lt = os.path.join(img_dir, 'wordlength_lt.png')
    word_cloud_img_path_lt = os.path.join(img_dir, 'wordcloud_lt.png')
    class_path_lt = os.path.join(img_dir, 'class_lt.png')

    return formatted_output_lt, max_word_lt, max_count_lt, min_word_lt, min_count_lt, word_length_img_path_lt, word_cloud_img_path_lt, class_path_lt


def visualize_data_lt(summary):
    img_dir_lt = os.path.expanduser('~/visualization_images')
    os.makedirs(img_dir_lt, exist_ok=True)


    # Plot histogram for word lengths
    word_lengths = [len(word) for word in word_tokenize(summary)]
    plt.figure(figsize=(12, 6))
    plt.hist(word_lengths, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Word Lengths')
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(img_dir_lt, 'word_length_distribution.png'))

    # Plot word cloud for most frequent words
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(summary)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Most Frequent Words')
    plt.savefig(os.path.join(img_dir_lt, 'word_cloud.png'))

    # Plot bar chart for top N most frequent words
    tokens = word_tokenize(summary)
    word_freq = pd.Series(tokens).value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=word_freq.values, y=word_freq.index, palette='viridis')
    plt.title('Top 10 Most Frequent Words')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.savefig(os.path.join(img_dir_lt, 'top_10_words_bar_chart.png'))

    # Add more visualizations as needed

    # Save the generated images to the directory
    word_length_img_path = os.path.join(img_dir_lt, 'word_length_distribution.png')
    word_cloud_img_path = os.path.join(img_dir_lt, 'word_cloud.png')
    top_10_words_img_path = os.path.join(img_dir_lt, 'top_10_words_bar_chart.png')
    prediction = lt_predict(summary, loaded_vectorizer, svm_classifier)

    return word_length_img_path, word_cloud_img_path, top_10_words_img_path, prediction

# Gradio Interface for user input and predictions
with gr.Blocks() as lt_demo:
    with gr.Column():
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox("", type="text", label="Summary")
                out_text = gr.components.Textbox(label="Predicted News", type="text")
            with gr.Column():
                run_button_text = gr.Button("Label News")
                visualize_button = gr.Button("Visualize")
        with gr.Accordion("Data Visualization", open=False):
            with gr.Row():
                out_img = gr.Image(type="pil", label="Word Length Distribution")
                out_img1 = gr.Image(type="pil", label="Word Cloud")
                out_img2 = gr.Image(type="pil", label="Top 10 Words Bar Chart")

    run_button_text.click(lambda summary: lt_predict(summary, loaded_vectorizer, svm_classifier), inputs=text_input, outputs=out_text)
    visualize_button.click(lambda summary: visualize_data_lt(summary), inputs=text_input, outputs=[out_img, out_img1, out_img2])

def download_df(file: pd.DataFrame, predictions: pd.DataFrame):
    # Combine the original text DataFrame (file) with the predictions DataFrame
    result_df = pd.concat([file.rename(columns={"summary": "text"}), predictions], axis=1)
    
    download_path = os.path.join(root_path, "lt_predicted_combined.csv")
    # result_df.to_csv(download_path, index=False)
    # print(f"Combined Predictions Downloaded to: {download_path}")
    return result_df

def download_df_data(file: pd.DataFrame, predictions: pd.DataFrame):
    # Combine the original text DataFrame (file) with the predictions DataFrame
    result_df = pd.concat([file.rename(columns={"summary": "text"}), predictions], axis=1)
    
    download_path = os.path.join(root_path, "lt_predicted_combined.csv")
    result_df.to_csv(download_path, index=False)
    print(f"Combined Predictions Downloaded to: {download_path}")
    gr.Info("File Downloaded")
    return result_df

# Gradio Interface for file upload and predictions
with gr.Blocks() as file_lt_demo:
    with gr.Row():
        with gr.Column():
            lt_df = gr.components.DataFrame(label="Web News", height=200)
        with gr.Column():
            upload_button = gr.UploadButton("Click to Upload a File", file_types=["csv"])
            run_button = gr.Button("Label News")
    with gr.Row():
        with gr.Column():
            lt_df_out = gr.DataFrame(visible=False)
            out = gr.components.DataFrame(label="Predicted News Data", height=200)
        with gr.Column():
            download_button = gr.Button("Download")
            visualize_button = gr.Button("Visualize")
    # Long Text Interface
    with gr.Tab(label="Data Visualization"):
        with gr.Accordion("List of Figures", open=False):
            with gr.Row():
                with gr.Column():
                    label_counts = gr.Textbox(label="Label Counts")
                    with gr.Row():
                        max_word = gr.Textbox(label="Most Frequent Word")
                        max_count = gr.Textbox(label="Word Count")
                        min_word = gr.Textbox(label="Least Frequent Word")
                        min_count = gr.Textbox(label="Word Count")
                    with gr.Row():
                        out_img = gr.Image(label="Word Length Distribution")
                        out_img1 = gr.Image(label="Word Cloud")
                        out_img2 = gr.Image(label="Class Distribution")

    # Adjust the lambda functions to call actual functions
    upload_button.upload(lambda file_path: pd.read_csv(file_path), inputs=upload_button, outputs=lt_df)
    # run_button.click(lambda file_df: st_predict_file(file_df, loaded_vectorizer, svm_classifier), inputs=st_df, outputs=out)
    # Adjust the lambda functions to call actual functions
    upload_button.upload(lambda file_path: pd.read_csv(file_path), inputs=upload_button, outputs=lt_df)

    run_button.click(lambda file_df: download_df(file_df, lt_predict_file(file_df, loaded_vectorizer, svm_classifier)), inputs=lt_df, outputs=out)

    download_button.click(lambda file_df: download_df_data(file_df, global_predictions), inputs=lt_df)

    visualize_button.click(lambda df: lt_visualize_file(df), inputs=None, outputs=[label_counts, max_word, max_count, min_word, min_count, out_img, out_img1, out_img2])