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
    print(predictions)

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

def lt_predict_file(df: pd.DataFrame, loaded_vectorizer, svm_classifier):
    global global_predictions
    global_predictions = df['summary'].apply(lambda summary: lt_predict(summary, loaded_vectorizer, svm_classifier))
    return global_predictions

# queries = [
#     'a team dredging employees changing line hoist drum employee pants leg became caught hoist line pulled around hoist drum employee sustained fatal injuries died within next minutes','at approximately december employee mechanic working number boiler house unit second employer the second employer refiner petroleum products employee attempting step open drain sump contained hot substance degrees fahrenheit when attempted step open sump stepped immersed right leg lower thigh employee received burns right leg hospitalized seven days','a power line worker mounting l bracket crossarm utility pole the bracket going hold cutout new transformer as employee tightening two top bolts bracket contacted power line neck he electrocuted','on february employee coworker installing metal decking onto steel beams skylight lobby area towson town center mall this serve containment subsequent asbestos removal employee fell approximately ft steel beams concrete floor he hospitalized university maryland shock trauma center fractured skull fractured nose fractured arms the investigation revealed employee wearing fall protection equipment time accident the coworker returned ground level via scissors lift see caused employee fall'
# ]


# for query in queries:
#     print(query, predict(query, loaded_vectorizer, svm_classifier))
#     print()

def visualize_data_lt(summary):
    img_dir = os.path.expanduser('~/visualization_images')
    os.makedirs(img_dir, exist_ok=True)


    # Plot histogram for word lengths
    word_lengths = [len(word) for word in word_tokenize(summary)]
    plt.figure(figsize=(12, 6))
    plt.hist(word_lengths, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Word Lengths')
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(img_dir, 'word_length_distribution.png'))

    # Plot word cloud for most frequent words
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(summary)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Most Frequent Words')
    plt.savefig(os.path.join(img_dir, 'word_cloud.png'))

    # Plot bar chart for top N most frequent words
    tokens = word_tokenize(summary)
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
    prediction = lt_predict(summary, loaded_vectorizer, svm_classifier)

    return word_length_img_path, word_cloud_img_path, top_10_words_img_path, prediction

lt_demo = gr.Interface(
    fn=visualize_data_lt,
    inputs=gr.components.Textbox(label='Input Web News'),
    outputs=[gr.Image(type="pil", label="Word Length Distribution"),
             gr.Image(type="pil", label="Word Cloud"),
             gr.Image(type="pil", label="Top 10 Words Bar Chart"),
             gr.components.Label(label="Text Predictions")],
    allow_flagging='never'
)

def download_df(file: pd.DataFrame, predictions: pd.DataFrame):
    # Combine the original text DataFrame (file) with the predictions DataFrame
    result_df = pd.concat([file, predictions], axis=1)
    
    download_path = os.path.join(root_path, "lt_predicted_combined.csv")
    result_df.to_csv(download_path)
    print(f"Combined Predictions Downloaded to: {download_path}")

# Gradio Interface for file upload and predictions
with gr.Blocks(css="#warning {background-color: red} .feedback {font-size: 74px} .gradio-container {max-width: none;") as file_lt_demo:
    with gr.Row():
        with gr.Column():
            lt_df = gr.components.DataFrame(label="Web News")
            upload_button = gr.UploadButton("Click to Upload a File", file_types=["csv"])
            run_button = gr.Button("Run")
        with gr.Column():
            # Replace file_out with st_df
            lt_df_out = gr.DataFrame(visible=False)
            out = gr.components.Textbox(label="Prediction", type="text")
            download_button = gr.Button("Download")

     # Adjust the lambda functions to call actual functions
    upload_button.upload(lambda file_path: pd.read_csv(file_path), inputs=upload_button, outputs=lt_df)
    run_button.click(lambda file_df: lt_predict_file(file_df, loaded_vectorizer, svm_classifier), inputs=lt_df, outputs=out)
    download_button.click(lambda file_df: download_df(file_df, global_predictions), inputs=lt_df)


# file_lt_demo = gr.Interface(
#     fn = predict_file,
#     inputs = gr.components.File(label="Long Text"),
#     outputs = gr.components.Textbox(label="answer", type="text"),
#     allow_flagging='never'
# 
# )
