from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
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
import io
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE

root_path = os.path.dirname(os.path.abspath(__file__))
label_mapping = {0: 'Not Related', 1: 'Traffic Incident', 2: 'Traffic Info'}

label_mapping = {1: 'Falls,Slips,Trips', 2: 'Expose to Harmful Substance', 3: 'Contact with objects/equipments'}

# Load the saved SVM classifier
st_svm_classifier = joblib.load('ShortText/st_svm_model.h5')
# Load the saved word2vec vectorizer
st_loaded_vectorizer = joblib.load('ShortText/word2vec_model.model')

# Load the saved SVM classifier
lt_svm_classifier = joblib.load('LongText/lt_svm_model.h5')
# Load the saved word2vec vectorizer
lt_loaded_vectorizer = joblib.load('LongText/word2vec_model.model')

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


def st_predict(tweet, st_loaded_vectorizer, st_svm_classifier):
    tokenized_tweet = [word.lower() for word in tweet.split()]
    vectorized_input_data = [average_word_vectors(tokenized_tweet, st_loaded_vectorizer)]
    predictions = st_svm_classifier.predict(vectorized_input_data)
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

def lt_predict(summary, lt_loaded_vectorizer, lt_svm_classifier):
    tokenized_summary = [word.lower() for word in summary.split()]
    vectorized_input_data = [average_word_vectors(tokenized_summary, lt_loaded_vectorizer)]
    predictions = lt_svm_classifier.predict(vectorized_input_data)
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
global_predictions_st = None
global_predictions_lt = None

def st_predict_file(df: pd.DataFrame, st_loaded_vectorizer, st_svm_classifier):
    global global_predictions_st
    global_predictions_st = df['tweet'].apply(lambda tweet: st_predict(tweet, st_loaded_vectorizer, st_svm_classifier))
    return global_predictions_st

def lt_predict_file(df: pd.DataFrame, lt_loaded_vectorizer, lt_svm_classifier):
    global global_predictions_lt
    global_predictions_lt = df['summary'].apply(lambda summary: lt_predict(summary, lt_loaded_vectorizer, lt_svm_classifier))
    return global_predictions_lt

def visualize_data_st(tweet):
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
    prediction = st_predict(tweet, st_loaded_vectorizer, st_svm_classifier)

    return word_length_img_path, word_cloud_img_path, top_10_words_img_path, prediction

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
    prediction = lt_predict(summary, lt_loaded_vectorizer, lt_svm_classifier)

    return word_length_img_path, word_cloud_img_path, top_10_words_img_path, prediction

# Update your Gradio interface
st_demo_main = gr.Interface(
    fn=visualize_data_st,
    inputs=gr.components.Textbox(label='Input short text'),
    outputs=[gr.Image(type="pil", label="Word Length Distribution"),
             gr.Image(type="pil", label="Word Cloud"),
             gr.Image(type="pil", label="Top 10 Words Bar Chart"),
             gr.components.Label(label="Text Predictions")],
    allow_flagging='never',    
)

lt_demo_main = gr.Interface(
    fn=visualize_data_lt,
    inputs=gr.components.Textbox(label='Input long text'),
    outputs=[gr.Image(type="pil", label="Word Length Distribution"),
             gr.Image(type="pil", label="Word Cloud"),
             gr.Image(type="pil", label="Top 10 Words Bar Chart"),
             gr.components.Label(label="Text Predictions")],
    allow_flagging='never'
)

# upload = gr.UploadButton("Click to Upload a File", file_types=["file"])
# inp_file=gr.components.File(label="Short Text")

def st_download_df(file: pd.DataFrame, predictions: pd.DataFrame):
    # Combine the original text DataFrame (file) with the predictions DataFrame
    result_df = pd.concat([file, predictions], axis=1)
    
    download_path = os.path.join(root_path, "st_predicted_combined.csv")
    result_df.to_csv(download_path)
    print(f"Combined Predictions Downloaded to: {download_path}")


def lt_download_df(file: pd.DataFrame, predictions: pd.DataFrame):
    # Combine the original text DataFrame (file) with the predictions DataFrame
    result_df = pd.concat([file, predictions], axis=1)
    
    download_path = os.path.join(root_path, "lt_predicted_combined.csv")
    result_df.to_csv(download_path)
    print(f"Combined Predictions Downloaded to: {download_path}")


# Define separate folders for short text and long text interfaces
short_text_folder = "ShortText_Figures"
long_text_folder = "LongText_Figures"

# Ensure the folders exist, create them if not
os.makedirs(short_text_folder, exist_ok=True)
os.makedirs(long_text_folder, exist_ok=True)

# Gradio Interface for file upload and predictions
with gr.Blocks() as file_main_demo:
    # Short Text Interface
    with gr.Tab(label="Tweet News"):
        with gr.Accordion("Model Analysis"):
            gr.Markdown("Figures")

            with gr.Column():
                df = pd.read_csv("Data1/pre_arr_dt_st.csv")
                mlb = MultiLabelBinarizer()
                labels = mlb.fit_transform(df['relation'].apply(lambda x: [x]))
                df_encoded = pd.concat([df['tweet'], pd.DataFrame(labels, columns=mlb.classes_)], axis=1)
                train_data, test_data = train_test_split(df_encoded, test_size=0.2, random_state=42)
                X_train = train_data['tweet']
                y_train = np.argmax(train_data.drop('tweet', axis=1).values, axis=1)
                X_test = test_data['tweet']
                y_test = np.argmax(test_data.drop('tweet', axis=1).values, axis=1)

                tokenized_tweet = [word_tokenize(tweet.lower()) for tweet in X_train]
                train_vectors = [average_word_vectors(tweet, st_loaded_vectorizer) for tweet in tokenized_tweet]
                X_train_word2vec = np.vstack(train_vectors)

                svm_model = SVC(kernel='linear')
                svm_model.fit(X_train_word2vec, y_train)

                tokenized_test_tweet = [word_tokenize(tweet.lower()) for tweet in X_test]
                test_vectors = [average_word_vectors(tweet, st_loaded_vectorizer) for tweet in tokenized_test_tweet]
                X_test_word2vec = np.vstack(test_vectors)

                y_pred_test = svm_model.predict(X_test_word2vec)

                accuracy_test1 = accuracy_score(y_test, y_pred_test)
                print(f"Round Test Accuracy: {accuracy_test1}")
                sns.countplot(x='relation', data=df, hue='relation')
                plt.title('Class Distribution')
                plt.savefig(os.path.join(short_text_folder, "class.png"))
                plt.close()

                cm = confusion_matrix(y_test, y_pred_test)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                plt.savefig(os.path.join(short_text_folder, "cm.png"))
                plt.close()

                all_tweets_text = ' '.join(df['tweet'])
                wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(all_tweets_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis('off')
                plt.title('Word Cloud')
                plt.savefig(os.path.join(short_text_folder, "wordcloud.png"))
                plt.close()

                tweet_lengths = df['tweet'].apply(len)
                plt.figure(figsize=(10, 5))
                plt.hist(tweet_lengths, bins=50, color='skyblue', edgecolor='black')
                plt.title('Distribution of Tweet Lengths')
                plt.xlabel('Tweet Length')
                plt.ylabel('Frequency')
                plt.savefig(os.path.join(short_text_folder, "wordlength.png"))
                plt.close()

                word2vec_model = gensim.models.Word2Vec.load('ShortText/word2vec_model.model')
                words = list(word2vec_model.wv.key_to_index.keys())
                vectors = [word2vec_model.wv[word] for word in words]

                top_words = 100
                words = words[:top_words]
                vectors = vectors[:top_words]
                vectors = np.array(vectors)

                tsne = TSNE(n_components=2, random_state=42)
                vectors_tsne = tsne.fit_transform(vectors)

                plt.figure(figsize=(10, 8))
                plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], marker='.')
                for i, word in enumerate(words):
                    plt.annotate(word, xy=(vectors_tsne[i, 0], vectors_tsne[i, 1]), fontsize=8)
                plt.title('t-SNE Visualization of Top 100 Word2Vec Embeddings')
                plt.savefig(os.path.join(short_text_folder, "TSNE.png"))
                plt.close()

                with gr.Blocks():
                    with gr.Row():
                        img1 = gr.Image("ShortText_Figures/cm.png")
                        img2 = gr.Image("ShortText_Figures/wordcloud.png")
                        img3 = gr.Image("ShortText_Figures/class.png")
                    with gr.Row():
                        img4 = gr.Image("ShortText_Figures/wordlength.png")
                        img5 = gr.Image("ShortText_Figures/TSNE.png")
        
        with gr.Row():
            gr.Text(f"Accuracy on Test Data: {accuracy_test1:.2%}")


    # Long Text Interface
    with gr.Tab(label="Web News"):
        with gr.Accordion("Model Analysis"):
            gr.Markdown("Figures")

            with gr.Column():
                df = pd.read_csv("Data2/pre_arr_dt_lt.csv")
                mlb = MultiLabelBinarizer()
                labels = mlb.fit_transform(df['tag'].apply(lambda x: [x]))
                df_encoded = pd.concat([df['summary'], pd.DataFrame(labels, columns=mlb.classes_)], axis=1)
                train_data, test_data = train_test_split(df_encoded, test_size=0.2, random_state=42)
                X_train = train_data['summary']
                y_train = np.argmax(train_data.drop('summary', axis=1).values, axis=1)
                X_test = test_data['summary']
                y_test = np.argmax(test_data.drop('summary', axis=1).values, axis=1)

                tokenized_summary = [word_tokenize(summary.lower()) for summary in X_train]
                train_vectors = [average_word_vectors(summary, lt_loaded_vectorizer) for summary in tokenized_summary]
                X_train_word2vec = np.vstack(train_vectors)

                svm_model = SVC(kernel='linear')
                svm_model.fit(X_train_word2vec, y_train)

                tokenized_test_summary = [word_tokenize(summary.lower()) for summary in X_test]
                test_vectors = [average_word_vectors(summary, lt_loaded_vectorizer) for summary in tokenized_test_summary]
                X_test_word2vec = np.vstack(test_vectors)

                y_pred_test = svm_model.predict(X_test_word2vec)

                accuracy_test2 = accuracy_score(y_test, y_pred_test)
                print(f"Round Test Accuracy: {accuracy_test2}")
                
                sns.countplot(x='tag', data=df, hue='tag')
                plt.title('Class Distribution')
                plt.savefig(os.path.join(long_text_folder, "class.png"))
                plt.close()

                cm = confusion_matrix(y_test, y_pred_test)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                plt.savefig(os.path.join(long_text_folder, "cm.png"))
                plt.close()

                all_tweets_text = ' '.join(df['summary'])
                wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(all_tweets_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis('off')
                plt.title('Word Cloud')
                plt.savefig(os.path.join(long_text_folder, "wordcloud.png"))
                plt.close()

                summary_lengths = df['summary'].apply(len)
                plt.figure(figsize=(10, 5))
                plt.hist(summary_lengths, bins=50, color='skyblue', edgecolor='black')
                plt.title('Distribution of Summary Lengths')
                plt.xlabel('Summary Length')
                plt.ylabel('Frequency')
                plt.savefig(os.path.join(long_text_folder, "wordlength.png"))
                plt.close()

                word2vec_model = gensim.models.Word2Vec.load('LongText/word2vec_model.model')
                words = list(word2vec_model.wv.key_to_index.keys())
                vectors = [word2vec_model.wv[word] for word in words]

                top_words = 100
                words = words[:top_words]
                vectors = vectors[:top_words]
                vectors = np.array(vectors)

                tsne = TSNE(n_components=2, random_state=42)
                vectors_tsne = tsne.fit_transform(vectors)

                plt.figure(figsize=(10, 8))
                plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], marker='.')
                for i, word in enumerate(words):
                    plt.annotate(word, xy=(vectors_tsne[i, 0], vectors_tsne[i, 1]), fontsize=8)
                plt.title('t-SNE Visualization of Top 100 Word2Vec Embeddings')
                plt.savefig(os.path.join(long_text_folder, "TSNE.png"))
                plt.close()

                with gr.Blocks():
                    with gr.Row():
                        img1 = gr.Image("LongText_Figures/cm.png")
                        img2 = gr.Image("LongText_Figures/wordcloud.png")
                        img3 = gr.Image("LongText_Figures/class.png")
                    with gr.Row():
                        img4 = gr.Image("LongText_Figures/wordlength.png")
                        img5 = gr.Image("LongText_Figures/TSNE.png")

        with gr.Row():
            gr.Text(f"Accuracy on Test Data: {accuracy_test2:.2%}")

