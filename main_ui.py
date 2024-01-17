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
from collections import Counter

root_path = os.path.dirname(os.path.abspath(__file__))
# short_text_folder = "ShortText_Figures"
long_text_folder = "LongText_Figures"

# os.makedirs(short_text_folder, exist_ok=True)
os.makedirs(long_text_folder, exist_ok=True)

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

def process_and_visualize_data_before_st(df):
    # Display label counts
    img_dir = os.path.expanduser('~/visualization_images_main')
    os.makedirs(img_dir, exist_ok=True)

    label_mapping = {0: '0-Not Related', 1: '1-Traffic Incident', 2: '2-Traffic Information'}
    label_counts = df['relation'].value_counts().rename(index=label_mapping)
    formatted_output_st = '\n'.join([f"{label}: {count}" for label, count in label_counts.items()])

    text_column = 'tweet'
    all_text = ' '.join(df[text_column].astype(str))
    words = word_tokenize(all_text)
    word_counts = Counter(words)

    # Find the word with the maximum count
    max_word_st, max_count_st = max(word_counts.items(), key=lambda x: x[1])

    # Find the word with the minimum count
    min_word_st, min_count_st = min(word_counts.items(), key=lambda x: x[1])

    # Display word cloud
    all_tweets_text = ' '.join(df['tweet'])
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(all_tweets_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Word Cloud')
    plt.savefig(os.path.join(img_dir, "wordcloud_main_st.png"))
    plt.close()

    # Display tweet length distribution
    tweet_lengths = df['tweet'].apply(len)
    plt.figure(figsize=(10, 5))
    plt.hist(tweet_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Tweet Lengths')
    plt.xlabel('Tweet Length')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(img_dir, "wordlength_main_st.png"))
    plt.close()

    # Display class distribution
    sns.countplot(x='relation', data=df, hue='relation')
    plt.title('Class Distribution')
    plt.savefig(os.path.join(img_dir, "class_main_st.png"))
    plt.close()

    word_length_img_path_main_st = os.path.join(img_dir, 'wordlength_main_st.png')
    word_cloud_img_path_main_st = os.path.join(img_dir, 'wordcloud_main_st.png')
    class_path_main_st = os.path.join(img_dir, 'class_main_st.png')

    return formatted_output_st, max_word_st, max_count_st, min_word_st, min_count_st, word_length_img_path_main_st, word_cloud_img_path_main_st, class_path_main_st

def process_and_visualize_data_before_lt(df):
    # Display label counts
    img_dir = os.path.expanduser('~/visualization_images_main')
    os.makedirs(img_dir, exist_ok=True)

    label_mapping = {1: '1-Falls,Slips,Trips', 2: '2-Expose to Harmful Substance', 3: '3-Contact with objects/equipments'}
    label_counts = df['tag'].value_counts().rename(index=label_mapping)
    formatted_output_lt = '\n'.join([f"{label}: {count}" for label, count in label_counts.items()])

    text_column = 'summary'
    all_text = ' '.join(df[text_column].astype(str))
    words = word_tokenize(all_text)
    word_counts = Counter(words)

    # Find the word with the maximum count
    max_word_lt, max_count_lt = max(word_counts.items(), key=lambda x: x[1])

    # Find the word with the minimum count
    min_word_lt, min_count_lt = min(word_counts.items(), key=lambda x: x[1])

    # Display word cloud
    all_summary_text = ' '.join(df['summary'])
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(all_summary_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Word Cloud')
    plt.savefig(os.path.join(img_dir, "wordcloud_main_lt.png"))
    plt.close()

    # Display summary length distribution
    tweet_lengths = df['summary'].apply(len)
    plt.figure(figsize=(10, 5))
    plt.hist(tweet_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Web News Lengths')
    plt.xlabel('Summary Length')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(img_dir, "wordlength_main_lt.png"))
    plt.close()

    # Display class distribution
    sns.countplot(x='tag', data=df, hue='tag')
    plt.title('Class Distribution')
    plt.savefig(os.path.join(img_dir, "class_main_lt.png"))
    plt.close()

    word_length_img_path_main_lt = os.path.join(img_dir, 'wordlength_main_lt.png')
    word_cloud_img_path_main_lt = os.path.join(img_dir, 'wordcloud_main_lt.png')
    class_path_main_lt = os.path.join(img_dir, 'class_main_lt.png')

    return formatted_output_lt, max_word_lt, max_count_lt, min_word_lt, min_count_lt, word_length_img_path_main_lt, word_cloud_img_path_main_lt, class_path_main_lt

def st_train_and_evaluate_model(df):
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

    accuracy_test = accuracy_score(y_test, y_pred_test)
    from sklearn.metrics import precision_recall_fscore_support
    # Assuming y_test and y_pred_test are your true labels and predicted labels, respectively
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')
    
    return {
        'f1_score': f1_score,
        'accuracy': accuracy_test,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'test_data': test_data  # Add this line to return the test_data
    }

def lt_train_and_evaluate_model(df):
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

    accuracy_test = accuracy_score(y_test, y_pred_test)
    from sklearn.metrics import precision_recall_fscore_support
    # Assuming y_test and y_pred_test are your true labels and predicted labels, respectively
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')

    return {
        'f1_score': f1_score,
        'accuracy': accuracy_test,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'test_data': test_data  # Add this line to return the test_data
    }

with gr.Blocks() as file_main_demo:
    with gr.Tab(label="Twitter News"):
        with gr.Column():
            with gr.Accordion("Data Preload and Evaluation", open=False):
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            st_df = gr.components.DataFrame(label="Twitter News", height=200)
                        with gr.Column():
                            upload_button = gr.UploadButton("Upload Data", file_types=["csv"])
                            button_before = gr.Button("Process Data")
                with gr.Accordion("Analysis Before Training", open=False):
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
            with gr.Accordion("Model Training and Evaluation", open=False):
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            out1 = gr.components.Textbox(label="F1 Score", type="text")
                            out2 = gr.components.Textbox(label="Model Accuracy", type="text")
                        with gr.Column():
                            with gr.Column():
                                button_train = gr.Button("Train Model")
                                out_test_data = gr.components.DataFrame(label="Test Data", height=200)
                with gr.Accordion("Analysis After Training", open=False):
                    with gr.Row():
                        outim3 = gr.Image(label="Confusion Matrix")
                        outim4 = gr.Image(label="TSNE for Word2Vec Embedding")

    upload_button.upload(lambda file_path: pd.read_csv(file_path), inputs=upload_button, outputs=st_df)
    button_before.click(lambda df: process_and_visualize_data_before_st(df), inputs=st_df, outputs=[label_counts, max_word, max_count, min_word, min_count, out_img, out_img1, out_img2])
    button_train.click(lambda file_df: st_train_model(file_df), inputs=[st_df], outputs=[out1, out2, outim3, outim4, out_test_data])

    with gr.Tab(label="Web News"):
        with gr.Column():
            with gr.Accordion("Data Preload and Evaluation", open=False):
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            lt_df = gr.components.DataFrame(label="Web News", height=200)
                        with gr.Column():
                            upload_button = gr.UploadButton("Upload Data", file_types=["csv"])
                            button_before = gr.Button("Process Data")
                with gr.Accordion("Analysis Before Training", open=False):
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
            with gr.Accordion("Model Training and Evaluation", open=False):
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            out1 = gr.components.Textbox(label="F1 Score", type="text")
                            out2 = gr.components.Textbox(label="Model Accuracy", type="text")
                        with gr.Column():
                            with gr.Column():
                                button_train = gr.Button("Train Model")
                                out_test_data = gr.components.DataFrame(label="Test Data", height=200)
                with gr.Accordion("Analysis After Training", open=False):
                    with gr.Row():
                        outim3 = gr.Image(label="Confusion Matrix")
                        outim4 = gr.Image(label="TSNE for Word2Vec Embedding")

    upload_button.upload(lambda file_path: pd.read_csv(file_path), inputs=upload_button, outputs=lt_df)
    button_before.click(lambda df: process_and_visualize_data_before_lt(df), inputs=lt_df, outputs=[label_counts, max_word, max_count, min_word, min_count, out_img, out_img1, out_img2])
    button_train.click(lambda file_df: lt_train_model(file_df), inputs=[lt_df], outputs=[out1, out2, outim3, outim4, out_test_data])

def st_train_model(file_df):
    results = st_train_and_evaluate_model(file_df)
    f1_score = results['f1_score']
    accuracy_test = results['accuracy']
    y_test = results['y_test']
    y_pred_test = results['y_pred_test']
    test_data = results['test_data']

    acc_percent = round(accuracy_test, 4)
    acc_per = acc_percent * 100
    acc_per_str = str(acc_per) + "%"

    f1_percent = round(f1_score, 4)
    f1_per = f1_percent * 100
    f1_per_str = str(f1_per) + "%"


    img_dir = os.path.expanduser('~/visualization_images_main')
    os.makedirs(img_dir, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(img_dir, "cm_main_st.png"))
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
    plt.savefig(os.path.join(img_dir, "TSNE_main_st.png"))
    plt.close()

    cm_main_st_path = os.path.join(img_dir, 'cm_main_st.png')
    tsne_main_st_path = os.path.join(img_dir, 'TSNE_main_st.png')
    test_data.columns = ['tweet', '0-Not Related', '1-Traffic Incident', '2-Traffic Infos']

    return f1_per_str, acc_per_str, cm_main_st_path, tsne_main_st_path, test_data

def lt_train_model(file_df):
    results = lt_train_and_evaluate_model(file_df)
    f1_score = results['f1_score']
    accuracy_test = results['accuracy']
    y_test = results['y_test']
    y_pred_test = results['y_pred_test']
    test_data = results['test_data']

    acc_percent = round(accuracy_test, 4)
    acc_per = acc_percent * 100
    acc_per_str = str(acc_per) + "%"

    f1_percent = round(f1_score, 4)
    f1_per = f1_percent * 100
    f1_per_str = str(f1_per) + "%"


    img_dir = os.path.expanduser('~/visualization_images_main')
    os.makedirs(img_dir, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(img_dir, "cm_main_lt.png"))
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
    plt.savefig(os.path.join(img_dir, "TSNE_main_lt.png"))
    plt.close()

    cm_main_lt_path = os.path.join(img_dir, 'cm_main_lt.png')
    tsne_main_lt_path = os.path.join(img_dir, 'TSNE_main_lt.png')
    test_data.columns = ['summary', '1-Falls,Slips,Trips', '2-Expose to Harmful Substance', '3-Contact with objects/equipments']

    return f1_per_str, acc_per_str, cm_main_lt_path, tsne_main_lt_path, test_data

