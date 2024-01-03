# Gradio Interface
# demo_func, demo, can be change to other name 
import gradio as gr
import joblib
from IPython import embed
import pickle
import pandas as pd
import os
import sklearn
from collections import Counter
import spacy
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.svm import SVC



def handle_clear1() -> tuple:
    return '', None, '', None

def handle_clear2() -> tuple:
    return '', None, '', None

def handle_submit1(file_path: str, text: str) -> tuple:
    print(file_path)
    print(text)
    return file_path, text

def handle_submit2(file_path: str, text: str) -> tuple:
    print(file_path)
    print(text)
    return file_path, text


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_shorttext = gr.components.Textbox(label = 'Input short text')
            input_shortimage = gr.components.Image(label = 'Input short image', type = 'filepath')
            with gr.Row():
                submit1 = gr.components.Button(value = 'Submit', variant = 'primary')
                clear1 = gr.components.Button(value = 'Clear', variant = 'stop')
        with gr.Column():
            output_shorttext = gr.components.Textbox(label = 'Output short text')
            output_shortimage = gr.components.Image(label = 'Output short image', type = 'filepath')

        with gr.Row():
            input_longtext = gr.components.Textbox(label = 'Input long text')
            input_longimage = gr.components.Image(label = 'Input long image', type = 'filepath')
            with gr.Row():
                submit2 = gr.components.Button(value = 'Submit', variant = 'primary')
                clear2 = gr.components.Button(value = 'Clear', variant = 'stop')
        with gr.Column():
            output_longtext = gr.components.Textbox(label = 'Output long text')
            output_longimage = gr.components.Image(label = 'Output long image', type = 'filepath')

    clear1.click(handle_clear1, [], [input_shorttext, input_shortimage, output_shorttext, output_shortimage])
    submit1.click(handle_submit1, [input_shortimage, input_shorttext], [output_shortimage, output_shorttext])

    clear2.click(handle_clear2, [], [input_longtext, input_longimage, output_longtext, output_longimage])
    submit2.click(handle_submit2, [input_longimage, input_longtext], [output_longimage, output_longtext])


