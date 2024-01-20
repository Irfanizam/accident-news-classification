from fastapi import FastAPI
import gradio as gr
from st_classification_ui import st_demo, file_st_demo
from lt_classification_ui import lt_demo, file_lt_demo  
from main_ui import file_main_demo
import numpy as np

app = FastAPI()

demo_tabbed = gr.TabbedInterface(
    [file_main_demo],
    ["Model Comparison"],
    title="Twitter News and Web News Classification System"
)

implementation_tabbed = gr.TabbedInterface(
    [st_demo, file_st_demo, lt_demo, file_lt_demo],
    ["Twitter News", "Twitter News File", "Web News", "Web News File"],
    title="Twitter News and Web News Classification System"
)

fullapp = gr.TabbedInterface([demo_tabbed, implementation_tabbed], ["Model Demonstration", "Implementation"])

# @app.get('/')
# def home():
#     return 'Gradio is running on /gradio', 200

# app = gr.mount_gradio_app(app, fullapp, '/gradio')

fullapp.launch(share=True)