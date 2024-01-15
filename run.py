from fastapi import FastAPI
import gradio as gr
from st_classification_ui import st_demo, file_st_demo
from lt_classification_ui import lt_demo, file_lt_demo  
from main_ui import st_demo_main, file_main_demo, lt_demo_main
import numpy as np
app = FastAPI()

# with gr.Blocks:
#     gr.Markdown("Main Menu")
#     with gr.Tab("Tweet News"):
#         st_demo
#     with gr.Tab("Web News"):
#         file_st_demo
#     with gr.Accordion("Open for More!"):
#         gr.Markdown("Look at me...")

demo_tabbed = gr.TabbedInterface(
    [file_main_demo],
    ["Exploratory Data Analysis"],
    title="News Classification System")

# Create the main Gradio interface
fullapp = gr.TabbedInterface([demo_tabbed, st_demo, file_st_demo, lt_demo, file_lt_demo], ["Main Page", "Twitter News", "Twitter News File", "Web News", "Web News File"])

@app.get('/')
def home():
    return 'Gradio is running on /gradio', 200

app = gr.mount_gradio_app(app, fullapp, '/gradio')

# Run the FastAPI app
# uvicorn run:app --reload
