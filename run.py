from fastapi import FastAPI
import gradio as gr

# from gradio_layouts import demo
from st_classification_ui import st_demo, file_st_demo
from lt_classification_ui import lt_demo, file_lt_demo

fullapp = gr.TabbedInterface([st_demo, file_st_demo, lt_demo, file_lt_demo ], ["Short Text", "ST File","Long Text", "LT File"])

app = FastAPI()

@app.get('/')
def home():
    return 'Gradio is running on /gradio', 200

app = gr.mount_gradio_app(app, fullapp, '/gradio')



# st_demo, lt_demo
# uvicorn run:app -- reload
