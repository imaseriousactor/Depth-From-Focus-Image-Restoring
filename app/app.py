import gradio as gr
import torch
from models.attention_unet import AttentionUNet

def predict(image):
    return image

demo = gr.Interface(fn=predict, inputs="image", outputs="image")
demo.launch()
