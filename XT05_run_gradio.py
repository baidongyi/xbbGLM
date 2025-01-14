
import gradio as gr

from XT04_my_run import chat


if __name__ == "__main__":

    demo = gr.ChatInterface(chat, multimodal=False)
    demo.launch(inbrowser=True, server_name="0.0.0.0", server_port=7860, show_error=True)
