import argparse
import gc
import os
from pathlib import Path

import gradio as gr
from app_modules.overwrites import postprocess
from app_modules.presets import description, small_and_beautiful_theme, title
from app_modules.utils import cancel_outputing, delete_last_conversation, reset_state, reset_textbox, transfer_input
from interface.hddr_llm_onnx_dml_interface import LLMOnnxDmlInterface

top_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
optimized_directory = os.path.join(top_directory, "models", "optimized")
available_models = {}
interface = None
binding_device = "dml"


def change_model_listener(new_model_name):
    global interface

    # if a model exists - shut it down before trying to create the new one
    if interface is not None:
        interface.shutdown()
        del interface
        gc.collect()

    d = available_models[new_model_name]
    interface = LLMOnnxDmlInterface(
        model_dir=d["model_dir"],
        device=binding_device,
    )
    interface.initialize()

    return [
        new_model_name,
        gr.update(visible="llava" in new_model_name),
        [],
        [],
        gr.update(value=""),
        "",
    ]


def change_image_visibility(new_model_name):
    if "llava" in new_model_name:
        return gr.update(visible=True)

    return gr.update(visible=False)


gr.Chatbot.postprocess = postprocess

with Path("chat_app/assets/custom.css").open() as f:
    custom_css = f.read()


def interface_predict(*args):
    res = interface.predict(*args)
    yield from res


def interface_retry(*args):
    res = interface.retry(*args)
    yield from res


def launch_chat_app(expose_locally: bool = False, device: str = "dml"):
    global binding_device
    binding_device = device

    # ruff: noqa: PTH208
    for model_name in os.listdir(optimized_directory):
        available_models[model_name] = {"model_dir": os.path.join(optimized_directory, model_name)}

    with gr.Blocks(css=custom_css, theme=small_and_beautiful_theme) as demo:
        history = gr.State([])
        user_question = gr.State("")
        with gr.Row():
            gr.HTML(title)
            status_display = gr.Markdown("Success", elem_id="status_display")

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row():
                    chatbot = gr.Chatbot(elem_id="chuanhu_chatbot", height=900)
                with gr.Row():
                    with gr.Column(scale=12):
                        user_input = gr.Textbox(show_label=False, placeholder="Enter text")
                    with gr.Column(min_width=70, scale=1):
                        submit_button = gr.Button("Send")
                    with gr.Column(min_width=70, scale=1):
                        cancel_button = gr.Button("Stop")
                with gr.Row():
                    empty_button = gr.Button(
                        "🧹 New Conversation",
                    )
                    retry_button = gr.Button("🔄 Regenerate")
                    delete_last_button = gr.Button("🗑️ Remove Last Turn")
            reset_args = {"fn": reset_textbox, "inputs": [], "outputs": [user_input, status_display]}
            with gr.Column(), gr.Column(min_width=50, scale=1), gr.Tab(label="Parameter Setting"):
                gr.Markdown("# Model")
                model_name = gr.Dropdown(
                    choices=list(available_models.keys()),
                    label="Model",
                    show_label=False,  # default="Empty STUB",
                    value=next(iter(available_models.keys())),
                )
                max_length_tokens = gr.Slider(
                    minimum=0,
                    maximum=4096,
                    value=256,
                    step=8,
                    interactive=True,
                    label="Max Generation Tokens",
                )
                max_context_length_tokens = gr.Slider(
                    minimum=0,
                    maximum=4096,
                    value=2048,
                    step=128,
                    interactive=True,
                    label="Max History Tokens",
                )
                token_printing_step = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=4,
                    step=1,
                    interactive=True,
                    label="Token Printing Step",
                )

                image = gr.Image(type="pil", visible=False)
                image.change(
                    reset_state,
                    outputs=[chatbot, history, status_display],
                    show_progress=True,
                )
                image.change(**reset_args)

                model_name.change(
                    change_model_listener,
                    inputs=[model_name],
                    outputs=[model_name, image, chatbot, history, user_input, status_display],
                )
        gr.Markdown(description)

        predict_args = {
            "fn": interface_predict,
            "inputs": [
                user_question,
                chatbot,
                history,
                max_length_tokens,
                max_context_length_tokens,
                token_printing_step,
                image,
            ],
            "outputs": [chatbot, history, status_display],
            "show_progress": True,
        }
        retry_args = {
            "fn": interface_retry,
            "inputs": [
                chatbot,
                history,
                max_length_tokens,
                max_context_length_tokens,
                token_printing_step,
            ],
            "outputs": [chatbot, history, status_display],
            "show_progress": True,
        }

        # Chatbot
        transfer_input_args = {
            "fn": transfer_input,
            "inputs": [user_input],
            "outputs": [user_question, user_input, submit_button],
            "show_progress": True,
        }

        predict_event1 = user_input.submit(**transfer_input_args).then(**predict_args)

        predict_event2 = submit_button.click(**transfer_input_args).then(**predict_args)

        empty_button.click(
            reset_state,
            outputs=[chatbot, history, status_display],
            show_progress=True,
        )
        empty_button.click(**reset_args)

        predict_event3 = retry_button.click(**retry_args)

        delete_last_button.click(
            delete_last_conversation,
            [chatbot, history],
            [chatbot, history, status_display],
            show_progress=True,
        )
        cancel_button.click(
            cancel_outputing,
            [],
            [status_display],
            cancels=[predict_event1, predict_event2, predict_event3],
        )

        demo.load(change_model_listener, inputs=[model_name], outputs=[model_name, image])

    demo.title = "LLM Chat UI"

    if expose_locally:
        demo.queue(concurrency_count=1).launch(server_name="0.0.0.0", server_port=7860)
    else:
        demo.queue(concurrency_count=1).launch(server_port=7860)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expose_locally", action="store_true")
    parser.add_argument("--device", type=str, choices=["dml", "cuda"], default="dml")
    args = parser.parse_args()
    launch_chat_app(args.expose_locally, args.device)
