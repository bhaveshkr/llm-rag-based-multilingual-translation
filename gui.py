import os
import shutil
import gradio as gr
from core.translation_using_llm import (
    translate_with_retrieval,
    SUPPORTED_LANGS,
    CACHE_DIR,
    prebuild_all_indexes
)

language_choices = list(SUPPORTED_LANGS.keys())


def clear_cache():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)
    return "✅ Cache cleared. Rebuilding indexes..."


def interface(text, src_lang, tgt_lang, show_bleu):
    result = translate_with_retrieval(text, src_lang, tgt_lang, show_bleu)
    if show_bleu:
        translation, bleu = result
        return translation, f"{bleu} BLEU"
    else:
        return result, ""


with gr.Blocks() as demo:
    gr.Markdown("## LLM & RAG-based Multilingual Translator")

    with gr.Row():
        text_input = gr.Textbox(label="Enter Text", lines=3)
    with gr.Row():
        src_lang = gr.Dropdown(language_choices, label="Source Language", value="en")
        tgt_lang = gr.Dropdown(language_choices, label="Target Language", value="fr")
    show_bleu = gr.Checkbox(label="Show BLEU Score", value=False)

    with gr.Row():
        translate_btn = gr.Button("Translate")
        clear_btn = gr.Button("Clear Cache")

    output_translation = gr.Textbox(label="Translation")
    output_bleu = gr.Textbox(label="BLEU Score")

    translate_btn.click(
        fn=interface,
        inputs=[text_input, src_lang, tgt_lang, show_bleu],
        outputs=[output_translation, output_bleu]
    )
    clear_btn.click(fn=clear_cache, inputs=[], outputs=output_bleu)

# Prebuild all indexes at startup
prebuild_all_indexes()
demo.launch()
