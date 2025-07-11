# LLM & RAG-Based Multilingual Translation

This project provides an end-to-end **Large Language Model (LLM)** and **Retrieval-Augmented Generation (RAG)**-based machine translation system using Hugging Face models, FAISS for similarity search, and Gradio for a user-friendly interface. All components run locally.

Uses Hugging Face model: 
1. mistral-7b-instruct.Q4_K_M.gguf

    Download and copy to model directory. 
    https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
    

2. facebook/nllb-200-distilled-600M



Example dataset: Helsinki-NLP/opus-100

---

## Features

-  Supports multiple languages (en, de, fr, hi, es)
-  Bidirectional translation (e.g., en → fr, fr → en)
-  FAISS-based retrieval from translation examples
-  Index caching to disk for fast reloads
-  Optional BLEU score evaluation
-  Cache clearing button in GUI

---

## Requirements

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Gradio GUI

To start the web interface:
```bash
python gui.py
```

Then open http://127.0.0.1:7860 in your browser.

## Start the API server
```bash
uvicorn api:app --reload
```
It will default to http://127.0.0.1:8000

Curl 
```bash
curl -X POST http://127.0.0.1:8000/translate/ \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Wie geht es dir?",
        "source_lang": "de",
        "target_lang": "en"
      }'
```


## You can call the translation logic from any Python script:
```python
from core.translation_using_llm import translate_with_retrieval

translation = translate_with_retrieval("Wie geht es dir?", "de", "en")
print(translation)
```

To get BLEU score:
```python
result, bleu = translate_with_retrieval("Wie geht es dir?", "de", "en", show_bleu=True)
print("Translation:", result)
print("BLEU Score:", bleu)
```
