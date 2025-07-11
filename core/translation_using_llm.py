import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

from llama_cpp import Llama

SUPPORTED_LANGS = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "hi": "Hindi",
    "es": "Spanish"
}

TOP_K = 5
CACHE_DIR = "rag_cache"
MODEL_PATH = "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    use_mlock=True,
    n_gpu_layers=30  # set to 0 if no GPU available
)

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

index_cache = {}
data_cache = {}


def _get_cache_paths(src_lang, tgt_lang):
    key = f"{src_lang}-{tgt_lang}"
    os.makedirs(CACHE_DIR, exist_ok=True)
    return (
        os.path.join(CACHE_DIR, f"{key}.faiss"),
        os.path.join(CACHE_DIR, f"{key}.pkl")
    )


def load_index(src_lang: str, tgt_lang: str):
    key = f"{src_lang}-{tgt_lang}"
    if key in index_cache:
        return index_cache[key], data_cache[key]

    faiss_path, data_path = _get_cache_paths(src_lang, tgt_lang)

    if os.path.exists(faiss_path) and os.path.exists(data_path):
        index = faiss.read_index(faiss_path)
        with open(data_path, "rb") as f:
            source_sents, target_sents = pickle.load(f)
    else:
        try:
            dataset = load_dataset("Helsinki-NLP/opus-100", f"{src_lang}-{tgt_lang}", split="train[:1000]")
            source_key, target_key = src_lang, tgt_lang
        except:
            dataset = load_dataset("Helsinki-NLP/opus-100", f"{tgt_lang}-{src_lang}", split="train[:1000]")
            source_key, target_key = tgt_lang, src_lang

        source_sents = [ex["translation"][source_key] for ex in dataset]
        target_sents = [ex["translation"][target_key] for ex in dataset]

        embeddings = embedder.encode(source_sents, convert_to_numpy=True, show_progress_bar=False)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, faiss_path)
        with open(data_path, "wb") as f:
            pickle.dump((source_sents, target_sents), f)

    index_cache[key] = index
    data_cache[key] = (source_sents, target_sents)
    return index, (source_sents, target_sents)


def format_prompt(query, examples, src, tgt):
    prompt = (f"You are a translation assistant. Translate from {SUPPORTED_LANGS[src]} "
              f"to {SUPPORTED_LANGS[tgt]}.\n\n### Retrieved Examples:\n")
    for i, (s, t) in enumerate(examples, 1):
        prompt += f"EX {i}: {s} → {t}\n"
    prompt += f"\nNow translate the following:\n{query} →"
    return prompt


def compute_bleu(ref, hyp):
    ref_tok = word_tokenize(ref.lower())
    hyp_tok = word_tokenize(hyp.lower())
    smooth = SmoothingFunction().method1
    return round(sentence_bleu(ref_tok, hyp_tok, smoothing_function=smooth) * 100, 2)


def translate_with_retrieval(query_text: str, src_lang: str, tgt_lang: str, show_bleu=False):
    if src_lang == tgt_lang:
        return "Source and target languages must be different.", None if show_bleu else "Source and target languages must be different."

    index, (source_sents, target_sents) = load_index(src_lang, tgt_lang)
    query_embedding = embedder.encode([query_text])
    _, indices = index.search(query_embedding, TOP_K)

    examples = [(source_sents[i], target_sents[i]) for i in indices[0]]
    prompt = format_prompt(query_text, examples, src_lang, tgt_lang)

    print(prompt)

    output = llm(prompt, max_tokens=100, temperature=0.7, stop=["\n"])
    hypothesis = output["choices"][0]["text"].strip()

    if show_bleu:
        reference = target_sents[indices[0][0]]
        bleu = compute_bleu(reference, hypothesis)
        return hypothesis, bleu
    return hypothesis


def prebuild_all_indexes():
    print("Prebuilding FAISS indexes...")
    for src in SUPPORTED_LANGS:
        for tgt in SUPPORTED_LANGS:
            if src != tgt:
                try:
                    load_index(src, tgt)
                    print(f"Cached: {src} → {tgt}")
                except Exception as e:
                    print(f"Failed {src} → {tgt}: {e}")
    print("Prebuild complete.")
