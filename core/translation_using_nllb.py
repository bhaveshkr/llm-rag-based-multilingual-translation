import os
import faiss
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')


SUPPORTED_LANGS = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "hi": "Hindi",
    "es": "Spanish"
}

TOP_K = 5
CACHE_DIR = "rag_cache"

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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
        print(f"Loading cached index: {key}")
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


def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())
    smoothie = SmoothingFunction().method4
    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
    return round(score * 100, 2)


def translate_with_retrieval(query_text: str, src_lang: str, tgt_lang: str, show_bleu=False):
    if src_lang == tgt_lang:
        return "Source and target languages must be different.", None if show_bleu else "Source and target languages must be different."

    index, (source_sents, target_sents) = load_index(src_lang, tgt_lang)
    query_embedding = embedder.encode([query_text])
    _, indices = index.search(query_embedding, TOP_K)

    examples = [f"{source_sents[i]} → {target_sents[i]}" for i in indices[0]]
    prompt = "\n".join(examples) + f"\n\n{query_text} →"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=128)
    hypothesis = tokenizer.decode(outputs[0], skip_special_tokens=True)

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
