from fastapi import FastAPI
from pydantic import BaseModel
from core.translation_using_llm import translate_with_retrieval, prebuild_all_indexes

app = FastAPI()

# Prebuild all indexes at startup
prebuild_all_indexes()


class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str


@app.post("/translate/")
async def translate(req: TranslationRequest):
    result = translate_with_retrieval(req.text, req.source_lang, req.target_lang)
    return {"translation": result}
