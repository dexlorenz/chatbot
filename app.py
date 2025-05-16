# app.py (FastAPI versiyonu - InMemoryCache ve RAG On/Off Testi ile)

import os
import time
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

import config
from fixed_responses_module import get_fixed_response
from llm_setup import initialize_rag_components
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(
    title="Modaselvim Chatbot API",
    description="Modaselvim için RAG tabanlı chatbot backend servisi.",
    version="0.1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

llm_instance_global = None
rag_chain_instance = None

class ChatRequest(BaseModel):
    user_message: str
    use_rag: bool = Field(default=True, description="RAG sistemini kullanıp kullanmayacağı (True/False)")

class ChatResponse(BaseModel):
    bot_response: str

class ErrorResponse(BaseModel):
    detail: str

@app.on_event("startup")
async def startup_event():
    global llm_instance_global, rag_chain_instance
    print("FastAPI uygulaması başlatılıyor...")
    
    set_llm_cache(InMemoryCache())
    print("Langchain için InMemoryCache aktif edildi.")
    
    temp_llm, temp_rag_chain = await initialize_rag_components()
    
    if temp_llm: # Sadece LLM'in yüklenmesi yeterli, RAG zinciri de LLM'i kullanır
        llm_instance_global = temp_llm
        if temp_rag_chain:
            rag_chain_instance = temp_rag_chain
            print("LLM ve RAG bileşenleri FastAPI startup event'inde başarıyla yüklendi.")
        else:
            # Bu durum, initialize_rag_components içinde RAG zinciri oluşturulamazsa oluşabilir,
            # ama LLM yine de yüklenmiş olabilir. RAG'sız modda çalışabilir.
            print("UYARI: LLM yüklendi ancak RAG zinciri FastAPI startup'ta yüklenemedi! RAG'sız modda çalışılabilir.")
    else:
        print("HATA: FastAPI startup'ta LLM yüklenemedi! Uygulama düzgün çalışmayabilir.")
        # İsteğe bağlı: raise RuntimeError("LLM yüklenemedi, uygulama başlatılamıyor.")

@app.get("/", response_class=HTMLResponse, summary="Chatbot Arayüzü", tags=["Frontend"])
async def get_home(request: Request):
    print("Ana sayfa isteği alındı.")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse, 
          summary="Chatbot ile Mesajlaşma", tags=["Chatbot"],
          responses={
              400: {"description": "Geçersiz istek", "model": ErrorResponse},
              500: {"description": "Sunucu içi hata", "model": ErrorResponse}
          })
async def chat_with_bot(chat_request: ChatRequest):
    if not llm_instance_global: 
        print("HATA: /chat endpoint'i çağrıldı ancak LLM başlatılmamış.")
        raise HTTPException(status_code=500, detail="Chatbot LLM'i başlatılamadı.")
    
    if chat_request.use_rag and not rag_chain_instance:
        print("HATA: /chat endpoint'i RAG ile çağrıldı ancak RAG zinciri başlatılmamış.")
        raise HTTPException(status_code=500, detail="Chatbot RAG sistemi başlatılamadı.")

    user_message_original = chat_request.user_message.strip()
    if not user_message_original:
        raise HTTPException(status_code=400, detail="user_message alanı boş olamaz.")
        
    print(f"\nKullanıcıdan gelen mesaj (FastAPI): '{user_message_original}', RAG Kullanımı: {chat_request.use_rag}")

    fixed_answer = get_fixed_response(user_message_original)
    if fixed_answer:
        print(f"Sabit cevap bulundu: '{fixed_answer}'")
        return ChatResponse(bot_response=fixed_answer)
    
    start_time = time.time()
    bot_response_text = ""

    try:
        if chat_request.use_rag:
            if not rag_chain_instance: # Ekstra güvenlik kontrolü
                raise HTTPException(status_code=500, detail="RAG zinciri mevcut değil, RAG ile işlem yapılamaz.")
            print("RAG AKTİF: RAG zinciri kullanılıyor...")
            bot_response_text = await rag_chain_instance.ainvoke({"question": user_message_original})
        else:
            print("RAG KAPALI: Doğrudan LLM çağrısı yapılıyor...")
            
            if "VERİLEN BİLGİLER (BAĞLAM):" in config.RAG_SYSTEM_PROMPT_TEXT:
                direct_system_prompt = config.RAG_SYSTEM_PROMPT_TEXT.split("VERİLEN BİLGİLER (BAĞLAM):")[0].strip()
            else:
                direct_system_prompt = config.RAG_SYSTEM_PROMPT_TEXT.strip()

            prompt_str_for_direct_llm = direct_system_prompt + "\n\nKULLANICI SORUSU: {question}\n\nCEVABIN:"
            direct_llm_prompt_template = PromptTemplate.from_template(prompt_str_for_direct_llm)
            
            direct_llm_chain = direct_llm_prompt_template | llm_instance_global | StrOutputParser()
            
            bot_response_text = await direct_llm_chain.ainvoke({"question": user_message_original})

    except Exception as e:
        print(f"Chatbot işlenirken HATA (RAG: {chat_request.use_rag}): {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Chatbot cevabı işlenirken bir sunucu hatası oluştu.")

    end_time = time.time()
    duration = end_time - start_time
    print(f"LLM Cevap Süresi (RAG: {chat_request.use_rag}): {duration:.4f} saniye")
    
    actual_response = bot_response_text.strip()
    print(f"Son Gönderilecek Cevap (FastAPI, RAG: {chat_request.use_rag}): '{actual_response}'")
    
    return ChatResponse(bot_response=actual_response)

if __name__ == "__main__":
    import uvicorn
    print("Uvicorn geliştirme sunucusu başlatılıyor http://127.0.0.1:8000 adresinde...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)