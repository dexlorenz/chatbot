# app.py (FastAPI versiyonu - InMemoryCache ve RAG On/Off Testi ile)

import os
import time
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field # Field import edildi

# inMemoryCache için importlar
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Kendi oluşturduğumuz modüllerden importlar
import config
from fixed_responses_module import get_fixed_response
from llm_setup import initialize_rag_components # Bu fonksiyon async
# RAG'sız durumda doğrudan LLM ve PromptTemplate kullanmak için:
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --- FastAPI Uygulamasını Başlatma ---
app = FastAPI(
    title="Modaselvim Chatbot API",
    description="Modaselvim için RAG tabanlı chatbot backend servisi.",
    version="0.1.0"
)

# --- HTML Şablonları İçin Ayar ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- Global Değişkenler ---
llm_instance_global = None # llm_setup'tan gelen global LLM örneği için
rag_chain_instance = None

# --- Pydantic Modelleri ---
class ChatRequest(BaseModel):
    user_message: str
    # RAG on/off testi için opsiyonel alan, varsayılanı True (RAG aktif)
    use_rag: bool = Field(default=True, description="RAG sistemini kullanıp kullanmayacağı (True/False)")

class ChatResponse(BaseModel):
    bot_response: str

class ErrorResponse(BaseModel):
    detail: str

# --- Uygulama Başlangıcında Çalışacak Olay (Startup Event) ---
@app.on_event("startup")
async def startup_event():
    global llm_instance_global, rag_chain_instance
    print("FastAPI uygulaması başlatılıyor...")
    
    # Langchain için InMemoryCache'i ayarla
    set_llm_cache(InMemoryCache())
    print("Langchain için InMemoryCache aktif edildi.")
    
    temp_llm, temp_rag_chain = await initialize_rag_components()
    
    if temp_llm and temp_rag_chain:
        llm_instance_global = temp_llm # initialize_rag_components'tan dönen LLM'i global değişkene ata
        rag_chain_instance = temp_rag_chain
        print("LLM ve RAG bileşenleri FastAPI startup event'inde başarıyla yüklendi.")
    else:
        print("HATA: FastAPI startup'ta LLM veya RAG zinciri yüklenemedi!")
        # raise RuntimeError("LLM veya RAG zinciri yüklenemedi, uygulama başlatılamıyor.")

# --- API Endpoint'leri ---
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
async def chat_with_bot(chat_request: ChatRequest): # ChatRequest artık use_rag alanını içeriyor
    # LLM örneğinin yüklenip yüklenmediğini kontrol et
    if not llm_instance_global: 
        print("HATA: /chat endpoint'i çağrıldı ancak LLM başlatılmamış.")
        raise HTTPException(status_code=500, detail="Chatbot LLM'i başlatılamadı.")
    
    # Eğer RAG kullanılacaksa ve RAG zinciri yüklenmemişse hata ver
    if chat_request.use_rag and not rag_chain_instance:
        print("HATA: /chat endpoint'i RAG ile çağrıldı ancak RAG zinciri başlatılmamış.")
        raise HTTPException(status_code=500, detail="Chatbot RAG sistemi başlatılamadı.")

    user_message_original = chat_request.user_message.strip()
    if not user_message_original:
        raise HTTPException(status_code=400, detail="user_message alanı boş olamaz.")
        
    print(f"\nKullanıcıdan gelen mesaj (FastAPI): '{user_message_original}', RAG Kullanımı: {chat_request.use_rag}")

    # 1. Kural Tabanlı Hızlı Cevap Kontrolü
    fixed_answer = get_fixed_response(user_message_original)
    if fixed_answer:
        print(f"Sabit cevap bulundu: '{fixed_answer}'")
        return ChatResponse(bot_response=fixed_answer)
    
    start_time = time.time()
    bot_response_text = ""

    try:
        if chat_request.use_rag:
            # 2.A. RAG AKTİF: RAG zincirini kullan
            print("RAG AKTİF: RAG zinciri kullanılıyor...")
            # RAG zincirine girdi olarak {'question': user_message_original} veriyoruz
            bot_response_text = await rag_chain_instance.ainvoke({"question": user_message_original})
        else:
            # 2.B. RAG KAPALI: Doğrudan LLM'i çağır
            print("RAG KAPALI: Doğrudan LLM çağrısı yapılıyor...")
            
            # RAG'sız durum için sistem mesajı (config.py'de tanımlanabilir)
            # Şimdilik RAG_SYSTEM_PROMPT_TEXT'in context'siz halini kullanalım
            # İdealde, config.py'de DIRECT_LLM_SYSTEM_PROMPT gibi ayrı bir değişken olmalı.
            if "VERİLEN BİLGİLER (BAĞLAM):" in config.RAG_SYSTEM_PROMPT_TEXT:
                direct_system_prompt = config.RAG_SYSTEM_PROMPT_TEXT.split("VERİLEN BİLGİLER (BAĞLAM):")[0].strip()
            else: # Eğer context bölümü yoksa, tüm sistem mesajını kullan
                direct_system_prompt = config.RAG_SYSTEM_PROMPT_TEXT.strip()

            prompt_str_for_direct_llm = direct_system_prompt + "\n\nKULLANICI SORUSU: {question}\n\nCEVABIN:"
            direct_llm_prompt_template = PromptTemplate.from_template(prompt_str_for_direct_llm)
            
            # Basit bir zincir: prompt | llm | output_parser
            # llm_instance_global'i kullanıyoruz (llm_setup.py'den gelen)
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

# --- Uygulamayı Çalıştırmak İçin ---
if __name__ == "__main__":
    import uvicorn
    print("Uvicorn geliştirme sunucusu başlatılıyor http://127.0.0.1:8000 adresinde...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)