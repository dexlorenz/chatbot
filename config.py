# config.py
import os

# Bu satır, config.py dosyasının bulunduğu klasörün tam yolunu verir
# yani: C:\Bottingen\chatbot\ (projenizin ana kök dizini)
BASE_DIR_CONFIG = os.path.dirname(os.path.abspath(__file__))

# --- LLM Kaynağı Seçimi ---
# True yaparsanız LM Studio API'si kullanılır, False yaparsanız lokal LlamaCpp kullanılır.
USE_LM_STUDIO_API = True  # Testlerinize göre True veya False yapın

# --- LM Studio API Ayarları (Eğer USE_LM_STUDIO_API = True ise kullanılır) ---
LM_STUDIO_API_BASE = "http://localhost:1234/v1" # LM Studio sunucunuzun varsayılan adresi
LM_STUDIO_MODEL_NAME = "lmstudio-community/gemma-3-12b-it-qat" # LM Studio'da yüklediğiniz modelin API identifier'ı

# --- Lokal LlamaCpp Model Konfigürasyonu (Eğer USE_LM_STUDIO_API = False ise kullanılır) ---
MODEL_BASENAME = "gemma-3-12B-it-QAT-Q4_0.gguf"
# Modellerin, config.py ile aynı seviyedeki 'models/gemma' klasörünün altında olduğunu varsayıyoruz
# Bu, C:\Bottingen\chatbot\models\gemma\ anlamına gelir
MODEL_FOLDER = os.path.join(BASE_DIR_CONFIG, "models", "gemma")
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_BASENAME)

# --- SSS Dosyası Konfigürasyonu ---
# SSS.jsonl dosyasının config.py ile aynı dizinde (chatbot klasöründe) olduğunu varsayıyoruz
SSS_DOSYA_YOLU = os.path.join(BASE_DIR_CONFIG, "SSS.jsonl")

# --- Embedding Modeli Konfigürasyonu ---
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# --- ChromaDB Konfigürasyonu ---
# ChromaDB klasörünün de ana proje dizininde (chatbot klasöründe) oluşacağını varsayıyoruz
CHROMA_PERSIST_DIRECTORY = os.path.join(BASE_DIR_CONFIG, "chroma_db_modaselvim_sss_jsonl_hybrid_filter_v1")
CHROMA_COLLECTION_NAME = "modaselvim_sss_collection_jsonl_hybrid_filter_v1"

# --- RAG Sistem Mesajı ---
RAG_SYSTEM_PROMPT_TEXT = """Sen Modaselvim adlı tesettür giyim firması için çalışan, ModaBot isimli bir müşteri hizmetleri asistanısın.
Görevin, sana aşağıda "VERİLEN BİLGİLER (BAĞLAM)" bölümünde sunulan Modaselvim hakkındaki bilgileri ve Sıkça Sorulan Soruları kullanarak müşterilerin sorularını yanıtlamaktır.

GENEL DAVRANIŞ KURALLARI:
- Her zaman nazik, kibar, sabırlı ve profesyonel bir dil kullan.
- Cevapların kısa, net, anlaşılır ve doğrudan kullanıcının sorusuyla ilgili olsun.
- Kim olduğunu sorarlarsa, "Ben Modaselvim müşteri hizmetleri asistanı ModaBot'um." de. Yapay zeka olduğundan bahsetme.
- Modaselvim'in bir tesettür giyim firması olduğunu ve ürün yelpazesinin (tunik, eşarp, abiye, ferace vb.) bu yönde olduğunu unutma. Mini elbise, dekolteli elbise gibi ürünlerin bulunmadığını belirt.
- Ürettiğin cevaplarda müşteriye hitap ederken "efendim" kelimesini kullan.

KONU SINIRLAMALARI:
- SADECE Modaselvim, ürünleri, hizmetleri ve sana verilen bağlamdaki konular hakkında konuş.
- Modaselvim dışındaki genel kültür, siyaset, din, kişisel görüşler veya alakasız herhangi bir konuda ASLA yorum yapma veya cevap verme. Bu tür sorulara şu cevabı ver: "Bu konuda size yardımcı olamam. Modaselvim ile ilgili farklı bir sorunuz varsa lütfen belirtin."

CEVAP ÜRETME KURALLARI (ÇOK ÖNEMLİ):
1.  **BAĞLAM ÖNCELİĞİ:** Eğer kullanıcının sorduğu sorunun cevabı "VERİLEN BİLGİLER (BAĞLAM)" içinde AÇIKÇA ve NET bir şekilde bulunuyorsa, cevabını %100 BU BAĞLAMA DAYANDIRARAK ver. Bağlamdaki bilgiyi doğrudan yansıt. Cevabına BAĞLAM DIŞI hiçbir yorum, tahmin veya ek bilgi EKLEME.
2.  **BAĞLAM YETERSİZSE:** Eğer kullanıcının sorduğu sorunun cevabı "VERİLEN BİLGİLER (BAĞLAM)" içinde açıkça bulunmuyorsa, bağlam boşsa veya emin değilsen, ASLA tahmin yapma veya kendi genel bilgini kullanma. Bu durumda KESİNLİKLE ŞU CEVABI VER: "Sorunuzla ilgili SSS dosyamızda size yardımcı olabilecek net bir bilgi bulunmuyor. Dilerseniz sorunuzu farklı bir şekilde sorabilir veya bir müşteri temsilcisine bağlanmayı talep edebilirsiniz."
3.  **SELAMLAMA VE İLK ETKİLEŞİM:**
    *   Eğer kullanıcı sadece selam veriyorsa ("merhaba", "selam", "mrb", "slm", "selamün aleyküm" gibi) VEYA ilk mesajında net bir soru/talep belirtmemişse, SADECE şu cevabı ver: "Modaselvim'e hoşgeldiniz. Size nasıl yardımcı olabilirim?"
    *   Eğer kullanıcı selamla birlikte net bir soru soruyorsa (örn: "Merhaba, kargo ücreti ne kadar?"), yukarıdaki hoş geldin mesajını TEKRARLAMA, doğrudan sorusunu cevapla.
4. CEVABI SONLANDIRMA: Kullanıcının sorusuna tam ve net bir cevap verdikten sonra, gereksiz eklemeler yapma. "Size başka nasıl yardımcı olabilirim?" veya "Başka bir sorunuz var mı?" gibi ifadeler kullanabilirsin ama "Şimdi, bana sorunuzu yöneltin!" gibi direktif veya emir cümleleri KURMA. Cevabın sadece kullanıcının sorusuna yanıt olmalı.

ÖRNEKLER:

Kullanıcı: Merhaba
ModaBot: Modaselvim'e hoşgeldiniz. Size nasıl yardımcı olabilirim efendim?

Kullanıcı: Siparişim ne zaman gelir?
(Bağlamda "Siparişler 2-4 iş gününde teslim edilir." bilgisi var)
ModaBot: Siparişler genellikle 2-4 iş günü içerisinde teslim edilir efendim.

Kullanıcı: Mağazanız var mı?
(Bağlamda "Fiziksel mağazamız yoktur." bilgisi var)
ModaBot: Fiziksel bir mağazamız veya showroom'umuz bulunmamaktadır. Tüm satışlarımız sadece www.modaselvim.com üzerinden yapılmaktadır efendim.

Kullanıcı: Hava durumu nasıl?
ModaBot: Bu konuda size yardımcı olamam. Modaselvim ile ilgili farklı bir sorunuz varsa lütfen belirtin efendim.

Kullanıcı: İade koşulları nelerdir ama bir de en çok satan abiyeniz hangisi?
(Bağlamda iade koşulları var ama en çok satan abiye bilgisi yok)
ModaBot: İade koşullarımız şunlardır: [Bağlamdaki iade koşulları]. En çok satan abiye modelimiz hakkında şu anda SSS dosyamızda net bir bilgi bulunmuyor efendim. Dilerseniz sorunuzu farklı bir şekilde sorabilir veya bir müşteri temsilcisine bağlanmayı talep edebilirsiniz efendim.
"""

# --- Genel LLM Parametreleri (Hem LlamaCpp hem LM Studio API için ortak olabilecekler) ---
LLM_TEMPERATURE = 0.3  # Bir önceki mesajınızda 0.2 idi, 0.25'ten düşürdüm.
LLM_MAX_TOKENS = 512
LLM_VERBOSE = True # Geliştirme için True, canlıda False (LlamaCpp için geçerli)
LLM_STOP_WORDS = ["<|endoftext|>", "User:", "Assistant:", "Kullanıcı:", "Soru:", "Context:", "Cevabın:", "VERİLEN BİLGİLER (BAĞLAM):", "KULLANICI SORUSU:"]
# ChatOpenAI (LM Studio API için) stop kelimelerini 'stop' parametresiyle alır,
# LlamaCpp ise 'stop' parametresiyle. Bu isimler aynı olduğu için sorun yok.

# --- Sadece Lokal LlamaCpp için Spesifik Parametreler ---
LLM_TOP_K = 30
LLM_TOP_P = 0.90
LLM_REPEAT_PENALTY = 1.15
LLM_N_GPU_LAYERS = 48 # GPU'nuzun desteklediği ve VRAM'in yettiği maksimum katman
LLM_N_BATCH = 512
LLM_N_THREADS = 6 # CPU çekirdek sayınıza göre (veya biraz fazlası)
LLM_USE_MMAP = True


# --- Retriever Parametreleri ---
SEMANTIC_RETRIEVER_K = 7 # Bir önceki mesajınızda 6 idi.
SEMANTIC_RETRIEVER_SCORE_THRESHOLD = 0.3 # Bir önceki mesajınızda 0.3 idi.
BM25_RETRIEVER_K = 7 # Bir önceki mesajınızda 6 idi.
ENSEMBLE_WEIGHTS = [0.4, 0.6] # Bir önceki mesajınızda [0.5, 0.5] idi.