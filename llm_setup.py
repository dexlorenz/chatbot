# llm_setup.py

import os
import traceback
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.documents import Document

from langchain.retrievers import BM25Retriever, EnsembleRetriever
# --- RERANKING İÇİN IMPORTLARI AKTİF TUT ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
# --- RERANKING İÇİN IMPORTLARI AKTİF TUT SONU ---

import config
from utils import metadata_func

# --- ASENKRON format_docs ---
async def async_format_docs_for_context_and_log(docs: list[Document]) -> str:
    print("\n--- ASYNC: LLM'E GÖNDERİLEN (FİLTRELENMİŞ) BAĞLAM (CONTEXT) ---")
    if not docs:
        print("--- (FİLTRELENMİŞ Retriever hiçbir chunk bulamadı veya geçerli chunk yok) ---")
        return " "
    
    context_str_list = []
    for i, doc in enumerate(docs):
        orijinal_soru_metadata = doc.metadata.get('orijinal_soru', 'Bilinmiyor (metadata yok)')
        print(f"--- ALINAN (FİLTRELENMİŞ) CHUNK {i+1} (Orijinal Soru: {orijinal_soru_metadata}) ---")
        print(f"İçerik (page_content): {doc.page_content}")
        print("-----------------------------")
        context_str_list.append(doc.page_content)
    print("--- (FİLTRELENMİŞ) BAĞLAM SONU ---\n")
    return "\n\n".join(context_str_list)


async def initialize_rag_components():
    """
    LLM ve ASENKRON RAG zinciri (reranking/filtreleme aktif) bileşenlerini başlatır ve döndürür.
    """
    local_llm_instance = None
    local_rag_chain = None

    try:
        # 1. Dil Modelini (LLM) Yükle
        print(f"'{config.MODEL_PATH}' modeli Langchain LlamaCpp ile yükleniyor...")
        local_llm_instance = LlamaCpp(
            model_path=config.MODEL_PATH, 
            n_gpu_layers=config.LLM_N_GPU_LAYERS, 
            n_ctx=4096,
            n_batch=config.LLM_N_BATCH, 
            use_mmap=config.LLM_USE_MMAP, 
            n_threads=config.LLM_N_THREADS,
            temperature=config.LLM_TEMPERATURE, 
            top_k=config.LLM_TOP_K, 
            top_p=config.LLM_TOP_P, 
            repeat_penalty=config.LLM_REPEAT_PENALTY,
            max_tokens=config.LLM_MAX_TOKENS, 
            verbose=config.LLM_VERBOSE,
            stop=config.LLM_STOP_WORDS
        )
        print("LLM (LlamaCpp) başarıyla yüklendi.")

        # 2. Embedding Modelini Yükle
        print(f"Embedding modeli ({config.EMBEDDING_MODEL_NAME}) yükleniyor...")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda' if local_llm_instance.n_gpu_layers > 0 else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding modeli başarıyla yüklendi.")

        # 3. Vektör Deposunu (ChromaDB) ve Belgeleri Hazırla/Yükle
        vector_store = None
        doc_chunks = []
        chroma_db_exists = os.path.exists(config.CHROMA_PERSIST_DIRECTORY) and \
                           (os.path.isdir(config.CHROMA_PERSIST_DIRECTORY) and len(os.listdir(config.CHROMA_PERSIST_DIRECTORY)) > 0)

        if chroma_db_exists:
            print(f"Mevcut ChromaDB '{config.CHROMA_PERSIST_DIRECTORY}' adresinden yükleniyor...")
            vector_store = Chroma(
                persist_directory=config.CHROMA_PERSIST_DIRECTORY,
                embedding_function=embeddings,
                collection_name=config.CHROMA_COLLECTION_NAME
            )
            print("Mevcut ChromaDB başarıyla yüklendi. BM25 için belgeler SSS dosyasından tekrar yüklenecek.")
            if not os.path.exists(config.SSS_DOSYA_YOLU):
                raise FileNotFoundError(f"Mevcut DB yüklendi ancak BM25 için SSS dosyası bulunamadı: {config.SSS_DOSYA_YOLU}")
            loader = JSONLoader(
                file_path=config.SSS_DOSYA_YOLU, jq_schema=' .S + " Cevap: " + .C', text_content=False,
                json_lines=True, metadata_func=metadata_func
            )
            doc_chunks = loader.load()
            if not doc_chunks: print("UYARI: BM25 için SSS dosyasından hiç doküman yüklenemedi.")
        else:
            print(f"Yeni bir ChromaDB oluşturuluyor ve '{config.CHROMA_PERSIST_DIRECTORY}' adresine kaydedilecek...")
            if not os.path.exists(config.SSS_DOSYA_YOLU):
                raise FileNotFoundError(f"SSS dosyası bulunamadı: {config.SSS_DOSYA_YOLU}")
            loader = JSONLoader(
                file_path=config.SSS_DOSYA_YOLU, jq_schema=' .S + " Cevap: " + .C', text_content=False,
                json_lines=True, metadata_func=metadata_func
            )
            documents_from_loader = loader.load()
            if not documents_from_loader:
                raise ValueError("SSS JSONL dosyasından hiçbir doküman yüklenemedi.")
            doc_chunks = documents_from_loader
            vector_store = Chroma.from_documents(
                documents=doc_chunks, embedding=embeddings,
                collection_name=config.CHROMA_COLLECTION_NAME, persist_directory=config.CHROMA_PERSIST_DIRECTORY
            )
            print(f"Yeni ChromaDB başarıyla oluşturuldu ve '{config.CHROMA_PERSIST_DIRECTORY}' adresine kaydedildi.")

        # 4. Temel Retriever'ları Oluştur
        semantic_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': config.SEMANTIC_RETRIEVER_SCORE_THRESHOLD, 'k': config.SEMANTIC_RETRIEVER_K}
        )
        if not doc_chunks:
             raise ValueError("BM25 Retriever oluşturulamadı çünkü hiç doküman (doc_chunks) bulunmuyor.")
        bm25_retriever = BM25Retriever.from_documents(documents=doc_chunks)
        bm25_retriever.k = config.BM25_RETRIEVER_K

        # 5. Ensemble (Hibrit) Retriever Oluştur
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=config.ENSEMBLE_WEIGHTS
        )
        print("Ensemble (Hibrit) Retriever başarıyla oluşturuldu.")
        
        # --- RERANKING/FİLTRELEME ADIMINI AKTİF ET ---
        print("LLMChainFilter tabanlı compressor oluşturuluyor...")
        llm_filter_compressor = LLMChainFilter.from_llm(local_llm_instance)
        print("LLMChainFilter compressor başarıyla oluşturuldu.")
        
        print("ContextualCompressionRetriever (LLMChainFilter ile) oluşturuluyor...")
        compression_retriever_with_filter = ContextualCompressionRetriever(
            base_compressor=llm_filter_compressor,
            base_retriever=ensemble_retriever  # Temel retriever olarak hibrit olanı kullan
        )
        # Zincirde kullanılacak son retriever bu olacak:
        final_retriever = compression_retriever_with_filter 
        print("ContextualCompressionRetriever (LLMChainFilter ile) başarıyla oluşturuldu. Filtreleme aktif.")
        # --- RERANKING/FİLTRELEME ADIMI SONU ---

        # 7. ASENKRON RAG Zincirini Kur
        prompt_template_str = config.RAG_SYSTEM_PROMPT_TEXT + """

VERİLEN BİLGİLER (BAĞLAM):
{context}

KULLANICI SORUSU: {question}

CEVABIN:"""
        rag_prompt = PromptTemplate.from_template(prompt_template_str)

        # RAG Zinciri (Bir önceki mesajdaki düzeltilmiş yapı)
        # Girdi: {"question": "kullanıcının sorusu string'i"}
        # Çıktı: LLM'den gelen cevap string'i
        setup_and_retrieval = RunnableParallel(
            context=(
                RunnableLambda(lambda x: x["question"]) 
                | RunnableLambda(final_retriever.aget_relevant_documents) # final_retriever artık compression_retriever_with_filter
                | RunnableLambda(async_format_docs_for_context_and_log)
            ),
            question=RunnableLambda(lambda x: x["question"])
        )

        local_rag_chain = (
            setup_and_retrieval 
            | rag_prompt        
            | local_llm_instance  
            | StrOutputParser()   
        )
        print("Langchain ASENKRON RAG zinciri (FİLTRELEME AKTİF) başarıyla oluşturuldu.")
        
        return local_llm_instance, local_rag_chain

    except Exception as e:
        print(f"RAG veya LLM kurulumu sırasında bir HATA oluştu: {e}")
        print(traceback.format_exc())
        return None, None