# llm_setup.py

import os
import traceback
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.documents import Document

# Langchain'in yeni import yolları için uyarıları dikkate alarak:
from langchain_community.retrievers import BM25Retriever # Güncel import
from langchain.retrievers import EnsembleRetriever

# --- RERANKING/FİLTRELEME İÇİN IMPORTLARI AKTİF ET ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
# --- RERANKING/FİLTRELEME İÇİN IMPORTLARI AKTİF ET SONU ---

import config
from utils import metadata_func, async_format_docs_for_context_and_log

async def initialize_rag_components():
    local_llm_instance = None
    local_rag_chain = None

    try:
        # 1. Dil Modelini (LLM) Yükle (Konfigürasyona göre)
        if config.USE_LM_STUDIO_API:
            print(f"LM Studio API'sine bağlanılıyor ({config.LM_STUDIO_API_BASE})...")
            try:
                local_llm_instance = ChatOpenAI(
                    model_name=config.LM_STUDIO_MODEL_NAME,
                    openai_api_base=config.LM_STUDIO_API_BASE,
                    openai_api_key="lm-studio-key",
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=config.LLM_MAX_TOKENS,
                )
                print("LM Studio API için LLM örneği başarıyla oluşturuldu.")
            except Exception as e:
                print(f"HATA: LM Studio API'sine bağlanırken veya LLM örneği oluşturulurken hata: {e}")
                print(traceback.format_exc())
                return None, None
        else: # Lokal LlamaCpp kullanılacaksa
            print(f"Lokal LlamaCpp modeli '{config.MODEL_PATH}' yükleniyor...")
            if not os.path.exists(config.MODEL_PATH):
                raise FileNotFoundError(f"Lokal model dosyası bulunamadı: {config.MODEL_PATH}")
            try:
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
                print("Lokal LlamaCpp LLM örneği başarıyla yüklendi.")
            except Exception as e:
                print(f"HATA: Lokal LlamaCpp modeli yüklenirken hata: {e}")
                print(traceback.format_exc())
                return None, None

        if not local_llm_instance:
            print("HATA: LLM örneği oluşturulamadı.")
            return None, None

        # 2. Embedding Modelini Yükle
        print(f"Embedding modeli ({config.EMBEDDING_MODEL_NAME}) yükleniyor...")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda' if (not config.USE_LM_STUDIO_API and hasattr(config, 'LLM_N_GPU_LAYERS') and config.LLM_N_GPU_LAYERS > 0) or \
                                            (config.USE_LM_STUDIO_API) else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding modeli başarıyla yüklendi.")

        # 3. Vektör Deposu ve Belgeler
        # ... (Bu kısım aynı kalacak, doc_chunks burada SSS dosyasından yükleniyor) ...
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
            if not doc_chunks: print(f"UYARI: BM25 için SSS dosyasından ({config.SSS_DOSYA_YOLU}) hiç doküman yüklenemedi.")
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

        # 4. Temel Retriever'lar
        # Reranker'a daha az, daha hedefe yönelik doküman vermek için k değerlerini düşünebiliriz.
        # Örneğin, config.py'de SEMANTIC_RETRIEVER_K ve BM25_RETRIEVER_K'yı 3-5 arasına çekebilirsiniz.
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
            weights=config.ENSEMBLE_WEIGHTS # Ağırlıkları config'den alıyoruz
        )
        print("Ensemble (Hibrit) Retriever başarıyla oluşturuldu.")
        
        # --- RERANKING/FİLTRELEME ADIMINI TEKRAR AKTİF ET ---
        print("LLMChainFilter tabanlı compressor ve ContextualCompressionRetriever oluşturuluyor...")
        # LLMChainFilter, her bir dokümanın soruya uygun olup olmadığını LLM'e sorar.
        # Bu, local_llm_instance'ı (ana LLM'imizi) kullanır.
        llm_filter_compressor = LLMChainFilter.from_llm(local_llm_instance)
        
        # ContextualCompressionRetriever, ensemble_retriever'dan gelen dokümanları
        # llm_filter_compressor ile işler.
        compression_retriever_with_filter = ContextualCompressionRetriever(
            base_compressor=llm_filter_compressor,
            base_retriever=ensemble_retriever # Temel retriever olarak hibrit retriever'ı kullanıyoruz
        )
        final_retriever = compression_retriever_with_filter # Artık ana retriever'ımız bu sıkıştırılmış/filtrelenmiş retriever
        print("ContextualCompressionRetriever (LLMChainFilter ile) başarıyla oluşturuldu.")
        # --- RERANKING/FİLTRELEME ADIMI SONU ---


        # 7. ASENKRON RAG Zincirini Kur (final_retriever artık compression_retriever_with_filter)
        prompt_template_str = config.RAG_SYSTEM_PROMPT_TEXT + """

VERİLEN BİLGİLER (BAĞLAM):
{context}

KULLANICI SORUSU: {question}

CEVABIN:"""
        rag_prompt = PromptTemplate.from_template(prompt_template_str)

        async def get_context_async(user_question_string: str) -> str:
            print(f"DEBUG: get_context_async (reranking aktif) çağrıldı, user_question_string: '{user_question_string}'")
            # final_retriever (ContextualCompressionRetriever) .ainvoke metodunu destekler
            # ve filtrelenmiş/sıkıştırılmış dokümanları döndürür.
            docs = await final_retriever.ainvoke(user_question_string)
            return await async_format_docs_for_context_and_log(docs)

        local_rag_chain = (
            {
                "context": RunnableLambda(lambda x: x["question"]) | RunnableLambda(get_context_async),
                "question": RunnableLambda(lambda x: x["question"])
            }
            | rag_prompt
            | local_llm_instance
            | StrOutputParser()
        )
        print("Langchain ASENKRON RAG zinciri (Hibrit Retriever ve LLMChainFilter ile) başarıyla oluşturuldu.")
        
        return local_llm_instance, local_rag_chain

    except Exception as e:
        print(f"RAG veya LLM kurulumu sırasında bir HATA oluştu: {e}")
        print(traceback.format_exc())
        return None, None