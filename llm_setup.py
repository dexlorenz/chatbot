# llm_setup.py

import os
import traceback
from langchain_community.llms import LlamaCpp
# ChatOpenAI importu kaldırıldı
from langchain_core.prompts import PromptTemplate, PromptTemplate as LLMChainFilterPromptTemplate # LLMChainFilter promptu için (gerçi artık kullanılmıyor ama dursun)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.documents import Document

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# --- RERANKING/FİLTRELEME İÇİN IMPORTLARI KALDIRDIK ---
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainFilter
# --- RERANKING/FİLTRELEME İÇİN IMPORTLARI KALDIRDIK SONU ---

import config
from utils import metadata_func, async_format_docs_for_context_and_log

async def initialize_rag_components():
    local_llm_instance = None
    local_rag_chain = None

    try:
        # 1. Dil Modelini (LLM) Yükle (Sadece Lokal LlamaCpp)
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
            model_kwargs={'device': 'cuda' if config.LLM_N_GPU_LAYERS > 0 else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding modeli başarıyla yüklendi.")

        # 3. Vektör Deposu ve Belgeler
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

        # --- RERANKING/FİLTRELEME ADIMI ARTIK YOK ---
        # print("LLMChainFilter tabanlı compressor ve ContextualCompressionRetriever oluşturuluyor...")
        # filter_prompt_str = """Aşağıdaki BELGE, sorulan SORU ile doğrudan ilgili mi?
# SADECE 'EVET' veya 'HAYIR' olarak cevap ver. Başka HİÇBİR ŞEY YAZMA. Açıklama EKLEME.

# BELGE:
# {document}

# SORU: {question}

# İlgili (EVET/HAYIR):"""
        # filter_prompt = LLMChainFilterPromptTemplate.from_template(filter_prompt_str)
        # llm_filter_compressor = LLMChainFilter.from_llm(
        #     llm=local_llm_instance,
        #     prompt=filter_prompt
        # )
        # compression_retriever_with_filter = ContextualCompressionRetriever(
        #     base_compressor=llm_filter_compressor,
        #     base_retriever=ensemble_retriever
        # )
        # final_retriever = compression_retriever_with_filter
        # print("ContextualCompressionRetriever (LLMChainFilter ile) başarıyla oluşturuldu.")
        # --- RERANKING/FİLTRELEME ADIMI SONU ---

        # ARTIK final_retriever DOĞRUDAN ensemble_retriever OLACAK
        final_retriever = ensemble_retriever
        print("Ana retriever olarak Ensemble Retriever kullanılacak (LLMChainFilter devredışı).")


        # 7. ASENKRON RAG Zincirini Kur (final_retriever artık ensemble_retriever)
        prompt_template_str = config.RAG_SYSTEM_PROMPT_TEXT + """

VERİLEN BİLGİLER (BAĞLAM):
{context}

KULLANICI SORUSU: {question}

CEVABIN:"""
        rag_prompt = PromptTemplate.from_template(prompt_template_str)

        async def get_context_async(user_question_string: str) -> str:
            print(f"DEBUG: get_context_async (LLMChainFilter devredışı) çağrıldı, user_question_string: '{user_question_string}'")
            # final_retriever (artık EnsembleRetriever) .ainvoke metodunu destekler
            # ve filtrelenmemiş dokümanları döndürür.
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
        print("Langchain ASENKRON RAG zinciri (Sadece Hibrit Retriever ile) başarıyla oluşturuldu.")

        return local_llm_instance, local_rag_chain

    except Exception as e:
        print(f"RAG veya LLM kurulumu sırasında bir HATA oluştu: {e}")
        print(traceback.format_exc())
        return None, None