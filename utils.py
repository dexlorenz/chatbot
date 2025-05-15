# utils.py

from langchain_core.documents import Document # Document tipini belirtmek için

# JSONLoader için metadata çıkarıcı fonksiyon (Bu zaten doğruydu)
def metadata_func(record: dict, metadata: dict) -> dict:
    # Gelen 'record' bir sözlük mü diye dikkatlice kontrol et
    if isinstance(record, dict):
        # "S" anahtarını .get() ile güvenli bir şekilde almayı dene
        original_question = record.get("S") # Soru için
        if original_question and isinstance(original_question, str): # Değerin var ve string olduğundan emin ol
            metadata["orijinal_soru"] = original_question
        # else:
        #     print(f"UYARI: metadata_func - 'S' anahtarı bulunamadı veya string değil: {record}")

        # Eğer cevap için de bir metadata eklemek isterseniz (örn: cevap_özeti)
        # original_answer = record.get("C") # Cevap için
        # if original_answer and isinstance(original_answer, str):
        #     metadata["cevap_ilk_50_karakter"] = original_answer[:50] + "..."
    else:
        print(f"UYARI: metadata_func dict beklerken {type(record)} aldı: {record}")
    return metadata

# Retriever'dan gelen belgeleri formatlayıp loglayan ASENKRON fonksiyon
async def async_format_docs_for_context_and_log(docs: list[Document]) -> str: # <<<--- ADI VE TANIMI ASENKRON OLARAK GÜNCELLENDİ
    print("\n--- ASYNC: LLM'E GÖNDERİLEN BAĞLAM (CONTEXT) ---")
    if not docs:
        print("--- (Retriever hiçbir chunk bulamadı veya geçerli chunk yok) ---")
        return " "
    
    context_str_list = []
    for i, doc in enumerate(docs):
        orijinal_soru_metadata = doc.metadata.get('orijinal_soru', 'Bilinmiyor (metadata yok)')
        print(f"--- ALINAN CHUNK {i+1} (Orijinal Soru: {orijinal_soru_metadata}) ---")
        print(f"İçerik (page_content): {doc.page_content}")
        print("-----------------------------")
        context_str_list.append(doc.page_content)
    print("--- BAĞLAM SONU ---\n")
    return "\n\n".join(context_str_list)