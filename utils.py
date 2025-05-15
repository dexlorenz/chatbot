# utils.py

from langchain_core.documents import Document # Document tipini belirtmek için

# JSONLoader için metadata çıkarıcı fonksiyon
def metadata_func(record: dict, metadata: dict) -> dict:
    if isinstance(record, dict):
        orijinal_soru = record.get("S")
        if orijinal_soru is not None:
            metadata["orijinal_soru"] = orijinal_soru
        # Başka metadata alanları eklenecekse buraya...
    else:
        metadata["hatali_record_tipi"] = str(type(record))
        metadata["hatali_record_degeri"] = str(record)
    return metadata

# Retriever'dan gelen belgeleri formatlayıp loglayan yardımcı fonksiyon
def format_docs_for_context_and_log(docs: list[Document]) -> str:
    print("\n--- LLM'E GÖNDERİLEN BAĞLAM (CONTEXT) ---")
    if not docs:
        print("--- (Retriever hiçbir chunk bulamadı veya geçerli chunk yok) ---")
        return " " # LLM'e boş bağlam gitmesi için (veya "Bilgi bulunamadı.")
    
    context_str_list = []
    for i, doc in enumerate(docs):
        orijinal_soru_metadata = doc.metadata.get('orijinal_soru', 'Bilinmiyor (metadata yok)')
        print(f"--- ALINAN CHUNK {i+1} (Orijinal Soru: {orijinal_soru_metadata}) ---")
        print(f"İçerik (page_content): {doc.page_content}")
        print("-----------------------------")
        context_str_list.append(doc.page_content)
    print("--- BAĞLAM SONU ---\n")
    return "\n\n".join(context_str_list)