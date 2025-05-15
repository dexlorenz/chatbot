# fixed_responses_module.py

# Cevabı sabit olan sorular ve yanıtları için bir sözlük
# Anahtarlar küçük harfe çevrilmiş ve soru işaretleri kaldırılmış olmalı
FIXED_RESPONSES = {
    "kargo kaç günde gelir": "Siparişler genellikle 2-4 iş günü içerisinde teslim edilir.",
    "kargo ücreti ne kadar": "500 TL ve üzeri alışverişlerde kargo ücretsizdir. Altındaki siparişler için kargo ücreti 30 TL'dir.",
    "iade süresi ne kadar": "Ürünlerinizi teslim aldıktan sonra 14 gün içerisinde iade edebilirsiniz.",
    "hangi kargo ile çalışıyorsunuz": "Yurtiçi Kargo ve Aras Kargo ile çalışıyoruz.",
    "merhaba": "Modaselvim'e hoşgeldiniz. Size nasıl yardımcı olabilirim?", # Selamlama için de eklenebilir
    "selam": "Modaselvim'e hoşgeldiniz. Size nasıl yardımcı olabilirim?",
    "merhabalar": "Modaselvim'e hoşgeldiniz. Size nasıl yardımcı olabilirim?",
    "selamun aleyküm": "Modaselvim'e hoşgeldiniz. Size nasıl yardımcı olabilirim?",
    "modaselvim nedir": "Modaselvim, tesettür giyim alanında online satış yapan bir e-ticaret platformudur.",
    # ... diğer sık sorulan ve cevabı sabit soruları buraya ekleyebilirsiniz ...
}

def normalize_for_fixed_lookup(text: str) -> str:
    """
    Kullanıcı girdisini FIXED_RESPONSES sözlüğünde arama yapmak için normalize eder.
    Küçük harfe çevirir, soru işaretini ve baştaki/sondaki boşlukları kaldırır.
    """
    if not isinstance(text, str): # Gelenin string olup olmadığını kontrol et
        return ""
    return text.lower().replace('?', '').replace('.', '').strip()

def get_fixed_response(user_message: str) -> str | None:
    """
    Verilen kullanıcı mesajı için sabit bir cevap olup olmadığını kontrol eder.
    Varsa cevabı, yoksa None döndürür.
    """
    normalized_message = normalize_for_fixed_lookup(user_message)
    return FIXED_RESPONSES.get(normalized_message)