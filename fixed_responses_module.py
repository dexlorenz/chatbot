# fixed_responses_module.py

# Cevabı sabit olan sorular ve yanıtları için bir sözlük
# Anahtarlar küçük harfe çevrilmiş ve soru işaretleri kaldırılmış olmalı
RAW_FIXED_RESPONSES_DATA = {
    "kargo kaç günde gelir": "Siparişler genellikle 2-4 iş günü içerisinde teslim edilir.",
    "kargo ücreti ne kadar": "500 TL ve üzeri alışverişlerde kargo ücretsizdir. Altındaki siparişler için kargo ücreti 30 TL'dir.",
    "iade süresi ne kadar": "Ürünlerinizi teslim aldıktan sonra 14 gün içerisinde iade edebilirsiniz.",
    "hangi kargo ile çalışıyorsunuz": "Yurtiçi Kargo ve Aras Kargo ile çalışıyoruz.",
    "merhaba": "Modaselvim'e hoşgeldiniz. Size nasıl yardımcı olabilirim?", # Selamlama için de eklenebilir
    "selam": "Modaselvim'e hoşgeldiniz. Size nasıl yardımcı olabilirim?",
    "merhabalar": "Modaselvim'e hoşgeldiniz. Size nasıl yardımcı olabilirim?",
    "selamun aleyküm": "Modaselvim'e hoşgeldiniz. Size nasıl yardımcı olabilirim?",
    "modaselvim nedir": "Modaselvim, tesettür giyim alanında online satış yapan bir e-ticaret platformudur.",
    "MODASELVİM ne zaman kuruldu?": "1996 yılında kurulan MODASELVİM, tekstil sektörünün çeşitli kollarında faaliyet göstermektedir.",
    "MODASELVİM hangi sektörde faaliyet göstermektedir?": "1996 yılında kurulan MODASELVİM, tekstil sektörünün çeşitli kollarında faaliyet göstermektedir.",
    "MODASELVİM'in kuruluş yılı nedir?": "1996 yılında kurulan MODASELVİM, tekstil sektörünün çeşitli kollarında faaliyet göstermektedir.",
    "MODASELVİM hangi alanlarda üretim yapmaktadır?": "Gerek yurt dışı gerek yurt içinde hazır giyim , konfeksiyon alanında alt ve üst grup olmak üzeri çeşitli ürün gruplarında üretim yapmaktadır.",
    "MODASELVİM'in üretim yaptığı ürün grupları nelerdir?": "Gerek yurt dışı gerek yurt içinde hazır giyim , konfeksiyon alanında alt ve üst grup olmak üzeri çeşitli ürün gruplarında üretim yapmaktadır.",
    "MODASELVİM yurt içinde hangi marka ile faaliyet göstermektedir?": "Genellikle yurtdışı satışa odaklanmış olan MODASELVİM yurt içinde de MODASELVİM markası ile 2013 yılında özellikle tesettür giyim alanında faaliyetlerine başlamıştır.",
    "MODASELVİM yurt içinde tesettür giyim alanında ne zaman faaliyete başladı?": "Genellikle yurtdışı satışa odaklanmış olan MODASELVİM yurt içinde de MODASELVİM markası ile 2013 yılında özellikle tesettür giyim alanında faaliyetlerine başlamıştır.",
    "MODASELVİM'in hedef ve misyonu nedir?": "Hedef ve misyonumuz en kaliteli ve ekonomik ürünleri müşterilerine hızlı ve güvenli bir şekilde ulaştırmaktır.",
    "MODASELVİM müşterilerine nasıl ürünler ulaştırmayı hedefler?": "Hedef ve misyonumuz en kaliteli ve ekonomik ürünleri müşterilerine hızlı ve güvenli bir şekilde ulaştırmaktır.",
    "MODASELVİM'in müşteri memnuniyeti ilkesi nedir?": "Yüzde yüz müşteri memnuniyeti ilkesi ile yola çıkan MODASELVİM ekibi olarak vazgeçilmez standartlarımız, özgün tasarımlarımız , kaliteli ve rahat kumaşlardan oluşan koleksiyonlarımız, sürdürülebilir yenilikleri ve gelişmeleri takip ederek güncel modeller ile müşterilerimize farklı olma ayrıcalığını sunmak, 6502 sayılı tüketiciyi koruma kanunlarına uygun hareket etmektir.",
    "MODASELVİM'in vazgeçilmez standartları nelerdir?": "Yüzde yüz müşteri memnuniyeti ilkesi ile yola çıkan MODASELVİM ekibi olarak vazgeçilmez standartlarımız, özgün tasarımlarımız , kaliteli ve rahat kumaşlardan oluşan koleksiyonlarımız, sürdürülebilir yenilikleri ve gelişmeleri takip ederek güncel modeller ile müşterilerimize farklı olma ayrıcalığını sunmak, 6502 sayılı tüketiciyi koruma kanunlarına uygun hareket etmektir.",
    "MODASELVİM hangi tüketiciyi koruma kanunlarına uygun hareket eder?": "Yüzde yüz müşteri memnuniyeti ilkesi ile yola çıkan MODASELVİM ekibi olarak vazgeçilmez standartlarımız, özgün tasarımlarımız , kaliteli ve rahat kumaşlardan oluşan koleksiyonlarımız, sürdürülebilir yenilikleri ve gelişmeleri takip ederek güncel modeller ile müşterilerimize farklı olma ayrıcalığını sunmak, 6502 sayılı tüketiciyi koruma kanunlarına uygun hareket etmektir.",
    "MODASELVİM'in firma adı nedir?": "Firma Adı:MODASELVİM TEKSTİL SAN. VE TİC.A.Ş.",
    "MODASELVİM'in telefon numarası nedir?": "Telefon:+90 (212) 550 52 52",
    "MODASELVİM'in firma adresi nedir?": "Firma Adresi:Yenibosna Merkez, Oruç Reis Sokağı No:5, 34197 Bahçelievler/İstanbul",
    "MODASELVİM'in e-posta adresi nedir?": "E-Posta:info@modaselvim.com",
    "MODASELVİM'in vergi dairesi hangisidir?": "Vergi Dairesi:Yenibosna",
    "MODASELVİM'in vergi numarası nedir?": "Vergi Numarası:6221479395",
    "MODASELVİM'in ticaret sicil numarası nedir?": "Ticaret Sicil No:55176-5",
    "MODASELVİM'in Mersis numarası nedir?": "Mersis Numarası:0622147939500001",
    "Ödememi nasıl yapabilirim?": "YURT İÇİ ALIŞVERİŞLER: Yurtiçi alışverişlerinizde ödemelerinizi; kredi kartınızla, banka kartınızla veya kapıda ödeme seçeneklerinden birisini kullanarak yapabilirsiniz. YURT DIŞI ALIŞVERİŞLER: Yurtdışı alışverişlerinizde ödemelerinizi; kredi kartı, banka kartınız (debit – ATM) ile tek çekim olarak yapabilirsiniz.",
    "Yurt içi alışverişlerde hangi ödeme seçenekleri mevcut?": "Yurtiçi alışverişlerinizde ödemelerinizi; kredi kartınızla, banka kartınızla veya kapıda ödeme seçeneklerinden birisini kullanarak yapabilirsiniz.",
    "Banka kartıyla nasıl ödeme yapabilirim?": "Banka kartı ile ödeme seçeneği: Alışverişlerinizde, banka kartınızı (ATM - Bankamatik) kredi kartı gibi kullanarak alışveriş yapabilirisiniz",
    "Kapıda ödeme seçeneği nedir?": "Kapıda Ödeme : Kapıda ödeme seçeneğinde, kargo şirketi kapıda nakit para tahsilatı yapabilmek için, 15,00 TL kapıda ödeme servis ücreti tahsil etmektedir.",
    "Kapıda ödeme servis ücreti ne kadar?": "Kapıda ödeme seçeneğinde, kargo şirketi kapıda nakit para tahsilatı yapabilmek için, 15,00 TL kapıda ödeme servis ücreti tahsil etmektedir.",
    "Kapıda ödeme ücreti nasıl ödenmez?": "Kapıda ödeme servis ücreti ödememek için,güvenli alışveriş sistemimiz üzerinden kredi kartı veya banka kartı ile alışveriş yapmanızı öneririz.",
    "Yurt içi alışverişlerde kredi kartına taksit yapılıyor mu?": "Yurtiçi alışverişlerde Kredi kartına Taksit: Kredi kartınız ile ödemelerinizde, siparişi tamamla sayfasında kredi kartlarınızı seçtikten sonra taksit seçeneklerini görebilirsiniz.",
    "Kredi kartı taksit seçeneklerini nerede görebilirim?": "Kredi kartınız ile ödemelerinizde, siparişi tamamla sayfasında kredi kartlarınızı seçtikten sonra taksit seçeneklerini görebilirsiniz.",
    "Yurt dışı alışverişlerde ödeme nasıl yapılır?": "Yurtdışı alışverişlerinizde ödemelerinizi; kredi kartı, banka kartınız (debit – ATM) ile tek çekim olarak yapabilirsiniz.",
    "Yurt dışı alışverişlerde banka kartı ile ödeme nasıl yapılır?": "Banka kartı ile ödeme seçeneği: Yurtdışı alışverişlerinizde de, banka kartınızı (ATM - Debit )kredi kartı gibi kullanarak alışveriş yapabilirisiniz.",
    "İade işlemini nasıl yapabilirim?": "Üyeliğinize giriş yaparak siparişlerim kısmından Değişim/İade Et butonuna tıklayınız. İade edilecek ürün, iade nedeni ve ürün adedini seçiniz. Kargo seçiminizi yapınız. İade kodunu not alınız.",
    "Ürün iadesi nasıl gerçekleştirilir?": "Üyeliğinize giriş yaparak siparişlerim kısmından Değişim/İade Et butonuna tıklayınız. İade edilecek ürün, iade nedeni ve ürün adedini seçiniz. Kargo seçiminizi yapınız. İade kodunu not alınız.",
    "İade kargo koduma nasıl ulaşabilirim?": "İade kargo kodunuza siparişlerim sayfasından ve cep telefonunuza gönderilen bilgilendirme mesajından ulaşabilirsiniz.",
    "İade edilecek ürünleri nasıl paketlemeliyim?": "İade kodu aynı olan ürünleri faturasıyla beraber aynı pakete koyup ve paketin sağlam olduğundan emin olunuz.",
    "İade süresi ne kadar?": "İade süresi sipariş teslim edildikten sonra 15 gündür.",
    "İade kodunu aldıktan sonra paketi ne kadar sürede kargoya vermeliyim?": "İade kodunu aldığınız tarihten itibaren 7 gün içinde paketinizi kargo şubesine teslim etmeniz gerekmektedir.",
    "Yurt dışı gönderimlerde kargo ücreti nasıl belirlenir?": "Yapacağınız alışverişlerde kargo ücretiniz gönderim yapılacak ülke politikasına, paketin ağırlığına ve ürün miktarına göre değişmektedir.",
    "Yurt dışı kargo tutarını nerede görebilirim?": "Kargo tutarını ödeme öncesi bölümde görebilirsiniz.",
    "Yurt dışına hangi kargo şirketleri ile teslimat yapılıyor?": "Anlaşmalı olduğumuz DHL Express, UPS ve PTS kargo şirketleri ile Dünya'nın her yerine 1 hafta ile 10 gün arasında teslimat gerçekleştirilmektedir. Anlaşmalı olduğumuz Smsa Express, Ups(Sadece Abd) , Dhl Express& Global, Asset, Ptt Kıbrıs kargo şirketleri ile Dünya'nın her yerine, 7-10 gün içerisinde teslimat gerçekleştirilmektedir.",
    "Yurt dışı teslimat süresi ne kadar?": "Dünya'nın her yerine 1 hafta ile 10 gün arasında teslimat gerçekleştirilmektedir. (Kargo tercihinize göre teslimat süresi değişiklik gösterebilir.) Dünya'nın her yerine, 7-10 gün içerisinde teslimat gerçekleştirilmektedir. Orders are delivered in 5-10 days after 24-48 hours of control.",
    "Taksitlendirme işlemlerinde hangi sözleşme hükümleri geçerlidir?": "Taksitlendirme işlemlerinde, alıcı ile kart sahibi banka arasında imzalamış bulunan sözleşmenin ilgili hükümleri geçerlidir. Kredi kartı ödeme tarihi banka ile alıcı arasındaki sözleşme hükümlerince belirlenir.",
    "Cayma hakkı ne kadar sürede kullanılabilir?": "Alıcı, sözleşme konusu mal/hizmetin kendisine veya gösterdiği adresteki kişi/kuruluşa tesliminden itibaren 14 (on dört)gün içinde cayma hakkını kullanabilir.",
    "Cayma hakkının kullanılması için ne yapılması gerekir?": "Cayma hakkının kullanılması için aynı süre içinde satıcının müşteri hizmetlerine e-posta veya telefon ile bildirimde bulunulması ve mal/hizmetin kullanılmamış olması şarttır.",
    "Fatura aslı gönderilmezse KDV iade edilir mi?": "Fatura aslı gönderilmezse alıcıya KDV ve varsa diğer yasal yükümlülükler iade edilemez.",
    "Cayma hakkı ile iade edilen ürünün teslimat bedeli kime aittir?": "Cayma hakkı nedeni ile iade edilen mal/hizmetin teslimat bedeli satıcı tarafından karşılanır.",
    "Hangi mal ve hizmetlerde cayma hakkı kullanılamaz?": "Niteliği itibarıyla iade edilemeyecek mal/hizmetler, hızla bozulan ve son kullanma tarihi geçen mal/hizmetler, tek kullanımlık mal/hizmetler, hijyenik mal/hizmetler, abiye mal/hizmetler, kopyalanabilir her türlü yazılım ve programlardır. Ayrıca, her türlü yazılım ve programlarında, Çeşitli medyaların (Dvd,Cd v.b), bilgisayar ve kırtasiye sarf malzemelerinde(toner, kartuş, şerit v.b) ile kozmetik malzemelerinde cayma hakkının kullanılabilmesi için mal/hizmetin ambalajının açılmamış, bozulmamış ve kullanılmamış olmaları şartı bulunmaktadır.",
    "Alıcı kredi kartı işlemlerinde temerrüde düşerse ne olur?": "Alıcı, kredi kartı ile yapmış olduğu işlemlerinde temerrüde düşmesi halinde kart sahibi bankanın kendisi ile yapmış olduğu kredi kartı sözleşmesi çerçevesinde faiz ödeyecek ve bankaya karşı sorumlu olacaktır. Bu durumda ilgili banka hukuki yollara başvurabilir; doğacak masrafları ve vekalet ücretini alıcıdan talep edebilir ve her koşulda alıcının borcundan dolayı temerrüde düşmesi halinde, alıcı, borcun gecikmeli ifasından dolayı satıcının oluşan zarar ve ziyanını ödemeyi kabul eder.",
    "Sözleşmeden kaynaklanabilecek ihtilaflarda yetkili mahkeme hangisidir?": "İş bu sözleşmeden kaynaklanabilecek ihtilaflarda, Sanayi ve Ticaret Bakanlığınca ilan edilen değere kadar Tüketici Hakem Heyetleri, belirtilen değer üstüne Tüketici Mahkemeleri; bulunamayan yerlerde Asliye Hukuk Mahkemeleri yetkilidir.",
    "Modaselvim'in fiziksel mağazası var mı?": "Fiziksel mağazamız yoktur satışlarımız sadece www.modaselvim.com üzerinden yapılmaktadır. Türkiye'nin hiçbir yerinde mağaza veya şubemiz bulunmamaktadır. Satışlarımız sadece internetten www.modaselvim.com adresinden yapılmaktadır.",
    "Sipariş oluşturmak için hangi bilgiler gerekli?": "İsim soy isim, adresinizi (Mahalle,sok/cad, kapı no /İl-ilçe ) 2 farklı cep telefon numarası, Ürün kodu, Renk Beden yazarsanız siparişiniz kapıda ödemeli olarak oluşturulacaktır.",
    "İade ve değişimde kargo ücreti var mı?": "Aldığınız kod ile ürünlerinizi ücretsiz gönderebilirsiniz. Değişimler kargo ücretsiz olup, her sipariş için yalnızca bir kez yapılmaktadır.",
    "Birden fazla ürünü iade/değişim yapacaksam nasıl göndermeliyim?": "İade değişime göndereceğiniz tüm ürünleri tek kargo ile göndermelisiniz. Daha sonradan göndereceğiniz ürünlerin kargo ücretini ödeyerek gönderebilirsiniz.",
    "İade ve değişim süresi ne kadar?": "Teslim tarihinizden sonra 15 gün içerisinde iade ve değişim yapılabilmektedir.",
    "İade ve değişim için ürünlerin durumu nasıl olmalı?": "Ürünlerin kullanılmamış ve tadilat görmemiş olması gerekmektedir.",
    "Bir sipariş için kaç kez değişim yapılabilir?": "Değişimler kargo ücretsiz olup, her sipariş için yalnızca bir kez yapılmaktadır.",
    "Yurt içi kargo ücreti ne kadar?": "Yurt içi tüm siparişlerde 34,90 TL kargo ücreti yansımaktadır.",
    "Kapıda ödeme tahsilat farkı ne kadar?": "Kapıda ödemeli tüm siparişlerde ayrıca 15,00 TL tahsilat farkı yansımaktadır.",
    "Kredi kartı ile siparişte kapıda ödeme bedeli yansır mı?": "Site üzerinden kredi kartı ile sipariş oluşturulduğunda kapıda ödeme bedeli yansımamaktadır.",
    "Paketim ne zaman kargoya teslim edilir?": "Paketiniz 24 saatlik hazırlık aşamasının ardından kargoya teslim edilir.",
    "Ürünüm kargoya verildiğinde bilgilendirme yapılır mı?": "Ürünleriniz kargoya verildiğinde sms ile bilgilendirilme yapılmaktadır.",
    "Hangi kargo firmaları ile teslimat yapılıyor ve süreleri nedir?": "MNG, ARAS, SÜRAT kargo ile 3 iş günü Kargoist firmasıyla 24- 48 saat ve PTT kargo ile köy kasaba adreslerine 7-10 iş günü içerisinde teslimat sağlanmaktadır. PTT, Aras , Mng, Sürat ve Kargoist kargo ile anlaşmamız bulunmaktadır.",
    "Sipariş sorgulamak için hangi bilgiler gerekli?": "Siparişe kayıtlı ; Telefon numaranızı veya TS ile başlayan sipariş numaranızı yazarsanız yardımcı olabiliriz.",
    "Tükenen ürünler tekrar stoğa girer mi?": "Stoklarımız zaman zaman güncellenmektedir. Ancak net bir tarih veremiyoruz. Sitemizden, facebook ve instagram hesaplarımızdan takipte kalmanızı rica ederiz.",
    "Yurt dışı kargo ücretsiz mi?": "Bazı avrupa ülkelerine 100€, Arap Ülkelerine 75$ üzeri kargo ücretsizdir. ABD ve Kanada ülkesine 100$ üzeri kargo ücretsizdir.",
    "Yurt dışı kargo tutarını nasıl öğrenebilirim?": "Kargo tutarını sepette ya da satın alma ekranında ülke seçerek görebilirsiniz.",
    "Yurt dışı ödeme seçenekleri nelerdir?": "Yurtdışı alışverişlerinizde ödemelerinizi; kredi kartı, banka kartınız (debit – ATM) ile yapabilirsiniz.",
    "Ürünler görseldeki ile aynı mı geliyor?": "Ürünlerimiz görseldeki ile aynı gönderilmektedir.",
    "Ürünle ilgili bir problem yaşarsam ne yapabilirim?": "Ürün ile alakalı herhangi bir problem yaşadığınız takdirde 15 gün içerisinde değişim veya iade yapılabilmektedir.",
    "Hangi kargo firmalarıyla anlaşmanız var?": "PTT, Aras , Mng, Sürat ve Kargoist kargo ile anlaşmamız bulunmaktadır.",
    "Köy ve kasabalara hangi kargo ile gönderim yapılıyor?": "Köy ve kasaba gönderileri PTT Kargo ile yapılmaktadır.",
    "Kapıda ödeme ek hizmet bedeli nedir?": "Kargo firmalarının kapıda tahsilat hizmeti için aldığı bir ücrettir.",
    "Kapıda ödeme ek hizmet bedelini ödememek için ne yapabilirim?": "Bu ücreti ödemek istemiyorsanız kredi kartınız ile https://www.modaselvim.com/ adresimizden sipariş verebilirsiniz.",
    "IBAN nedir?": "IBAN, banka hesabınızın TR ile başlayan 24 haneli numaradır. Bu numarayı hesap cüzdanınızdan, bankanızın çağrı merkezi veya ATM’lerinden öğrenebilirsiniz.",
    "IBAN numaramı verdikten sonra iadem ne zaman yapılır?": "IBAN numaranız sisteme kaydedildi. 48 saat içinde bankaya iadeniz ödenecektir. Bankanız 7-10 iş günü içerisinde hesabınıza yansıtacaktır.",
    "Siparişim 'Hazırlık aşaması' ne demek?": "Ürününüz paketlenmek üzere depoya sevk edilmiştir. İşlemleri tamamlandığında doğrudan kargoya verilecek ve sms ile bilgilendirileceksiniz.",
    "Ürün değişimi veya para iadesi talebimi nasıl iletebilirim?": "Ürün değişimi istiyorsanız istediğiniz ürünün; kodunu, rengini ve bedenini yazmanız yeterlidir. Para iadesi istiyorsanız iban numarası ve iban sahibinin ismini soy ismini alabilir miyiz? (Kredi kartı ile verilen siparişlerde alış-veriş yapılan karta iade edilir.)",
    "Hasarlı ürün iade/değişimi nasıl yapılır?": "Hasarlı ürününüzü 30 günlük yasal süreciniz dolmadan Üyeliğinize giriş yaparak siparişler alanında Değişim / iade et butonundaki adımları izleyerek iade kodu alıp tarafımıza gönderebilirsiniz. Ürün ulaştığında incelenerek dönüş yapılacaktır .",
    "Yurt dışı iade talebimi nasıl iletebilirim?": "Merhaba, 05300915749 nolu whatsapp hattımıza iade sebebi ile birlikte siparişinize kayıtlı ad-soyad ve sipariş numaranızı iletmenizi rica ederiz.",
    "Siparişimi iptal edebilir miyim?": "Sipariş işlemleriniz tamamlanmıştır bu aşamada maalesef kargo iptali gerçekleştiremiyoruz. Ürünü deneyip öyle karar vermeniz daha iyi olur ki beğenmezseniz ücretsiz değişim hakkınız veya iade hakkınız mevcut.",
    "Siparişim kargoya verilmişse iptal edebilir miyim?": "Siparişiniz şu an kargoya verilmiş bu indirimli ürünümüzün avantajını kaçırmamanız için ürünü denedikten sonra karar vermeniz sizin için daha avantajlı olacaktır.",
    "Köy dağıtımına tabii kargolar için ne yapılabilir?": "İkamet ettiğiniz bölgeye, kargo şubesi belli günlerde teslimat yapmaktadır. Aciliyetiniz varsa şubeden teslim alabilirsiniz.",
    "Siparişimdeki ürün temin edilemezse ne olur?": "Siparişinizdeki ürün temin edilemediği için iptal edilmiştir. Aksaklık için özür dileriz. Dilerseniz ürünün farklı renklerini veya yeni ürünlerimizi inceleyerek sipariş oluşturabilirsiniz .",
    "Geciken kargom için ne yapabilirim?": "Kargo hazırlanma süreciniz devam etmektedir. Kampanya dönemlerinde hazırlık süreçlerinde gecikmeler yaşanabiliyor. Acil çıkış talebinizi ilgili birimimize iletiyoruz. En hızlı şekilde ürünlerinizi size ulaştırmaya çalışmaktayız.",
    "Kişiye özel indirim yapılıyor mu?": "Ürünlerimiz genel anlamda indirimlidir. Sistemsel olarak kişiye özel fiyat değişikliği yapamıyoruz.",
    "Doğru bedeni nasıl seçebilirim?": "Beden kilo ile doğru orantılı olmayabiliyor. Sizleri yanıltmak istemeyiz. Ürün özellikleri kısmından ürüne özel beden tablomuzu inceleyerek ölçülere göre sipariş vermenizi tavsiye ederiz",
    "Kredi kartı iadem ne zaman yansır?": "Merhaba, iade onay tarihinden itibaren 3-4 iş günü içerisinde kartınıza yansımaktadır. Alışverişiniz taksitli ise; bir sonraki ay ekstrenize taksitli bir şekilde yansır. Banka kartlarına yapılan iadelerde 10-15 gün içerisinde yansıyabilmektedir. Kartınız kredi kartı özelliğindeyse iade yapıldıktan sonra bankanız 3 iş günü içerisinde, banka kartı ise 7-10 iş günü içerisinde yansıtacaktır. İadeniz 48 saat içerisinde bankanıza sağlanıyor olacaktır. Kartınız kredi kartı özelliğindeyse iade yapıldıktan sonra bankanız size 3 iş günü içerisinde, banka kartı ise iban numaralarına ortalama 7 ila 10 iş günü içerisinde yansıtılacaktır.",
    "Siparişim neden oluşturulmadı?": "Siparişinizi onaylamadan web sitemizden çıkış yaptığınız için siparişiniz oluşturulmamıştır.",
    "Şeffaf veya kontrollü kargo anlaşmanız var mı?": "Kontrollü kargo anlaşmamız bulunmamaktadır. Paketinizi ancak teslim alarak açabilirsiniz. Herhangi bir sorunda bize 24 saat içinde bilgi vermeniz gerekir. İade değişim hakkınız bulunmaktadır .",
    "Mesajlarıma neden geç cevap veriliyor?": "Mesajlara en eski mesajdan başlayarak sırasıyla cevap verilmektedir. Bizden yanıt gelmeden tekrar yazarsanız sistem sizi en üste atar. Bu durumda yanıt alma süreniz uzar.",
    "İademi nasıl alırım? (Ödeme şekline göre)": "Kapıda nakit ödemeli siparişlerde iadeler iban numarasına yapılmaktadır. Kredi kartlı siparişlerde iade kredi kartına yapılmaktadır.",
    "Ürünümü iadeye gönderdim, kargo takip numaramı nasıl bulabilirim?": "Gönderdiğinize dair kargo takip numarası iletebilir misiniz ? Kargo takip numaranızı bilmiyorsanız kargo firması ile iletişime geçerek talep edebilirsiniz. Takip numarası yanlıştır. Kargoyu bize gönderirken şubeden aldığınız takip numarası gerekir. Eğer yoksa kargo şubenizi arayarak talep edebilirsiniz.",
    "İade ürünüm size ulaştı mı?": "Ürün tarafımıza ulaşmıştır. Kalite kontrolü yapılarak sisteme işlenir. Telefonunuza sms ile bilgilendirilmesi yapılarak talebiniz alınacaktır.",
    "İngilizce Instagram sayfanız var mı?": "Hello, For English information you can follow our English Instagram page https://www.instagram.com/modaselvim_en/?hl=tr . Thank you.",
    "Toptan satış yapıyor musunuz?": "Toptan satışımız sadece yurtdışına mevcuttur. Toptan alışverişler için 905050763615 nolu whatsapp hattımızdan temsilcilerimize ulaşabilirsiniz. Hello, Wholesale purchases are only possible to Turkey, you should have a partnership with cargo company in Turkey. System automatically makes the discount for wholesale purchases up to %15 depending on the quantity and amount. You cannot return but can exchange in 15 days from delivery and only the 20% of the products.",
    "Aras Kargo takip linki nedir?": "Kargo takip numaranızla https://www.araskargo.com.tr linkine tıklayarak kargonuzun nerede olduğunu görebilir, bulunduğu şube ile iletişime geçebilirsiniz.",
    "PTT Kargo takip linki ve telefon numarası nedir?": "barkod numaranızla https://gonderitakip.ptt.gov.tr/ linke tıklayarak kargonuzu takip edebilirsiniz . 444 1 788 nolu telefon numarasını arayarak kargonuz hakkında bilgi alabilirsiniz.",
    "Sürat Kargo takip linki nedir?": "Kargo takip numaranızla https://www.suratkargo.com.tr/ linkine tıklayarak kargonuzun nerede olduğunu görebilir, bulunduğu şube ile iletişime geçebilirsiniz",
    "UPS Kargo takip linki nedir?": "UPS kargo _________ takip numaranız ile https://www.ups.com.tr/gonderi_takip.aspx linke tıklayarak kargo hareketlerini takip edebilirsiniz.",
    "DHL Express Kargo takip linki nedir?": "DHL kargo _________takip numaranız ile https://www.dhl.com.tr/tr/express.html linke tıklayarak kargo hareketlerini takip edebilirsiniz.",
    "İngilizce Facebook sayfanız var mı?": "Hello, For English information you can follow our English Instagram page m.me/modaselvimen Thank you.",
    "IBAN bilgilerim hatalıysa iadem nasıl yapılır?": "Değerli Müşterimiz, MODASELVİM den yapmış olduğunuz .... numaralı siparişinizdeki iadeniz IBAN bilgileri hatalı olduğu için iade yapılamamaktadır. IBAN ve bu IBAN 'a ait Ad Soyad bilgisini iletmenizi rica ederiz.",
    "Talebim net anlaşılamazsa ne olur?": "Değerli Müşterimiz, MODASELVİM den yapmış olduğunuz numaralı siparişiniz bize ulaşmış olup talebiniz net anlaşılamamış ve ürün bedeli üyeliğinize çek olarak tanımlanmıştır. Bu değişim çeki ile 1 yıl içinde kargo ücretsiz olarak yeni siparişinizi oluşturabilirsiniz ya da talebiniz bize buradan iletebilirsiniz.",
    "Hamileler için özel ürünleriniz var mı?": "Merhaba gebelik dönemine özel ürünlerimiz bulunmamaktadır. Dilerseniz sayfamızın http://www.modaselvim.com/elbise bölümünden sizin için uygun olduğunu düşündüğünüz ürünü inceleyebilirsiniz.",
    "Değişime yolladığım paket ne zaman işleme alınır?": "Paketiniz ulaştığında, 48 saatlik kalite kontrol süreci ardından talebiniz doğrultusunda işleme yapılacaktır. Yoğun dönemlerde bu süre değişiklik gösterebilir ayrıca sms ile bilgilendirileceksiniz.",
    "Ürün dar kalıp mıdır?": "Ürün dar kalıptır. Normalde kullandığınız bedenden bir beden büyük alınması tavsiye edilir.",
    "Ürün tam kalıp mıdır?": "Ürün tam kalıp olup normalde kullandığınız bedeni sipariş vermeniz tavsiye edilir.",
    "Kampanya ve yeniliklerden nasıl haberdar olabilirim?": "Kişiye özel bilgilendirme yapamıyoruz. Kampanya ve yeniliklerden haberdar olmak için mobil uygulamamızı indirebilir, bizi www.modaselvim.com adresimizden veya sosyal medya hesaplarımızdan (INSTAGRAM FACEBOOK) takip edebilirsiniz.",
    "Üyeliğimi nasıl iptal edebilirim?": "https://www.modaselvim.com/uye-girisi-sayfasi.xhtml linke tıklayıp üyeliğe girerek üyelik iptali alanından üyeliğinizi silebilirsiniz. Dilediğiniz zaman üyeliğinizi yeniden oluşturabilirsiniz.",
    "Ürün depo çıkışı ne demek?": "Ürün depo çıkışınız sağlanmış, kargo firmasına teslim edilmiştir. Kargo firması sisteme kayıt yaptığında size sms ile bilgisi gelecektir.",
    "Çalışma saatleriniz nedir?": "Hafta İçi 09:00-19:00 ve Cumartesi 09:00-15:00 , Facebook & Instagram üzerinden 7/24 hizmet alabilirsiniz.",
    "Değişim siparişlerinde kargo ücreti var mı?": "Değişim siparişleri kargo ücretsizdir. Çekinizle aynı veya daha alt tutarda bir ürün seçtiğinizde tarafınızdan kargo ücreti ve ek hizmet bedeli alınmıyor. Ancak daha yüksek tutarda bir ürün seçtiğinizde sadece 15,00 TL ek hizmet bedeli yansımaktadır.",
    "Bayram dönemi iade/değişim süresi nasıl işliyor?": "İade -Değişim süremiz 15 gündür. Bayram tatili iade değişim süresine dahil değildir.",
    "Sesli mesaj gönderebilir miyim?": "Sesli mesaj dinleyememekteyiz. Talebinizi yazılı olarak iletir misiniz?",
    "Siparişim dağıtıma çıkarıldı ne demek?": "Siparişiniz kargo firması tarafından dağıtıma çıkarılmıştır. Bugün teslimatı sağlanacaktır.",
    "Üye olmadan sipariş verebilir miyim?": "Sipariş işlemleriniz ve takip edebilmeniz için üye olmanız gerekmektedir. Üyelikler ücretsiz olup kampanyalardan haberdar olmanızı sağlar. Dilediğiniz zaman üyeliğinizi iptal edebilirsiniz.",
    "Kargom için tekrar teslimat talebinde bulunabilir miyim?": "Tekrar teslimat talebiniz kargo firmasına iletilmiştir. Kargo firmasının olumlu - olumsuz dönüşüne istinaden sipariş oluşturduğunuz numaradan sizi arayacağız.",
    "Whatsapp iletişim numaranız nedir?": "908508115252 numaralı whatsapp hattımızdan bizlerle iletişime geçebilirsiniz.",
    "Dünya geneline teslimat yapıyor musunuz? (İngilizce)": "We do deliver worldwide all abroad. Orders are delivered in 5-10 days after 24-48 hours of control.",
    "Sipariş verdikten sonra değişiklik yapabilir miyim? (İngilizce)": "Hello We cannot change the order after it is placed. We can either cancel the whole order or ship just as it is. If we cancel, your refund is made within 3-7 days.",
    "Toptan alımlarda indirim oranları nelerdir?": "5% for 5.000₺ - 10.000₺ 7% for 10.000₺ - 50.000₺ 10% for 50.000₺ - 100.000₺ 15% for 100.000₺ and above.",
    "Ürün ayırma işlemi yapıyor musunuz?": "Ürün ayırma işlemi yapamamaktayız. Kalite kontrol sürecinde mevcut stok durumları değişiklik gösterebilmektedir.",
    "Kullanıcı kaynaklı olmayan hasarlı ürünü gönderebilir miyim?": "Ürün etiketi üzerindeyse ve kullanıcı kaynaklı hasarı yoksa kalite kontrolde incelenmesi için tarafımıza gönderebilirsiniz. Kalite kontrol aşamasından sonra olumlu-olumsuz sizlere bilgi verilecektir.",
    "Komisyonlu satışınız var mı?": "Komisyonlu satışımız mevcut değildir. Satışlarımız tek noktada modaselvim.com da yapılmaktadır.",
    "Sepete özel indirim var mı?": "Bugüne özel ve sezon sonu ürünleri hariç tüm üründe sepete özel indirim mevcuttur. Ürünü sepete eklediğinizde indirimli fiyatını görüntüleyebilir ve siparişinizi oluşturabilirsiniz.",
    "Yurt dışı ve yurt içi satış fiyatları neden farklı?": "Yurt dışı satış fiyatımız ile yurt içi satış fiyatımız farklılık göstermektedir. Yurt dışı siparişlerinde ürün fiyatı; ülkeye, teslimat giderlerine ve gümrük bedeli dahil edilerek sistem tarafından hesaplanmaktadır.",
    "Hediye çeki nasıl kullanılır?": "Üyeliğinize, gönderdiğiniz ürün ücreti değerinde değişim çeki tanımlanmıştır. Dilediğiniz ürünü sepete ekleyip, sepet sayfasının alt kısmında bulunan değişim çekinizi aktifleştirerek siparişinizi tamamlayabilirsiniz.",
    "Yurt içi ödeme seçenekleri nelerdir?": "Yurtiçi alışverişlerinizde ; banka / kredi kartınızla, veya KAPIDA nakit/ kredi kartı, EFT&amp;Havale seçenekleri ile ödemenizi sağlayabilirsiniz.",
    "Kapıda kredi kartı ile ödeme seçeneği var mı?": "PTT kargo hariç kapıda kredi kartına tek çekim ile ödeme mevcuttur",
    "IBAN numaramı sosyal medyadan verebilir miyim?": "Bilgi güvenliği nedeniyle siparişe kayıtlı olmayan numara ve sosyal medya hesaplarından iban numarası alamıyoruz.",
    "Dekontumu nasıl alabilirim?": "Bilgi güvenliği nedeniyle dekontunuzu buradan iletemiyoruz. Dekontunuzu siparişinize veya üyeliğinize kayıtlı telefon numarasına mı yoksa mail adresinize mi göndermemizi istersiniz ?",
    "Değişim talebimi sosyal medyadan iletebilir miyim?": "Bilgi güvenliği nedeniyle siparişe kayıtlı olmayan numara ve sosyal medya hesaplarından değişim talebinizi alamıyoruz.",
    "MNG Kargo takip linki nedir?": "kargo takip numaranızla MNG kargo web sitesinden https://www.mngkargo.com.tr/ linke tıklayarak kargonuzun nerede olduğunu görebilir, bulunduğu şube ile iletişime geçebilirsiniz.",
    "Geçmiş sezon ürünlerini bulabilir miyim?": "Merhaba, İlettiğiniz ürün geçmiş sezona aittir, stoklarımızda mevcut değildir. https://www.modaselvim.com/ linkten yeni ürünlerimizi inceleyip siparişinizi oluşturabilirsiniz.",
    "Bayrama siparişim yetişir mi?": "Aras ve Mng kargoda en geç 13.06.2024 tarihine kadar kargoya verilen siparişler de yakın bölge (Marmara Bölgesi) bayramdan önce teslimat yapılacaktır. (Mobil alan hariç – Mobil alanlara bayramdan sonra teslimatı yapılacaktır.) Kargoist kargoda en geç 14.06.2024 tarihine kadar kargoya verilen siparişler de yakın bölge (Marmara Bölgesi) bayramdan önce teslimat yapılacaktır. PTT kargoda en geç 11.06.2024 tarihine kadar kargoya verilen siparişler de yakın bölge (Marmara Bölgesi) bayramdan önce teslimat yapılacaktır. (Köy dağıtımı tabii bekleyenler hariç – Köy ve kasabalara bayramdan sonra teslimatı yapılacaktır.)",
    "Bayramda kargo şubeleri açık mı?": "16 Haziran– 19 Haziran arası PTT , MNG, ARAS, KARGOİST kargo şubeler kapalı ve dağıtımlar yapılmayacaktır. 20.06.2024 tarihinde normal mesaisine başlayacaktır.",
    "Kargo paketim hasarlı geldiyse ne yapmalıyım?": "Kargo paketinde hasar varsa vakit kaybetmeden paket ile birlikte teslimat şubesine giderek paketi hasarlı teslim aldığınıza dair tutanak tutturmalısınız. Aksi takdirde kargo firması hasarı kabul etmemektedir.",
    "Kargo ücreti neden artıyor?": "Kargo ücreti , kargo firmaları artan taşıma maliyetlerine göre belirlenmektedir.",
    "Teslim alınan kargonun ücret iadesi yapılır mı?": "Kargo sizin tarafınıza teslim edildiğinde alınan kargo ücret iadesi yapılmamaktadır.",
    "EFT/Havale ile ödeme nasıl yapılır?": "Sayın ...... ...... numaralı ...... tutarındaki siparişinizin çıkış işleminin tamamlanması için İş Bankası, - Modaselvim Tekstil San. ve Tic A.Ş. IBAN: TR740006400000110571324118 Hesabına 48 saat içinde ödeme işlemi gerçekleştirmeniz beklenmektedir. Ödemesi tamamlanmayan siparişler sistem tarafından 48 saat sonra otomatik iptal edilmektedir. Not: Havale/ EFT işlemi yaparken açıklama kısmına TS ile başlayan sipariş numaranızı yazmanızı rica ederiz.",
    "Ohal ilan edilen iller için iade/değişim süresi nedir?": "Ohal ilan edilen illerimiz için yurtiçi siparişlerinde ürünü teslim aldığınız günden itibaren 3 ay içerisinde iade ve değişim yapılabilmektedir.",
    "Geniş kalıp ürünlerde hangi bedeni seçmeliyim?": "Ürün geniş kalıp olup bir beden küçük alınması önerilir.",
    "Kargoist takip linki nedir?": "Ürününüz Kargoist kargoda. kargo takip numaranızla Kargoist kargo web sitesinden https://kargoist.com/ linke tıklayarak kargonuzun nerede olduğunu görebilir, bulunduğu şube ile iletişime geçebilirsiniz.",
    "Hollanda için DHL global kargo takip linki nedir?": "DHL global kargo _________takip numaranız ile https://www.dhl.com/tr-tr/home/tracking.html linke tıklayarak kargo hareketlerini takip edebilirsiniz.",
    "İngiltere için DHL global kargo takip linki nedir?": "DHL global kargo _________takip numaranız ile https://www.evri.com/track-a-parcel linke tıklayarak kargo hareketlerini takip edebilirsiniz.",
        # ... diğer sık sorulan ve cevabı sabit soruları buraya ekleyebilirsiniz ...,
}

def normalize_for_fixed_lookup(text: str) -> str:
    """
    Kullanıcı girdisini FIXED_RESPONSES sözlüğünde arama yapmak için normalize eder.
    Küçük harfe çevirir, soru işaretini ve baştaki/sondaki boşlukları kaldırır.
    """
    if not isinstance(text, str): # Gelenin string olup olmadığını kontrol et
        return ""
    return text.lower().replace('?', '').replace('.', '').strip()

# Normalize edilmiş anahtarlarla FIXED_RESPONSES sözlüğünü oluşturan fonksiyon
def _initialize_normalized_fixed_responses() -> dict:
    normalized_responses = {}
    for question, answer in RAW_FIXED_RESPONSES_DATA.items():
        normalized_key = normalize_for_fixed_lookup(question)
        normalized_responses[normalized_key] = answer
    return normalized_responses

# Asıl kullanılacak, normalize edilmiş FIXED_RESPONSES sözlüğü
# Bu, modül ilk import edildiğinde bir kere çalışır ve sözlüğü oluşturur.
FIXED_RESPONSES = _initialize_normalized_fixed_responses()

def get_fixed_response(user_message: str) -> str | None:
    """
    Verilen kullanıcı mesajı için sabit bir cevap olup olmadığını kontrol eder.
    Varsa cevabı, yoksa None döndürür.
    """
    normalized_message = normalize_for_fixed_lookup(user_message)
    return FIXED_RESPONSES.get(normalized_message)


print(f"Fixed Responses Modülü: {len(FIXED_RESPONSES)} adet normalize edilmiş sabit cevap yüklendi.")
if len(RAW_FIXED_RESPONSES_DATA) != len(FIXED_RESPONSES):
    print(f"UYARI: RAW_FIXED_RESPONSES_DATA ({len(RAW_FIXED_RESPONSES_DATA)} adet) ile normalize edilmiş FIXED_RESPONSES ({len(FIXED_RESPONSES)} adet) arasında eleman sayısı farkı var. Bu, normalize edilmiş bazı anahtarların çakıştığı anlamına gelebilir!")