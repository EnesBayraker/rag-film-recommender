🎥 RAG Tabanlı Film Öneri Sistemi
📌 Projenin Amacı

Bu proje, kullanıcının doğal dilde yazdığı sorgulara göre en uygun filmleri öneren bir RAG (Retrieval-Augmented Generation) tabanlı öneri motorudur.
Klasik anahtar kelime eşleşmesinin ötesine geçerek film özetleri, türler, temalar ve anlatı unsurlarını semantik olarak analiz eder ve benzerlik tabanlı öneriler sunar.

📊 Veri Seti Hakkında

Kaynak: TMDB 5000 Movies Dataset (Kaggle)

İçerik: Film adı, çıkış yılı, özet, tür, anahtar kelimeler, yönetmen/yazar/oyuncu bilgileri, IMDb puanı ve popülerlik

Hazırlık:

JSON benzeri alanlar ayrıştırıldı (genres, keywords, cast, crew)

Yönetmen, yazarlar ve başlıca oyuncular çıkarıldı

Her film için açıklama, tür, anahtar kelimeler ve kadro bilgilerinden oluşan tek bir doküman (doc) üretildi

Bu dokümanlar embedding modeli ile vektörleştirildi ve FAISS ile arama yapılabilir hale getirildi

Veri seti repoya eklenmemiştir. Kod çalıştırıldığında veri otomatik olarak işlenir.

🧠 Kullanılan Yöntemler

Embedding: intfloat/multilingual-e5-small (Sentence Transformers)

Vektör Arama: FAISS (Inner Product Similarity)

Query İşleme: Sorgu tipi tespiti, otomatik query expansion, tematik kelime analizi

Skorlama: Semantik benzerlik + IMDb puanı + popülerlik + tür/tema uyumu

Arayüz: Gradio

🧪 Elde Edilen Sonuçlar

“movies like Interstellar” tipi sorgularda atmosfer, tema ve anlatı bakımından benzer filmler önerir.

“romantic comedy”, “detective thriller”, “space movies” gibi sorgularda tür & tema filtreleri ile isabet oranı artırılmıştır.

Bazı durumlarda hâlâ alakasız sonuçlar dönebilir, bu sistemin bilinen sınırlamasıdır ve gelecekte geliştirilecektir.

⚙️ Çalıştırma Kılavuzu (Önerilen: app.py – Hugging Face Versiyonu)
🔹 1) Sanal ortam oluşturma ve bağımlılıkların kurulumu
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
pip install -r requirements.txt

🔹 2) Uygulamayı çalıştırma
python app.py


Bu komutla Gradio arayüzü başlatılır ve tarayıcıda uygulama açılır. Buradan doğal dil sorguları ile film önerileri alabilirsiniz.

🔹 3) (Opsiyonel) Notebook sürümü

film_rag(1).ipynb dosyası, geliştirme sürecini ve veri hazırlama adımlarını detaylı olarak göstermektedir. Çalıştırılabilir olmak zorunda değildir, ancak incelenebilir olması beklenmektedir.

🧱 Çözüm Mimarisi
📁 Veri (TMDB / Kaggle)
      ↓
🧹 Ön İşleme & Doküman oluşturma
      ↓
🔎 Embedding (E5-small)
      ↓
📚 FAISS Vektör İndeksi
      ↓
🤖 Query Analizi & Expansion
      ↓
⚖️ Tür/tema filtreleme + Skor hesaplama
      ↓
🌐 Gradio Arayüzü

🌐 Web Arayüzü & Ürün Kılavuzu

Canlı demo (Hugging Face Spaces):
👉 🎬 Film Öneri Sistemi - Live Demo

Kullanım örnekleri:

romantic comedy

detective thriller

movies like Interstellar

post-apocalyptic emotional dramas

💡 İpucu: İngilizce sorgular daha isabetli sonuçlar verir. Türkçe sorgular da desteklenir ancak embedding modeli İngilizce’de çok daha başarılıdır.

📉 Sınırlamalar ve Geliştirme Planı

Bazı sorgularda hâlâ alakasız sonuçlar görülebilir.

Daha iyi sonuçlar için daha büyük embedding modelleri ve reranker katmanı planlanmaktadır.

Kullanıcı etkileşimine göre kişiselleştirme özelliği gelecekte eklenecektir.

🔄 Hugging Face’e Geçerken Yapılan Küçük Değişiklikler

Kaggle’dan veri çekme işlemi yerine CSV dosyaları doğrudan repoya eklendi.

Sorgu genişletme ve filtreleme mantığı iyileştirildi.

Kod yapısı tek dosyada (app.py) çalışacak şekilde sadeleştirildi.

🔗 Bağlantılar

💻 Canlı Demo: https://huggingface.co/spaces/etherwa/film_rag

