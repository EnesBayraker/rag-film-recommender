# 🎥 RAG Tabanlı Film Öneri Sistemi

> ℹ️ **Proje Hakkında:** Bu çalışma, **Akbank Gen-AI Bootcamp'i** kapsamında, **Global AI Hub** eğitmenleri tarafından verilen bir proje görevi olarak geliştirilmiştir. Bir konsept kanıtlama (Proof of Concept) çalışması olup gelecekteki geliştirmelere açıktır.

Bu proje, RAG (Retrieval-Augmented Generation) tekniğini kullanarak film önerileri sunan bir denemedir. Projenin amacı, filmlerin metinsel verilerini (özet, tema, tür vb.) semantik olarak analiz edip benzerlik tabanlı öneriler sunmaktır.

Ancak, sistem henüz karmaşık doğal dil sorgularını yorumlamada başarılı değildir. En isabetli sonuçlar için, **`"romantic comedy"`**, **`"small town mystery"`** veya **`"space horror"`** gibi belirli tür veya temaları ifade eden kısa anahtar kelimelerle kullanılması tavsiye edilir.

-----

## 📊 Veri Seti Hakkında

  - **Kaynak:** [TMDB 5000 Movies Dataset (Kaggle)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
  - **İçerik:** Film adı, çıkış yılı, özet, tür, anahtar kelimeler, yönetmen/yazar/oyuncu bilgileri, IMDb puanı ve popülerlik.
  - **Hazırlık Aşamaları:**
      - JSON benzeri alanlar (`genres`, `keywords`, `cast`, `crew`) ayrıştırıldı.
      - Yönetmen, yazarlar ve başlıca oyuncular çıkarıldı.
      - Her film için `açıklama`, `tür`, `anahtar kelimeler` ve `kadro` bilgilerinden oluşan tek bir doküman (`doc`) üretildi.
      - Bu dokümanlar, embedding modeli ile vektörleştirildi ve **FAISS** ile arama yapılabilir hale getirildi.

> **Not:** Veri seti repoya eklenmemiştir. Kod çalıştırıldığında veri otomatik olarak işlenir.

## 🧠 Kullanılan Yöntemler

  - **Embedding:** `intfloat/multilingual-e5-small` (Sentence Transformers)
  - **Vektör Arama:** FAISS (Inner Product Similarity)
  - **Query İşleme:** Sorgu tipi tespiti, otomatik query expansion, tematik kelime analizi.
  - **Skorlama:** Semantik benzerlik + IMDb puanı + popülerlik + tür/tema uyumu.
  - **Arayüz:** Gradio

## 🧪 Elde Edilen Sonuçlar

  - `“movies like Interstellar”` tipi sorgularda atmosfer, tema ve anlatı bakımından benzer filmler önerir.
  - `“romantic comedy”`, `“detective thriller”`, `“space movies”` gibi sorgularda tür & tema filtreleri ile isabet oranı artırılmıştır.
  - Bazı durumlarda hâlâ alakasız sonuçlar dönebilir, bu sistemin bilinen sınırlamasıdır ve gelecekte geliştirilecektir.

## ⚙️ Çalıştırma Kılavuzu

Önerilen çalıştırma yöntemi `app.py` dosyasıdır.

#### 1\. Sanal Ortam Oluşturma ve Bağımlılıkların Kurulumu

```sh
# Sanal ortam oluşturun
python -m venv venv

# Sanal ortamı aktive edin
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

#### 2\. Uygulamayı Çalıştırma

```sh
python app.py
```

Bu komutla Gradio arayüzü başlatılır ve tarayıcıda uygulama açılır.

#### 3\. (Opsiyonel) Notebook Sürümü

`film_rag(1).ipynb` dosyası, geliştirme sürecini ve veri hazırlama adımlarını detaylı olarak göstermektedir. Çalıştırılabilir olmak zorunda değildir, ancak incelenebilir olması beklenmektedir.

## 🧱 Çözüm Mimarisi

📁 **Veri Toplama** → 🧹 **Veri Ön İşleme** → 🧠 **Embedding** → 📚 **FAISS İndeksi** → 🔍 **Query İşleme** → ⚖️ **Skorlama** → 🌐 **Arayüz**

-----

## 💡 İpuçları ve Bilinen Sınırlamalar

  - **İngilizce Sorgular:** İngilizce sorgular daha isabetli sonuçlar verir. Türkçe sorgular da desteklenir ancak embedding modeli İngilizce'de çok daha başarılıdır.
  - **Alakasız Sonuçlar:** Bazı sorgularda hâlâ alakasız sonuçlar görülebilir.
  - **Geliştirme Planı:** Daha iyi sonuçlar için daha büyük embedding modelleri ve bir `reranker` katmanı eklenmesi planlanmaktadır.

## 🔄 Hugging Face’e Geçerken Yapılan Değişiklikler

  - Kaggle’dan veri çekme işlemi yerine CSV dosyaları doğrudan repoya eklendi.
  - Sorgu genişletme ve filtreleme mantığı iyileştirildi.
  - Kod yapısı tek dosyada (`app.py`) çalışacak şekilde sadeleştirildi.

## 🔗 Bağlantılar

  - **💻 Canlı Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/etherwa/film_rag)
