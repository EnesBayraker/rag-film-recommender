# 🎥 RAG Tabanlı Film Öneri Sistemi

> ℹ️ **Proje Hakkında:** Bu çalışma, **Akbank Generative AI Bootcamp'i** kapsamında, **Global AI Hub** eğitmenleri tarafından verilen bir proje görevi olarak geliştirilmiştir. Bir konsept kanıtlama (Proof of Concept) çalışması olup gelecekteki geliştirmelere açıktır.

Bu proje, RAG (Retrieval-Augmented Generation) tekniğini kullanarak film önerileri sunan bir denemedir. Projenin amacı, filmlerin metinsel verilerini (özet, tema, tür vb.) semantik olarak analiz edip benzerlik tabanlı öneriler sunmaktır.

Ancak, sistem henüz karmaşık doğal dil sorgularını yorumlamada başarılı değildir. En isabetli sonuçlar için, **`"romantic comedy"`**, **`"small town mystery"`** veya **`"space horror"`** gibi belirli tür veya temaları ifade eden kısa anahtar kelimelerle kullanılması tavsiye edilir.

---

## 🚀 Projeyi Deneyimleme ve İnceleme

Bu projenin nasıl çalıştığını görmek ve kodlarını incelemek için iki ana yol bulunmaktadır.

### 1. Canlı Demo (Önerilen Yöntem)
Projenin çalışan halini görmek ve test etmek için en kolay ve hızlı yol, Hugging Face üzerine yüklenmiş olan canlı demoyu kullanmaktır.

- **➡️ [Buradan Canlı Demoya Ulaşabilirsiniz](https://huggingface.co/spaces/etherwa/film_rag)**

### 2. Geliştirme Sürecini İnceleme (Jupyter Notebook)
Veri setinin nasıl hazırlandığını, modelin nasıl denendiğini ve projenin adımlarını detaylı olarak görmek isterseniz, repodaki `film_rag(1).ipynb` dosyasını inceleyebilirsiniz. Bu dosya, projenin "mutfağını" gösterir ve kodun Google Colab ortamında nasıl geliştirildiğini anlamak için faydalıdır.

### ⚠️ Önemli Not: Yerel Çalıştırma Hakkında
Bu repodaki `app.py` dosyası, büyük boyutları nedeniyle GitHub'a yüklenmemiş olan veri setlerine ihtiyaç duymaktadır. Bu yüzden, **repoyu indirip doğrudan `python app.py` komutuyla çalıştırmayı denemek hata verecektir.** Projenin çalıştırılması yalnızca yukarıda linki verilen Hugging Face'teki demo üzerinden mümkündür.

---

## 📊 Veri Seti Hakkında

- **Kaynak:** [TMDB 5000 Movies Dataset (Kaggle)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Hazırlık:** Her film için `açıklama`, `tür`, `anahtar kelimeler` ve `kadro` bilgilerinden oluşan tek bir metin dokümanı üretilmiş, bu dokümanlar vektörleştirilerek **FAISS** ile aranabilir hale getirilmiştir.


## 🧠 Proje Mimarisi

* **📁 Veri Toplama** – TMDB veri seti (Kaggle)
* **🧹 Veri Ön İşleme** – JSON ayrıştırma, `doc` oluşturma
* **🧠 Embedding** – Sentence Transformers (`intfloat/multilingual-e5-small`)
* **📚 FAISS** – Vektör arama indeksi oluşturma
* **🔍 Query İşleme** – Query expansion + filtreleme
* **⚖️ Skorlama** – Semantik benzerlik + IMDb puanı + popülerlik
* **🌐 Arayüz** – Gradio ile web tabanlı öneri sistemi




## 💡 İpuçları ve Bilinen Sınırlamalar

- **İngilizce Sorgular:** Embedding modelinin doğası gereği, İngilizce sorgular Türkçe sorgulara göre daha isabetli sonuçlar vermektedir.
- **Geliştirme Planı:** Gelecekte daha büyük embedding modelleri ve bir `reranker` katmanı eklenerek sistemin isabet oranı artırılabilir.

## 🔗 Bağlantılar

- **💻 Canlı Demo:** [https://huggingface.co/spaces/etherwa/film_rag](https://huggingface.co/spaces/etherwa/film_rag)

