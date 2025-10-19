import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr
import ast
import re

# ============================
# 📂 1. Veri Yükleme ve Hazırlama
# ============================

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

def parse_json_like(s):
    try:
        return ast.literal_eval(s) if isinstance(s, str) else []
    except Exception:
        return []

# JSON benzeri alanları parse et
movies['genres']   = movies['genres'].apply(parse_json_like)
movies['keywords'] = movies['keywords'].apply(parse_json_like)
credits['cast']    = credits['cast'].apply(parse_json_like)
credits['crew']    = credits['crew'].apply(parse_json_like)

# Birleştir
df = movies.merge(credits, left_on='id', right_on='movie_id', how='left')

# Title kolonunu garantiye al
if 'title' not in df.columns:
    if 'title_x' in df.columns:
        df['title'] = df['title_x']
    elif 'title_y' in df.columns:
        df['title'] = df['title_y']

# Güvenlik: temel sütun tipleri
for col in ['vote_average', 'vote_count', 'popularity', 'runtime']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Yardımcı çıkarımlar
def get_director(crew):
    for c in crew or []:
        if isinstance(c, dict) and c.get('job') == 'Director':
            return c.get('name')
    return None

def get_writers(crew):
    return [c.get('name') for c in (crew or []) if isinstance(c, dict) and c.get('department') == 'Writing'][:3]

def top_cast(cast, k=5):
    return [c.get('name') for c in (cast or []) if isinstance(c, dict)][:k]

df['director']      = df['crew'].apply(get_director)
df['writers']       = df['crew'].apply(get_writers)
df['top_cast']      = df['cast'].apply(top_cast)
df['genre_names']   = df['genres'].apply(lambda xs: [x.get('name') for x in xs] if isinstance(xs, list) else [])
df['keyword_names'] = df['keywords'].apply(lambda xs: [x.get('name') for x in xs] if isinstance(xs, list) else [])

def build_doc(row):
    parts = []
    ov = row.get('overview')
    if isinstance(ov, str) and ov.strip():
        parts.append(f"🎬 Özet: {ov}")
    if row.get('genre_names'):
        parts.append(f"📁 Türler: {', '.join(row['genre_names'])}")
    if row.get('keyword_names'):
        parts.append(f"✨ Anahtar Kelimeler: {', '.join(row['keyword_names'])}")
    if row.get('director'):
        parts.append(f"🎥 Yönetmen: {row['director']}")
    if row.get('writers'):
        parts.append(f"✍️ Yazarlar: {', '.join(row['writers'])}")
    if row.get('top_cast'):
        parts.append(f"⭐ Oyuncular: {', '.join(row['top_cast'])}")
    return "\n".join(parts)

df['doc']  = df.apply(build_doc, axis=1)
df['year'] = df['release_date'].fillna('').astype(str).str[:4]

keep = ['id','title','year','vote_average','vote_count','popularity','runtime','genre_names','top_cast','director','doc']
df_clean = df[keep].copy()
df_clean = df_clean[df_clean['doc'].astype(str).str.len() > 0].reset_index(drop=True)

# Hız için bazı ön-hesaplamalar
df_clean['genres_lower'] = df_clean['genre_names'].apply(lambda gs: [g.lower() for g in gs])

# ============================
# 🧠 2. Embedding ve FAISS Index
# ============================

MODEL = "intfloat/multilingual-e5-small"
st = SentenceTransformer(MODEL)

def to_passage(x: str) -> str:
    return "passage: " + x

texts = df_clean['doc'].tolist()
BATCH = 128
embs = []
for i in range(0, len(texts), BATCH):
    batch = [to_passage(t) for t in texts[i:i+BATCH]]
    vecs = st.encode(batch, normalize_embeddings=True, convert_to_numpy=True)
    embs.append(vecs)

embs = np.vstack(embs).astype('float32')
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

# ============================
# 🧰 3. Yardımcılar (tema + tür algılayıcıları)
# ============================

# Tematik kelime havuzları (belirgin filtre için)
THEME_KEYWORDS = {
    "detective": ["detective", "investigation", "investigates", "murder", "homicide", "crime", "police", "noir", "case", "whodunit", "serial killer"],
    "mystery": ["mystery", "secret", "secrets", "disappearance", "missing", "unsolved", "twist", "investigation"],
    "slasher": ["slasher", "killer", "serial killer", "stab", "blood", "gore", "chainsaw", "mask"],
    "space": ["space", "astronaut", "spaceship", "galaxy", "planet", "cosmos", "universe", "interstellar", "mission"],
    "post-apocalyptic": ["apocalypse", "post-apocalyptic", "dystopia", "wasteland", "outbreak", "survivor", "collapse", "after the fall"],
    "emotional": ["tearjerker", "tragic", "sad", "cry", "heartbreaking", "loss", "grief", "moving", "touching"],
    "heist": ["heist", "robbery", "bank", "vault", "crew", "plan", "score"],
    "coming-of-age": ["teen", "high school", "youth", "adolescence", "growing up", "coming of age"],
    "thriller": ["thriller", "tension", "cat and mouse", "chase", "suspense"],
    "romance": ["love", "relationship", "couple", "dating", "wedding", "marriage", "romantic", "heart"],
    "war": ["war", "soldier", "battle", "frontline", "world war", "veteran"],
    "sports": ["sport", "team", "coach", "tournament", "championship", "boxing", "baseball", "soccer", "basketball"],
    "biography": ["biopic", "based on true story", "biography", "true story", "real life"],
}

# Metinden tema geçer mi?
def doc_has_any(doc_text: str, keywords: list) -> bool:
    if not isinstance(doc_text, str):
        return False
    d = doc_text.lower()
    return any(kw in d for kw in keywords)

# Basit başlık yakalama (movies like ... için)
def find_title_mentioned(q_lower: str) -> str | None:
    # "movies like X", "similar to X", "X gibi", "X benzeri" vb.
    # Basitçe tüm film isimlerini dolaşıp substring arıyoruz (performans olarak yeterli).
    q = re.sub(r"[^a-z0-9\s:]", " ", q_lower)
    q = re.sub(r"\s+", " ", q).strip()
    for title in df_clean['title'].astype(str).str.lower().tolist():
        t = re.sub(r"[^a-z0-9\s:]", " ", title)
        t = re.sub(r"\s+", " ", t).strip()
        # Çok kısa (<=2 char) başlıklar ve aşırı generic olanları atla
        if len(t) <= 2:
            continue
        if t in q or all(tok in q for tok in t.split()[:2]):
            return title
    return None

# ============================
# 🔍 4. Film Öneri Fonksiyonu (GELİŞMİŞ)
# ============================

def suggest_movies(query, topn=5, min_rating=6.5, genre=None):
    q_lower = (query or "").lower().strip()
    topn = int(max(1, min(int(topn), 20)))  # güvenli aralık
    expanded_query = query or ""

    # --- Sorgu tipi belirleme
    similarity_triggers = ["movies like", "similar to", "like ", "benzeri", "gibi", "if you liked", "similar movies to"]
    is_similarity_query = any(trigger in q_lower for trigger in similarity_triggers)

    # --- Benzer film modu: başlık yakala ve dokümanını sorguya ekle
    ref_title = None
    if is_similarity_query:
        ref_title = find_title_mentioned(q_lower)
        if ref_title:
            ref_row = df_clean[df_clean['title'].astype(str).str.lower() == ref_title].head(1)
            if len(ref_row):
                expanded_query += " " + ref_row.iloc[0]['doc']
        expanded_query += " similar style, tone, pacing, themes, and narrative elements"

    # --- Tema/Konu zenginleştirmeleri
    if any(word in q_lower for word in ["romantic", "romance"]):
        expanded_query += " movies about love, relationships, couples"
    if "comedy" in q_lower:
        expanded_query += " funny, humorous, light-hearted"
    if any(word in q_lower for word in ["detective", "mystery", "noir"]):
        expanded_query += " detective investigation crime mystery whodunit"
    if any(word in q_lower for word in ["sci-fi", "science fiction", "scifi", "space"]):
        expanded_query += " futuristic, space, technology, exploration, astronauts"
    if "slasher" in q_lower:
        expanded_query += " slasher killer gore serial killer"
    if "thriller" in q_lower:
        expanded_query += " thriller suspense tension"
    if "post-apocalyptic" in q_lower or "post apocalyptic" in q_lower or "apocalypse" in q_lower:
        expanded_query += " dystopia apocalypse survival"
    if "emotional" in q_lower:
        expanded_query += " emotional tearjerker tragic heartbreaking"
    if "heist" in q_lower:
        expanded_query += " heist robbery plan crew"
    if "war" in q_lower:
        expanded_query += " war soldier battle"
    if "sports" in q_lower:
        expanded_query += " sport team coach championship"
    if "biography" in q_lower or "biopic" in q_lower:
        expanded_query += " biography based on true story"

    # --- Embedding araması
    q_vec = st.encode(["query: " + expanded_query], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    D, I = index.search(q_vec, 200)  # daha geniş aday havuzu
    df_result = df_clean.iloc[I[0]].copy()
    df_result['similarity'] = D[0]

    # --- Temel filtreler
    if min_rating:
        try:
            min_rating_val = float(min_rating)
        except Exception:
            min_rating_val = 0.0
        df_result = df_result[df_result['vote_average'] >= min_rating_val]

    if genre:
        g = genre.lower().strip()
        if g:
            df_result = df_result[df_result['genres_lower'].apply(lambda gs: g in gs)]

    # --- Tür algılama: rom-com gibi ikili koşullar
    # Kullanıcının yazdığı ifadelerden tür çıkarımı
    wants_romance = any(w in q_lower for w in ["romantic", "romance"])
    wants_comedy  = "comedy" in q_lower
    wants_mystery = "mystery" in q_lower or "detective" in q_lower or "noir" in q_lower
    wants_thriller = "thriller" in q_lower
    wants_horror = "horror" in q_lower or "slasher" in q_lower
    wants_scifi = any(w in q_lower for w in ["sci-fi", "scifi", "science fiction", "space"])
    wants_space_only = "space" in q_lower
    wants_war = "war" in q_lower
    wants_animation = "animation" in q_lower
    wants_crime = "crime" in q_lower
    wants_drama = "drama" in q_lower
    wants_action = "action" in q_lower
    wants_adventure = "adventure" in q_lower
    wants_biography = "biography" in q_lower or "biopic" in q_lower
    wants_sports = "sports" in q_lower
    wants_post_apoc = ("post-apocalyptic" in q_lower) or ("post apocalyptic" in q_lower) or ("apocalypse" in q_lower)
    wants_emotional = "emotional" in q_lower

    # Zorunlu tür filtreleri (varsa)
    def has_genre(gs, name):
        return name in gs

    if wants_romance and wants_comedy:
        df_result = df_result[df_result['genres_lower'].apply(lambda gs: has_genre(gs, "romance") and has_genre(gs, "comedy"))]
    else:
        # tekil istekler
        if wants_romance:
            df_result = df_result[df_result['genres_lower'].apply(lambda gs: has_genre(gs, "romance"))]
        if wants_comedy:
            df_result = df_result[df_result['genres_lower'].apply(lambda gs: has_genre(gs, "comedy"))]

    if wants_mystery:
        df_result = df_result[df_result['genres_lower'].apply(lambda gs: has_genre(gs, "mystery") or has_genre(gs, "thriller"))]

    if wants_thriller:
        df_result = df_result[df_result['genres_lower'].apply(lambda gs: has_genre(gs, "thriller"))]

    if wants_horror:
        df_result = df_result[df_result['genres_lower'].apply(lambda gs: has_genre(gs, "horror"))]

    if wants_scifi:
        # Sci-Fi zorunlu, "space" istenmişse ayrıca tema filtresi
        df_result = df_result[df_result['genres_lower'].apply(lambda gs: has_genre(gs, "science fiction"))]
        if wants_space_only:
            df_result = df_result[df_result['doc'].apply(lambda x: doc_has_any(str(x), THEME_KEYWORDS["space"]))]

    if wants_war:
        df_result = df_result[df_result['genres_lower'].apply(lambda gs: has_genre(gs, "war"))]

    if wants_animation:
        df_result = df_result[df_result['genres_lower'].apply(lambda gs: has_genre(gs, "animation"))]

    if wants_crime:
        df_result = df_result[df_result['genres_lower'].apply(lambda gs: has_genre(gs, "crime"))]

    if wants_action:
        df_result = df_result[df_result['genres_lower'].apply(lambda gs: has_genre(gs, "action"))]

    if wants_adventure:
        df_result = df_result[df_result['genres_lower'].apply(lambda gs: has_genre(gs, "adventure"))]

    if wants_biography:
        df_result = df_result[df_result['genres_lower'].apply(lambda gs: has_genre(gs, "history") or has_genre(gs, "biography") if "biography" in gs else True)]
        # ayrıca tema ile daralt
        df_result = df_result[df_result['doc'].apply(lambda x: doc_has_any(str(x), THEME_KEYWORDS["biography"]))]

    if wants_post_apoc:
        df_result = df_result[df_result['doc'].apply(lambda x: doc_has_any(str(x), THEME_KEYWORDS["post-apocalyptic"]))]

    if wants_emotional:
        df_result = df_result[df_result['doc'].apply(lambda x: doc_has_any(str(x), THEME_KEYWORDS["emotional"]))]

    # Ek tematik sıkılaştırma (dedektif, slasher vb.)
    if "detective" in q_lower:
        df_result = df_result[df_result['doc'].apply(lambda x: doc_has_any(str(x), THEME_KEYWORDS["detective"]))]
    if "slasher" in q_lower:
        df_result = df_result[df_result['doc'].apply(lambda x: doc_has_any(str(x), THEME_KEYWORDS["slasher"]))]

    # Boş kalırsa erken dönüş
    if df_result.empty:
        return f"🎯 Sorgu: {query}\nSonuç bulunamadı."

    # --- Bonuslar ve nihai skor
    # Genre match bonus (kullanıcının istediği tür(ler)le güçlü uyum)
    def romcom_bonus(gs):
        return 1 if (wants_romance and wants_comedy and has_genre(gs, "romance") and has_genre(gs, "comedy")) else 0

    def scifi_space_bonus(row):
        gs = row['genres_lower']
        if wants_scifi and has_genre(gs, "science fiction"):
            if wants_space_only and doc_has_any(str(row['doc']), THEME_KEYWORDS["space"]):
                return 1
            return 0.5
        return 0

    def mystery_bonus(gs):
        return 1 if (wants_mystery and (has_genre(gs, "mystery") or has_genre(gs, "thriller"))) else 0

    def horror_bonus(gs):
        return 1 if (wants_horror and has_genre(gs, "horror")) else 0

    def detective_bonus(doc):
        return 1 if ("detective" in q_lower and doc_has_any(str(doc), THEME_KEYWORDS["detective"])) else 0

    def emotional_bonus(doc):
        return 1 if (wants_emotional and doc_has_any(str(doc), THEME_KEYWORDS["emotional"])) else 0

    # Popülerlik normalizasyonu güvenli
    pop_max = float(df_result['popularity'].max()) if 'popularity' in df_result.columns and len(df_result) else 0.0
    pop_norm = (df_result['popularity'] / pop_max) if pop_max > 0 else 0.0

    # Bonusları hesapla
    df_result['genre_match_bonus'] = df_result['genres_lower'].apply(romcom_bonus) \
                                   + df_result['genres_lower'].apply(mystery_bonus) \
                                   + df_result['genres_lower'].apply(horror_bonus)
    df_result['theme_bonus'] = df_result.apply(scifi_space_bonus, axis=1) \
                              + df_result['doc'].apply(detective_bonus) \
                              + df_result['doc'].apply(emotional_bonus)

    # Benzerliği, puanı, popülerliği ve bonusları harmanla
    df_result['score'] = (
        0.55 * df_result['similarity'] +
        0.28 * (df_result['vote_average'] / 10.0) +
        0.12 * (pop_norm if isinstance(pop_norm, pd.Series) else 0) +
        0.05 * (df_result['genre_match_bonus'] + df_result['theme_bonus'])
    )

    # "movies like ..." ise aynı filmi ilk sıradan hariç tut
    if ref_title:
        df_result = df_result[df_result['title'].astype(str).str.lower() != ref_title]

    # Nihai sonuçlar
    df_result = df_result.sort_values('score', ascending=False).head(topn)

    out = [f"🎯 Sorgu: {query}\n"]
    for _, row in df_result.iterrows():
        genres = ", ".join(row['genre_names']) if isinstance(row['genre_names'], list) else ""
        desc = str(row['doc']).split("\n")[0].replace("🎬 Özet: ", "")
        desc = (desc[:220] + "...") if len(desc) > 220 else desc
        out.append(f"{len(out)}. {row['title']} ({row['year']}) — ⭐ {row['vote_average']:.1f}\n   🎭 {genres}\n   📖 {desc}\n")

    return "\n".join(out) if len(out) > 1 else f"🎯 Sorgu: {query}\nSonuç bulunamadı."

# ============================
# 🌐 5. Gradio Arayüzü
# ============================

def recommend_interface(query, genre, min_rating, topn):
    genre = (genre or "").strip() or None
    return suggest_movies(query, topn=int(topn), min_rating=float(min_rating), genre=genre)

demo = gr.Interface(
    fn=recommend_interface,
    inputs=[
        gr.Textbox(label="🎬 Film Araması", placeholder="ör: romantic comedy, detective thriller, space movies, movies like Interstellar"),
        gr.Textbox(label="🎭 Tür (opsiyonel)", placeholder="ör: Animation, Drama, Horror"),
        gr.Slider(0, 10, value=6.5, step=0.1, label="⭐ Minimum Puan"),
        gr.Slider(1, 20, value=5, step=1, label="📊 Kaç film?")
    ],
    outputs=gr.Markdown(label="🎞️ Önerilen Filmler"),
    title="🎥 RAG Film Öneri Sistemi (Gelişmiş)",
    description="E5 embedding + FAISS ile semantik arama. Tür/tema algılama, 'movies like …' benzerlik modu ve bonuslu skorlama."
)

if __name__ == "__main__":
    demo.launch()
