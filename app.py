from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz
import pymysql
import os
import gc
from datetime import datetime
from dotenv import load_dotenv
import requests
import json
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
 
app = Flask(__name__)
CORS(app)  # izinkan request dari Laravel
 
# Cache untuk Top-10 realtime
_CACHE_REALTIME = {
    "global"    : {"data": None, "last_updated": 0},
    "indonesia" : {"data": None, "last_updated": 0},
}
_CACHE_METRICS_MULTI = {}
_CACHE_TTL = 300  

# Cache untuk MAE dan RMSE
_CACHE_METRICS = {
    "data"        : None,
    "last_updated": 0,
}
_METRICS_TTL = 86400  

# =============================================================================
# KONFIGURASI -
# =============================================================================
 
DB_CONFIG = {
    "host"     : "localhost",
    "user"     : "root",
    "password" : "root",           
    "database" : "rekomendasi_film",
    "charset"  : "utf8mb4",
}
 
# Path ke file model hasil dari Google Colab
# Salin file .npy dan .csv dari Google Drive ke folder ini
MODEL_DIR  = "./model"
OUTPUT_DIR = "./dataset/processed/ml1m"
 
# Konfigurasi model
DEFAULT_K      = 20    
DEFAULT_TOP_N  = 10     
LAMBDA         = 0.7    
 
 
# =============================================================================
# LOAD MODEL SAAT STARTUP
# =============================================================================
 
print("=" * 55)
print("  LOADING MODEL IBCF...")
print("=" * 55)
 
# Load similarity matrix (gunakan model terbaik = weighted/timestamp)
path_sim_weight = os.path.join(MODEL_DIR, "similarity_weighted.npy")
if os.path.exists(path_sim_weight):
    SIM_WEIGHTED = np.load(path_sim_weight, mmap_mode='r')
    print(f"  ✅ Similarity matrix dimuat: {SIM_WEIGHTED.shape}")
else:
    SIM_WEIGHTED = None
    print(f"  ⚠️  similarity_weighted.npy tidak ditemukan di {MODEL_DIR}")

path_sim_baseline = os.path.join(MODEL_DIR, "similarity_baseline.npy")
if os.path.exists(path_sim_baseline):
    SIM_BASELINE = np.load(path_sim_baseline, mmap_mode='r')
    print(f"  ✅ Similarity matrix dimuat: {SIM_BASELINE.shape}")
else:
    SIM_BASELINE = None
    print(f"  ⚠️  similarity_baseline.npy tidak ditemukan di {MODEL_DIR}")
 
# Load train data untuk rebuild sparse matrix
path_train = os.path.join(OUTPUT_DIR, "train_flask.csv")
if os.path.exists(path_train):
    df_train = pd.read_csv(path_train,
        usecols=["user_encoded", "movie_encoded", "rating"])
    N_USERS  = df_train["user_encoded"].max() + 1
    N_MOVIES = df_train["movie_encoded"].max() + 1
 
    MATRIX_USER_ITEM = csr_matrix(
        (df_train["rating"].values.astype(np.float32),
         (df_train["user_encoded"].values,
          df_train["movie_encoded"].values)),
        shape=(N_USERS, N_MOVIES),
        dtype=np.float32
    )
    print(f"  ✅ User-Item matrix: {MATRIX_USER_ITEM.shape}")
    del df_train
    gc.collect()
else:
    MATRIX_USER_ITEM = None
    N_USERS  = 0
    N_MOVIES = 0
    print(f"  ⚠️  train_flask.csv tidak ditemukan di {OUTPUT_DIR}")
 
# Load mapping & info film
path_map_movie = os.path.join(OUTPUT_DIR, "mapping_movie.csv")
path_map_user  = os.path.join(OUTPUT_DIR, "mapping_user.csv")


DF_MAP_MOVIE  = pd.read_csv(path_map_movie)  if os.path.exists(path_map_movie)  else None
DF_MAP_USER   = pd.read_csv(path_map_user)   if os.path.exists(path_map_user)   else None
 

print(f"  ✅ Mapping movie : {'Dimuat' if DF_MAP_MOVIE  is not None else 'Tidak ada'}")
print(f"  ✅ Mapping user  : {'Dimuat' if DF_MAP_USER   is not None else 'Tidak ada'}")

print("=" * 55)

path_sim_id_weighted = os.path.join(MODEL_DIR, "sim_indonesia_weighted.npy")
path_sim_id_baseline = os.path.join(MODEL_DIR, "sim_indonesia_baseline.npy")
path_matrix_id       = os.path.join(MODEL_DIR, "mat_user_item_indonesia.npz")
path_test_id         = os.path.join("dataset","processed", "indonesia", "test_indonesia.csv")

SIM_ID_WEIGHTED = np.load(path_sim_id_weighted) if os.path.exists(path_sim_id_weighted) else None
SIM_ID_BASELINE = np.load(path_sim_id_baseline) if os.path.exists(path_sim_id_baseline) else None
MATRIX_ID       = load_npz(path_matrix_id) if os.path.exists(path_matrix_id) else None
DF_TEST_INDO    = pd.read_csv(path_test_id) if os.path.exists(path_test_id) else None

N_USERS_ID  = MATRIX_ID.shape[0] if MATRIX_ID is not None else 0
N_MOVIES_ID = MATRIX_ID.shape[1] if MATRIX_ID is not None else 0

print(f"  ✅ Model Indonesia : {'Dimuat' if SIM_ID_WEIGHTED is not None else 'Tidak ada'}") 

print(f"  DEBUG cwd = {os.getcwd()}")
print(f"  DEBUG path_test_id = {path_test_id}")
print(f"  DEBUG exists = {os.path.exists(path_test_id)}")
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
 
def get_db():
    """Buat koneksi ke MySQL."""
    return pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
 
 
def get_user_encoded(user_id):
    """Konversi user_id asli → user_encoded."""
    if DF_MAP_USER is None:
        return None
    row = DF_MAP_USER[DF_MAP_USER["userId"] == user_id]
    return int(row["user_encoded"].values[0]) if len(row) > 0 else None
 
 
def get_movie_encoded(movie_id):
    """Konversi movie_id asli → movie_encoded."""
    if DF_MAP_MOVIE is None:
        return None
    row = DF_MAP_MOVIE[DF_MAP_MOVIE["movieId"] == movie_id]
    return int(row["movie_encoded"].values[0]) if len(row) > 0 else None
 
 
def prediksi_rating(user_encoded, movie_encoded, K=DEFAULT_K, model="weighted"):
    """Prediksi rating menggunakan model weighted atau baseline."""
    SIM_MATRIX = SIM_WEIGHTED if model == "weighted" else SIM_BASELINE

    if SIM_MATRIX is None or MATRIX_USER_ITEM is None:
        return None
    if user_encoded >= N_USERS or movie_encoded >= N_MOVIES:
        return None

    sim_scores   = SIM_MATRIX[movie_encoded]
    rated_movies = MATRIX_USER_ITEM[user_encoded].nonzero()[1]

    if len(rated_movies) == 0:
        return None

    sim_rated = sim_scores[rated_movies]

    if len(rated_movies) > K:
        top_k_idx    = np.argsort(sim_rated)[::-1][:K]
        rated_movies = rated_movies[top_k_idx]
        sim_rated    = sim_rated[top_k_idx]

    mask         = sim_rated > 0
    sim_rated    = sim_rated[mask]
    rated_movies = rated_movies[mask]

    if len(sim_rated) == 0:
        return None

    user_ratings = np.array(
        MATRIX_USER_ITEM[user_encoded, rated_movies].todense()
    ).flatten()

    pembilang = np.dot(sim_rated.astype(np.float32), user_ratings)
    penyebut  = np.sum(np.abs(sim_rated))

    if penyebut == 0:
        return None

    pred = float(pembilang / penyebut)
    return round(max(0.5, min(5.0, pred)), 2)
 
 
def format_film(row):
    """Format data film dari MySQL menjadi dict."""
    return {
        "id"          : row["id"],
        "title"       : row["title"],
        "genre"       : row["genre"],
        "year"        : row["year"],
        "poster"      : row["poster"] or "",
        "description" : row["description"] or "",
        "avg_rating"  : float(row["avg_rating"]) if row["avg_rating"] else 0.0,
        "total_rating": int(row["total_rating"]) if row["total_rating"] else 0,
    }

CACHE_PATH = os.path.join(MODEL_DIR, "..", "tmdb_cache.json")

def _load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache):
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

TMDB_CACHE = _load_cache()
print(f"  ✅ TMDb cache   : {len(TMDB_CACHE)} entri dimuat")


def ambil_detail_tmdb(title, year=None):
    """Cari film di TMDb berdasarkan judul, kembalikan poster_url & deskripsi.
    Hasil disimpan ke cache supaya tidak perlu request berulang."""
    cache_key = f"{title}|{year}"

    if cache_key in TMDB_CACHE:
        cached = TMDB_CACHE[cache_key]
        return cached.get("poster"), cached.get("description")

    if not TMDB_API_KEY:
        return None, None

    try:
        params = {"api_key": TMDB_API_KEY, "query": title, "language": "id-ID"}
        if year:
            params["year"] = int(year)

        resp = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=5)
        resp.raise_for_status()
        hasil_list = resp.json().get("results", [])

        if not hasil_list:
            TMDB_CACHE[cache_key] = {"poster": None, "description": None}
            _save_cache(TMDB_CACHE)
            return None, None

        film = hasil_list[0]
        poster_path = film.get("poster_path")
        poster_url  = f"{TMDB_IMG_BASE}{poster_path}" if poster_path else None
        deskripsi   = film.get("overview") or None

        TMDB_CACHE[cache_key] = {"poster": poster_url, "description": deskripsi}
        _save_cache(TMDB_CACHE)

        return poster_url, deskripsi

    except Exception:
        return None, None

def pastikan_film_ada(db, movie_id):
    """Cek apakah movie_id valid.
    Untuk film numerik (Global): pastikan ada di movies atau movies_ml1m.
    Untuk film Indonesia (M0xx): langsung izinkan karena tidak ada foreign key.
    """
    # Film Indonesia (string seperti M035) — langsung izinkan
    try:
        movie_id_int = int(movie_id)
        is_numeric   = True
    except (ValueError, TypeError):
        return True  # film Indonesia, tidak perlu cek lebih lanjut

    # Film Global (numerik) — pastikan ada di tabel movies
    cur = db.cursor()
    cur.execute("SELECT id FROM movies WHERE id = %s", (movie_id_int,))
    if cur.fetchone():
        return True

    # Cari di movies_ml1m dan auto-insert ke movies
    try:
        cur.execute(
            "SELECT id, title, genres, year FROM movies_ml1m WHERE id = %s",
            (movie_id_int,)
        )
        film = cur.fetchone()
        if film:
            title  = str(film["title"])
            genres = str(film["genres"]).replace("|", ", ") if film["genres"] else ""
            tahun  = film["year"]

            judul_bersih          = title.rsplit(" (", 1)[0] if title.endswith(")") else title
            poster_url, deskripsi = ambil_detail_tmdb(judul_bersih, tahun)

            cur.execute("""
                INSERT IGNORE INTO movies (id, title, genre, year, description, poster, sumber)
                VALUES (%s, %s, %s, %s, %s, %s, 'movielens')
            """, (movie_id_int, title, genres, tahun, deskripsi or '', poster_url or ''))
            db.commit()
            return True
    except Exception as e:
        print(f"  ⚠️ Gagal cari di movies_ml1m: {e}")

    return False  # film numerik tidak ditemukan di mana pun

    # ---------------------------------------------------------------
    # Sumber 2: cari di top10_films.csv (fallback)
    # ---------------------------------------------------------------
    if DF_TOP10 is not None:
        baris = DF_TOP10[DF_TOP10["movieId"] == movie_id]
        if len(baris) > 0:
            row    = baris.iloc[0]
            title  = str(row.get("title")) if pd.notna(row.get("title")) else f"Film #{movie_id}"
            genres = str(row.get("genres")).replace("|", ", ") if pd.notna(row.get("genres")) else ""
            tahun  = int(row.get("tahun_rilis")) if pd.notna(row.get("tahun_rilis")) else None

            judul_bersih          = title.rsplit(" (", 1)[0] if title.endswith(")") else title
            poster_url, deskripsi = ambil_detail_tmdb(judul_bersih, tahun)

            try:
                cur.execute("""
                    INSERT INTO movies (id, title, genre, year, description, poster, sumber)
                    VALUES (%s, %s, %s, %s, %s, %s, 'movielens')
                """, (movie_id, title, genres, tahun, deskripsi or '', poster_url or ''))
                db.commit()
                return True
            except Exception as e:
                print(f"  ⚠️ Gagal insert dari CSV: {e}")

    return False  # tidak ketemu di mana pun
 
def hitung_top10(region="global", limit=30):
    """Hitung Top-10 real-time berdasarkan time-decay weighting.
    region: 'global' (MovieLens 1M) atau 'indonesia'
    """

    # ---------------------------------------------------------------
    # Konfigurasi per region
    # ---------------------------------------------------------------
    if region == "indonesia":
        MIN_RATING = 10
    else:
        MIN_RATING = 20

    agg_key = "movie_id"

    # ---------------------------------------------------------------
    # Query data rating dari database
    # ---------------------------------------------------------------
    db  = get_db()
    cur = db.cursor()

    if region == "indonesia":
        cur.execute("""
            SELECT movie_id, movie_title AS title, genre, year,
                   rating, timestamp
            FROM ratings_dataset_indonesia
            WHERE timestamp IS NOT NULL

            UNION ALL

            SELECT ri.movie_id, ri.movie_title, ri.genre, ri.year,
                   r.rating, r.timestamp
            FROM ratings r
            JOIN ratings_dataset_indonesia ri
                ON CAST(r.movie_id AS CHAR) = ri.movie_id
            WHERE r.timestamp IS NOT NULL
        """)
        cols = ["movie_id", "title", "genre", "year", "rating", "timestamp"]

    else:
        cur.execute("""
            SELECT movie_id, rating, timestamp
            FROM ratings_ml1m
            WHERE timestamp IS NOT NULL

            UNION ALL

            SELECT movie_id, rating, timestamp
            FROM ratings
            WHERE timestamp IS NOT NULL
        """)
        cols = ["movie_id", "rating", "timestamp"]

    rows = cur.fetchall()
    db.close()

    print(f"  DEBUG region={region}, MIN_RATING={MIN_RATING}, total rows={len(rows)}")

    if not rows:
        return []

    # ---------------------------------------------------------------
    # Bangun DataFrame
    # ---------------------------------------------------------------
    df = pd.DataFrame(rows, columns=cols)
    df["rating"]    = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(float)

    # ---------------------------------------------------------------
    # Hitung time-decay weight
    # ---------------------------------------------------------------
    ts_max   = df["timestamp"].max()
    ts_min   = df["timestamp"].min()
    ts_range = ts_max - ts_min if ts_max != ts_min else 1

    df["delta_t_norm"]    = (ts_max - df["timestamp"]) / ts_range
    df["weight"]          = np.exp(-LAMBDA * df["delta_t_norm"])
    df["weighted_rating"] = df["rating"] * df["weight"]

    # ---------------------------------------------------------------
    # Agregasi per film
    # ---------------------------------------------------------------
    if region == "indonesia":
        info = df.groupby(agg_key).first()[
            ["title", "genre", "year"]
        ].reset_index()

    agg = df.groupby(agg_key).agg(
        avg_rating   = ("rating",          "mean"),
        total_rating = ("rating",          "count"),
        weighted_avg = ("weighted_rating", "mean"),
    ).reset_index()

    if region == "indonesia":
        agg = agg.merge(info, on=agg_key, how="left")

    agg = agg[agg["total_rating"] >= MIN_RATING].copy()

    print(f"  DEBUG agg setelah filter: {len(agg)} film")

    if agg.empty:
        return []

    # ---------------------------------------------------------------
    # Hitung skor akhir
    # ---------------------------------------------------------------
    max_weighted = agg["weighted_avg"].max()
    agg["skor"]  = (agg["weighted_avg"] / max_weighted * 10).round(4)

    top         = agg.nlargest(limit, "skor").reset_index(drop=True)
    top["rank"] = top.index + 1

    # ---------------------------------------------------------------
    # Bangun hasil akhir dengan poster & deskripsi dari TMDb
    # ---------------------------------------------------------------
    hasil = []

    if region == "global":
        # Ambil judul dari tabel movies_ml1m
        movie_ids = top["movie_id"].dropna().astype(int).tolist()
        film_map  = {}

        if movie_ids:
            try:
                db2  = get_db()
                cur2 = db2.cursor()
                placeholders = ",".join(["%s"] * len(movie_ids))
                cur2.execute(
                    f"SELECT id, title, genres, year FROM movies_ml1m WHERE id IN ({placeholders})",
                    movie_ids
                )
                film_map = {row["id"]: row for row in cur2.fetchall()}
                db2.close()
            except Exception as e:
                print(f"  ⚠️ Gagal ambil judul film: {e}")

        for _, row in top.iterrows():
            movie_id = int(row["movie_id"]) if pd.notna(row.get("movie_id")) else None
            film     = film_map.get(movie_id, {})

            title  = film.get("title", f"Film #{movie_id}")
            genres = str(film.get("genres", "")).replace("|", ", ") if film.get("genres") else "Tidak diketahui"
            tahun  = film.get("year")

            judul_bersih          = title.rsplit(" (", 1)[0] if title.endswith(")") else title
            poster_url, deskripsi = ambil_detail_tmdb(judul_bersih, tahun)

            hasil.append({
                "rank"        : int(row["rank"]),
                "movie_id"    : movie_id,
                "title"       : title,
                "genre"       : genres,
                "year"        : int(tahun) if tahun and pd.notna(tahun) else None,
                "avg_rating"  : round(float(row["avg_rating"]), 2),
                "weighted_avg": round(float(row["weighted_avg"]), 4),
                "total_rating": int(row["total_rating"]),
                "skor"        : round(float(row["skor"]), 4),
                "poster"      : poster_url or "",
                "description" : deskripsi or "Deskripsi tidak tersedia.",
            })

    else:
        # Indonesia — judul sudah ada di DataFrame
        for _, row in top.iterrows():
            title  = str(row["title"]) if pd.notna(row.get("title")) else str(row["movie_id"])
            genres = str(row["genre"]).replace("|", ", ") if pd.notna(row.get("genre")) else "Tidak diketahui"
            tahun  = int(row["year"]) if pd.notna(row.get("year")) else None

            poster_url, deskripsi = ambil_detail_tmdb(title, tahun)

            hasil.append({
                "rank"        : int(row["rank"]),
                "movie_id"    : str(row["movie_id"]),
                "title"       : title,
                "genre"       : genres,
                "year"        : tahun,
                "avg_rating"  : round(float(row["avg_rating"]), 2),
                "weighted_avg": round(float(row["weighted_avg"]), 4),
                "total_rating": int(row["total_rating"]),
                "skor"        : round(float(row["skor"]), 4),
                "poster"      : poster_url or "",
                "description" : deskripsi or "Deskripsi tidak tersedia.",
            })

    return hasil


def hitung_mae_rmse(sample_size=None):
    """Hitung MAE dan RMSE dari SELURUH test set (bukan sample)."""
    db  = get_db()
    cur = db.cursor()

    cur.execute("""
        SELECT user_encoded, movie_encoded, rating
        FROM ratings_test_ml1m
        WHERE user_encoded IS NOT NULL
          AND movie_encoded IS NOT NULL
          AND user_encoded  < %s
          AND movie_encoded < %s
    """, (N_USERS, N_MOVIES))
    rows = cur.fetchall()
    db.close()

    if not rows:
        return None

    y_true = []
    y_pred = []

    for row in rows:
        pred = prediksi_rating(
            int(row["user_encoded"]),
            int(row["movie_encoded"]),
            K=DEFAULT_K
        )
        if pred is not None:
            y_true.append(float(row["rating"]))
            y_pred.append(pred)

    if len(y_true) < 10:
        return None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "mae"        : round(mae,  4),
        "rmse"       : round(rmse, 4),
        "sample_size": len(y_true),
        "total_test_rows": len(rows),
        "model"      : "IBCF + Time-Decay Timestamp (λ=0.7)",
        "k"          : DEFAULT_K,
    }

DF_MAP_USER_ID  = pd.read_csv("dataset/processed/indonesia/mapping_user_indo.csv")
DF_MAP_MOVIE_ID = pd.read_csv("dataset/processed/indonesia/mapping_movie_indo.csv")

def get_user_encoded_indonesia(user_id):
    row = DF_MAP_USER_ID[DF_MAP_USER_ID["user_id"] == user_id]
    return int(row["user_encoded"].values[0]) if len(row) > 0 else None

def get_movie_encoded_indonesia(movie_id):
    row = DF_MAP_MOVIE_ID[DF_MAP_MOVIE_ID["movie_id"] == movie_id]
    return int(row["movie_encoded"].values[0]) if len(row) > 0 else None

def hitung_mae_rmse_production(region="global"):
    """MAE/RMSE terhadap rating ASLI dari pengguna website (dinamis)."""
    db  = get_db()
    cur = db.cursor()
    cur.execute("SELECT user_id, movie_id, rating FROM ratings")
    rows = cur.fetchall()
    db.close()

    if not rows:
        return None

    y_true = []
    y_pred = []

    for row in rows:
        movie_id_raw = str(row["movie_id"])
        is_global    = movie_id_raw.isdigit()

        if region == "global" and not is_global:
            continue
        if region == "indonesia" and is_global:
            continue

        if region == "global":
            user_encoded  = get_user_encoded(row["user_id"])
            movie_encoded = get_movie_encoded(int(movie_id_raw))
            if user_encoded is None or movie_encoded is None:
                continue
            pred = prediksi_rating(user_encoded, movie_encoded, K=DEFAULT_K)
        else:
            # Untuk Indonesia: perlu mapping user_id & movie_id khusus dataset Indonesia
            user_encoded  = get_user_encoded_indonesia(row["user_id"])
            movie_encoded = get_movie_encoded_indonesia(movie_id_raw)
            if user_encoded is None or movie_encoded is None:
                continue
            pred = prediksi_rating_indonesia(user_encoded, movie_encoded, K=DEFAULT_K)

        if pred is not None:
            y_true.append(float(row["rating"]))
            y_pred.append(pred)

    if len(y_true) < 5:
        return None

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "mae": round(mae, 4), "rmse": round(rmse, 4),
        "sample_size": len(y_true), "total_rating_pengguna": len(rows),
        "model": "IBCF + Time-Decay Timestamp (λ=0.7)", "k": DEFAULT_K,
        "sumber": "rating_pengguna_realtime", "region": region,
    }

def prediksi_rating_indonesia(user_encoded, movie_encoded, K=DEFAULT_K, model="weighted"):
    SIM_MATRIX = SIM_ID_WEIGHTED if model == "weighted" else SIM_ID_BASELINE
    if SIM_MATRIX is None or MATRIX_ID is None:
        return None
    if user_encoded >= N_USERS_ID or movie_encoded >= N_MOVIES_ID:
        return None
    sim_scores   = SIM_MATRIX[movie_encoded]
    rated_movies = MATRIX_ID[user_encoded].nonzero()[1]
    if len(rated_movies) == 0:
        return None
    sim_rated = sim_scores[rated_movies]
    if len(rated_movies) > K:
        top_k_idx    = np.argsort(sim_rated)[::-1][:K]
        rated_movies = rated_movies[top_k_idx]
        sim_rated    = sim_rated[top_k_idx]
    mask = sim_rated > 0
    sim_rated, rated_movies = sim_rated[mask], rated_movies[mask]
    if len(sim_rated) == 0:
        return None
    user_ratings = np.array(MATRIX_ID[user_encoded, rated_movies].todense()).flatten()
    pembilang = np.dot(sim_rated.astype(np.float32), user_ratings)
    penyebut  = np.sum(np.abs(sim_rated))
    if penyebut == 0:
        return None
    pred = float(pembilang / penyebut)
    return round(max(0.5, min(5.0, pred)), 2)


def hitung_mae_rmse_indonesia():
    if DF_TEST_INDO is None:
        return None
    y_true, y_pred = [], []
    for _, row in DF_TEST_INDO.iterrows():
        pred = prediksi_rating_indonesia(int(row["user_encoded"]), int(row["movie_encoded"]), K=DEFAULT_K)
        if pred is not None:
            y_true.append(float(row["rating"]))
            y_pred.append(pred)
    if len(y_true) < 10:
        return None
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {
        "mae": round(mae, 4), "rmse": round(rmse, 4),
        "sample_size": len(y_true), "total_test_rows": len(DF_TEST_INDO),
        "model": "IBCF + Time-Decay Timestamp (λ=0.7)", "k": DEFAULT_K, "region": "indonesia",
    }

# =============================================================================
# ENDPOINT 1 — CEK STATUS API
# GET /api/status
# =============================================================================
 
@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status"          : "ok",
        "model_loaded"    : SIM_MATRIX is not None,
        "matrix_loaded"   : MATRIX_USER_ITEM is not None,
        "n_users"         : int(N_USERS),
        "n_movies"        : int(N_MOVIES),
        "sim_matrix_shape": list(SIM_MATRIX.shape) if SIM_MATRIX is not None else None,
        "timestamp"       : datetime.now().isoformat(),
    })
 
 
# =============================================================================
# ENDPOINT 2 — DAFTAR FILM
# GET /api/movies?page=1&limit=20&genre=Action&search=batman
# =============================================================================
 
@app.route("/api/movies", methods=["GET"])
def daftar_film():
    page   = int(request.args.get("page",   1))
    limit  = int(request.args.get("limit",  20))
    genre  = request.args.get("genre",  "")
    search = request.args.get("search", "")
    offset = (page - 1) * limit
 
    try:
        db  = get_db()
        cur = db.cursor()
 
        # Query dengan filter opsional
        where_clauses = []
        params        = []
 
        if genre:
            where_clauses.append("genre LIKE %s")
            params.append(f"%{genre}%")
 
        if search:
            where_clauses.append("title LIKE %s")
            params.append(f"%{search}%")
 
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
 
        # Hitung total film
        cur.execute(f"SELECT COUNT(*) as total FROM movies {where_sql}", params)
        total = cur.fetchone()["total"]
 
        # Ambil film dengan rata-rata rating
        sql = f"""
            SELECT
                m.*,
                ROUND(AVG(r.rating), 2) AS avg_rating,
                COUNT(r.id)             AS total_rating
            FROM movies m
            LEFT JOIN ratings r ON m.id = r.movie_id
            {where_sql}
            GROUP BY m.id
            ORDER BY avg_rating DESC
            LIMIT %s OFFSET %s
        """
        cur.execute(sql, params + [limit, offset])
        films = [format_film(row) for row in cur.fetchall()]
 
        db.close()
 
        return jsonify({
            "status"  : "ok",
            "data"    : films,
            "total"   : total,
            "page"    : page,
            "limit"   : limit,
            "total_pages": int(np.ceil(total / limit)),
        })
 
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
 
 
# =============================================================================
# ENDPOINT 3 — DETAIL FILM
# GET /api/movies/<movie_id>
# =============================================================================
 
@app.route("/api/movies/<string:movie_id>", methods=["GET"])
def detail_film(movie_id):
    print(f"  DEBUG detail_film dipanggil: movie_id={movie_id}, isdigit={movie_id.isdigit()}")
    try:
        db  = get_db()
        cur = db.cursor()

        # 1. Coba cari di tabel MySQL movies dulu
        # movie_id di MySQL selalu integer, jadi hanya dicoba kalau formatnya angka
        if movie_id.isdigit():
            cur.execute("""
                SELECT
                    m.*,
                    ROUND(AVG(r.rating), 2) AS avg_rating,
                    COUNT(r.id)             AS total_rating
                FROM movies m
                LEFT JOIN ratings r ON m.id = r.movie_id
                WHERE m.id = %s
                GROUP BY m.id
            """, (int(movie_id),))
            film = cur.fetchone()

            if film:
                cur.execute("""
                    SELECT r.rating, r.created_at, u.name AS user_name
                    FROM ratings r
                    JOIN users u ON r.user_id = u.id
                    WHERE r.movie_id = %s
                    ORDER BY r.created_at DESC
                    LIMIT 10
                """, (int(movie_id),))
                ulasan = cur.fetchall()
                db.close()

                return jsonify({
                    "status": "ok",
                    "data"  : {**format_film(film), "ulasan": ulasan},
                })

        db.close()

        # 2. Tidak ketemu di MySQL -> coba cari di top10_films.csv (Global, movieId angka)
        if movie_id.isdigit() and DF_TOP10 is not None:
            baris = DF_TOP10[DF_TOP10["movieId"] == int(movie_id)]
            if len(baris) > 0:
                row = baris.iloc[0]

                title = row.get("title")
                title = str(title) if pd.notna(title) else f"Film #{int(row['movie_encoded'])}"

                genres = row.get("genres")
                genres = str(genres).replace("|", ", ") if pd.notna(genres) else "Tidak diketahui"

                tahun = row.get("tahun_rilis")
                tahun = int(tahun) if pd.notna(tahun) else None

                judul_bersih = title.rsplit(" (", 1)[0] if title.endswith(")") else title
                poster_url, deskripsi = ambil_detail_tmdb(judul_bersih, tahun)

                return jsonify({
                    "status": "ok",
                    "data": {
                        "id"          : movie_id,
                        "title"       : title,
                        "genre"       : genres,
                        "year"        : tahun,
                        "poster"      : poster_url or "",
                        "description" : deskripsi or "Deskripsi tidak tersedia.",
                        "avg_rating"  : round(float(row["avg_rating"]), 2),
                        "total_rating": int(row["total_rating"]),
                        "ulasan"      : [],
                    },
                })

        # 3. Coba cari di top10_films_indonesia.csv (kode seperti M036)
        if DF_TOP10_ID is not None:
            baris = DF_TOP10_ID[DF_TOP10_ID["movie_id"] == movie_id]
            if len(baris) > 0:
                row = baris.iloc[0]

                title = row.get("title") or row.get("movie_title")
                title = str(title) if pd.notna(title) else movie_id

                genres = row.get("genre")
                genres = str(genres).replace("|", ", ") if pd.notna(genres) else "Tidak diketahui"

                tahun = row.get("year")
                tahun = int(tahun) if pd.notna(tahun) else None

                poster_url, deskripsi = ambil_detail_tmdb(title, tahun)

                return jsonify({
                    "status": "ok",
                    "data": {
                        "id"          : movie_id,
                        "title"       : title,
                        "genre"       : genres,
                        "year"        : tahun,
                        "poster"      : poster_url or "",
                        "description" : deskripsi or "Deskripsi tidak tersedia.",
                        "avg_rating"  : round(float(row["avg_rating"]), 2),
                        "total_rating": int(row["total_rating"]),
                        "ulasan"      : [],
                    },
                })
            
                
            # 3. Coba cari di tabel movies_ml1m (MovieLens 1M)
        if movie_id.isdigit():
            try:
                db3  = get_db()
                cur3 = db3.cursor()
                cur3.execute("""
                    SELECT
                        m.id, m.title, m.genres AS genre, m.year,
                        ROUND(AVG(rating_gabungan.rating), 2) AS avg_rating,
                        COUNT(rating_gabungan.rating)         AS total_rating
                    FROM movies_ml1m m
                    LEFT JOIN (
                        SELECT movie_id, rating FROM ratings_ml1m
                        UNION ALL
                        SELECT movie_id, rating FROM ratings WHERE movie_id = %s
                    ) AS rating_gabungan ON m.id = rating_gabungan.movie_id
                    WHERE m.id = %s
                    GROUP BY m.id
                """, (movie_id, int(movie_id)))
                film = cur3.fetchone()

                if film:
                    # Ambil ulasan dari tabel ratings (user website)
                    cur3.execute("""
                        SELECT r.rating, r.created_at, u.name AS user_name
                        FROM ratings r
                        JOIN users u ON r.user_id = u.id
                        WHERE r.movie_id = %s
                        ORDER BY r.created_at DESC
                        LIMIT 10
                    """, (int(movie_id),))
                    ulasan = cur3.fetchall()
                    db3.close()

                    title  = str(film["title"])
                    genres = str(film["genre"]).replace("|", ", ") if film["genre"] else "Tidak diketahui"
                    tahun  = film["year"]

                    judul_bersih          = title.rsplit(" (", 1)[0] if title.endswith(")") else title
                    poster_url, deskripsi = ambil_detail_tmdb(judul_bersih, tahun)

                    return jsonify({
                        "status": "ok",
                        "data"  : {
                            "id"          : int(movie_id),
                            "title"       : title,
                            "genre"       : genres,
                            "year"        : int(tahun) if tahun else None,
                            "poster"      : poster_url or "",
                            "description" : deskripsi or "Deskripsi tidak tersedia.",
                            "avg_rating"  : float(film["avg_rating"]) if film["avg_rating"] else 0.0,
                            "total_rating": int(film["total_rating"]),
                            "ulasan"      : ulasan or [],
                        },
                    })
                db3.close()
            except Exception as e:
                print(f"  ⚠️ Error cari di movies_ml1m: {e}")
        
        # 4. Cari di ratings_dataset_indonesia (film Indonesia posisi 11+)
        if not movie_id.isdigit():
            try:
                db4  = get_db()
                cur4 = db4.cursor()
                cur4.execute("""
                    SELECT
                        movie_id,
                        movie_title,
                        genre,
                        year,
                        ROUND(AVG(rating), 2) AS avg_rating,
                        COUNT(*)              AS total_rating
                    FROM ratings_dataset_indonesia
                    WHERE movie_id = %s
                    GROUP BY movie_id, movie_title, genre, year
                    LIMIT 1
                """, (movie_id,))
                film = cur4.fetchone()
                db4.close()

                if film:
                    title  = str(film["movie_title"]) if film["movie_title"] else movie_id
                    genres = str(film["genre"]).replace("|", ", ") if film["genre"] else "Tidak diketahui"
                    tahun  = int(film["year"]) if film["year"] else None

                    poster_url, deskripsi = ambil_detail_tmdb(title, tahun)

                    return jsonify({
                        "status": "ok",
                        "data"  : {
                            "id"          : movie_id,
                            "title"       : title,
                            "genre"       : genres,
                            "year"        : tahun,
                            "poster"      : poster_url or "",
                            "description" : deskripsi or "Deskripsi tidak tersedia.",
                            "avg_rating"  : float(film["avg_rating"]) if film["avg_rating"] else 0.0,
                            "total_rating": int(film["total_rating"]),
                            "ulasan"      : [],
                        },
                    })
            except Exception as e:
                print(f"  ⚠️ Gagal cari di ratings_dataset_indonesia: {e}")

        # 5. Tidak ketemu di mana pun
        return jsonify({"status": "error", "message": "Film tidak ditemukan"}), 404

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
 
 
# =============================================================================
# ENDPOINT 4 — SIMPAN RATING BARU
# POST /api/ratings
# Body JSON: { "user_id": 1, "movie_id": 10, "rating": 4.5 }
# =============================================================================
 
@app.route("/api/ratings", methods=["POST"])
def simpan_rating():
    data = request.get_json()

    user_id  = data.get("user_id")
    movie_id = data.get("movie_id")
    rating   = data.get("rating")

    # Validasi input
    if not all([user_id, movie_id, rating]):
        return jsonify({
            "status" : "error",
            "message": "user_id, movie_id, dan rating wajib diisi"
        }), 400

    if not (0.5 <= float(rating) <= 5.0):
        return jsonify({
            "status" : "error",
            "message": "Rating harus antara 0.5 dan 5.0"
        }), 400

    try:
        db = get_db()

        # Pastikan film ada (satu kali saja, tanpa int())
        if not pastikan_film_ada(db, movie_id):
            db.close()
            return jsonify({
                "status" : "error",
                "message": f"Film dengan id {movie_id} tidak dikenal"
            }), 404

        cur = db.cursor()

        # Cek apakah user sudah pernah rating film ini
        cur.execute(
            "SELECT id FROM ratings WHERE user_id=%s AND movie_id=%s",
            (user_id, str(movie_id))
        )
        existing = cur.fetchone()

        timestamp_now = int(datetime.now().timestamp())

        if existing:
            cur.execute("""
                UPDATE ratings
                SET rating=%s, timestamp=%s, created_at=NOW()
                WHERE user_id=%s AND movie_id=%s
            """, (rating, timestamp_now, user_id, str(movie_id)))
            aksi = "updated"
        else:
            cur.execute("""
                INSERT INTO ratings (user_id, movie_id, rating, timestamp, created_at)
                VALUES (%s, %s, %s, %s, NOW())
            """, (user_id, str(movie_id), rating, timestamp_now))
            aksi = "created"

        db.commit()

        # Reset cache Top-10 dan trigger recompute di background
        _CACHE_REALTIME["global"]["last_updated"]    = 0
        _CACHE_REALTIME["indonesia"]["last_updated"] = 0
        print("  🔄 Cache Top-10 direset karena ada rating baru")

        db.close()

        return jsonify({
            "status" : "ok",
            "message": f"Rating berhasil {aksi}",
            "data"   : {
                "user_id" : user_id,
                "movie_id": movie_id,
                "rating"  : rating,
                "aksi"    : aksi,
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
# =============================================================================
# ENDPOINT — TOP-10 REAL-TIME 
# =============================================================================

@app.route("/api/top-films/realtime", methods=["GET"])
def top_films_realtime():
    import time
    region = request.args.get("region", "global")
    limit  = int(request.args.get("limit", 10))
    offset = int(request.args.get("offset",  0))

    if region not in ["global", "indonesia"]:
        region = "global"

    cache = _CACHE_REALTIME[region]
    now   = time.time()

    if cache["data"] is None or (now - cache["last_updated"]) > _CACHE_TTL:
        print(f"  ⏳ Menghitung ulang Top-10 {region}...")
        try:
            cache["data"]         = hitung_top10(region, 30)  # selalu simpan 30
            cache["last_updated"] = now
            print(f"  ✅ Top-10 {region} selesai dihitung")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        print(f"  ✅ Top-10 {region} diambil dari cache")

    # Pastikan data tidak None sebelum len()
    semua_data = cache["data"] or []
    data       = semua_data[offset : offset + limit]

    return jsonify({
        "status" : "ok",
        "region" : region,
        "total"  : len(data),
        "source" : "realtime",
        "data"   : data,
    })
 
# =============================================================================
# ENDPOINT 5 — REKOMENDASI FILM
# GET /api/recommend/<user_id>?k=20&top_n=10
# =============================================================================
@app.route("/api/recommend/<int:user_id>", methods=["GET"])
def rekomendasi(user_id):
    K     = int(request.args.get("k",     DEFAULT_K))
    top_n = int(request.args.get("top_n", DEFAULT_TOP_N))

    if SIM_WEIGHTED is None and SIM_BASELINE is None:
        return jsonify({
            "status" : "error",
            "message": "Model belum dimuat."
        }), 503

    user_encoded = get_user_encoded(user_id)
    if user_encoded is None:
        return jsonify({
            "status" : "error",
            "message": f"User ID {user_id} tidak ditemukan"
        }), 404

    # Ambil film yang sudah dirating user
    try:
        db  = get_db()
        cur = db.cursor()
        cur.execute(
            "SELECT movie_id FROM ratings WHERE user_id = %s",
            (user_id,)
        )
        rated_movie_ids = {row["movie_id"] for row in cur.fetchall()}
        db.close()
    except:
        rated_movie_ids = set()

    # ---------------------------------------------------------------
    # Generate prediksi untuk KEDUA model
    # ---------------------------------------------------------------
    def generate_rekomendasi(model_name):
        prediksi_list = []
        for movie_enc in range(N_MOVIES):
            if DF_MAP_MOVIE is not None:
                row = DF_MAP_MOVIE[DF_MAP_MOVIE["movie_encoded"] == movie_enc]
                if len(row) > 0:
                    movie_id_asli = str(int(row["movieId"].values[0]))
                    if movie_id_asli in rated_movie_ids:
                        continue

            pred = prediksi_rating(user_encoded, movie_enc, K, model_name)
            if pred is not None:
                prediksi_list.append((movie_enc, pred))

        prediksi_list.sort(key=lambda x: x[1], reverse=True)
        top = prediksi_list[:top_n]

        hasil = []
        try:
            db  = get_db()
            cur = db.cursor()
            for rank, (movie_enc, skor) in enumerate(top, 1):
                movie_id_asli = None
                if DF_MAP_MOVIE is not None:
                    row = DF_MAP_MOVIE[DF_MAP_MOVIE["movie_encoded"] == movie_enc]
                    if len(row) > 0:
                        movie_id_asli = int(row["movieId"].values[0])

                film_data = {}
                if movie_id_asli:
                    cur.execute("""
                        SELECT id, title, genres AS genre, year
                        FROM movies_ml1m
                        WHERE id = %s
                    """, (movie_id_asli,))
                    film = cur.fetchone()
                    if film:
                        title  = str(film["title"])
                        genres = str(film["genre"]).replace("|", ", ") if film["genre"] else "Tidak diketahui"
                        tahun  = film["year"]
                        judul_bersih = title.rsplit(" (", 1)[0] if title.endswith(")") else title
                        poster_url, deskripsi = ambil_detail_tmdb(judul_bersih, tahun)
                        film_data = {
                            "id"         : movie_id_asli,
                            "title"      : title,
                            "genre"      : genres,
                            "year"       : int(tahun) if tahun else None,
                            "poster"     : poster_url or "",
                            "description": deskripsi or "",
                        }

                hasil.append({
                    "rank"          : rank,
                    "skor_prediksi" : skor,
                    "movie_encoded" : movie_enc,
                    **film_data,
                })
            db.close()
        except Exception as e:
            print(f"  ⚠️ Error ambil detail film: {e}")

        return hasil

    # Jalankan kedua model
    hasil_weighted = generate_rekomendasi("weighted") if SIM_WEIGHTED is not None else []
    hasil_baseline = generate_rekomendasi("baseline") if SIM_BASELINE is not None else []

    return jsonify({
        "status"  : "ok",
        "user_id" : user_id,
        "k"       : K,
        "top_n"   : top_n,
        "weighted": hasil_weighted,  # dengan timestamp
        "baseline": hasil_baseline,  # tanpa timestamp
    })
 
 
# =============================================================================
# ENDPOINT 6 — RATING USER (history rating milik user)
# GET /api/ratings/<user_id>
# =============================================================================
 
# @app.route("/api/ratings/<int:user_id>", methods=["GET"])
# def rating_user(user_id):
#     try:
#         db  = get_db()
#         cur = db.cursor()
 
#         cur.execute("""
#             SELECT
#                 r.id, r.rating, r.created_at,
#                 m.id AS movie_id, m.title, m.genre, m.year, m.poster
#             FROM ratings r
#             JOIN movies m ON r.movie_id = m.id
#             WHERE r.user_id = %s
#             ORDER BY r.created_at DESC
#         """, (user_id,))
 
#         data = cur.fetchall()
#         db.close()
 
#         return jsonify({
#             "status"  : "ok",
#             "user_id" : user_id,
#             "total"   : len(data),
#             "data"    : data,
#         })
 
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500
 
# =============================================================================
# ENDPOINT 7 — TOP 10 FILM TERPOPULER BERDASARKAN RATING + TIMESTAMP
# GET /api/top-films
# =============================================================================

path_top10_id = os.path.join(os.path.dirname(OUTPUT_DIR), "top10_films_indonesia.csv")
DF_TOP10_ID   = pd.read_csv(path_top10_id) if os.path.exists(path_top10_id) else None
print(f"  ✅ Top 10 Indonesia : {'Dimuat' if DF_TOP10_ID is not None else 'Tidak ada'}")

@app.route("/api/top-films", methods=["GET"])
def top_films():
    region = request.args.get("region", "global")

    df_dipilih = DF_TOP10_ID if region == "indonesia" else DF_TOP10

    if df_dipilih is None:
        return jsonify({
            "status" : "error",
            "message": f"Data top 10 untuk region '{region}' tidak ditemukan"
        }), 404

    hasil = []
    for _, row in df_dipilih.iterrows():

        if region == "indonesia":
            # Struktur kolom CSV Indonesia: movie_id (kode string), genre, year
            movie_id = row.get("movie_id")
            movie_id = str(movie_id) if pd.notna(movie_id) else None

            title = row.get("title") or row.get("movie_title")
            title = str(title) if pd.notna(title) else f"Film #{int(row['movie_encoded'])}"

            genres = row.get("genre")
            genres = str(genres).replace("|", ", ") if pd.notna(genres) else "Tidak diketahui"

            tahun = row.get("year")
            tahun = int(tahun) if pd.notna(tahun) else None

        else:
            # Struktur kolom CSV Global: movieId (angka), genres, tahun_rilis
            movie_id = row.get("movieId")
            movie_id = int(movie_id) if pd.notna(movie_id) else None

            title = row.get("title")
            title = str(title) if pd.notna(title) else f"Film #{int(row['movie_encoded'])}"

            genres = row.get("genres")
            genres = str(genres).replace("|", ", ") if pd.notna(genres) else "Tidak diketahui"

            tahun = row.get("tahun_rilis")
            tahun = int(tahun) if pd.notna(tahun) else None

        judul_bersih = title.rsplit(" (", 1)[0] if title.endswith(")") else title
        poster_url, deskripsi = ambil_detail_tmdb(judul_bersih, tahun)

        hasil.append({
            "rank"          : int(row["rank"]),
            "movie_encoded" : int(row["movie_encoded"]),
            "movie_id"      : movie_id,
            "title"         : title,
            "genre"         : genres,
            "year"          : tahun,
            "avg_rating"    : round(float(row["avg_rating"]), 2),
            "total_rating"  : int(row["total_rating"]),
            "skor"          : round(float(row["skor"]), 4),
            "poster"        : poster_url or "",
            "description"   : deskripsi or "Deskripsi tidak tersedia.",
        })

    return jsonify({
        "status" : "ok",
        "region" : region,
        "total"  : len(hasil),
        "data"   : hasil,
    })

# =============================================================================
# ENDPOINT — MAE & RMSE
# GET /api/metrics
# =============================================================================

@app.route("/api/metrics", methods=["GET"])
def metrics():
    import time
    force  = request.args.get("force", "false").lower() == "true"
    region = request.args.get("region", "global")       # global | indonesia
    source = request.args.get("source", "test")          # test | production
    now    = time.time()

    cache_key = f"{region}_{source}"
    if cache_key not in _CACHE_METRICS_MULTI:
        _CACHE_METRICS_MULTI[cache_key] = {"data": None, "last_updated": 0}
    cache = _CACHE_METRICS_MULTI[cache_key]

    if force or cache["data"] is None or (now - cache["last_updated"]) > _METRICS_TTL:
        print(f"  ⏳ Menghitung MAE & RMSE (region={region}, source={source})...")
        try:
            if source == "production":
                hasil = hitung_mae_rmse_production(region=region)
            elif region == "indonesia":
                hasil = hitung_mae_rmse_indonesia()
            else:
                hasil = hitung_mae_rmse(sample_size=None)

            if hasil:
                cache["data"] = hasil
                cache["last_updated"] = now
            else:
                return jsonify({
                    "status" : "error",
                    "message": "Tidak cukup data untuk menghitung metrik"
                }), 500
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        print(f"  ✅ Metrics ({region}/{source}) diambil dari cache")

    return jsonify({
        "status": "ok",
        "data"  : cache["data"],
    })
# =============================================================================
# ENDPOINT — SEARCH FILM
# GET /api/search?q=inception&region=global&limit=10
# =============================================================================

@app.route("/api/search", methods=["GET"])
def search_film():
    query  = request.args.get("q", "").strip()
    region = request.args.get("region", "global")
    limit  = int(request.args.get("limit", 10))

    if not query or len(query) < 2:
        return jsonify({
            "status" : "error",
            "message": "Kata kunci pencarian minimal 2 karakter"
        }), 400

    try:
        hasil = []

        if region == "indonesia":
            # Cari di ratings_dataset_indonesia
            db  = get_db()
            cur = db.cursor()
            cur.execute("""
                SELECT
                    movie_id,
                    movie_title,
                    genre,
                    year,
                    ROUND(AVG(rating), 2) AS avg_rating,
                    COUNT(*)              AS total_rating
                FROM ratings_dataset_indonesia
                WHERE movie_title LIKE %s
                GROUP BY movie_id, movie_title, genre, year
                ORDER BY avg_rating DESC
                LIMIT %s
            """, (f"%{query}%", limit))
            rows = cur.fetchall()
            db.close()

            for row in rows:
                title  = str(row["movie_title"]) if row["movie_title"] else row["movie_id"]
                genres = str(row["genre"]).replace("|", ", ") if row["genre"] else "Tidak diketahui"
                tahun  = int(row["year"]) if row["year"] else None

                poster_url, deskripsi = ambil_detail_tmdb(title, tahun)

                hasil.append({
                    "movie_id"    : str(row["movie_id"]),
                    "title"       : title,
                    "genre"       : genres,
                    "year"        : tahun,
                    "avg_rating"  : float(row["avg_rating"]) if row["avg_rating"] else 0.0,
                    "total_rating": int(row["total_rating"]),
                    "poster"      : poster_url or "",
                    "description" : deskripsi or "Deskripsi tidak tersedia.",
                })

        else:
            # Cari di movies_ml1m (Global) + movies (film tambahan admin)
            db  = get_db()
            cur = db.cursor()
            cur.execute("""
                SELECT
                    m.id, m.title, m.genres, m.year,
                    ROUND(AVG(r.rating), 2) AS avg_rating,
                    COUNT(r.id)             AS total_rating
                FROM movies_ml1m m
                LEFT JOIN ratings_ml1m r ON m.id = r.movie_id
                WHERE m.title LIKE %s
                GROUP BY m.id, m.title, m.genres, m.year

                UNION

                SELECT
                    m.id, m.title, m.genre AS genres, m.year,
                    ROUND(AVG(r.rating), 2) AS avg_rating,
                    COUNT(r.id)             AS total_rating
                FROM movies m
                LEFT JOIN ratings r ON m.id = r.movie_id
                WHERE m.title LIKE %s
                  AND m.id NOT IN (SELECT id FROM movies_ml1m)
                GROUP BY m.id, m.title, m.genre, m.year

                ORDER BY avg_rating DESC
                LIMIT %s
            """, (f"%{query}%", f"%{query}%", limit))
            rows = cur.fetchall()
            db.close()

            for row in rows:
                title  = str(row["title"])
                genres = str(row["genres"]).replace("|", ", ") if row["genres"] else "Tidak diketahui"
                tahun  = int(row["year"]) if row["year"] else None

                judul_bersih          = title.rsplit(" (", 1)[0] if title.endswith(")") else title
                poster_url, deskripsi = ambil_detail_tmdb(judul_bersih, tahun)

                hasil.append({
                    "movie_id"    : int(row["id"]),
                    "title"       : title,
                    "genre"       : genres,
                    "year"        : tahun,
                    "avg_rating"  : float(row["avg_rating"]) if row["avg_rating"] else 0.0,
                    "total_rating": int(row["total_rating"]),
                    "poster"      : poster_url or "",
                    "description" : deskripsi or "Deskripsi tidak tersedia.",
                })

        return jsonify({
            "status" : "ok",
            "query"  : query,
            "region" : region,
            "total"  : len(hasil),
            "data"   : hasil,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
 
# =============================================================================
# JALANKAN SERVER
# =============================================================================
 
if __name__ == "__main__":
    print("\n  Flask API berjalan di http://localhost:5000")
    print("  Tekan CTRL+C untuk berhenti\n")
    app.run(
        host  = "0.0.0.0",
        port  = 5000,
        debug = True,
    )