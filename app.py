from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
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
_CACHE_TTL = 300  # detik — hitung ulang maksimal setiap 5 menit

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
OUTPUT_DIR = "./dataset/processed"
 
# Konfigurasi model
DEFAULT_K      = 20    
DEFAULT_TOP_N  = 10     
LAMBDA         = 0.3    
 
 
# =============================================================================
# LOAD MODEL SAAT STARTUP
# =============================================================================
 
print("=" * 55)
print("  LOADING MODEL IBCF...")
print("=" * 55)
 
# Load similarity matrix (gunakan model terbaik = weighted/timestamp)
path_sim = os.path.join(MODEL_DIR, "similarity_weighted.npy")
if os.path.exists(path_sim):
    SIM_MATRIX = np.load(path_sim, mmap_mode='r')
    print(f"  ✅ Similarity matrix dimuat: {SIM_MATRIX.shape}")
else:
    SIM_MATRIX = None
    print(f"  ⚠️  similarity_weighted.npy tidak ditemukan di {MODEL_DIR}")
 
# Load train data untuk rebuild sparse matrix
path_train = os.path.join(OUTPUT_DIR, "train.csv")
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
    print(f"  ⚠️  train.csv tidak ditemukan di {OUTPUT_DIR}")
 
# Load mapping & info film
path_map_movie = os.path.join(OUTPUT_DIR, "mapping_movie.csv")
path_map_user  = os.path.join(OUTPUT_DIR, "mapping_user.csv")
path_info_film = os.path.join(OUTPUT_DIR, "info_film.csv")
 
DF_MAP_MOVIE  = pd.read_csv(path_map_movie)  if os.path.exists(path_map_movie)  else None
DF_MAP_USER   = pd.read_csv(path_map_user)   if os.path.exists(path_map_user)   else None
DF_INFO_FILM  = pd.read_csv(path_info_film)  if os.path.exists(path_info_film)  else None
 
print(f"  ✅ Mapping movie : {'Dimuat' if DF_MAP_MOVIE  is not None else 'Tidak ada'}")
print(f"  ✅ Mapping user  : {'Dimuat' if DF_MAP_USER   is not None else 'Tidak ada'}")
print(f"  ✅ Info film     : {'Dimuat' if DF_INFO_FILM  is not None else 'Tidak ada'}")
print("=" * 55)
 
 
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
 
 
def prediksi_rating(user_encoded, movie_encoded, K=DEFAULT_K):
    """Prediksi rating user terhadap film menggunakan IBCF."""
    if SIM_MATRIX is None or MATRIX_USER_ITEM is None:
        return None
    if user_encoded >= N_USERS or movie_encoded >= N_MOVIES:
        return None
 
    sim_scores   = SIM_MATRIX[movie_encoded]
    rated_movies = MATRIX_USER_ITEM[user_encoded].nonzero()[1]
 
    if len(rated_movies) == 0:
        return None
 
    sim_rated = sim_scores[rated_movies]
 
    # Ambil K tetangga terbaik
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
 
def hitung_top10(region="global", limit=10):
    """Hitung Top-10 real-time. Dipanggil saat cache expired atau ada rating baru."""
    db  = get_db()
    cur = db.cursor()

    if region == "indonesia":
        cur.execute("""
            SELECT movie_id, movie_title AS title, genre, year,
                   rating, timestamp, NULL AS movie_encoded
            FROM ratings_dataset_indonesia
            WHERE timestamp IS NOT NULL

            UNION ALL

            SELECT ri.movie_id, ri.movie_title, ri.genre, ri.year,
                   r.rating, r.timestamp, NULL
            FROM ratings r
            JOIN ratings_dataset_indonesia ri
                ON CAST(r.movie_id AS CHAR) = ri.movie_id
            WHERE r.timestamp IS NOT NULL
        """)
        MIN_RATING = 10
        agg_key    = "movie_id"

    else:
        cur.execute("""
            SELECT CAST(movie_id AS CHAR) AS movie_id,
                   NULL, NULL, NULL,
                   rating, timestamp, movie_encoded
            FROM ratings_dataset
            WHERE timestamp IS NOT NULL
              AND movie_encoded IS NOT NULL

            UNION ALL

            SELECT CAST(r.movie_id AS CHAR), NULL, NULL, NULL,
                   r.rating, r.timestamp, rd.movie_encoded
            FROM ratings r
            JOIN ratings_dataset rd ON r.movie_id = rd.movie_id
            WHERE r.timestamp IS NOT NULL
              AND rd.movie_encoded IS NOT NULL
        """)
        MIN_RATING = 50
        agg_key    = "movie_encoded"

    rows = cur.fetchall()
    db.close()

    if not rows:
        return []

    cols = ["movie_id", "title", "genre", "year", "rating", "timestamp", "movie_encoded"]
    df   = pd.DataFrame(rows, columns=cols)
    df["rating"]    = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(float)

    ts_max   = df["timestamp"].max()
    ts_min   = df["timestamp"].min()
    ts_range = ts_max - ts_min if ts_max != ts_min else 1

    df["delta_t_norm"]    = (ts_max - df["timestamp"]) / ts_range
    df["weight"]          = np.exp(-LAMBDA * df["delta_t_norm"])
    df["weighted_rating"] = df["rating"] * df["weight"]

    if region == "indonesia":
        info = df.groupby(agg_key).first()[["title", "genre", "year"]].reset_index()

    agg = df.groupby(agg_key).agg(
        avg_rating   = ("rating",          "mean"),
        total_rating = ("rating",          "count"),
        weighted_avg = ("weighted_rating", "mean"),
    ).reset_index()

    if region == "indonesia":
        agg = agg.merge(info, on=agg_key, how="left")

    agg = agg[agg["total_rating"] >= MIN_RATING].copy()
    if agg.empty:
        return []

    max_weighted = agg["weighted_avg"].max()
    agg["skor"]  = (agg["weighted_avg"] / max_weighted * 10).round(4)
    top          = agg.nlargest(limit, "skor").reset_index(drop=True)
    top["rank"]  = top.index + 1

    if region == "global":
        if DF_MAP_MOVIE is not None:
            top = top.merge(DF_MAP_MOVIE, on="movie_encoded", how="left")
        else:
            top["movieId"] = None

        if DF_INFO_ML is not None:
            top = top.merge(
                DF_INFO_ML[["movieId", "title", "genres", "tahun_rilis"]],
                on="movieId", how="left"
            )
        else:
            top["title"] = None
            top["genres"] = None
            top["tahun_rilis"] = None

        if DF_INFO_NETFLIX is not None:
            top = top.merge(
                DF_INFO_NETFLIX[["movie_encoded", "title_netflix", "tahun_rilis_netflix"]],
                on="movie_encoded", how="left"
            )
            top["title"]       = top["title"].fillna(top["title_netflix"])
            top["tahun_rilis"] = top["tahun_rilis"].fillna(top["tahun_rilis_netflix"])
            top["genres"]      = top["genres"].fillna("Netflix Collection")
            top = top.drop(columns=["title_netflix", "tahun_rilis_netflix"])

        top["title"]  = top["title"].fillna("Film #" + top["movie_encoded"].astype(str))
        top["genres"] = top["genres"].fillna("Tidak diketahui")

    hasil = []
    for _, row in top.iterrows():
        if region == "global":
            title    = str(row["title"])
            genres   = str(row["genres"]).replace("|", ", ") if pd.notna(row.get("genres")) else "Tidak diketahui"
            tahun    = int(row["tahun_rilis"]) if pd.notna(row.get("tahun_rilis")) else None
            movie_id = int(row["movieId"]) if pd.notna(row.get("movieId")) else None
        else:
            title    = str(row["title"]) if pd.notna(row.get("title")) else str(row["movie_id"])
            genres   = str(row["genre"]).replace("|", ", ") if pd.notna(row.get("genre")) else "Tidak diketahui"
            tahun    = int(row["year"]) if pd.notna(row.get("year")) else None
            movie_id = str(row["movie_id"])

        judul_bersih          = title.rsplit(" (", 1)[0] if title.endswith(")") else title
        poster_url, deskripsi = ambil_detail_tmdb(judul_bersih, tahun)

        hasil.append({
            "rank"        : int(row["rank"]),
            "movie_id"    : movie_id,
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

        # 4. Tidak ketemu di mana pun
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
        db  = get_db()
        cur = db.cursor()
 
        # Cek apakah user sudah pernah rating film ini
        cur.execute(
            "SELECT id FROM ratings WHERE user_id=%s AND movie_id=%s",
            (user_id, movie_id)
        )
        existing = cur.fetchone()
 
        timestamp_now = int(datetime.now().timestamp())
 
        if existing:
            # Update rating yang sudah ada
            cur.execute("""
                UPDATE ratings
                SET rating=%s, timestamp=%s, created_at=NOW()
                WHERE user_id=%s AND movie_id=%s
            """, (rating, timestamp_now, user_id, movie_id))
            aksi = "updated"
        else:
            # Insert rating baru
            cur.execute("""
                INSERT INTO ratings (user_id, movie_id, rating, timestamp, created_at)
                VALUES (%s, %s, %s, %s, NOW())
            """, (user_id, movie_id, rating, timestamp_now))
            aksi = "created"
 
        db.commit()
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

    if region not in ["global", "indonesia"]:
        region = "global"

    cache  = _CACHE_REALTIME[region]
    now    = time.time()

    # Hitung ulang hanya kalau cache expired
    if cache["data"] is None or (now - cache["last_updated"]) > _CACHE_TTL:
        print(f"  ⏳ Menghitung ulang Top-10 {region}...")
        try:
            cache["data"]         = hitung_top10(region, limit)
            cache["last_updated"] = now
            print(f"  ✅ Top-10 {region} selesai dihitung")
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        print(f"  ✅ Top-10 {region} diambil dari cache")

    return jsonify({
        "status" : "ok",
        "region" : region,
        "total"  : len(cache["data"]),
        "source" : "realtime",
        "data"   : cache["data"],
    })
 
# =============================================================================
# ENDPOINT 5 — REKOMENDASI FILM
# GET /api/recommend/<user_id>?k=20&top_n=10
# =============================================================================
 
@app.route("/api/recommend/<int:user_id>", methods=["GET"])
def rekomendasi(user_id):
    K     = int(request.args.get("k",     DEFAULT_K))
    top_n = int(request.args.get("top_n", DEFAULT_TOP_N))
 
    if SIM_MATRIX is None or MATRIX_USER_ITEM is None:
        return jsonify({
            "status" : "error",
            "message": "Model belum dimuat. Pastikan file .npy tersedia di folder model/"
        }), 503
 
    # Konversi user_id → user_encoded
    user_encoded = get_user_encoded(user_id)
    if user_encoded is None:
        return jsonify({
            "status" : "error",
            "message": f"User ID {user_id} tidak ditemukan di data training"
        }), 404
 
    # Ambil film yang sudah dirating user dari DB
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
 
    # Generate prediksi untuk semua film yang belum dirating
    prediksi_list = []
 
    for movie_enc in range(N_MOVIES):
        # Konversi movie_encoded → movie_id asli untuk cek DB
        if DF_MAP_MOVIE is not None:
            row = DF_MAP_MOVIE[DF_MAP_MOVIE["movie_encoded"] == movie_enc]
            if len(row) > 0:
                movie_id_asli = int(row["movieId"].values[0])
                if movie_id_asli in rated_movie_ids:
                    continue  # skip film yang sudah ditonton
 
        pred = prediksi_rating(user_encoded, movie_enc, K)
        if pred is not None:
            prediksi_list.append((movie_enc, pred))
 
    # Urutkan & ambil top_n
    prediksi_list.sort(key=lambda x: x[1], reverse=True)
    top_rekomendasi = prediksi_list[:top_n]
 
    if not top_rekomendasi:
        return jsonify({
            "status"  : "ok",
            "user_id" : user_id,
            "data"    : [],
            "message" : "Tidak ada rekomendasi. User perlu memberi rating lebih banyak film.",
        })
 
    # Ambil detail film dari MySQL
    try:
        db  = get_db()
        cur = db.cursor()
 
        hasil = []
        for rank, (movie_enc, skor) in enumerate(top_rekomendasi, 1):
            movie_id_asli = None
            if DF_MAP_MOVIE is not None:
                row = DF_MAP_MOVIE[DF_MAP_MOVIE["movie_encoded"] == movie_enc]
                if len(row) > 0:
                    movie_id_asli = int(row["movieId"].values[0])
 
            film_data = {}
            if movie_id_asli:
                cur.execute("""
                    SELECT
                        m.*,
                        ROUND(AVG(r.rating), 2) AS avg_rating,
                        COUNT(r.id)             AS total_rating
                    FROM movies m
                    LEFT JOIN ratings r ON m.id = r.movie_id
                    WHERE m.id = %s
                    GROUP BY m.id
                """, (movie_id_asli,))
                film = cur.fetchone()
                if film:
                    film_data = format_film(film)
 
            hasil.append({
                "rank"          : rank,
                "skor_prediksi" : skor,
                "movie_encoded" : movie_enc,
                **film_data,
            })
 
        db.close()
 
        return jsonify({
            "status"  : "ok",
            "user_id" : user_id,
            "k"       : K,
            "top_n"   : top_n,
            "data"    : hasil,
        })
 
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
 
 
# =============================================================================
# ENDPOINT 6 — RATING USER (history rating milik user)
# GET /api/ratings/<user_id>
# =============================================================================
 
@app.route("/api/ratings/<int:user_id>", methods=["GET"])
def rating_user(user_id):
    try:
        db  = get_db()
        cur = db.cursor()
 
        cur.execute("""
            SELECT
                r.id, r.rating, r.created_at,
                m.id AS movie_id, m.title, m.genre, m.year, m.poster
            FROM ratings r
            JOIN movies m ON r.movie_id = m.id
            WHERE r.user_id = %s
            ORDER BY r.created_at DESC
        """, (user_id,))
 
        data = cur.fetchall()
        db.close()
 
        return jsonify({
            "status"  : "ok",
            "user_id" : user_id,
            "total"   : len(data),
            "data"    : data,
        })
 
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
 
# =============================================================================
# ENDPOINT 7 — TOP 10 FILM TERPOPULER BERDASARKAN RATING + TIMESTAMP
# GET /api/top-films
# =============================================================================

# Load top10 saat startup
path_top10 = os.path.join(OUTPUT_DIR, "top10_films.csv")
DF_TOP10   = pd.read_csv(path_top10) if os.path.exists(path_top10) else None
print(f"  ✅ Top 10 films : {'Dimuat' if DF_TOP10 is not None else 'Tidak ada'}")

path_top10_id = os.path.join(OUTPUT_DIR, "top10_films_indonesia.csv")
DF_TOP10_ID   = pd.read_csv(path_top10_id) if os.path.exists(path_top10_id) else None
print(f"  ✅ Top 10 Indonesia : {'Dimuat' if DF_TOP10_ID is not None else 'Tidak ada'}")

# Load info film Netflix
path_info_nf = os.path.join(OUTPUT_DIR, "info_film_netflix.csv")
DF_INFO_NETFLIX = pd.read_csv(path_info_nf) if os.path.exists(path_info_nf) else None
print(f"  ✅ Info Netflix   : {'Dimuat' if DF_INFO_NETFLIX is not None else 'Tidak ada'}")

# Load info film MovieLens (kalau belum ada)
path_info_ml = os.path.join(OUTPUT_DIR, "info_film.csv")
DF_INFO_ML = pd.read_csv(path_info_ml) if os.path.exists(path_info_ml) else None
print(f"  ✅ Info MovieLens : {'Dimuat' if DF_INFO_ML is not None else 'Tidak ada'}")

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