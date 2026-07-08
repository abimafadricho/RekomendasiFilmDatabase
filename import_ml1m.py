import pandas as pd
import pymysql
import os
import re

# ==============================================================
# KONFIGURASI — sesuaikan path dengan lokasi file Anda
# ==============================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RATINGS_PATH = os.path.join(BASE_DIR, "dataset", "processed", "ratings.dat")
MOVIES_PATH  = os.path.join(BASE_DIR, "dataset", "processed", "movies.dat")
BATCH_SIZE   = 5000

DB_CONFIG = {
    "host"    : "localhost",
    "user"    : "root",
    "password": "root",
    "database": "rekomendasi_film",
    "charset" : "utf8mb4",
}
# ==============================================================

def ekstrak_tahun(title):
    """Ekstrak tahun dari judul film format 'Toy Story (1995)'."""
    match = re.search(r'\((\d{4})\)$', str(title).strip())
    return int(match.group(1)) if match else None


def import_movies():
    print("\n[1/2] Import movies.dat...")

    df = pd.read_csv(
        MOVIES_PATH,
        sep           = "::",
        header        = None,
        names         = ["movieId", "title", "genres"],
        engine        = "python",
        encoding      = "latin-1",
    )

    df["year"] = df["title"].apply(ekstrak_tahun)
    total      = len(df)
    print(f"  ✅ Total film: {total:,}")

    db  = pymysql.connect(**DB_CONFIG)
    cur = db.cursor()

    cur.execute("TRUNCATE TABLE movies_ml1m")
    db.commit()

    inserted = 0
    for i in range(0, total, BATCH_SIZE):
        chunk = df.iloc[i:i+BATCH_SIZE].where(pd.notna(df.iloc[i:i+BATCH_SIZE]), None)
        rows  = [
            (
                int(row["movieId"]),
                str(row["title"]),
                str(row["genres"]) if row["genres"] else None,
                row["year"],
            )
            for _, row in chunk.iterrows()
        ]
        cur.executemany("""
            INSERT IGNORE INTO movies_ml1m (id, title, genres, year)
            VALUES (%s, %s, %s, %s)
        """, rows)
        db.commit()
        inserted += len(rows)
        print(f"  📥 {inserted:,} / {total:,} film")

    db.close()
    print(f"  ✅ Movies selesai: {inserted:,} film")


def import_ratings():
    print("\n[2/2] Import ratings.dat...")

    df = pd.read_csv(
        RATINGS_PATH,
        sep      = "::",
        header   = None,
        names    = ["userId", "movieId", "rating", "timestamp"],
        engine   = "python",
        encoding = "latin-1",
    )

    total = len(df)
    print(f"  ✅ Total rating: {total:,}")

    db  = pymysql.connect(**DB_CONFIG)
    cur = db.cursor()

    cur.execute("TRUNCATE TABLE ratings_ml1m")
    db.commit()

    inserted = 0
    for i in range(0, total, BATCH_SIZE):
        chunk = df.iloc[i:i+BATCH_SIZE]
        rows  = [
            (
                int(row["userId"]),
                int(row["movieId"]),
                float(row["rating"]),
                int(row["timestamp"]),
            )
            for _, row in chunk.iterrows()
        ]
        cur.executemany("""
            INSERT INTO ratings_ml1m (user_id, movie_id, rating, timestamp)
            VALUES (%s, %s, %s, %s)
        """, rows)
        db.commit()
        inserted += len(rows)
        pct = (inserted / total) * 100
        print(f"  📥 {inserted:,} / {total:,} rating ({pct:.1f}%)")

    db.close()
    print(f"  ✅ Ratings selesai: {inserted:,} rating")


if __name__ == "__main__":
    print("=" * 55)
    print("  IMPORT MOVIELENS 1M KE MySQL")
    print("=" * 55)

    if not os.path.exists(RATINGS_PATH):
        print(f"  ❌ File tidak ditemukan: {RATINGS_PATH}")
        print("     Sesuaikan RATINGS_PATH dan MOVIES_PATH di atas")
        exit(1)

    import_movies()
    import_ratings()

    print("\n" + "=" * 55)
    print("  ✅ Import selesai!")
    print("=" * 55)