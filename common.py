"""
Module partagé entre le script d'indexation (index_photos.py) et l'API (main.py).
Gère : le modèle de détection/embedding de visages (InsightFace),
       l'index de similarité (FAISS),
       et les métadonnées (SQLite : quel visage vient de quelle photo).
"""

import os
import sqlite3
import numpy as np
import faiss

# --- Chemins -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(BASE_DIR, "photos")       # photos brutes déposées par la comm
DATA_DIR = os.path.join(BASE_DIR, "data")           # index FAISS + base SQLite
INDEX_PATH = os.path.join(DATA_DIR, "faces.faiss")
DB_PATH = os.path.join(DATA_DIR, "faces.db")

EMBEDDING_DIM = 512          # dimension des embeddings ArcFace (buffalo_l)
DET_SIZE = (640, 640)        # taille de détection : bon compromis vitesse/précision sur CPU
SIMILARITY_THRESHOLD = 0.38  # seuil cosinus ArcFace : à ajuster après tests (0.35-0.45 typique)
TOP_K = 50                   # nb de visages les plus proches à examiner par recherche

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PHOTOS_DIR, exist_ok=True)


def get_face_app():
    """Charge le modèle InsightFace (détection + embedding), CPU uniquement."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=DET_SIZE)
    return app


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalise un embedding pour que le produit scalaire FAISS = similarité cosinus."""
    vec = vec.astype("float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            faiss_id INTEGER PRIMARY KEY,   -- position dans l'index FAISS
            photo_path TEXT NOT NULL,       -- chemin relatif dans PHOTOS_DIR
            bbox TEXT,                      -- boîte englobante du visage "x1,y1,x2,y2"
            added_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS indexed_files (
            photo_path TEXT PRIMARY KEY,    -- fichiers déjà traités, pour ne pas les réindexer
            n_faces INTEGER,
            indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def load_or_create_index() -> faiss.Index:
    """Recherche exacte par produit scalaire (= cosinus, car vecteurs normalisés).
    Largement suffisant en vitesse jusqu'à plusieurs centaines de milliers de visages."""
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def save_index(index: faiss.Index):
    faiss.write_index(index, INDEX_PATH)
