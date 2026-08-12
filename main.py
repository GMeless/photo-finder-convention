"""
API de recherche de photos par selfie.

Lancer avec :  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import io
import os
import zipfile
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from common import (
    BASE_DIR, PHOTOS_DIR, get_face_app, get_db, load_or_create_index,
    normalize, SIMILARITY_THRESHOLD, TOP_K,
)

INDEX_HTML_PATH = os.path.join(BASE_DIR, "static", "index.html")

app = FastAPI(title="Recherche de photos par selfie")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Les photos originales sont servies directement pour affichage/téléchargement
app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")

# État chargé une fois au démarrage. Le script d'indexation tourne à part ;
# on recharge l'index périodiquement via /admin/reload (ou en redémarrant le service).
_state = {}


@app.on_event("startup")
def startup():
    _state["face_app"] = get_face_app()
    _state["index"] = load_or_create_index()
    print(f"Index chargé : {_state['index'].ntotal} visages.")


@app.post("/admin/reload")
def reload_index():
    """À appeler après une passe d'indexation pour que l'API voie les nouvelles photos,
    sans redémarrer le service."""
    _state["index"] = load_or_create_index()
    return {"n_faces": _state["index"].ntotal}


@app.get("/")
def root():
    return FileResponse(INDEX_HTML_PATH)


@app.post("/search")
async def search(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Image illisible.")

    faces = _state["face_app"].get(img)
    if not faces:
        raise HTTPException(404, "Aucun visage détecté sur le selfie. Réessayez avec plus de lumière et le visage bien cadré.")

    # On prend le plus grand visage détecté (le plus proche de la caméra = le photographe)
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    query_emb = normalize(faces[0].embedding).reshape(1, -1)

    index = _state["index"]
    if index.ntotal == 0:
        return {"matches": []}

    k = min(TOP_K, index.ntotal)
    scores, ids = index.search(query_emb, k)

    conn = get_db()
    seen_photos = {}  # photo_path -> meilleur score
    for score, faiss_id in zip(scores[0], ids[0]):
        if faiss_id == -1 or score < SIMILARITY_THRESHOLD:
            continue
        row = conn.execute(
            "SELECT photo_path FROM faces WHERE faiss_id = ?", (int(faiss_id),)
        ).fetchone()
        if row is None:
            continue
        photo_path = row[0]
        if photo_path not in seen_photos or score > seen_photos[photo_path]:
            seen_photos[photo_path] = float(score)

    results = sorted(seen_photos.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "matches": [
            {"url": f"/photos/{path}", "path": path, "score": round(score, 3)}
            for path, score in results
        ]
    }


@app.post("/zip")
async def zip_photos(payload: dict = Body(...)):
    """Reçoit {"paths": ["a.jpg", "sous_dossier/b.jpg", ...]} et renvoie une archive ZIP.
    Les chemins doivent être ceux renvoyés par /search (champ "path"), relatifs à photos/."""
    paths = payload.get("paths", [])
    if not paths:
        raise HTTPException(400, "Aucune photo spécifiée.")

    photos_root = os.path.realpath(PHOTOS_DIR)
    buf = io.BytesIO()
    added = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path in paths:
            full_path = os.path.realpath(os.path.join(PHOTOS_DIR, rel_path))
            # Empêche toute tentative de sortir du dossier photos/ (ex: "../../secret.txt")
            if not full_path.startswith(photos_root + os.sep):
                continue
            if os.path.isfile(full_path):
                zf.write(full_path, arcname=os.path.basename(full_path))
                added += 1

    if added == 0:
        raise HTTPException(404, "Aucune photo valide trouvée.")

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=mes_photos.zip"},
    )
