"""
Script d'indexation.

À lancer :
  - une première fois pour indexer un lot existant
  - puis en boucle (cron, ou `while true; do python index_photos.py; sleep 30; done`)
    pendant l'événement pour indexer les nouvelles photos au fil de l'eau.

Il ne retraite jamais un fichier déjà indexé (table indexed_files),
donc on peut le relancer aussi souvent qu'on veut sans dupliquer.
"""

import os
import time
import cv2

from common import (
    PHOTOS_DIR, get_face_app, get_db, load_or_create_index,
    save_index, normalize,
)

VALID_EXT = {".jpg", ".jpeg", ".png"}


def list_new_photos(conn):
    already = {row[0] for row in conn.execute("SELECT photo_path FROM indexed_files")}
    new_files = []
    for root, _, files in os.walk(PHOTOS_DIR):
        for f in files:
            if os.path.splitext(f)[1].lower() not in VALID_EXT:
                continue
            rel_path = os.path.relpath(os.path.join(root, f), PHOTOS_DIR)
            if rel_path not in already:
                new_files.append(rel_path)
    return sorted(new_files)


def main():
    conn = get_db()
    index = load_or_create_index()
    new_files = list_new_photos(conn)

    if not new_files:
        print("Aucune nouvelle photo à indexer.")
        return

    print(f"{len(new_files)} nouvelle(s) photo(s) à indexer...")
    face_app = get_face_app()

    t0 = time.time()
    total_faces = 0

    for i, rel_path in enumerate(new_files, 1):
        full_path = os.path.join(PHOTOS_DIR, rel_path)
        img = cv2.imread(full_path)
        if img is None:
            print(f"  [!] Impossible de lire {rel_path}, ignoré.")
            conn.execute(
                "INSERT OR REPLACE INTO indexed_files (photo_path, n_faces) VALUES (?, 0)",
                (rel_path,),
            )
            continue

        faces = face_app.get(img)
        for face in faces:
            emb = normalize(face.embedding)
            faiss_id = index.ntotal
            index.add(emb.reshape(1, -1))
            bbox = ",".join(str(int(v)) for v in face.bbox)
            conn.execute(
                "INSERT INTO faces (faiss_id, photo_path, bbox) VALUES (?, ?, ?)",
                (faiss_id, rel_path, bbox),
            )
        total_faces += len(faces)

        conn.execute(
            "INSERT OR REPLACE INTO indexed_files (photo_path, n_faces) VALUES (?, ?)",
            (rel_path, len(faces)),
        )

        if i % 20 == 0:
            conn.commit()
            save_index(index)
            print(f"  ... {i}/{len(new_files)} photos traitées")

    conn.commit()
    save_index(index)

    elapsed = time.time() - t0
    print(f"Terminé : {len(new_files)} photos, {total_faces} visages indexés en {elapsed:.1f}s "
          f"({elapsed/max(len(new_files),1):.2f}s/photo).")


if __name__ == "__main__":
    main()
