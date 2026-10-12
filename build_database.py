# build_database.py
import cv2
import numpy as np
import sqlite3
import pickle
import os
from pathlib import Path
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ── إعداد النماذج ─────────────────────────────
yolo_model = YOLO("yolov8n-face.pt")
face_app   = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(640, 640))  # -1 = CPU

DB_PATH      = "faces.db"
DATASET_PATH = "database"          # ← غير ده لمسار الفولدرات عندك
IMG_EXTS     = {".jpg", ".jpeg", ".png"}

# ── إنشاء الـ DB ──────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# ── استخراج embedding من صورة ────────────────
def get_embedding(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # YOLOv8: detect الوجوه
    results = yolo_model(img, conf=0.4, verbose=False)[0]
    boxes   = results.boxes.xyxy.cpu().numpy().astype(int)

    if len(boxes) == 0:
        return None

    # خد أكبر وش (الأوضح في الصورة)
    areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes]
    x1,y1,x2,y2 = boxes[int(np.argmax(areas))]

    # margin صغير حوالين الوش
    h, w = img.shape[:2]
    pad  = 30
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(w, x2+pad), min(h, y2+pad)

    crop  = img[y1:y2, x1:x2]
    faces = face_app.get(crop)

    if not faces:
        # جرب على الصورة كلها لو الـ crop مش شغال
        faces = face_app.get(img)
    if not faces:
        return None

    return faces[0].normed_embedding

# ── بناء الـ DB من الفولدرات ──────────────────
def build_database():
    init_db()
    dataset = Path(DATASET_PATH)

    if not dataset.exists():
        print(f"❌ المسار '{DATASET_PATH}' مش موجود")
        return

    # امسح الـ DB القديمة لو موجودة
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM faces")
    conn.commit()
    conn.close()

    people_dirs = [d for d in dataset.iterdir() if d.is_dir()]
    if not people_dirs:
        print("❌ مفيش فولدرات في الـ dataset")
        return

    total_enrolled = 0
    total_skipped  = 0

    for person_dir in sorted(people_dirs):
        name   = person_dir.name
        images = [f for f in person_dir.iterdir() if f.suffix.lower() in IMG_EXTS]

        if not images:
            print(f"  ⚠  {name}: مفيش صور")
            continue

        print(f"\n👤 {name}  ({len(images)} صورة)")

        embeddings = []
        for img_path in images:
            emb = get_embedding(img_path)
            if emb is not None:
                embeddings.append(emb)
                print(f"   ✓ {img_path.name}")
            else:
                print(f"   ✗ {img_path.name}  (مفيش وش اتلقى)")
                total_skipped += 1

        if not embeddings:
            print(f"   ⚠  مفيش embedding اتحسب لـ {name}، هيتخطى")
            continue

        # Average embedding من كل الصور
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding /= np.linalg.norm(avg_embedding)   # normalize

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO faces (name, embedding) VALUES (?, ?)",
            (name, pickle.dumps(avg_embedding))
        )
        conn.commit()
        conn.close()

        print(f"   ✅ saved {name} ({len(embeddings)}/{len(images)} photo used)")
        total_enrolled += 1

    print(f"\n{'─'*45}")
    print(f"✅ saved  : {total_enrolled} person")
    print(f"⚠  skipped : {total_skipped}")
    print(f"📁 saved in DB: {DB_PATH}")

# ── تحقق من الـ DB بعد البناء ─────────────────
def verify_database():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, name FROM faces").fetchall()
    conn.close()
    print(f"\n📋 DB have ({len(rows)} person):")
    for row in rows:
        print(f"   [{row[0]}] {row[1]}")

if __name__ == "__main__":
    build_database()
    verify_database()