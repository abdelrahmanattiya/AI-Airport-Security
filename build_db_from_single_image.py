import cv2
import numpy as np
import sqlite3
import pickle
from pathlib import Path
from ultralytics import YOLO
from insightface.app import FaceAnalysis

yolo_model = YOLO("yolov8n-face.pt")
yolo_model.to("cuda")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

DB_PATH      = "faces.db"
DATASET_PATH = "database"
IMG_EXTS     = {".jpg", ".jpeg", ".png"}

# ── Augmentations على الـ crop ────────────────
def augment(crop):
    """بتوّلد نسخ مختلفة من الوش الواحد"""
    variants = [crop]  # الأصلي

    # 1. flip أفقي (كأنه بص ناحية تانية)
    variants.append(cv2.flip(crop, 1))

    # 2. تدوير بزوايا مختلفة
    h, w = crop.shape[:2]
    cx, cy = w // 2, h // 2
    for angle in [-15, -10, 10, 15]:
        M   = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot = cv2.warpAffine(crop, M, (w, h))
        variants.append(rot)

    # 3. brightness مختلفة (إضاءة مختلفة)
    for factor in [0.75, 1.25]:
        bright = np.clip(crop.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        variants.append(bright)

    # 4. perspective transform خفيف (محاكاة الزاوية)
    for shift in [-0.07, 0.07]:
        pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        pts2 = np.float32([
            [w*max(0,  shift), 0],
            [w*min(1,1+shift), 0],
            [0, h],
            [w, h]
        ])
        M   = cv2.getPerspectiveTransform(pts1, pts2)
        per = cv2.warpPerspective(crop, M, (w, h))
        variants.append(per)

    return variants

# ── استخراج embedding مع augmentation ─────────
def get_embeddings(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return []

    results = yolo_model(img, conf=0.3, verbose=False)[0]
    boxes   = results.boxes.xyxy.cpu().numpy().astype(int)
    if len(boxes) == 0:
        return []

    # أكبر وش
    areas    = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes]
    x1,y1,x2,y2 = boxes[int(np.argmax(areas))]

    h, w = img.shape[:2]
    pw   = int((x2-x1) * 0.2)
    ph   = int((y2-y1) * 0.2)
    nx1  = max(0, x1-pw);  ny1 = max(0, y1-ph)
    nx2  = min(w, x2+pw);  ny2 = min(h, y2+ph)
    crop = img[ny1:ny2, nx1:nx2]

    if crop.size == 0:
        return []

    # نعمل augmentation على الـ crop
    variants   = augment(crop)
    embeddings = []

    for variant in variants:
        faces = face_app.get(variant)
        if faces:
            embeddings.append(faces[0].embedding)

    return embeddings

# ── بناء الـ DB ───────────────────────────────
def build():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    conn.execute("DELETE FROM faces")
    conn.commit()
    conn.close()

    dataset    = Path(DATASET_PATH)
    total_ok   = 0
    total_skip = 0

    for person_dir in sorted(dataset.iterdir()):
        if not person_dir.is_dir():
            continue

        name   = person_dir.name
        images = [f for f in person_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
        print(f"\n👤 {name}  ({len(images)} صورة)")

        all_embeddings = []
        for img_path in images:
            embs = get_embeddings(img_path)
            if embs:
                all_embeddings.extend(embs)
                print(f"   ✓ {img_path.name} ← {len(embs)} variant")
            else:
                print(f"   ✗ {img_path.name}")
                total_skip += 1

        if not all_embeddings:
            print(f"   ⚠ مفيش embedding لـ {name}")
            continue

        # average من كل الـ variants
        avg = np.mean(all_embeddings, axis=0)
        avg /= np.linalg.norm(avg)

        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)",
                     (name, pickle.dumps(avg)))
        conn.commit()
        conn.close()

        print(f"   ✅ {name} — {len(all_embeddings)} embedding (من {len(images)} صورة)")
        total_ok += 1

    print(f"\n{'─'*45}")
    print(f"✅ {total_ok} شخص | ⚠ {total_skip} صورة اتخطت")

if __name__ == "__main__":
    build()