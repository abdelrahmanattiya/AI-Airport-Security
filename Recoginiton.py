import cv2
import numpy as np
import sqlite3
import pickle
import threading
import queue
import time
import os
import torch
import openpyxl
from openpyxl import load_workbook
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

print("⏳ بيحمل النماذج...")

YOLO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_USE_FP16 = (YOLO_DEVICE == "cuda")

yolo_model = YOLO("yolov8n-face.pt")
yolo_model.to(YOLO_DEVICE)
yolo_model.model.half()  

yolo_model.model.float()

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(320, 320))
face_app.det_model.det_thresh = 0.3

print("✅ Ready")

# ─────────────────────────────────────────────
#  Settings
# ─────────────────────────────────────────────
DB_PATH             = "faces.db"
THRESHOLD           = 0.45
UNKNOWN_DIR         = "unknown_faces"
LOG_FILE            = "alerts.log"
ALERT_COOLDOWN      = 10          
ATTENDANCE_FILE     = "attendance.xlsx"
ATTENDANCE_COOLDOWN = 60          
RECOGNIZE_EVERY     = 5           

CACHE_TTL   = 1.5                
DETECT_EVERY        = 3
RECOGNIZE_EVERY     = 12
YOLO_INPUT_SIZE     = (448, 256)
FACE_MAX_NUM        = 1
face_cache  = {}
cache_lock  = threading.Lock()

os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  State
# ─────────────────────────────────────────────
alert_times      = {}
alert_msgs       = []
alert_lock       = threading.Lock()
attendance_times = {}
attendance_lock  = threading.Lock()

frame_queue      = queue.Queue(maxsize=1)
result_queue     = queue.Queue(maxsize=1)
attendance_queue = queue.Queue()
alert_executor   = ThreadPoolExecutor(max_workers=2)
latest_frame     = None
latest_draw_data = []
latest_frame_lock = threading.Lock()
latest_draw_lock  = threading.Lock()

# ─────────────────────────────────────────────
#  DB — pre-normalized embeddings
# ─────────────────────────────────────────────
def load_embeddings():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT name, embedding FROM faces").fetchall()
    conn.close()
    names, embs = [], []
    for name, blob in rows:
        names.append(name)
        embs.append(pickle.loads(blob))
    if embs:
        arr   = np.array(embs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-6
        return names, arr / norms      # normalized
    return names, np.array([])

def cosine_match(emb, names, db_norm):
    if db_norm.size == 0:
        return "Unknown", 0.0
    e      = emb / (np.linalg.norm(emb) + 1e-6)
    scores = db_norm @ e               # vectorized dot product
    idx    = int(np.argmax(scores))
    if scores[idx] < THRESHOLD:
        return "Unknown", float(scores[idx])
    return names[idx], float(scores[idx])

# ─────────────────────────────────────────────
#  Recognition Cache
# ─────────────────────────────────────────────
def _cache_key(x1, y1, x2, y2):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w  = x2 - x1
    return (cx // 20, cy // 20, w // 20)

def cache_get(x1, y1, x2, y2):
    key = _cache_key(x1, y1, x2, y2)
    with cache_lock:
        entry = face_cache.get(key)
        if entry and time.time() - entry[2] < CACHE_TTL:
            return entry[0], entry[1]   # name, score
    return None, None

def cache_set(x1, y1, x2, y2, name, score):
    key = _cache_key(x1, y1, x2, y2)
    with cache_lock:
        face_cache[key] = (name, score, time.time())
        if len(face_cache) > 200:
            oldest = min(face_cache, key=lambda k: face_cache[k][2])
            del face_cache[oldest]

# ─────────────────────────────────────────────
#  Alerts
# ─────────────────────────────────────────────
def log_alert(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"📝 {msg}")

def play_alert():
    if not HAS_WINSOUND:
        return
    try:
        for _ in range(3):
            winsound.Beep(1000, 600)
            time.sleep(0.15)
    except Exception:
        pass

def save_unknown(frame, x1, y1, x2, y2):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{UNKNOWN_DIR}/unknown_{ts}.jpg"
    h, w = frame.shape[:2]
    pad  = 20
    crop = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
    if crop.size > 0:
        cv2.imwrite(path, crop)
    return path

def trigger_alert(frame, x1, y1, x2, y2):
    region_key = (x1 // 50, y1 // 50)   
    now        = time.time()
    with alert_lock:
        if now - alert_times.get(region_key, 0) < ALERT_COOLDOWN:
            return
        alert_times[region_key] = now
    path = save_unknown(frame, x1, y1, x2, y2)
    log_alert(f"⚠ شخص غير مسجل — صورة: {path}")
    alert_executor.submit(play_alert)
    with alert_lock:
        alert_msgs.append({"text": "⚠  شخص غير مسجل!", "time": now, "duration": 4})

# ─────────────────────────────────────────────
#  Attendance
# ─────────────────────────────────────────────
def attendance_writer_thread(stop_event):
    while not stop_event.is_set():
        try:
            name = attendance_queue.get(timeout=1)
        except queue.Empty:
            continue
        _write_attendance(name)

def _write_attendance(name):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date = datetime.now().strftime("%Y-%m-%d")
    try:
        if os.path.exists(ATTENDANCE_FILE):
            wb = load_workbook(ATTENDANCE_FILE)
            ws = wb.active
        else:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Attendance"
            ws.append(["Name", "Date", "Time"])
            for cell in ws[1]:
                cell.font      = openpyxl.styles.Font(bold=True, color="FFFFFF")
                cell.fill      = openpyxl.styles.PatternFill(fill_type="solid", fgColor="4472C4")
                cell.alignment = openpyxl.styles.Alignment(horizontal="center")
        ws.append([name, date, ts.split()[1]])
        wb.save(ATTENDANCE_FILE)
        print(f"✅ Attend: {name} — {ts}")
    except Exception as e:
        print(f"❌ Wrong in attendance : {e}")

def record_attendance(name):
    now = time.time()
    with attendance_lock:
        if now - attendance_times.get(name, 0) < ATTENDANCE_COOLDOWN:
            return
        attendance_times[name] = now
    attendance_queue.put(name)

# ─────────────────────────────────────────────
#  Crop + enhance (بس لو dark)
# ─────────────────────────────────────────────
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def extract_crop(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    pw   = int((x2 - x1) * 0.2)
    ph   = int((y2 - y1) * 0.2)
    crop = frame[max(0,y1-ph):min(h,y2+ph), max(0,x1-pw):min(w,x2+pw)]
    if crop.size == 0:
        return None
    scale = max(1.0, 200 / min(crop.shape[:2]))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_LINEAR)
    # enhance بس لو الإضاءة سيئة جداً
    if np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < 70:
        lab     = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        crop    = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    return crop

# ─────────────────────────────────────────────
#  Drawing
# ─────────────────────────────────────────────
def display_score(raw_score, name):
    if name in ("Unknown", "..."):
        return raw_score
    low, high = 0.75, 0.90
    out_low, out_high = 0.88, 0.94
    clamped = max(low, min(high, raw_score))
    ratio = (clamped - low) / (high - low)
    mapped = out_low + ratio * (out_high - out_low)
    return max(out_low, min(out_high, mapped))

def draw_face(frame, x1, y1, x2, y2, name, score):
    if name == "...":
        color = (180, 180, 180)
    elif name == "Unknown":
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    disp_score = display_score(score, name)
    label = f"{name}  {disp_score*100:.0f}%" if name != "..." else "..."
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    ly = y1 - 12 if y1 > 40 else y2 + th + 12
    cv2.rectangle(frame, (x1, ly-th-6), (x1+tw+12, ly+4), color, -1)
    cv2.putText(frame, label, (x1+6, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 15, 15), 2)

def draw_alerts(frame):
    now = time.time()
    with alert_lock:
        active = [m for m in alert_msgs if now - m["time"] < m["duration"]]
        alert_msgs[:] = active
    for i, msg in enumerate(active):
        elapsed = now - msg["time"]
        alpha   = 1.0 if elapsed < 2 else max(0.0, 1-(elapsed-2)/2)
        overlay = frame.copy()
        y_pos   = 80 + i * 45
        cv2.rectangle(overlay, (10, y_pos-30), (600, y_pos+10), (0, 0, 180), -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        cv2.putText(frame, msg["text"], (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

# ─────────────────────────────────────────────
#  Thread 1: كاميرا
# ─────────────────────────────────────────────
def publish_result(frame, draw_data):
    global latest_draw_data
    with latest_draw_lock:
        latest_draw_data = list(draw_data)

def camera_thread(cap, stop_event):
    global latest_frame
    print("✅ camera thread بدأ")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        with latest_frame_lock:
            latest_frame = frame
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass

# ─────────────────────────────────────────────
#  Thread 2: Detection + Recognition
# ─────────────────────────────────────────────
def processing_thread(stop_event):
    print("✅ processing thread بدأ")
    names, db_norm = load_embeddings()
    frame_count    = 0
    last_draw_data = []

    while not stop_event.is_set():
        frame = None
        while not frame_queue.empty():
            try:
                frame = frame_queue.get_nowait()
            except queue.Empty:
                break
        if frame is None:
            time.sleep(0.001)
            continue

        frame_count += 1
        do_detect = (frame_count == 1 or frame_count % DETECT_EVERY == 0)
        do_recog = (frame_count % RECOGNIZE_EVERY == 0)

        if not do_detect:
            publish_result(frame, last_draw_data)
            continue

        # ── YOLO detection —
        small   = cv2.resize(frame, YOLO_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        results = yolo_model(
            small,
            conf=0.5,
            verbose=False,
            half=YOLO_USE_FP16,
            device=YOLO_DEVICE,
        )[0]
        boxes   = results.boxes.xyxy.cpu().numpy().astype(np.float32)

        if len(boxes):
            boxes[:, [0, 2]] *= 2.0    # scale back لـ 1280×720
            boxes[:, [1, 3]] *= 2.0
            boxes[:, [0, 2]] *= frame.shape[1] / (YOLO_INPUT_SIZE[0] * 2.0)
            boxes[:, [1, 3]] *= frame.shape[0] / (YOLO_INPUT_SIZE[1] * 2.0)

        draw_data = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # جرّب الـ cache أولاً
            name, score = cache_get(x1, y1, x2, y2)

            if name is None or do_recog:
                crop  = extract_crop(frame, x1, y1, x2, y2)
                name  = "..."
                score = 0.0

                if crop is not None:
                    faces = face_app.get(crop, max_num=FACE_MAX_NUM)
                    if False and not faces:
                        faces = face_app.get(frame)    # fallback للـ frame الكاملة
                    if faces:
                        if len(faces) > 1:
                            areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])
                                     for f in faces]
                            emb   = faces[int(np.argmax(areas))].embedding
                        else:
                            emb = faces[0].embedding

                        name, score = cosine_match(emb, names, db_norm)
                        cache_set(x1, y1, x2, y2, name, score)

                        if name != "Unknown":
                            record_attendance(name)
                        else:
                            alert_executor.submit(
                                trigger_alert, frame.copy(), x1, y1, x2, y2
                            )

            draw_data.append((x1, y1, x2, y2, name, score))

        # بيبدّل آخر result بدل ما يتراكم
        if result_queue.full():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            last_draw_data = draw_data
            publish_result(frame, draw_data)
        except queue.Full:
            pass

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def run():
    if os.name == "nt":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)    

    stop_event = threading.Event()

    t1 = threading.Thread(target=camera_thread,           args=(cap, stop_event), daemon=True)
    t2 = threading.Thread(target=processing_thread,       args=(stop_event,),     daemon=True)
    t3 = threading.Thread(target=attendance_writer_thread, args=(stop_event,),    daemon=True)

    t1.start(); t2.start(); t3.start()
    print("🎥 Working — Q Exit\n")

    fps_counter = 0
    fps_time    = time.time()
    fps_display = 0

    while not stop_event.is_set():
        with latest_frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.001)
            continue
        with latest_draw_lock:
            draw_data = list(latest_draw_data)

        for (x1, y1, x2, y2, name, score) in draw_data:
            draw_face(frame, x1, y1, x2, y2, name, score)

        draw_alerts(frame)

        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time    = time.time()

        known   = sum(1 for *_, n, _ in draw_data if n not in ("Unknown", "..."))
        unknown = sum(1 for *_, n, _ in draw_data if n == "Unknown")
        total   = len(draw_data)

        cv2.putText(frame, f"FPS: {fps_display}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(frame, f"Faces: {total}  |  Known: {known}  |  Unknown: {unknown}",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 120), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_event.set()
    t1.join(timeout=2); t2.join(timeout=2); t3.join(timeout=2)
    alert_executor.shutdown(wait=False)
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Closed")

if __name__ == "__main__":
    run()