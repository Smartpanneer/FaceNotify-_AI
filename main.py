import cv2
import numpy as np
import smtplib
import os
from datetime import datetime, timedelta
from email.message import EmailMessage
from insightface.app import FaceAnalysis
import mysql.connector

# ─────────────────────────────────────────────────────────────
#  DATABASE CONFIG
# ─────────────────────────────────────────────────────────────
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1972@Vkp",
    database="praveen"
)
cursor = db.cursor(dictionary=True)
print("✅ Database connected")

# ─────────────────────────────────────────────────────────────
#  EMAIL CONFIG
# ─────────────────────────────────────────────────────────────
SENDER_EMAIL    = "vpraveen1005@gmail.com"
SENDER_PASSWORD = "itavwjduxsnqcnex"
RECEIVER_EMAIL  = "panneer.ece2020@gmail.com"

# ─────────────────────────────────────────────────────────────
#  TIMING CONFIG
# ─────────────────────────────────────────────────────────────
BREAK_START      = "11:47"   # break begins  (no alerts during break)
BREAK_END        = "11:49"   # break ends    (alerts fire after this)
ALERT_INTERVAL   = 10        # minutes between repeated alerts for same student

# ─────────────────────────────────────────────────────────────
#  FACE MODEL
# ─────────────────────────────────────────────────────────────
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

# ─────────────────────────────────────────────────────────────
#  LOAD STUDENT FACE EMBEDDINGS
# ─────────────────────────────────────────────────────────────
known_embeddings = []
known_ids        = []
STUDENTS_FOLDER  = "students"

for file in os.listdir(STUDENTS_FOLDER):
    img_path = os.path.join(STUDENTS_FOLDER, file)
    img      = cv2.imread(img_path)
    if img is None:
        continue
    faces = face_app.get(img)
    if faces:
        known_embeddings.append(faces[0].embedding)
        known_ids.append(os.path.splitext(file)[0])

print(f"✅ Loaded {len(known_ids)} students:", known_ids)

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def _parse_time(t_str):
    return datetime.strptime(t_str, "%H:%M").time()

def is_break_time():
    """Returns True while break is in progress (alerts suppressed)."""
    now  = datetime.now().time()
    return _parse_time(BREAK_START) <= now <= _parse_time(BREAK_END)

def is_after_break():
    """Returns True only after break has ended (alerts allowed)."""
    return datetime.now().time() > _parse_time(BREAK_END)

def get_student_details(enroll_id):
    cursor.execute("SELECT * FROM student_info WHERE enroll_id = %s", (enroll_id,))
    return cursor.fetchone()

def best_face_frame(cap, n_frames=5):
    """
    Grab n_frames and return the one where the detected face
    has the largest bounding-box area (= closest / clearest shot).
    Falls back to the last captured frame if no face found.
    """
    best_frame = None
    best_area  = -1
    for _ in range(n_frames):
        ret, f = cap.read()
        if not ret:
            continue
        faces = face_app.get(f)
        if faces:
            b    = faces[0].bbox.astype(int)
            area = (b[2]-b[0]) * (b[3]-b[1])
            if area > best_area:
                best_area  = area
                best_frame = f.copy()
        else:
            if best_frame is None:
                best_frame = f.copy()
    return best_frame

# ─────────────────────────────────────────────────────────────
#  EMAIL FUNCTION
# ─────────────────────────────────────────────────────────────
def send_email(image_path, student, alert_type="initial"):
    """
    alert_type:
      'initial'   → first detection after break
      'followup'  → student still in canteen 10 min later
    """
    try:
        msg = EmailMessage()

        if alert_type == "initial":
            msg['Subject'] = "🚨 Alert: Student in Canteen After Break"
            body_header    = "Student detected IN CANTEEN after break time."
        else:
            mins = ALERT_INTERVAL
            msg['Subject'] = f"⚠️ Follow-up: Student STILL in Canteen ({mins} min)"
            body_header    = (f"Student is STILL in the canteen "
                              f"{mins} minutes after first alert!")

        msg['From'] = SENDER_EMAIL
        msg['To']   = RECEIVER_EMAIL

        msg.set_content(f"""
Hello Sir,

{body_header}

──────────────────────────────────
Name       : {student['stu_name']}
Enrollment : {student['enroll_id']}
Department : {student['dept']}
Year       : {student['stu_year']}
Section    : {student['section']}
Time       : {datetime.now().strftime("%d-%m-%Y  %H:%M:%S")}
──────────────────────────────────

Please take necessary action.

Regards,
AI Monitoring System
""")

        with open(image_path, 'rb') as f:
            msg.add_attachment(f.read(), maintype='image',
                               subtype='jpeg', filename='capture.jpg')

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)

        tag = "✅ Initial" if alert_type == "initial" else "✅ Follow-up"
        print(f"{tag} email sent for {student['stu_name']} at "
              f"{datetime.now().strftime('%H:%M:%S')}")

    except Exception as e:
        print("❌ Email failed:", e)

# ─────────────────────────────────────────────────────────────
#  STATE TRACKERS
# ─────────────────────────────────────────────────────────────
# student_id → datetime of last email sent
last_email_time: dict[str, datetime] = {}

def should_send_alert(student_id) -> tuple[bool, str]:
    """
    Returns (True, alert_type) when an alert should fire,
    (False, '') otherwise.

    Rules:
    - During break  → never send
    - After break   → send 'initial' the first time
    - After break   → send 'followup' every ALERT_INTERVAL minutes if still seen
    """
    if not is_after_break():
        return False, ""

    if student_id not in last_email_time:
        return True, "initial"

    elapsed = datetime.now() - last_email_time[student_id]
    if elapsed >= timedelta(minutes=ALERT_INTERVAL):
        return True, "followup"

    return False, ""

# ─────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
# Increase capture resolution for better quality images
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🎥 Camera started — Press Q to quit")
print(f"   Break window : {BREAK_START} → {BREAK_END}")
print(f"   Re-alert every {ALERT_INTERVAL} minutes if student remains")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_app.get(frame)

    # ── Status overlay ──────────────────────────────────────
    now_str = datetime.now().strftime("%H:%M:%S")
    if is_break_time():
        status_txt   = f"BREAK TIME — monitoring paused  {now_str}"
        status_color = (0, 200, 200)
    elif is_after_break():
        status_txt   = f"POST-BREAK monitoring ACTIVE  {now_str}"
        status_color = (0, 255, 100)
    else:
        status_txt   = f"Pre-break monitoring  {now_str}"
        status_color = (200, 200, 200)

    cv2.putText(frame, status_txt, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    for face in faces:
        bbox      = face.bbox.astype(int)
        embedding = face.embedding

        student    = None
        name       = "Unknown"
        student_id = None

        # ── Face matching ──────────────────────────────────
        best_sim = 0
        for i, known_emb in enumerate(known_embeddings):
            sim = np.dot(known_emb, embedding) / (
                np.linalg.norm(known_emb) * np.linalg.norm(embedding))
            if sim > best_sim:
                best_sim = sim
                if sim > 0.5:
                    student_id = known_ids[i]

        if student_id:
            student = get_student_details(student_id)
            name    = student["stu_name"] if student else "Not in DB"

        # ── Alert logic ────────────────────────────────────
        if student and student_id:
            fire, alert_type = should_send_alert(student_id)
            if fire:
                # Capture best quality frame before saving
                quality_frame = best_face_frame(cap, n_frames=5)
                if quality_frame is None:
                    quality_frame = frame

                image_path = f"{student_id}_after_break.jpg"
                cv2.imwrite(image_path, quality_frame)
                print(f"📸 Captured quality image for {name}")

                send_email(image_path, student, alert_type)
                last_email_time[student_id] = datetime.now()

        # ── Draw bounding box ──────────────────────────────
        box_color = (0, 255, 100) if student else (0, 100, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      box_color, 2)

        if student:
            display = (f"{student['stu_name']}  |  {student['dept']}  "
                       f"|  Y{student['stu_year']}  |  {student['section']}")
            # Show next-alert countdown
            if student_id in last_email_time:
                elapsed = (datetime.now() - last_email_time[student_id]).seconds // 60
                remaining = ALERT_INTERVAL - elapsed
                display += f"  [next alert ~{remaining}m]"
        else:
            display = name

        cv2.putText(frame, display,
                    (bbox[0], max(bbox[1] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    cv2.imshow("AI Canteen Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
db.close()
print("🛑 Monitor stopped.")