import cv2, serial, time
from collections import deque

# ===============================
# ARDUINO CONNECTION
# ===============================
PORT = 'COM18'  # Change to your Arduino port
BAUD = 9600
try:
    arduino = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print(f"✅ Connected to Arduino on {PORT}")
    connected = True
except:
    print(f"⚠️ Arduino not found on {PORT}, running in simulation mode.")
    arduino = None
    connected = False

# ===============================
# CAMERA + FACE DETECTOR
# ===============================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

CENTER_X, CENTER_Y = 320, 240
DEAD_ZONE = 80
PIXELS_PER_STEP = 8
command_count = 0
last_cmd = time.time()

# ===============================
# HELPER FUNCTIONS
# ===============================
def send_command(steps=0, direction=0):
    """Send command to Arduino: R:x, L:x, or S"""
    global command_count
    if not connected:
        print(f"[SIM] {'RIGHT' if direction>0 else 'LEFT' if direction<0 else 'STOP'} {steps}")
        return
    try:
        cmd = "S\n" if steps == 0 else f"{'R' if direction>0 else 'L'}:{steps}\n"
        arduino.write(cmd.encode())
        command_count += 1
    except Exception as e:
        print("Send error:", e)

def calc_move(face_x):
    """Return steps and direction based on horizontal offset"""
    offset = face_x - CENTER_X
    if abs(offset) < DEAD_ZONE:
        return 0, 0
    steps = max(2, min(30, abs(offset)//PIXELS_PER_STEP))
    direction = 1 if offset > 0 else -1
    return steps, direction

# ===============================
# MAIN LOOP
# ===============================
print("🟢 Face Tracking System Started")
print("Press 'q' to quit, '+' or '-' to change sensitivity.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

    # Draw center + dead zone
    cv2.rectangle(frame, (CENTER_X - DEAD_ZONE, 0), (CENTER_X + DEAD_ZONE, 480), (0, 255, 255), 2)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        cx, cy = x + w//2, y + h//2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        steps, direction = calc_move(cx)
        if time.time() - last_cmd > 0.1:
            send_command(steps, direction)
            last_cmd = time.time()

        msg = "CENTERED" if steps == 0 else ("RIGHT →" if direction > 0 else "← LEFT")
        cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        send_command(0, 0)
        cv2.putText(frame, "NO FACE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Info
    cv2.putText(frame, f"Commands: {command_count}", (480, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"Sensitivity: {PIXELS_PER_STEP}px/step", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.imshow("Face Tracking", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('+'):
        PIXELS_PER_STEP = max(3, PIXELS_PER_STEP - 1)
    elif k == ord('-'):
        PIXELS_PER_STEP = min(20, PIXELS_PER_STEP + 1)
    elif k == ord('s'):
        send_command(0, 0)

# ===============================
# CLEANUP
# ===============================
if connected:
    send_command(0, 0)
    arduino.close()
cap.release()
cv2.destroyAllWindows()
print("🛑 Stopped Face Tracking System")
