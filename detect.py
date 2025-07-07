from ultralytics import YOLO
import cv2, numpy as np, math

model = YOLO("yolov8n-pose.pt")

def calc_angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    ab, cb = a - b, c - b
    mag = np.linalg.norm(ab) * np.linalg.norm(cb)
    if mag == 0: 
        return None            # return None if vectors collapse
    return int(                 # whole‑degree integer
        math.degrees(
            np.arccos(np.clip(np.dot(ab, cb) / mag, -1, 1))
        )
    )

# COCO keypoints (left side for demo)
L_SH, L_EL, L_WR = 5, 7, 9
L_HIP, L_KNEE    = 11, 13
L_ANK            = 15  # not used here

def elbow_angle(kp):
    if kp[[L_SH, L_EL, L_WR], 2].min() > 0.5:
        return calc_angle(kp[L_SH][:2], kp[L_EL][:2], kp[L_WR][:2])
    return None

def hip_angle(kp):
    if kp[[L_SH, L_HIP, L_KNEE], 2].min() > 0.5:
        return calc_angle(kp[L_SH][:2], kp[L_HIP][:2], kp[L_KNEE][:2])
    return None

def shoulder_angle(kp):
    if kp[[L_EL, L_SH, L_HIP], 2].min() > 0.5:
        return calc_angle(kp[L_EL][:2], kp[L_SH][:2], kp[L_HIP][:2])
    return None

# ── video / webcam loop ─────────────────────────────────────────
for res in model(source=0, stream=True, show=False):
    if res.keypoints is None or res.keypoints.data.shape[0] == 0:
        continue

    frame = cv2.flip(res.orig_img, 1)     # un‑mirror
    kp    = res.keypoints.data[0].cpu().numpy()

    e = elbow_angle(kp)
    h = hip_angle(kp)
    s = shoulder_angle(kp)

    parts = []
    if e is not None: parts.append(f"Elbow:{e}")
    if h is not None: parts.append(f"Hip:{h}")
    if s is not None: parts.append(f"Shoulder:{s}")

    if parts:
        cv2.putText(frame, "  ".join(parts), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Angles  (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
