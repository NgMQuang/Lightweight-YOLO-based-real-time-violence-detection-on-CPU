import cv2
import numpy as np
import time
from ultralytics import YOLO

# ===== TRACKER FACTORY =====
TRACKER_TYPE = "MOSSE"  # or "MEDIANFLOW"

def make_tracker():
    if TRACKER_TYPE == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()
    elif TRACKER_TYPE == "MEDIANFLOW":
        return cv2.legacy.TrackerMedianFlow_create()
    raise ValueError(f"Unknown tracker: {TRACKER_TYPE}")

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)

def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=float)

def l1_center_dist_matrix(tracked_boxes, detected_boxes):
    def centers(b):
        return np.stack([(b[:,0]+b[:,2])*0.5, (b[:,1]+b[:,3])*0.5], axis=1)
    tc = centers(tracked_boxes)
    dc = centers(detected_boxes)
    diff = np.abs(tc[:,None,:] - dc[None,:,:])
    return diff.sum(axis=2)

def merge_overlapping_tracks(tracks, iou_threshold=0.45):
    """
    Merge pairs of tracks whose boxes overlap above iou_threshold.
    Keeps the higher-conf track, discards the other.
    Runs repeatedly until no more merges needed.
    """
    if len(tracks) <= 1:
        return tracks

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = (ax2-ax1) * (ay2-ay1)
        area_b = (bx2-bx1) * (by2-by1)
        return inter / (area_a + area_b - inter)

    changed = True
    while changed:
        changed = False
        keep = [True] * len(tracks)
        for i in range(len(tracks)):
            if not keep[i]:
                continue
            for j in range(i+1, len(tracks)):
                if not keep[j]:
                    continue
                if iou(tracks[i]['box'], tracks[j]['box']) >= iou_threshold:
                    # keep higher conf, absorb the other's conf slightly
                    winner, loser = (i, j) if tracks[i]['conf'] >= tracks[j]['conf'] else (j, i)
                    tracks[winner]['conf'] = max(tracks[winner]['conf'], tracks[loser]['conf'])
                    keep[loser] = False
                    changed = True
        tracks = [t for i, t in enumerate(tracks) if keep[i]]

    return tracks

# ===== CONFIG =====
FPS_VIDEO         = 30
TOTAL_TIME_DETECT = 2.5
FRAME_PER_DETECT  = 8
DETECT_INTERVAL   = int(FPS_VIDEO * TOTAL_TIME_DETECT / FRAME_PER_DETECT)

MAX_TRACKS   = 5
CONF_ON      = 0.25
CONF_OFF     = 0.1
k            = 5.0
STICK_WEIGHT = 0.7
alpha        = 0.8

COLORS = [
    (0,   0,   255),
    (0,   255, 0  ),
    (255, 0,   0  ),
    (0,   255, 255),
    (255, 0,   255),
]

# ===== LOAD =====
model = YOLO("best.pt")
cap   = cv2.VideoCapture("demovid/vid6.avi")

# Each track:
#   box    : np.array xyxy   (smoothed)
#   conf   : float
#   show   : bool
#   tracker: cv2 tracker instance or None
#   tracker_ok: bool  (False = tracker failed, use last known box until YOLO)
tracks    = []
raw_boxes = []
raw_confs = []

frame_id   = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    is_detect_frame = (frame_id % DETECT_INTERVAL == 0)

    # ===== BETWEEN-FRAME TRACKING =====
    if not is_detect_frame:
        for t in tracks:
            if t['tracker'] is None or not t['tracker_ok']:
                # tracker failed or not yet initialized → hold position
                continue

            ok, new_xywh = t['tracker'].update(frame)
            if ok:
                new_box = xywh_to_xyxy(new_xywh)
                # sanity check: box must stay within frame
                if (new_box[0] >= 0 and new_box[1] >= 0 and
                    new_box[2] <= W  and new_box[3] <= H  and
                    new_box[2] > new_box[0] and new_box[3] > new_box[1]):
                    t['box'] = alpha * t['box'] + (1 - alpha) * new_box
                    t['tracker_ok'] = True
                else:
                    t['tracker_ok'] = False  # out-of-bounds → treat as failed
            else:
                t['tracker_ok'] = False  # tracker lost target

    # ===== YOLO DETECTION =====
    if is_detect_frame:
        results = model(frame, conf=0.1, verbose=False)
        r       = results[0]

        raw_boxes = []
        raw_confs = []

        if len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])

            if tracks:
                prev     = np.array([t['box'] for t in tracks])
                dist_mat = l1_center_dist_matrix(prev, boxes) / (W + H)
                best_stick = np.clip(1.0 - k * dist_mat.min(axis=0), 0.0, 1.0)
            else:
                best_stick = np.zeros(len(boxes))

            scores  = confs * areas * (1.0 + STICK_WEIGHT * best_stick)
            top_idx = np.argsort(scores)[::-1][:MAX_TRACKS]
            raw_boxes = boxes[top_idx]
            raw_confs = confs[top_idx]

        # ===== MATCH YOLO → TRACKS =====
        if len(raw_boxes) > 0:
            det_boxes = np.array(raw_boxes)
            det_confs = np.array(raw_confs)

            if not tracks:
                for i in range(len(det_boxes)):
                    tr = make_tracker()
                    tr.init(frame, xyxy_to_xywh(det_boxes[i]))
                    tracks.append({
                        'box':        det_boxes[i],
                        'conf':       det_confs[i],
                        'show':       False,
                        'tracker':    tr,
                        'tracker_ok': True,
                    })
                    tracks = merge_overlapping_tracks(tracks) 
            else:
                prev     = np.array([t['box'] for t in tracks])
                dist_mat = l1_center_dist_matrix(prev, det_boxes) / (W + H)

                matched_det = set()
                matched_trk = set()

                flat_order = np.argsort(dist_mat, axis=None)
                for idx in flat_order:
                    ti, di = divmod(int(idx), dist_mat.shape[1])
                    if ti in matched_trk or di in matched_det:
                        continue
                    # Re-init tracker on every YOLO frame for matched tracks
                    # (corrects any tracker drift)
                    tr = make_tracker()
                    tr.init(frame, xyxy_to_xywh(det_boxes[di]))
                    tracks[ti]['box']        = alpha * tracks[ti]['box'] + (1-alpha) * det_boxes[di]
                    tracks[ti]['conf']       = alpha * tracks[ti]['conf'] + (1-alpha) * det_confs[di]
                    tracks[ti]['tracker']    = tr
                    tracks[ti]['tracker_ok'] = True
                    matched_trk.add(ti)
                    matched_det.add(di)
                    if len(matched_trk) == min(len(tracks), len(det_boxes)):
                        break

                # Unmatched detections → new tracks
                for di in range(len(det_boxes)):
                    if di not in matched_det and len(tracks) < MAX_TRACKS:
                        tr = make_tracker()
                        tr.init(frame, xyxy_to_xywh(det_boxes[di]))
                        tracks.append({
                            'box':        det_boxes[di],
                            'conf':       det_confs[di],
                            'show':       False,
                            'tracker':    tr,
                            'tracker_ok': True,
                        })

                # Unmatched tracks → decay
                for ti in range(len(tracks)):
                    if ti not in matched_trk:
                        tracks[ti]['conf']       *= 0.5
                        tracks[ti]['tracker_ok']  = False  # don't trust tracker either

        else:
            # YOLO found nothing → decay all tracks
            for t in tracks:
                t['conf']       *= 0.5
                t['tracker_ok']  = False

        # Prune dead tracks
        tracks = [t for t in tracks if t['conf'] >= CONF_OFF]
        tracks = merge_overlapping_tracks(tracks) 

    # ===== HYSTERESIS =====
    for t in tracks:
        if t['conf'] >= CONF_ON:
            t['show'] = True
        elif t['conf'] < CONF_OFF:
            t['show'] = False

    # ===== DRAW =====
    for i, t in enumerate(tracks):
        if not t['show']:
            continue
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = map(int, t['box'])
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        # Show tracker status in label
        status = "OK" if t['tracker_ok'] else "HOLD"
        cv2.putText(frame, f"#{i} {t['conf']:.2f} [{status}]", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

elapsed = time.time() - start_time
print(f"Processed {frame_id} frames in {elapsed:.2f}s ({frame_id/elapsed:.2f} FPS)")

cap.release()
cv2.destroyAllWindows()