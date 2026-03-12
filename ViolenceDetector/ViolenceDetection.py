import cv2
import numpy as np
import onnxruntime as ort
from utilities import *
import logging
from collections import deque
import time

#System parameters
TOTAL_TIME_DETECT = 2.5 #1.64 for HKF and Movies    # Detection window duration (seconds)
FRAME_PER_DETECT  = 8                               # Number of classifier inputs per window

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ViolenceDetector:
    def __init__(self, video_path, yolo_model = 'violence_yolo.onnx', temporalhead = 'temporal_classifier.onnx', tracker='MEDIANFLOW', **kwargs):

        # open video
        self.cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(0)   # '0' or path to video
        if not self.cap.isOpened():
            raise FileNotFoundError("Video file not found")

        # ===== CONFIG =====
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps_video = fps if fps > 0 else 30 # frame rate of the input video (default to 30 if unknown)
        self.detect_interval = max(1, int(self.fps_video * TOTAL_TIME_DETECT / FRAME_PER_DETECT))  # Frames between each detection (e.g., every 10 frames for 30 FPS)
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        logger.info(f"Video detected: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                    f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {self.fps_video}fps")
        
        self.YoLO_session, self.TemporalClassifier_session = self.load_onnx_models(yolo_model, temporalhead)
        self.yolo_input_name, self.yolo_output_names, self.gap_input_name, self.gap_output_name = self.name_onnx_model()
        

        # Tracking parameters
        self.tracker      = tracker                                          # Tracker type (e.g., "KCF", "MOSSE", "MEDIANFLOW")
        self.max_tracks   = kwargs.get("max_tracks", 5)                      # Maximum simultaneous tracks
        self.conf_on      = kwargs.get("conf_on", 0.25)                      # Confidence threshold to show track
        self.conf_off     = kwargs.get("conf_off", 0.1)                      # Confidence threshold to show/hide tracks (hysteresis)
        self.k            = kwargs.get("k", 5.0)                             # Distance penalty multiplier (higher = prefer existing tracks)
        self.stick_weight = kwargs.get("stick_weight", 0.7)                  # Weight of stickiness in scoring
        self.alpha        = kwargs.get("alpha", 0.8)                         # EMA smoothing factor for box coordinates (0.8 = 80% old, 20% new)
        self.tracker_failure_decay = kwargs.get("tracker_failure_decay", 0.5)# Confidence decay factor when tracker fails (0.5 = halve confidence)

        self.colors = kwargs.get("colors",  [
            (0,   0,   255),
            (0,   255, 0  ),
            (255, 0,   0  ),
            (0,   255, 255),
            (255, 0,   255),
        ])

        self.resolution = kwargs.get("resolution", (256, 320)) # YOLO model input size (height, width)

        # Initialization
        self.frame_id   = 0
        self.tracks    = []
        self.feats     = deque(maxlen=FRAME_PER_DETECT)  # store last N features for classifier input

    # Load ONNX model
    def load_onnx_models(self, yolo_model, temporalhead)->tuple[ort.InferenceSession, ort.InferenceSession]:
        """Load ONNX models with proper error handling."""
        model_paths = {
            'yolo': yolo_model,
            'gap': temporalhead
        }
        
        try:
            yolo_session = ort.InferenceSession(
                model_paths['yolo'],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            gap_session = ort.InferenceSession(
                model_paths['gap'],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            return yolo_session, gap_session
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            exit(1)
        except Exception as e:
            logger.error(f"Failed to load ONNX models: {e}")
            exit(1)

    def name_onnx_model(self)->tuple[str, list[str], str, str]:
        """Extract input and output names from ONNX sessions."""
        yolo_input_name = self.YoLO_session.get_inputs()[0].name
        yolo_output_names = [o.name for o in self.YoLO_session.get_outputs()]

        gap_input_name = self.TemporalClassifier_session.get_inputs()[0].name
        gap_output_name = self.TemporalClassifier_session.get_outputs()[0].name
        return yolo_input_name,yolo_output_names,gap_input_name,gap_output_name
    
    def Detect(self, frame)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run YOLO detection and return rescaled boxes, feature vector, and areas."""
        # Resize to model size
        img, scale, pad_x, pad_y = letterbox_image(frame, self.resolution)

        # Convert uint8 → float32 with proper shape
        img_input = img.astype(np.float32) / 255.0  # uint8 to float32
        img_input = np.transpose(img_input, (2, 0, 1))  # HWC → CHW
        img_input = np.expand_dims(img_input, 0)  # Add batch dimension

        yolo_outputs = self.YoLO_session.run(
            self.yolo_output_names, 
            {self.yolo_input_name: img_input}
        )

        detections = yolo_outputs[0]     # (1, 5, 6)
        feature    = yolo_outputs[-1]     # (1, 896, 15)

        detections = np.squeeze(detections, axis=0)
        feature    = np.squeeze(feature, axis=0)
        
        
        boxes = rescale_boxes(detections, scale, pad_x, pad_y, (self.H, self.W))  # rescale to original frame size
        boxes, areas = merge_overlapping_boxes(boxes)  # merge overlapping detections into one (optional, can help reduce noise)
        
        return boxes, feature, areas
    
    def score_track(self, boxes, areas)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(boxes) == 0:
            return np.empty((0,4)), np.array([]), np.array([])
        confs = boxes[:, 4]  # Extract confidence scores from the last column
        boxes = boxes[:, :4]  # Extract box coordinates

        if self.tracks:
            prev     = np.array([t['box'] for t in self.tracks])
            dist_mat = l1_center_dist_matrix(prev, boxes) / (self.W + self.H)
            best_stick = np.clip(1.0 - self.k * dist_mat.min(axis=0), 0.0, 1.0)
        else:
            best_stick = np.zeros(len(boxes))

        scores  = confs * areas * (1.0 + self.stick_weight * best_stick)
        top_idx = np.argsort(scores)[::-1][:self.max_tracks]  # Get indices of top scoring boxes
        return boxes,confs,top_idx

    def tracking(self, frame, det_boxes, det_confs, i):
        tr = make_tracker(self.tracker)
        tr.init(frame, xyxy_to_xywh(det_boxes[i]))
        self.tracks.append({
                            'box':        det_boxes[i],
                            'conf':       det_confs[i],
                            'show':       False,
                            'tracker':    tr,
                            'tracker_ok': True,
                        })
    
    def run(self):
        """The debugging and analysis mode with visualization for system behavior understanding."""

        # Timing tracking
        timing_stats = {
            'tracking': [],
            'detection': [],
            'matching': [],
            'classifier': [],
            'visualization': [],
            'total_frame': []
        }
        
        frame_count = 0

        while True: #For safety, put a condition to break loop if needed (e.g., max frames)
            frame_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                break
            is_detect_frame = (self.frame_id % self.detect_interval == 0)
            is_classifier_frame = (self.frame_id % self.detect_interval == self.detect_interval//2)

            # ===== BETWEEN-FRAME TRACKING =====
            tracking_start = time.time()
            
            if not is_detect_frame:
                for t in self.tracks:
                    if t['tracker'] is None or not t['tracker_ok']:
                        # tracker failed or not yet initialized → hold position
                        continue

                    ok, new_xywh = t['tracker'].update(frame)
                    if ok:
                        new_box = xywh_to_xyxy(new_xywh)
                        # sanity check: box must stay within frame
                        if (new_box[0] >= 0 and new_box[1] >= 0 and
                            new_box[2] <= self.W  and new_box[3] <= self.H  and
                            new_box[2] > new_box[0] and new_box[3] > new_box[1]):
                            t['box'] = self.alpha * t['box'] + (1 - self.alpha) * new_box
                            t['tracker_ok'] = True
                        else:
                            t['tracker_ok'] = False  # out-of-bounds → treat as failed
                    else:
                        t['tracker_ok'] = False  # tracker lost target
                        t['conf'] *= self.tracker_failure_decay # decay confidence if tracker fails
            
            tracking_time = time.time() - tracking_start
            timing_stats['tracking'].append(tracking_time)

            # ===== YOLO DETECTION =====
            detection_start = time.time()
            
            if is_detect_frame:
                boxes, feature, areas = self.Detect(frame)
                self.feats.append(feature)

                raw_boxes = []
                raw_confs = []

                if len(boxes) > 0:

                    boxes, confs, top_idx = self.score_track(boxes, areas)
                    raw_boxes = boxes[top_idx]
                    raw_confs = confs[top_idx]

                detection_time = time.time() - detection_start
                timing_stats['detection'].append(detection_time)

                # ===== MATCH YOLO → TRACKS =====
                matching_start = time.time()
                
                if len(raw_boxes) > 0:
                    det_boxes = np.array(raw_boxes)
                    det_confs = np.array(raw_confs)

                    if not self.tracks:
                        for i in range(len(det_boxes)):
                            self.tracking(frame, det_boxes, det_confs, i)
                            self.tracks = merge_overlapping_tracks(self.tracks) 
                    else:
                        prev     = np.array([t['box'] for t in self.tracks])
                        dist_mat = l1_center_dist_matrix(prev, det_boxes) / (self.W + self.H)

                        matched_det = set()
                        matched_trk = set()

                        flat_order = np.argsort(dist_mat, axis=None)
                        for idx in flat_order:
                            ti, di = divmod(int(idx), dist_mat.shape[1])
                            if ti in matched_trk or di in matched_det:
                                continue
                            # Re-init tracker on every YOLO frame for matched tracks
                            # (corrects any tracker drift)
                            tr = make_tracker(self.tracker)
                            tr.init(frame, xyxy_to_xywh(det_boxes[di]))
                            self.tracks[ti]['box']        = self.alpha * self.tracks[ti]['box'] + (1-self.alpha) * det_boxes[di]
                            self.tracks[ti]['conf']       = self.alpha * self.tracks[ti]['conf'] + (1-self.alpha) * det_confs[di]
                            self.tracks[ti]['tracker']    = tr
                            self.tracks[ti]['tracker_ok'] = True
                            matched_trk.add(ti)
                            matched_det.add(di)
                            if len(matched_trk) == min(len(self.tracks), len(det_boxes)):
                                break

                        # Unmatched detections → new tracks
                        for di in range(len(det_boxes)):
                            if di not in matched_det and len(self.tracks) < self.max_tracks:
                                self.tracking(frame, det_boxes, det_confs, di)

                        # Unmatched tracks → decay
                        for ti in range(len(self.tracks)):
                            if ti not in matched_trk:
                                self.tracks[ti]['conf']       *= self.tracker_failure_decay
                                self.tracks[ti]['tracker_ok']  = False  # don't trust tracker either

                else:
                    # YOLO found nothing → decay all tracks
                    for t in self.tracks:
                        t['conf']       *= self.tracker_failure_decay
                        t['tracker_ok']  = False

                # Prune dead tracks
                self.tracks = [t for t in self.tracks if t['conf'] >= self.conf_off]
                self.tracks = merge_overlapping_tracks(self.tracks) 
                
                matching_time = time.time() - matching_start
                timing_stats['matching'].append(matching_time)
            else:
                timing_stats['detection'].append(0.0)
                timing_stats['matching'].append(0.0)

            # ===== CLASSIFIER INFERENCE =====
            classifier_start = time.time()
            
            if is_classifier_frame and len(self.feats) >= FRAME_PER_DETECT:
                # Run classifier on the last 8 features
                seq = np.asarray(self.feats)  # shape: (8,896, 15)
                seq = np.expand_dims(seq, 0)    # (1, 8, 896, 15)
                gap_output = self.TemporalClassifier_session.run(
                    [self.gap_output_name],
                    {self.gap_input_name: seq}
                )

                logits = gap_output[0]   # shape (1,2)

                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

                violence_prob = probs[0][1]   # class index 1 = violence
                logger.info(f"Frame {self.frame_id}: Violence probability = {violence_prob:.4f} "
                        f"(Active tracks: {len(self.tracks)})")
            
                # Optional: Alert on high confidence
                if violence_prob > 0.8:
                    logger.warning(f"HIGH VIOLENCE DETECTED!")
                    cv2.putText(frame, f"⚠ VIOLENCE: {violence_prob:.2f}", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                                (0, 0, 255), 3)
            
            classifier_time = time.time() - classifier_start
            timing_stats['classifier'].append(classifier_time)
            
            # ===== HYSTERESIS =====
            for t in self.tracks:
                if t['conf'] >= self.conf_on:
                    t['show'] = True
                elif t['conf'] < self.conf_off:
                    t['show'] = False

            # ===== DRAW =====
            viz_start = time.time()
            
            for i, t in enumerate(self.tracks):
                if not t['show']:
                    continue
                color = self.colors[i % len(self.colors)]
                x1, y1, x2, y2 = map(int, t['box'])
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                # Show tracker status in label
                status = "OK" if t['tracker_ok'] else "HOLD"
                cv2.putText(frame, f"#{i} {t['conf']:.2f} [{status}]", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("demo", frame)
            
            viz_time = time.time() - viz_start
            timing_stats['visualization'].append(viz_time)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            total_frame_time = time.time() - frame_start
            timing_stats['total_frame'].append(total_frame_time)
            
            # Log timing every 30 frames
            if frame_count % 30 == 0:
                self._log_timing_stats(timing_stats, frame_count)

            self.frame_id += 1

        self.cap.release()
        cv2.destroyAllWindows()
        
        # Final timing report
        logger.info("=" * 80)
        logger.info("FINAL TIMING ANALYSIS")
        logger.info("=" * 80)
        self._log_timing_stats(timing_stats, frame_count, final=True)

    def _log_timing_stats(self, timing_stats, frame_count, final=False):
        """Log timing statistics for performance analysis."""
        
        def safe_avg(times):
            """Calculate average, handling empty lists."""
            return np.mean(times) if times else 0.0
        
        def safe_max(times):
            """Calculate max, handling empty lists."""
            return np.max(times) if times else 0.0
        
        def safe_min(times):
            """Calculate min, handling empty lists."""
            return np.min(times) if times else 0.0
        
        tracking_avg = safe_avg(timing_stats['tracking'])
        detection_avg = safe_avg(timing_stats['detection'])
        matching_avg = safe_avg(timing_stats['matching'])
        classifier_avg = safe_avg(timing_stats['classifier'])
        viz_avg = safe_avg(timing_stats['visualization'])
        total_avg = safe_avg(timing_stats['total_frame'])
        
        tracking_max = safe_max(timing_stats['tracking'])
        detection_max = safe_max(timing_stats['detection'])
        matching_max = safe_max(timing_stats['matching'])
        classifier_max = safe_max(timing_stats['classifier'])
        viz_max = safe_max(timing_stats['visualization'])
        total_max = safe_max(timing_stats['total_frame'])
        
        tracking_min = safe_min(timing_stats['tracking'])
        detection_min = safe_min(timing_stats['detection'])
        matching_min = safe_min(timing_stats['matching'])
        classifier_min = safe_min(timing_stats['classifier'])
        viz_min = safe_min(timing_stats['visualization'])
        total_min = safe_min(timing_stats['total_frame'])
        
        # Calculate target FPS
        target_fps = 1.0 / total_avg if total_avg > 0 else 0
        
        # Timing breakdown percentages
        total_non_zero = tracking_avg + detection_avg + matching_avg + classifier_avg + viz_avg
        
        if total_non_zero > 0:
            tracking_pct = (tracking_avg / total_non_zero) * 100
            detection_pct = (detection_avg / total_non_zero) * 100
            matching_pct = (matching_avg / total_non_zero) * 100
            classifier_pct = (classifier_avg / total_non_zero) * 100
            viz_pct = (viz_avg / total_non_zero) * 100
        else:
            tracking_pct = detection_pct = matching_pct = classifier_pct = viz_pct = 0
        
        status = "FINAL REPORT" if final else f"REPORT (Frame {frame_count})"
        logger.info(f"\n{status}")
        logger.info(f"Frames processed: {frame_count}")
        logger.info(f"Average FPS: {target_fps:.2f}")
        logger.info("")
        logger.info("COMPONENT TIMING (milliseconds):")
        logger.info(f"  Tracking:      avg={tracking_avg*1000:.3f}ms  min={tracking_min*1000:.3f}ms  max={tracking_max*1000:.3f}ms  ({tracking_pct:.1f}%)")
        logger.info(f"  Detection:     avg={detection_avg*1000:.3f}ms  min={detection_min*1000:.3f}ms  max={detection_max*1000:.3f}ms  ({detection_pct:.1f}%)")
        logger.info(f"  Matching:      avg={matching_avg*1000:.3f}ms  min={matching_min*1000:.3f}ms  max={matching_max*1000:.3f}ms  ({matching_pct:.1f}%)")
        logger.info(f"  Classifier:    avg={classifier_avg*1000:.3f}ms  min={classifier_min*1000:.3f}ms  max={classifier_max*1000:.3f}ms  ({classifier_pct:.1f}%)")
        logger.info(f"  Visualization: avg={viz_avg*1000:.3f}ms  min={viz_min*1000:.3f}ms  max={viz_max*1000:.3f}ms  ({viz_pct:.1f}%)")
        logger.info("")
        logger.info(f"  TOTAL FRAME:   avg={total_avg*1000:.3f}ms  min={total_min*1000:.3f}ms  max={total_max*1000:.3f}ms")
        logger.info("=" * 80)

    def val(self):
        max_violence_prob = 0.0
        climax_frame_id = -1
        start_time = time.time()
        while True: #For safety, put a condition to break loop if needed (e.g., max frames)
            ret, frame = self.cap.read()
            if not ret:
                if climax_frame_id == -1:
                    for i in range(8 - len(self.feats)):  # Append last feature if video ends before classifier frame
                        self.feats.append(self.feats[-1])
                    seq = np.asarray(self.feats)  # shape: (8, 896, 15)
                    seq = np.expand_dims(seq, 0)    # (1, 8, , 896, 15)
                    gap_output = self.TemporalClassifier_session.run(
                        [self.gap_output_name],
                        {self.gap_input_name: seq}
                    )

                    logits = gap_output[0]   # shape (1,2)

                    exp_logits = np.exp(logits)
                    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

                    violence_prob = probs[0][1]   # class index 1 = violence

                    if violence_prob > max_violence_prob:
                        max_violence_prob = violence_prob
                        climax_frame_id = self.frame_id
                break
            is_detect_frame = (self.frame_id % self.detect_interval == 0)
            is_classifier_frame = (self.frame_id % self.detect_interval == self.detect_interval//2)

            # ===== YOLO DETECTION =====
            if is_detect_frame:
                boxes, feature, areas = self.Detect(frame)
                self.feats.append(feature)

            # ===== CLASSIFIER INFERENCE =====
            if is_classifier_frame and len(self.feats) >= FRAME_PER_DETECT:
                # Run classifier on the last 8 features
                seq = np.asarray(self.feats)  # shape: (8, 896, 15)
                seq = np.expand_dims(seq, 0)    # (1, 8, 896, 15)
                gap_output = self.TemporalClassifier_session.run(
                    [self.gap_output_name],
                    {self.gap_input_name: seq}
                )

                logits = gap_output[0]      # shape (1,2)
                exp = np.exp(logits - np.max(logits))
                probs = exp / exp.sum(axis=1, keepdims=True)

                violence_prob = probs[0][1]   # class 1 = violence

                if violence_prob > max_violence_prob:
                    max_violence_prob = violence_prob
                    climax_frame_id = self.frame_id

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.frame_id += 1

        self.cap.release()
        cv2.destroyAllWindows()

        # ===== REPORTING =====
        end_time = time.time()
        logger.info(f"Validation complete. Total frames: {self.frame_id}, Max violence probability: {max_violence_prob:.4f}, Climax frame: {climax_frame_id}, Processing time: {end_time - start_time:.2f} seconds")
        return max_violence_prob, climax_frame_id

if __name__ == "__main__":
    video_path = "demo.mp4"  # Set to None or "0" for webcam
    detector = ViolenceDetector(video_path)
    detector.run()