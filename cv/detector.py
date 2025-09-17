# detector.py
# Usage (typical pattern in your operate loop):
#   det = ObjectDetector("best.pt", use_gpu=True, conf=0.25, imgsz=640, iou=0.7)

# - Writes a JSON line per frame into lab_output/pred.txt with keys:
#     {"pose":[[x],[y],[theta]], "predfname":"...", "im_w":W, "im_h":H, "detections":[
#        {"name":"Capsicum","bbox_xyxy":[x1,y1,x2,y2],"bbox_xywh":[cx,cy,w,h],"conf":0.97}, ...
#     ]}


import os
import json
import cv2
import numpy as np
import torch

def _canonical_label(raw: str) -> str:
    #normalize labels of fruits
    k = str(raw).strip().lower().replace('_','').replace(' ','').replace('-','')
    mapping = {
        'redapple': 'Red Apple',
        'greenapple': 'Green Apple',
        'orange': 'Orange',
        'mango': 'Mango',
        'capsicum': 'Capsicum',
        'lemon': 'Lemon',
        'yellowlemon': 'Lemon',
        'greenlemon': 'Green Lemon',
    }
    return mapping.get(k, raw if raw else "Unknown")

class ObjectDetector:
    def __init__(self, ckpt_path, use_gpu=False, conf=0.25, imgsz=640, iou=0.7):
        self.device = 0 if (use_gpu and torch.cuda.is_available()) else "cpu"
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.iou = float(iou)

        self.backend = None
        self.yolo = None
        self.model = None
        self.names = None

        os.makedirs("lab_output", exist_ok=True)
        self.pred_pose_path = os.path.join("lab_output", "pred.txt")
        self._fp = open(self.pred_pose_path, "w", buffering=1)  # line-buffered

        self.pred_count = 0
        self._load_weights(ckpt_path)

    def _try_load_ultralytics(self, ckpt_path: str) -> bool:
        #try to import the yolo library
        #if successful initializes YOLO model from tbhe ckpt path
        try:
            from ultralytics import YOLO
        except Exception:
            return False
        try:
            self.yolo = YOLO(ckpt_path)
            try:
               
                self.names = getattr(self.yolo.model, "names", None) or getattr(self.yolo, "names", None)
            except Exception:
                self.names = None
            self.backend = "ultralytics"
            print(f"[detector_opt] Ultralytics backend ready. names={self.names}")
            return True
        except Exception as e:
            print(f"[detector_opt] Ultralytics load failed: {e}")
            self.yolo = None
            return False

    def _try_load_torchvision(self, ckpt_path: str) -> bool:
        
        return False

    def _load_weights(self, ckpt_path: str):
        #import model
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if self._try_load_ultralytics(ckpt_path):
            return
        if self._try_load_torchvision(ckpt_path):
            return
        raise RuntimeError("Unsupported checkpoint format for detector_opt.py")

    def detect(self, bgr: np.ndarray):
        #utilizes model weight to detect fruits and make bounding boxes
        if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)
        if self.backend == "ultralytics":
            rs = self.yolo.predict(source=bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                   device=self.device, verbose=False)
            r = rs[0]
            if r.boxes is None or len(r.boxes) == 0:
                return (np.zeros((0,4), np.float32),
                        np.zeros((0,), np.float32),
                        np.zeros((0,), np.int32))
            b = r.boxes
            boxes = b.xyxy.detach().cpu().numpy().astype(np.float32)
            confs = b.conf.detach().cpu().numpy().astype(np.float32)
            clss  = b.cls.detach().cpu().numpy().astype(np.int32)
            return boxes, confs, clss
        raise RuntimeError("Detector backend not initialized.")

    def detect_single_image(self, bgr: np.ndarray):
        boxes, confs, clss = self.detect(bgr)
        pred_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        vis = bgr.copy()
        dets = []

        for i in range(len(boxes)):
            x1,y1,x2,y2 = boxes[i].astype(int).tolist()
            conf = float(confs[i])
            cid = int(clss[i])
            raw_label = None
            if self.names is not None and 0 <= cid < len(self.names):
                raw_label = self.names[cid]
            nice = _canonical_label(raw_label)

           
            bbox_xyxy = [float(x1), float(y1), float(x2), float(y2)]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = (x2 - x1)
            h  = (y2 - y1)
            bbox_xywh = [float(cx), float(cy), float(w), float(h)]

            dets.append({
                "name": nice,
                "bbox_xyxy": bbox_xyxy,
                "bbox_xywh": bbox_xywh,
                "conf": conf
            })

            
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"{nice} {conf:.2f}", (x1, max(20,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

          
            pred_mask[max(0,y1):max(0,y2), max(0,x1):max(0,x2)] = cid+1

        return pred_mask, vis, dets

    def write_image(self, pred_img: np.ndarray, state, lab_output_dir: str, detections, im_w=None, im_h=None):
        # save the prediction label images
        os.makedirs(lab_output_dir, exist_ok=True)
        out_name = f"pred_{self.pred_count}.png"
        out_path = os.path.join(lab_output_dir, out_name)
        self.pred_count += 1
        cv2.imwrite(out_path, pred_img)

        H, W = int(pred_img.shape[0]), int(pred_img.shape[1])
        if im_w is not None: W = int(im_w)
        if im_h is not None: H = int(im_h)

        
        if isinstance(state, (list, tuple, np.ndarray)):
            pose_list = state
        else:
            pose_list = [state]
        if isinstance(pose_list, np.ndarray):
            pose_list = pose_list.tolist()
        # for every prediction label image, save the state of the robot (its position when pressing "p")
        
        img_dict = {
            "pose": pose_list,
            "predfname": out_path,
            "im_w": W,
            "im_h": H,
            "detections": detections if detections is not None else []
        }
        self._fp.write(json.dumps(img_dict) + "\n")
        return out_name

    def process_and_write_image(self, bgr: np.ndarray, state, lab_output_dir: str, im_w=None, im_h=None):
        #detects objects in the image and saves the visualization and data together with detection results
        pred_mask, vis, dets = self.detect_single_image(bgr)
        out_name = self.write_image(vis, state, lab_output_dir, dets, im_w, im_h)
        return pred_mask, vis, dets, out_name

    def close(self):
        try:
            if self._fp:
                self._fp.flush()
                self._fp.close()
                self._fp = None
        except Exception:
            pass
