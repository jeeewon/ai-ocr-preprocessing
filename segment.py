"""
segment.py — 문서 영역 탐지 및 원근 보정
==========================================
YOLOv8s-seg 파인튜닝 모델로 문서 영역 segmentation 후 perspective warp 적용.

단독 실행:
  python segment.py --model best.pt --image car_reg.jpg
  python segment.py --model best.pt --input-dir ./photos --output-dir ./results
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
import time


# ──────────────────────────────────────────
# Perspective warp 유틸
# ──────────────────────────────────────────

def order_corners(pts: np.ndarray) -> np.ndarray:
    """4개 코너 → [TL, TR, BR, BL] 순으로 정렬"""
    pts  = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    d    = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    rect[1] = pts[np.argmin(d)]   # TR
    rect[3] = pts[np.argmax(d)]   # BL
    return rect


def mask_to_corners(mask: np.ndarray) -> np.ndarray | None:
    """
    세그멘테이션 마스크 → 4개 코너 추출.
    Douglas-Peucker 근사 → 실패 시 minAreaRect fallback.
    """
    m = (mask * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt  = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)

    for eps in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2))

    rect = cv2.minAreaRect(cnt)
    box  = cv2.boxPoints(rect).astype(np.float32)
    return order_corners(box)


def perspective_warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """4 코너 기반 perspective transform → 정면 뷰"""
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, M, (w, h))


def mask_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크 외곽선대로 크롭 (배경 흰색). warp 실패 시 fallback."""
    m = (mask * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    coords = cv2.findNonZero(m)
    if coords is None:
        return image.copy()
    x, y, w, h = cv2.boundingRect(coords)

    result         = np.full_like(image, 255)
    result[mask]   = image[mask]
    return result[y:y+h, x:x+w]


# ──────────────────────────────────────────
# DocumentSegmentor
# ──────────────────────────────────────────

class DocumentSegmentor:
    def __init__(self, model_path: str, conf: float = 0.25, device: str = "cpu"):
        self.model  = YOLO(model_path)
        self.conf   = conf
        self.device = device
        print(f"[YOLO] Model loaded: {model_path}")

    def predict(
        self,
        image:          np.ndarray,
        min_area_ratio: float = 0.05,
        iou_threshold:  float = 0.5,
        min_aspect:     float = 0.2,
        max_aspect:     float = 5.0,
    ) -> list[dict]:
        """
        탐지된 모든 객체 반환 (confidence 내림차순 정렬).

        필터:
          1. 면적 비율  : min_area_ratio 미만 제거
          2. Aspect ratio: min_aspect ~ max_aspect 범위 밖 제거
          3. IoU 중복 제거: iou_threshold 이상 겹치면 conf 낮은 쪽 제거

        Returns: list[dict]
          - mask      : np.ndarray (H, W) bool
          - corners   : np.ndarray (4, 2) | None
          - confidence: float
          - warped    : np.ndarray | None  (perspective warp 결과)
          - mask_crop : np.ndarray | None  (마스크 크롭 결과, warp fallback)
        """
        results  = self.model(image, conf=self.conf, device=self.device, verbose=False)
        img_area = image.shape[0] * image.shape[1]
        detections = []

        for result in results:
            if result.masks is None:
                continue
            for mask_data, conf in zip(result.masks.data, result.boxes.conf):
                c    = float(conf)
                m    = mask_data.cpu().numpy()
                mask = cv2.resize(
                    m, (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

                # 필터 1: 면적 비율
                if mask.sum() / img_area < min_area_ratio:
                    continue

                # 필터 2: Aspect ratio
                coords = cv2.findNonZero((mask * 255).astype(np.uint8))
                if coords is not None:
                    _, _, mw, mh = cv2.boundingRect(coords)
                    aspect = mw / (mh + 1e-6)
                    if not (min_aspect < aspect < max_aspect):
                        continue

                corners = mask_to_corners(mask)
                warped  = perspective_warp(image, corners) if corners is not None else None
                cropped = mask_crop(image, mask)

                detections.append({
                    "mask":       mask,
                    "corners":    corners,
                    "confidence": c,
                    "warped":     warped,
                    "mask_crop":  cropped,
                })

        detections.sort(key=lambda x: x["confidence"], reverse=True)

        # 필터 3: IoU 중복 제거
        kept = []
        for det in detections:
            duplicate = False
            for kept_det in kept:
                intersection = np.logical_and(det["mask"], kept_det["mask"]).sum()
                union        = np.logical_or(det["mask"],  kept_det["mask"]).sum()
                if intersection / (union + 1e-6) > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)

        return kept


# ──────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────

OBJ_COLORS = [
    (50,  220, 100),
    (255, 100,  50),
    (100, 150, 255),
    (255, 200,  50),
    (200,  80, 255),
]


def visualize(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """모든 탐지 객체를 색상 구분해서 하나의 이미지에 시각화"""
    vis = image.copy()

    for i, det in enumerate(detections):
        color_mask = OBJ_COLORS[i % len(OBJ_COLORS)]
        color_line = tuple(min(255, c + 80) for c in color_mask)

        if det["mask"] is not None:
            overlay = vis.copy()
            overlay[det["mask"]] = color_mask
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
            m = (det["mask"] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (255, 80, 0), 2)

        if det["corners"] is not None:
            pts    = det["corners"].astype(np.int32)
            labels = ["TL", "TR", "BR", "BL"]
            cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
            for pt, lbl in zip(pts, labels):
                cv2.circle(vis, tuple(pt), 10, (0, 0, 255), -1)
                cv2.putText(vis, lbl, (pt[0]+6, pt[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if det["corners"] is not None:
            cx = int(det["corners"][:, 0].mean())
            cy = int(det["corners"][:, 1].mean())
        elif det["mask"] is not None:
            coords = np.argwhere(det["mask"])
            cy, cx = coords.mean(axis=0).astype(int)
        else:
            continue

        label = f"#{i}  {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(vis, (cx - 4, cy - th - 6), (cx + tw + 4, cy + 4), (20, 20, 20), -1)
        cv2.putText(vis, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_mask, 2, cv2.LINE_AA)

    cv2.putText(vis, f"detected: {len(detections)}",
                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return vis


def make_grid_2x2(
    original:     np.ndarray,
    detected:     np.ndarray,
    warped:       np.ndarray | None,
    mask_cropped: np.ndarray | None,
    obj_idx:      int   = 0,
    obj_conf:     float = 0.0,
    cell_h:       int   = 700,
) -> np.ndarray:
    """
    단독 실행용 디버그 그리드:
    ┌──────────┬──────────┐
    │  원본    │  검출    │
    ├──────────┼──────────┤
    │ 4코너보정│ 마스크크롭│
    └──────────┴──────────┘
    """
    DIVIDER_W     = 4
    DIVIDER_COLOR = 60

    def to_h(img, h):
        if img is None:
            return np.full((h, int(h * 0.75), 3), 200, dtype=np.uint8)
        r = h / img.shape[0]
        return cv2.resize(img, (max(1, int(img.shape[1] * r)), h))

    def add_label(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 36), (30, 30, 30), -1)
        cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return out

    tl = add_label(to_h(original,     cell_h), "Original")
    tr = add_label(to_h(detected,     cell_h), f"Detection  obj#{obj_idx} conf={obj_conf:.3f}")
    bl = add_label(to_h(warped,       cell_h), f"4-Corner Warp  obj#{obj_idx}")
    br = add_label(to_h(mask_cropped, cell_h), f"Mask Crop  obj#{obj_idx}")

    cell_w = max(tl.shape[1], tr.shape[1], bl.shape[1], br.shape[1])

    def pad_w(img, w):
        if img.shape[1] == w:
            return img
        pad = np.full((img.shape[0], w - img.shape[1], 3), 200, dtype=np.uint8)
        return np.hstack([img, pad])

    tl, tr, bl, br = [pad_w(x, cell_w) for x in [tl, tr, bl, br]]

    vdiv      = np.full((cell_h, DIVIDER_W, 3), DIVIDER_COLOR, dtype=np.uint8)
    hdiv_full = np.full((DIVIDER_W, (cell_w * 2 + DIVIDER_W), 3), DIVIDER_COLOR, dtype=np.uint8)
    top       = np.hstack([tl, vdiv, tr])
    bot       = np.hstack([bl, vdiv, br])
    return np.vstack([top, hdiv_full, bot])


# ──────────────────────────────────────────
# CLI (단독 실행용)
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="문서 영역 탐지 및 원근 보정")
    parser.add_argument("--model",          required=True, help="학습된 best.pt 경로")
    parser.add_argument("--image",          type=str,      help="단일 이미지 경로")
    parser.add_argument("--input-dir",      type=str,      help="배치 처리할 이미지 폴더")
    parser.add_argument("--output-dir",     type=str,      default="output")
    parser.add_argument("--conf",           type=float,    default=0.25)
    parser.add_argument("--device",         type=str,      default="cpu")
    parser.add_argument("--min-area-ratio", type=float,    default=0.05)
    parser.add_argument("--iou-threshold",  type=float,    default=0.5)
    parser.add_argument("--min-aspect",     type=float,    default=0.2)
    parser.add_argument("--max-aspect",     type=float,    default=5.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg = DocumentSegmentor(args.model, conf=args.conf, device=args.device)
    predict_kwargs = {
        "min_area_ratio": args.min_area_ratio,
        "iou_threshold":  args.iou_threshold,
        "min_aspect":     args.min_aspect,
        "max_aspect":     args.max_aspect,
    }

    def _process(img_path: str):
        image       = cv2.imread(img_path)
        stem        = Path(img_path).stem
        t0          = time.time()
        detections  = seg.predict(image, **predict_kwargs)
        elapsed     = time.time() - t0
        n = len(detections)
        print(f"{'✓' if n > 0 else '✗'} {Path(img_path).name} | detected={n} | {elapsed*1000:.0f}ms")
        vis = visualize(image, detections)
        for i, det in enumerate(detections):
            grid   = make_grid_2x2(image, vis, det["warped"], det["mask_crop"],
                                   obj_idx=i, obj_conf=det["confidence"])
            suffix = f"_obj{i}_result.jpg" if n > 1 else "_result.jpg"
            cv2.imwrite(str(output_dir / f"{stem}{suffix}"), grid)
        if n == 0:
            cv2.imwrite(str(output_dir / f"{stem}_result.jpg"),
                        make_grid_2x2(image, vis, None, None))

    if args.image:
        _process(args.image)
    elif args.input_dir:
        exts   = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        images = []
        for ext in exts:
            images.extend(Path(args.input_dir).glob(ext))
        images  = sorted(set(images))
        success = 0
        print(f"\nProcessing {len(images)} images\n")
        for img_path in images:
            _process(str(img_path))
            success += 1
        print(f"\nDone: {success}/{len(images)}")
    else:
        parser.print_help()
