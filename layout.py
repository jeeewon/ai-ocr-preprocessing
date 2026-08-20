"""
layout.py — 문서 레이아웃 분석기
==================================
PP-StructureV2 LayoutDetection으로 문서 레이아웃 영역을 감지하고 크롭.

단독 실행:
  python layout.py --image result.jpg --output-dir ./layout_out
  python layout.py --input-dir ./final --output-dir ./layout_out --save-crops --save-viz
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


# ──────────────────────────────────────────
# 색상 맵 (클래스별 bbox 시각화용)
# ──────────────────────────────────────────

COLOR_MAP: dict[str, tuple] = {
    "title":         (0,   80, 255),
    "text":          (0,  200,  80),
    "header":        (180, 180,   0),
    "footer":        (150, 150,   0),
    "table":         (200,   0, 200),
    "table_caption": (220,  50, 220),
    "figure":        (255, 120,   0),
    "figure_title":  (255, 160,  50),
    "image":         (255, 160,  50),
    "reference":     (80,   80, 200),
    "equation":      (0,  200, 200),
    "page_number":   (100, 100,   0),
}
DEFAULT_COLOR = (120, 120, 120)


# ──────────────────────────────────────────
# DocLayoutAnalyzer
# ──────────────────────────────────────────

class DocLayoutAnalyzer:
    def __init__(self, conf: float = 0.0, device: str = "cpu"):
        from paddleocr import LayoutDetection
        self.engine = LayoutDetection()
        self.conf   = conf
        print("[DocLayout] PP-StructureV2 LayoutDetection loaded")

    def predict(self, img: np.ndarray) -> list[dict]:
        """
        레이아웃 박스 감지.

        Returns:
            [{"box": [x1,y1,x2,y2], "label": str, "conf": float}, ...]
        """
        result = self.engine.predict(img)
        if not result:
            return []

        res = result[0]
        raw = res.get("boxes", []) if isinstance(res, dict) else getattr(res, "boxes", [])

        boxes = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            coord = item.get("coordinate", item.get("bbox", None))
            if coord is None:
                continue
            score = float(item.get("score", 0.0))
            if score < self.conf:
                continue
            x1, y1, x2, y2 = map(int, coord[:4])
            label = str(item.get("label", "unknown")).lower()
            boxes.append({"box": [x1, y1, x2, y2], "label": label, "conf": score})

        return boxes


# ──────────────────────────────────────────
# 시각화 / 저장 유틸
# ──────────────────────────────────────────

def draw_layout_boxes(img: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """이미지에 layout bbox + label 오버레이."""
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        color = COLOR_MAP.get(b["label"], DEFAULT_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        txt = f"{b['label']} {b['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_layout_crops(
    img:        np.ndarray,
    boxes:      list[dict],
    stem:       str,
    output_dir: Path,
) -> list[Path]:
    """클래스별 크롭 이미지 저장."""
    output_dir.mkdir(parents=True, exist_ok=True)
    img_h, img_w = img.shape[:2]
    saved = []
    for j, b in enumerate(boxes):
        x1, y1, x2, y2 = b["box"]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_w, x2); y2 = min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        p    = output_dir / f"{stem}_{b['label']}_{j}.jpg"
        cv2.imwrite(str(p), crop)
        saved.append(p)
    return saved


# ──────────────────────────────────────────
# 단일 이미지 처리 (CLI용)
# ──────────────────────────────────────────

def _process_image(
    img_path:   Path,
    analyzer:   DocLayoutAnalyzer,
    output_dir: Path,
    save_crops: bool,
    save_viz:   bool,
    verbose:    bool,
) -> dict:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"이미지 로드 실패: {img_path}")

    boxes = analyzer.predict(img)

    if verbose:
        print(f"  Detected layouts: {len(boxes)}")
        for b in boxes:
            print(f"    {b['label']:<16} conf={b['conf']:.2f}  box={b['box']}")

    saved_crops = []
    if save_crops and boxes:
        saved_crops = save_layout_crops(img, boxes, img_path.stem, output_dir / "crops")

    viz_path = None
    if save_viz and boxes:
        viz_dir = output_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz      = draw_layout_boxes(img, boxes)
        viz_path = viz_dir / f"{img_path.stem}_layout.jpg"
        cv2.imwrite(str(viz_path), viz)

    return {"boxes": boxes, "crops": saved_crops, "viz": viz_path}


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP-StructureV2 레이아웃 분석기")
    parser.add_argument("--image",      type=str,                help="단일 이미지 경로")
    parser.add_argument("--input-dir",  type=str,                help="배치 처리 폴더")
    parser.add_argument("--output-dir", type=str, default="./layout_output")
    parser.add_argument("--conf",       type=float, default=0.0, help="confidence threshold")
    parser.add_argument("--device",     type=str, default="cpu", help="cpu / cuda (현재 미사용)")
    parser.add_argument("--save-crops", action="store_true",     help="클래스별 크롭 저장")
    parser.add_argument("--save-viz",   action="store_true",     help="bbox 오버레이 이미지 저장")
    parser.add_argument("--verbose",    action="store_true",     help="상세 로그")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = DocLayoutAnalyzer(conf=args.conf, device=args.device)

    if args.image:
        img_path = Path(args.image)
        print(f"\n[Input] {img_path.name}")
        result = _process_image(img_path, analyzer, output_dir,
                                save_crops=args.save_crops,
                                save_viz=args.save_viz,
                                verbose=True)
        print(f"Detected: {len(result['boxes'])}")
        if result["crops"]:
            print(f"Crops saved: {len(result['crops'])} → {output_dir / 'crops'}")
        if result["viz"]:
            print(f"Viz saved: {result['viz']}")

    elif args.input_dir:
        images = sorted([
            f for f in Path(args.input_dir).iterdir()
            if f.suffix in IMAGE_EXTENSIONS
        ])
        print(f"Processing {len(images)} images")
        print("-" * 60)

        total_boxes = total_crops = errors = 0
        for i, img_path in enumerate(images, 1):
            try:
                result   = _process_image(img_path, analyzer, output_dir,
                                          save_crops=args.save_crops,
                                          save_viz=args.save_viz,
                                          verbose=args.verbose)
                n_boxes  = len(result["boxes"])
                n_crops  = len(result["crops"])
                total_boxes += n_boxes
                total_crops += n_crops
                print(f"[{i:>4}/{len(images)}] {img_path.name:<45} boxes={n_boxes}  crops={n_crops}")
            except Exception as e:
                errors += 1
                print(f"[{i:>4}/{len(images)}] {img_path.name:<45} ERROR: {e}")

        print("-" * 60)
        print(f"Done: total {total_boxes} layouts detected, {total_crops} crops saved, errors={errors}")
        if args.save_crops:
            print(f"Crops saved: {(output_dir / 'crops').resolve()}")
        if args.save_viz:
            print(f"Viz saved: {(output_dir / 'viz').resolve()}")
    else:
        parser.print_help()
