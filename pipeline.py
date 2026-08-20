"""
pipeline.py — 메인 파이프라인
==============================
document source 분류 → (camera/screenshot) YOLO 크롭 → 방향 보정 → 합본분리 → 기울기 보정

실행:
  # 단일 이미지
  python pipeline.py --yolo-model weights/best.pt --image car_reg.jpg

  # 폴더 전체
  python pipeline.py --yolo-model weights/best.pt --input-dir ./photos --output-dir ./results

  # 디버그 그리드 + 레이아웃 분석
  python pipeline.py --yolo-model weights/best.pt --input-dir ./photos \\
    --layout-analysis --debug --device cuda --verbose 2>&1 | tee ./verbose.log
"""

import cv2
import numpy as np
import argparse
import time
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from classify  import ScanCameraClassifier, ClassificationResult
from segment   import DocumentSegmentor, visualize
from preprocess import correct_orientation, split_combined_scan, deskew
from layout    import DocLayoutAnalyzer, draw_layout_boxes


def setup_logger(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """터미널 + 파일 동시 출력 로거 설정"""
    log_dir  = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"[Log] {log_path}")
    return logger


try:
    from paddleocr import DocImgOrientationClassification
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("[Warning] paddleocr not installed → orientation correction disabled")


# ──────────────────────────────────────────
# 결과 dataclass
# ──────────────────────────────────────────

@dataclass
class PipelineResult:
    image_path:      str
    source:          str            # scan / camera / screenshot
    source_conf:     float
    source_features: dict           # border_white_ratio 등 수치
    yolo_conf:       float          # scan이면 0.0
    orientation:     Optional[int]  # 0 / 90 / 180 / 270 / None
    orient_conf:     float
    orient_applied:  bool           # 실제 보정 적용 여부
    final_image:     Optional[np.ndarray]
    elapsed_ms:      float

    # 디버그용 중간 이미지
    orig_image:    Optional[np.ndarray] = field(default=None, repr=False)
    layout_boxes:  list                 = field(default_factory=list, repr=False)
    yolo_vis:      Optional[np.ndarray] = field(default=None, repr=False)
    crop_image:    Optional[np.ndarray] = field(default=None, repr=False)
    before_orient: Optional[np.ndarray] = field(default=None, repr=False)

    def __str__(self):
        return (
            f"source={self.source}({self.source_conf:.0%})  "
            f"yolo_conf={self.yolo_conf:.3f}  "
            f"orientation={self.orientation}deg({self.orient_conf:.2f})  "
            f"elapsed={self.elapsed_ms:.0f}ms"
        )


# ──────────────────────────────────────────
# 디버그 그리드 시각화
# ──────────────────────────────────────────

CELL_H    = 600
DIVIDER_W = 4
DIVIDER_C = 60
BANNER_H  = 28
FONT      = cv2.FONT_HERSHEY_SIMPLEX

SOURCE_COLORS = {
    "scan":       (52,  199,  89),
    "camera":     (255, 149,   0),
    "screenshot": (90,  200, 255),
}


def _resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.full((h, int(h * 0.75), 3), 180, dtype=np.uint8)
    r = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * r)), h))


def _pad_to_w(img: np.ndarray, w: int) -> np.ndarray:
    if img.shape[1] >= w:
        return img
    pad = np.full((img.shape[0], w - img.shape[1], 3), 180, dtype=np.uint8)
    return np.hstack([img, pad])


def _banner(img: np.ndarray, lines: list[str], color=(220, 220, 220)) -> np.ndarray:
    out    = img.copy()
    total  = BANNER_H * len(lines)
    banner = np.full((total, out.shape[1], 3), 25, dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(banner, line,
                    (6, BANNER_H * i + BANNER_H - 7),
                    FONT, 0.42, color, 1, cv2.LINE_AA)
    return np.vstack([banner, out])


def _placeholder(h: int, w: int, text: str) -> np.ndarray:
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2),
                FONT, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
    return img


def make_debug_grid(result: PipelineResult, orient_conf_th: float) -> np.ndarray:
    """
    3x2 디버그 그리드:
    ┌──────────┬──────────┬──────────┐
    │  원본    │  YOLO    │  warp    │
    ├──────────┼──────────┼──────────┤
    │  source  │ 보정전→후 │  최종    │
    └──────────┴──────────┴──────────┘
    """
    # [0,0] 원본
    c00 = _banner(
        _resize_to_h(result.orig_image, CELL_H),
        [f"[Original]  {Path(result.image_path).name}"],
    )

    # [0,1] YOLO 검출
    if result.yolo_vis is not None:
        c01 = _banner(
            _resize_to_h(result.yolo_vis, CELL_H),
            [f"[YOLO]  conf={result.yolo_conf:.3f}"],
        )
    else:
        c01 = _banner(
            _placeholder(CELL_H + BANNER_H, int(CELL_H * 0.75), "YOLO SKIPPED (scan)"),
            ["[YOLO]  skipped"],
        )

    # [0,2] crop 결과
    if result.crop_image is not None:
        c02 = _banner(_resize_to_h(result.crop_image, CELL_H), ["[Crop]  perspective warp"])
    else:
        c02 = _banner(
            _placeholder(CELL_H + BANNER_H, int(CELL_H * 0.75), "CROP SKIPPED (scan)"),
            ["[Crop]  skipped"],
        )

    # [1,0] Source 판단 지표
    feat    = result.source_features
    s_color = SOURCE_COLORS.get(result.source, (200, 200, 200))
    src_bg  = np.full((CELL_H, int(CELL_H * 0.75), 3), 30, dtype=np.uint8)

    label_text = result.source.upper()
    scale, thick = 1.6, 2
    (lw, _), _ = cv2.getTextSize(label_text, FONT, scale, thick)
    cv2.putText(src_bg, label_text, ((src_bg.shape[1] - lw) // 2, 60),
                FONT, scale, s_color, thick, cv2.LINE_AA)
    cv2.putText(src_bg, f"conf: {result.source_conf:.0%}",
                ((src_bg.shape[1] - lw) // 2, 95),
                FONT, 0.5, s_color, 1, cv2.LINE_AA)

    TICK = {True: "(+)", False: "(-)"}

    def _feat_line(key, val, threshold, higher_is_scan):
        if val is None:
            return f"  {key:<22} None"
        passed = (val >= threshold) if higher_is_scan else (val < threshold)
        mark   = TICK[passed]
        return f"  {key:<22} {val:.3f}  {mark}" if isinstance(val, float) else f"  {key:<22} {val}  {mark}"

    feat_lines = [
        f"  {'feature':<22} value",
        f"  {'-'*34}",
    ]
    if "border_white_ratio"  in feat: feat_lines.append(_feat_line("border_white_ratio",  feat["border_white_ratio"],  0.9,  True))
    if "contour_ratio"       in feat: feat_lines.append(_feat_line("contour_ratio",        feat["contour_ratio"],       0.85, True))
    if "brightness_variance" in feat: feat_lines.append(_feat_line("brightness_variance",  feat["brightness_variance"], 800,  False))
    if "illum_uniformity"    in feat: feat_lines.append(_feat_line("illum_uniformity",     feat["illum_uniformity"],    150,  False))
    if "sharpness"           in feat: feat_lines.append(_feat_line("sharpness",            feat["sharpness"],           300,  True))
    if "dpi"                 in feat:
        dpi_v = feat["dpi"]
        feat_lines.append(f"  {'dpi':<22} {dpi_v}  {'(+)' if (dpi_v and dpi_v >= 600) else '(-)'}")
    if "cam_override"        in feat: feat_lines.append("  >> cam_override: True")
    if "full_frame_override" in feat: feat_lines.append("  >> full_frame_override: True")

    y0 = 120
    for line in feat_lines:
        cv2.putText(src_bg, line, (8, y0), FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        y0 += 20

    c10 = _banner(src_bg, ["[Source Classifier]"])

    # [1,1] 방향 보정 전 → 후
    if result.before_orient is not None and result.final_image is not None:
        bef      = _resize_to_h(result.before_orient, CELL_H)
        aft      = _resize_to_h(result.final_image,   CELL_H)
        div      = np.full((CELL_H, DIVIDER_W, 3), 120, dtype=np.uint8)
        combined = np.hstack([bef, div, aft])
        applied  = "applied" if result.orient_applied else "skipped (low conf)"
        th_mark  = "(OK)" if result.orient_conf >= orient_conf_th else f"(NG th={orient_conf_th})"
        c11 = _banner(combined, [
            f"[Orientation]  angle={result.orientation}deg  conf={result.orient_conf:.3f} {th_mark}",
            f"  before → after  ({applied})",
        ])
    else:
        c11 = _banner(_placeholder(CELL_H + BANNER_H * 2, int(CELL_H * 1.5), "N/A"),
                      ["[Orientation]", "  N/A"])

    # [1,2] 최종 결과
    final_for_debug = (draw_layout_boxes(result.final_image, result.layout_boxes)
                       if result.layout_boxes else result.final_image)
    c12 = _banner(_resize_to_h(final_for_debug, CELL_H), [
        f"[Final]  elapsed={result.elapsed_ms:.0f}ms",
        f"  source={result.source}  angle={result.orientation}deg",
    ])

    # 행별 높이 통일 후 합치기
    row0_h = max(c00.shape[0], c01.shape[0], c02.shape[0])
    row1_h = max(c10.shape[0], c11.shape[0], c12.shape[0])

    def fit_h(img, h):
        if img.shape[0] == h:
            return img
        pad = np.full((h - img.shape[0], img.shape[1], 3), 25, dtype=np.uint8)
        return np.vstack([img, pad])

    c00, c01, c02 = [fit_h(c, row0_h) for c in [c00, c01, c02]]
    c10, c11, c12 = [fit_h(c, row1_h) for c in [c10, c11, c12]]

    col_w = [
        max(c00.shape[1], c10.shape[1]),
        max(c01.shape[1], c11.shape[1]),
        max(c02.shape[1], c12.shape[1]),
    ]
    c00 = _pad_to_w(c00, col_w[0]); c10 = _pad_to_w(c10, col_w[0])
    c01 = _pad_to_w(c01, col_w[1]); c11 = _pad_to_w(c11, col_w[1])
    c02 = _pad_to_w(c02, col_w[2]); c12 = _pad_to_w(c12, col_w[2])

    vd = lambda h: np.full((h, DIVIDER_W, 3), DIVIDER_C, dtype=np.uint8)
    hd = lambda w: np.full((DIVIDER_W, w,  3), DIVIDER_C, dtype=np.uint8)

    row0    = np.hstack([c00, vd(row0_h), c01, vd(row0_h), c02])
    row1    = np.hstack([c10, vd(row1_h), c11, vd(row1_h), c12])
    return np.vstack([row0, hd(row0.shape[1]), row1])


# ──────────────────────────────────────────
# 저장
# ──────────────────────────────────────────

def save_results(
    results:        list[PipelineResult],
    output_dir:     Path,
    save_debug:     bool  = False,
    orient_conf_th: float = 0.0,
) -> tuple[list[Path], list[Path], list[Path]]:
    """객체별로 final/debug/layout 저장. 파일명에 obj index 붙임 (2개 이상일 때)."""
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    final_paths  = []
    debug_paths  = []
    layout_paths = []
    stem         = Path(results[0].image_path).stem
    multi        = len(results) > 1

    for i, result in enumerate(results):
        if result.final_image is None:
            continue

        suffix  = f"_obj{i}" if multi else ""
        final_p = final_dir / f"{stem}{suffix}.jpg"
        cv2.imwrite(str(final_p), result.final_image)
        final_paths.append(final_p)

        if save_debug:
            debug_dir = output_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            grid    = make_debug_grid(result, orient_conf_th)
            debug_p = debug_dir / f"{stem}{suffix}_debug.jpg"
            cv2.imwrite(str(debug_p), grid)
            debug_paths.append(debug_p)

        if result.layout_boxes:
            layout_dir = output_dir / "layout"
            layout_dir.mkdir(parents=True, exist_ok=True)
            img_h, img_w = result.final_image.shape[:2]
            for j, b in enumerate(result.layout_boxes):
                x1, y1, x2, y2 = b["box"]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop   = result.final_image[y1:y2, x1:x2]
                crop_p = layout_dir / f"{stem}{suffix}_{b['label']}_{j}.jpg"
                cv2.imwrite(str(crop_p), crop)
                layout_paths.append(crop_p)

    return final_paths, debug_paths, layout_paths


# ──────────────────────────────────────────
# DocumentPipeline
# ──────────────────────────────────────────

class DocumentPipeline:
    def __init__(
        self,
        yolo_model_path: str,
        yolo_conf:       float = 0.25,
        device:          str   = "cpu",
        orient_conf_th:  float = 0.0,
        score_threshold: int   = 2,
        min_area_ratio:  float = 0.05,
        iou_threshold:   float = 0.5,
        min_aspect:      float = 0.2,
        max_aspect:      float = 5.0,
        verbose:         bool  = False,
    ):
        self.verbose        = verbose
        self.orient_conf_th = orient_conf_th
        self.predict_kwargs = {
            "min_area_ratio": min_area_ratio,
            "iou_threshold":  iou_threshold,
            "min_aspect":     min_aspect,
            "max_aspect":     max_aspect,
        }

        self.source_clf = ScanCameraClassifier(score_threshold=score_threshold)
        self.segmentor  = DocumentSegmentor(yolo_model_path, conf=yolo_conf, device=device)

        self.ocr      = None
        self.text_ocr = None
        if PADDLE_AVAILABLE:
            self.ocr = DocImgOrientationClassification(device="cpu")
            try:
                from paddleocr import PaddleOCR as _PaddleOCR
                self.text_ocr = _PaddleOCR(
                    use_textline_orientation=False,
                    lang="korean",
                    device="cpu",
                )
            except Exception as e:
                print(f"[Warning] text_ocr initialization failed: {e}")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _get_orientation(self, img: np.ndarray) -> tuple[Optional[int], float]:
        if self.ocr is None:
            return None, 0.0
        try:
            results = list(self.ocr.predict(img))
            res     = results[0]
            angle   = int(res["label_names"][0])
            conf    = float(res["scores"][0])
            return angle, conf
        except Exception as e:
            self._log(f"  [Orientation] 오류: {e}")
            return None, 0.0

    def run(self, image_path: str) -> list[PipelineResult]:
        t0   = time.time()
        path = str(Path(image_path).resolve())

        self._log(f"\n{'='*60}")
        self._log(f"[입력] {Path(image_path).name}")
        self._log(f"{'='*60}")

        # ── 1단계: source 분류 ────────────────────────────
        source_result: ClassificationResult = self.source_clf.classify(path, verbose=self.verbose)
        source       = source_result.label
        source_conf  = source_result.confidence
        source_feat  = source_result.features
        self._log(f"[Source]      {source} ({source_conf:.0%})")

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"이미지 로드 실패: {path}")
        orig_image = img.copy()

        # ── 2단계: YOLO 크롭 (scan은 스킵) ───────────────
        detections = []
        yolo_vis   = None

        if source == "scan":
            self._log("[YOLO]        스킵 (scan)")
            detections, pre_angle, pre_conf = self._run_scan_path(img)
        else:
            detections, yolo_vis = self._run_camera_path(img)
            pre_angle = pre_conf = None

        # ── 3단계: 객체별 방향 보정 + deskew ─────────────
        results = []
        for i, det in enumerate(detections):
            result = self._process_detection(
                i, det, img, source, pre_angle, pre_conf, path,
                source_conf, source_feat, orig_image, yolo_vis, t0,
            )
            results.append(result)

        return results

    def _run_scan_path(self, img: np.ndarray):
        """scan 경로: orientation → 합본분리 → detections 생성"""
        pre_angle, pre_conf = self._get_orientation(img)
        self._log(f"[Orient pre]  angle={pre_angle}deg  conf={pre_conf:.3f}")

        img_for_split = img
        if pre_angle is not None and pre_conf >= self.orient_conf_th:
            img_for_split = correct_orientation(img, pre_angle)
            self._log(f"[Rotate pre]  {pre_angle}deg 보정 후 합본 분리")

        pages = split_combined_scan(img_for_split, ocr_instance=self.text_ocr, verbose=self.verbose)
        self._log(f"[합본]        {len(pages)}페이지")

        if len(pages) == 1:
            detections = [{"warped": img_for_split if pre_angle and pre_angle != 0 else None,
                           "mask_crop": None, "confidence": 0.0, "mask": None, "corners": None}]
        else:
            detections = [
                {"warped": p, "mask_crop": None, "confidence": 0.0, "mask": None, "corners": None}
                for p in pages
            ]
        return detections, pre_angle, pre_conf

    def _run_camera_path(self, img: np.ndarray):
        """camera/screenshot 경로: YOLO → 합본분리 → detections 생성"""
        detections = self.segmentor.predict(img, **self.predict_kwargs)
        yolo_vis   = visualize(img, detections)
        self._log(f"[YOLO]        detected={len(detections)}")

        if not detections:
            detections = [{"warped": None, "mask_crop": None, "confidence": 0.0, "mask": None, "corners": None}]
            self._log("[YOLO]        검출 실패 → 원본 사용")

        expanded = []
        for det in detections:
            warped = det.get("warped")
            if warped is None:
                warped = det.get("mask_crop")
            if warped is None:
                expanded.append(det)
                continue
            wh, ww = warped.shape[:2]
            if wh / ww >= 2.0:
                pages = split_combined_scan(warped, ocr_instance=self.text_ocr, verbose=self.verbose)
                self._log(f"[합본]        {len(pages)}페이지 (camera/screenshot)")
                if len(pages) > 1:
                    for p in pages:
                        expanded.append({"warped": p, "mask_crop": None,
                                         "confidence": det["confidence"],
                                         "mask": None, "corners": None})
                    continue
            expanded.append(det)

        return expanded, yolo_vis

    def _process_detection(
        self, i, det, img, source, pre_angle, pre_conf,
        path, source_conf, source_feat, orig_image, yolo_vis, t0,
    ) -> PipelineResult:
        """단일 detection에 대해 orientation + deskew 적용 후 PipelineResult 반환."""
        yolo_conf = det["confidence"]

        if det["warped"] is not None:
            target_img = det["warped"]
            crop_image = det["warped"].copy()
            self._log(f"[Crop #{i}]    4-corner warp  conf={yolo_conf:.3f}")
        elif det["mask_crop"] is not None:
            target_img = det["mask_crop"]
            crop_image = det["mask_crop"].copy()
            self._log(f"[Crop #{i}]    mask crop  conf={yolo_conf:.3f}")
        else:
            target_img = img
            crop_image = None
            self._log(f"[Crop #{i}]    원본 사용")

        before_orient = target_img.copy()

        # scan 분리 페이지는 pre orientation 이미 적용됨 → 재보정 스킵
        if source == "scan" and det["warped"] is not None:
            angle, orient_conf = pre_angle, pre_conf
            final_img          = target_img
            orient_applied     = (pre_angle is not None and pre_angle != 0)
            self._log(f"[Orient #{i}]  scan 분리 페이지, pre 보정 적용됨")
        else:
            angle, orient_conf = self._get_orientation(target_img)
            self._log(f"[Orient #{i}]  angle={angle}deg  conf={orient_conf:.3f}")
            final_img      = target_img
            orient_applied = False
            if angle is not None and orient_conf >= self.orient_conf_th:
                final_img      = correct_orientation(target_img, angle)
                orient_applied = True
                self._log(f"[Rotate #{i}]  {angle}deg 보정 완료")
            elif angle is not None:
                self._log(f"[Rotate #{i}]  스킵 (conf={orient_conf:.3f} < th={self.orient_conf_th})")

        if source == "scan":
            final_img = deskew(final_img)
            self._log(f"[Deskew #{i}]  미세 기울기 보정 완료")

        elapsed = max(0.0, (time.time() - t0) * 1000)

        return PipelineResult(
            image_path=path,
            source=source,
            source_conf=source_conf,
            source_features=source_feat,
            yolo_conf=yolo_conf,
            orientation=angle,
            orient_conf=orient_conf,
            orient_applied=orient_applied,
            final_image=final_img,
            elapsed_ms=elapsed,
            orig_image=orig_image,
            yolo_vis=yolo_vis,
            crop_image=crop_image,
            before_orient=before_orient,
        )


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Document pipeline: source 분류 → YOLO 크롭 → 방향 보정"
    )
    parser.add_argument("--yolo-model",     default="weights/best.pt", help="YOLO 가중치 경로")
    parser.add_argument("--image",          type=str,                 help="단일 이미지 경로")
    parser.add_argument("--input-dir",      type=str,                 help="배치 처리 폴더")
    parser.add_argument("--output-dir",     type=str, default="./output")
    parser.add_argument("--device",         type=str, default="cpu",  help="cpu / cuda")
    parser.add_argument("--yolo-conf",      type=float, default=0.25)
    parser.add_argument("--score-th",       type=int,   default=2,    help="scan 판별 최소 score (기본: 2/6)")
    parser.add_argument("--orient-conf-th", type=float, default=0.0,  help="방향 보정 최소 confidence")
    parser.add_argument("--min-area-ratio", type=float, default=0.05)
    parser.add_argument("--iou-threshold",  type=float, default=0.5)
    parser.add_argument("--min-aspect",     type=float, default=0.2)
    parser.add_argument("--max-aspect",     type=float, default=5.0)
    parser.add_argument("--verbose",         action="store_true",     help="단계별 상세 로그")
    parser.add_argument("--debug",           action="store_true",     help="디버그 그리드 이미지 저장")
    parser.add_argument("--layout-analysis", action="store_true",     help="PP-StructureV2 레이아웃 분석")
    parser.add_argument("--layout-conf",     type=float, default=0.0, help="DocLayout confidence threshold")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir, verbose=args.verbose)

    layout_analyzer = DocLayoutAnalyzer(conf=args.layout_conf, device=args.device) if args.layout_analysis else None

    pipeline = DocumentPipeline(
        yolo_model_path=args.yolo_model,
        yolo_conf=args.yolo_conf,
        device=args.device,
        orient_conf_th=args.orient_conf_th,
        score_threshold=args.score_th,
        min_area_ratio=args.min_area_ratio,
        iou_threshold=args.iou_threshold,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        verbose=args.verbose,
    )

    def _run(img_path: str):
        results = pipeline.run(img_path)
        if layout_analyzer:
            for r in results:
                if r.final_image is not None:
                    r.layout_boxes = layout_analyzer.predict(r.final_image)
        return results

    if args.image:
        results = _run(args.image)
        final_ps, debug_ps, layout_ps = save_results(
            results, output_dir, save_debug=args.debug, orient_conf_th=args.orient_conf_th,
        )
        for i, r in enumerate(results):
            logger.info(f"[Result #{i}] {r}")
        for p in final_ps:  logger.info(f"[Saved] {p}")
        for p in debug_ps:  logger.info(f"[Debug] {p}")
        for p in layout_ps: logger.info(f"[Layout] {p}")

    elif args.input_dir:
        exts   = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        images = sorted(set(
            p for ext in exts for p in Path(args.input_dir).glob(ext)
        ))
        logger.info(f"Processing {len(images)} images")
        logger.info("-" * 60)

        counts = {"scan": 0, "camera": 0, "screenshot": 0, "error": 0}
        for i, img_path in enumerate(images, 1):
            try:
                results = _run(str(img_path))
                save_results(results, output_dir, save_debug=args.debug,
                             orient_conf_th=args.orient_conf_th)
                counts[results[0].source] += 1
                logger.info(f"[{i:>3}/{len(images)}] {img_path.name:<40} obj={len(results)}  {results[0]}")
            except Exception as e:
                counts["error"] += 1
                logger.info(f"[{i:>3}/{len(images)}] {img_path.name:<40} ERROR: {e}")

        logger.info("-" * 60)
        logger.info(f"Done: scan={counts['scan']}, camera={counts['camera']}, "
                    f"screenshot={counts['screenshot']}, error={counts['error']}")
        logger.info(f"final saved: {(output_dir / 'final').resolve()}")
        if args.debug:
            logger.info(f"debug saved: {(output_dir / 'debug').resolve()}")
        if args.layout_analysis:
            logger.info(f"layout saved: {(output_dir / 'layout').resolve()}")
    else:
        parser.print_help()