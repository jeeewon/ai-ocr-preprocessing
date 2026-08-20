"""
classify.py — 문서 소스 분류기
================================
scan / camera / screenshot 3종 분류.

단독 실행:
  python classify.py image.jpg --verbose
  python classify.py ./photos --output-dir ./output --verbose
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False


DocumentSource = Literal["scan", "camera", "screenshot"]

VIZ_COLORS = {
    "scan":       (52, 199, 89),
    "camera":     (255, 149, 0),
    "screenshot": (90, 200, 255),
}
STAGE_LABEL = {
    "exif":       "[EXIF]",
    "screenshot": "[Screenshot]",
    "score":      "[Score]",
}


@dataclass
class ClassificationResult:
    label:      DocumentSource
    stage:      Literal["exif", "screenshot", "score"]
    confidence: float           # 0.0 ~ 1.0
    features:   dict            # 각 feature 수치값

    def __str__(self):
        feat_str = "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in self.features.items()
        )
        return (f"label={self.label} | stage={self.stage} | "
                f"confidence={self.confidence:.0%} | {feat_str}")


# ──────────────────────────────────────────
# ScanCameraClassifier
# ──────────────────────────────────────────

class ScanCameraClassifier:
    """
    판별 순서:
      1. EXIF 메타데이터  → scan / camera 확정 시 즉시 반환
      2. 회전 보정        → 이후 분석 정확도 향상을 위한 전처리
      3. Screenshot 감지 → PDF 뷰어 캡처 등 조기 반환
      4. Scoring         → 6개 feature 투표로 최종 판별
         - border_white_ratio  : 가장자리 흰 픽셀(>240) 비율
         - contour_ratio       : 최대 윤곽선 면적 비율
         - brightness_variance : 전체 밝기 분산
         - illum_uniformity    : 블록별 밝기 분산 (조명 균일도)
         - sharpness           : Laplacian 분산 (선명도)
         - DPI metadata        : 600dpi 이상이면 스캔 가능성↑
    """

    def __init__(
        self,
        border_threshold:   float = 0.9,
        contour_threshold:  float = 0.85,
        variance_threshold: float = 800,
        score_threshold:    int   = 2,      # score >= 이 값이면 scan
    ):
        self.border_threshold   = border_threshold
        self.contour_threshold  = contour_threshold
        self.variance_threshold = variance_threshold
        self.score_threshold    = score_threshold

    # ── 전처리: 회전 보정 ─────────────────────────────────────

    def _correct_rotation(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Sobel 엣지 방향성으로 90도 회전 감지 및 보정.
        정상 문서는 수평 엣지 > 수직 엣지.
        회전된 문서는 반대 → h/v ratio < 0.75이면 보정.
        """
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        ratio   = float(np.sum(np.abs(sobel_y))) / (float(np.sum(np.abs(sobel_x))) + 1e-6)
        if ratio < 0.75:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), True
        return img, False

    # ── 전처리: Screenshot 감지 ───────────────────────────────

    def _is_screenshot(self, img: np.ndarray) -> bool:
        """
        두 가지 패턴으로 스크린샷 감지:
        (A) PDF 뷰어 캡처: 어두운 UI 테두리(mean<80) + 밝은 내부 문서(mean>150)
        (B) 일반 스크린샷: 인접 픽셀 flat 비율 높음 + 표준 화면 비율
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # (A) PDF 뷰어 패턴
        border = np.concatenate([
            gray[:15, :].ravel(), gray[-15:, :].ravel(),
            gray[:, :15].ravel(), gray[:, -15:].ravel(),
        ])
        interior      = gray[h//4:3*h//4, w//4:3*w//4]
        is_pdf_viewer = float(np.mean(border)) < 80 and float(np.mean(interior)) > 150

        # (B) 일반 스크린샷 패턴
        diff_h     = np.abs(gray[:, 1:].astype(int) - gray[:, :-1].astype(int))
        diff_v     = np.abs(gray[1:, :].astype(int) - gray[:-1, :].astype(int))
        flat_ratio = (np.sum(diff_h == 0) / diff_h.size * 0.5 +
                      np.sum(diff_v == 0) / diff_v.size * 0.5)
        aspect          = w / h
        is_screen_ratio = any(abs(aspect - a) < 0.05 for a in [16/9, 16/10, 4/3, 3/2, 21/9])
        is_flat_screen  = (flat_ratio > 0.80 and is_screen_ratio) or flat_ratio > 0.90

        return is_pdf_viewer or is_flat_screen

    # ── EXIF ──────────────────────────────────────────────────

    def _check_exif(self, image_path: str) -> Optional[DocumentSource]:
        if not PIEXIF_AVAILABLE:
            return None
        scan_kw   = ["scanner", "scan", "wia", "twain", "epson sc", "canon scanner", "hp scan"]
        camera_kw = ["samsung", "apple", "iphone", "google", "xiaomi", "nikon", "sony", "canon eos", "fuji"]
        try:
            exif   = piexif.load(image_path)
            ifd    = exif.get("0th", {})
            fields = " ".join([
                ifd.get(piexif.ImageIFD.Make,     b"").decode(errors="ignore"),
                ifd.get(piexif.ImageIFD.Model,    b"").decode(errors="ignore"),
                ifd.get(piexif.ImageIFD.Software, b"").decode(errors="ignore"),
            ]).lower()
            if any(k in fields for k in scan_kw):   return "scan"
            if any(k in fields for k in camera_kw): return "camera"
            if exif.get("GPS"):                      return "camera"
        except Exception:
            pass
        return None

    # ── Scoring features ──────────────────────────────────────

    def _border_white_ratio(self, img: np.ndarray, border_ratio: float = 0.1) -> float:
        """가장자리 픽셀 중 240 초과(흰색에 가까운) 비율."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        b    = int(min(h, w) * border_ratio)
        px   = np.concatenate([
            gray[:b,  :].flatten(), gray[-b:, :].flatten(),
            gray[:, :b].flatten(),  gray[:, -b:].flatten(),
        ])
        return float(np.sum(px > 240) / len(px))

    def _contour_ratio(self, img: np.ndarray) -> float:
        """최대 윤곽선 면적 / 전체 이미지 면적."""
        gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges       = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        return float(cv2.contourArea(largest) / (img.shape[0] * img.shape[1]))

    def _brightness_variance(self, img: np.ndarray) -> float:
        """전체 밝기 분산. 스캔본은 균일해서 낮고, 카메라는 조명 차이로 높음."""
        return float(np.var(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

    def _illumination_uniformity(self, img: np.ndarray) -> float:
        """
        블록별 평균 밝기의 분산.
        스캔은 조명 균일 → 낮음 / 카메라는 그림자·반사 → 높음.
        """
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w   = gray.shape
        bh, bw = h // 4, w // 4
        means  = [
            float(np.mean(gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]))
            for i in range(4) for j in range(4)
        ]
        return float(np.var(means))

    def _sharpness(self, img: np.ndarray) -> float:
        """
        Laplacian 분산으로 선명도 측정.
        스캔은 텍스트 엣지 선명 → 높음 / 카메라 촬영본은 블러 → 낮음.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _is_full_frame_scan(self, img: np.ndarray,
                             contour_ratio: float,
                             illum_uniformity: float,
                             sharpness: float) -> bool:
        """
        화면을 꽉 채운 스캔본 감지.
        border_white_ratio가 낮아 일반 scoring에서 오판되므로 별도 처리.
        """
        return (contour_ratio > 0.92
                and illum_uniformity < 150
                and sharpness > 300)

    def _get_dpi(self, image_path: str) -> Optional[float]:
        """PIL로 DPI 메타데이터 읽기. 스캔본은 보통 300dpi 이상."""
        if not PIL_AVAILABLE:
            return None
        try:
            dpi = Image.open(image_path).info.get("dpi")
            return float(dpi[0]) if dpi else None
        except Exception:
            return None

    # ── 메인 classify ─────────────────────────────────────────

    def classify(self, image_path: str, verbose: bool = False) -> ClassificationResult:
        path = str(Path(image_path).resolve())

        # 1단계: EXIF → 확실한 경우 즉시 반환
        exif_label = self._check_exif(path)
        if verbose:
            print(f"  [EXIF]       -> {exif_label}")
        if exif_label is not None:
            return ClassificationResult(
                label=exif_label, stage="exif", confidence=1.0, features={}
            )

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"이미지 로드 실패: {path}")

        # 2단계: 회전 보정 (label 결정 아님, 후속 분석 정확도용)
        img, rotated = self._correct_rotation(img)
        if verbose:
            print(f"  [Rotation]   -> {'corrected' if rotated else 'upright'}")

        # 3단계: Screenshot → 즉시 반환
        if self._is_screenshot(img):
            if verbose:
                print(f"  [Screenshot] -> detected")
            return ClassificationResult(
                label="screenshot", stage="screenshot", confidence=1.0,
                features={"screenshot": True}
            )
        if verbose:
            print(f"  [Screenshot] -> None")

        # 4단계: Scoring — 6개 feature 투표
        score    = 0
        features = {}

        bwr = self._border_white_ratio(img)
        features["border_white_ratio"] = round(bwr, 3)
        if bwr > self.border_threshold:
            score += 1

        cr = self._contour_ratio(img)
        features["contour_ratio"] = round(cr, 3)
        if cr > self.contour_threshold:
            score += 1

        var = self._brightness_variance(img)
        features["brightness_variance"] = round(var, 1)
        if var < self.variance_threshold:
            score += 1

        dpi = self._get_dpi(path)
        features["dpi"] = dpi
        if dpi and dpi >= 600:  # 카메라도 300dpi 저장 → 스캔 전용 신호는 600 이상
            score += 1

        illum = self._illumination_uniformity(img)
        features["illum_uniformity"] = round(illum, 1)
        if illum < 150:
            score += 1

        sharp = self._sharpness(img)
        features["sharpness"] = round(sharp, 1)
        if sharp > 300:
            score += 1

        # 카메라 촬영 확정: 문서가 배경 속에 작게 떠있는 경우
        is_floating_doc = (
            bwr < 0.05
            and cr < 0.30
            and illum > 500
            and var > 2000
        )
        if is_floating_doc:
            if verbose:
                print(f"  [CamOverride]-> floating document detected, forcing camera")
            return ClassificationResult(
                label="camera", stage="score", confidence=1.0,
                features={**features, "cam_override": True}
            )

        # 풀프레임 스캔 확정: 문서가 화면을 꽉 채워 border_white_ratio가 낮게 나오는 경우
        if self._is_full_frame_scan(img, cr, illum, sharp):
            if verbose:
                print(f"  [FullFrame]  -> full-frame scan detected, overriding score")
            return ClassificationResult(
                label="scan", stage="score", confidence=1.0,
                features={**features, "full_frame_override": True}
            )

        label      = "scan" if score >= self.score_threshold else "camera"
        confidence = score / 6

        if verbose:
            print(f"  [Score]      -> {score}/6  (threshold>={self.score_threshold})")
            for k, v in features.items():
                print(f"                  {k} = {v}")

        return ClassificationResult(
            label=label, stage="score", confidence=confidence, features=features
        )


# ──────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────

def _draw_overlay(image: np.ndarray, result: ClassificationResult, filename: str = "") -> np.ndarray:
    vis   = image.copy()
    h, w  = vis.shape[:2]
    color = VIZ_COLORS[result.label]

    bt = max(6, min(w, h) // 80)
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), color, bt)

    banner_h = max(90, h // 8)
    overlay  = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = w / 1200

    label_text  = result.label.upper()
    label_scale = max(1.4, base_scale * 2.2)
    label_thick = max(2, int(label_scale * 2))
    (lw_px, _), _ = cv2.getTextSize(label_text, font, label_scale, label_thick)
    cv2.putText(vis, label_text,
                (20, int(banner_h * 0.65)),
                font, label_scale, color, label_thick, cv2.LINE_AA)

    meta_text  = f"{STAGE_LABEL[result.stage]}  conf:{result.confidence:.0%}"
    if filename:
        meta_text += f"  | {filename}"
    meta_scale = max(0.55, base_scale * 0.85)
    meta_thick = max(1, int(meta_scale * 1.5))
    cv2.putText(vis, meta_text,
                (lw_px + 35, int(banner_h * 0.65)),
                font, meta_scale, (210, 210, 210), meta_thick, cv2.LINE_AA)

    return vis


def _save_visualization(image: np.ndarray, result: ClassificationResult,
                         src_path: str, output_dir: str) -> str:
    src     = Path(src_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis      = _draw_overlay(image, result, filename=src.name)
    out_path = out_dir / f"{src.stem}_classified{src.suffix}"
    cv2.imwrite(str(out_path), vis)
    return str(out_path)


# ──────────────────────────────────────────
# 배치 처리
# ──────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def classify_directory(
    input_dir:          str,
    output_dir:         str = "./output",
    verbose:            bool = False,
    save_visualization: bool = True,
    classifier:         Optional[ScanCameraClassifier] = None,
) -> dict[str, ClassificationResult]:
    if classifier is None:
        classifier = ScanCameraClassifier()

    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"디렉터리가 아닙니다: {input_dir}")

    image_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])
    if not image_files:
        print(f"[Warning] No image files found: {input_dir}")
        return {}

    print(f"Processing {len(image_files)} images\n" + "-" * 60)

    results = {}
    counts  = {"scan": 0, "camera": 0, "screenshot": 0, "error": 0}

    for i, img_path in enumerate(image_files, 1):
        try:
            if verbose:
                print(f"\n{'='*60}")
                print(f"[{i:>3}/{len(image_files)}] {img_path.name}")
                print(f"{'='*60}")

            result = classifier.classify(str(img_path), verbose=verbose)

            if save_visualization:
                img = cv2.imread(str(img_path))
                _save_visualization(img, result, str(img_path), output_dir)

            results[img_path.name] = result
            counts[result.label]  += 1
            conf_str = f"{result.confidence:.0%}"

            if verbose:
                print(f"-> FINAL: {result.label:<10}  stage={result.stage:<10}  conf={conf_str}")
            else:
                print(f"[{i:>3}/{len(image_files)}] {img_path.name:<40} "
                      f"-> {result.label:<10}  stage={result.stage:<10}  conf={conf_str}")

        except Exception as e:
            counts["error"] += 1
            print(f"[{i:>3}/{len(image_files)}] {img_path.name:<40} -> ERROR: {e}")

    print("-" * 60)
    print(f"Done: scan={counts['scan']}, camera={counts['camera']}, "
          f"screenshot={counts['screenshot']}, error={counts['error']}")
    if save_visualization:
        print(f"Visualization saved: {Path(output_dir).resolve()}")

    return results


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="문서 소스 분류: scan / camera / screenshot"
    )
    parser.add_argument("input",         help="입력 이미지 또는 디렉터리 경로")
    parser.add_argument("--output-dir",  default="./output", help="시각화 저장 폴더 (기본: ./output)")
    parser.add_argument("--no-viz",      action="store_true", help="시각화 저장 비활성화")
    parser.add_argument("--verbose",     action="store_true", help="단계별 상세 로그 출력")
    parser.add_argument("--quiet",       action="store_true", help="로그 출력 비활성화")
    parser.add_argument("--border-th",   type=float, default=0.9,   help="border_white_ratio 임계값")
    parser.add_argument("--contour-th",  type=float, default=0.85,  help="contour_ratio 임계값")
    parser.add_argument("--variance-th", type=float, default=800.0, help="brightness_variance 임계값")
    parser.add_argument("--score-th",    type=int,   default=2,     help="scan 판별 최소 score (기본: 2/6)")
    args = parser.parse_args()

    clf = ScanCameraClassifier(
        border_threshold=args.border_th,
        contour_threshold=args.contour_th,
        variance_threshold=args.variance_th,
        score_threshold=args.score_th,
    )

    input_path = Path(args.input)

    if input_path.is_dir():
        classify_directory(
            input_dir=str(input_path),
            output_dir=args.output_dir,
            verbose=args.verbose,
            save_visualization=not args.no_viz,
            classifier=clf,
        )
    else:
        result = clf.classify(
            image_path=str(input_path),
            verbose=args.verbose or not args.quiet,
        )
        if not args.no_viz:
            img = cv2.imread(str(input_path))
            _save_visualization(img, result, str(input_path), args.output_dir)
        print(result)
