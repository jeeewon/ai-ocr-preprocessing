"""
preprocess.py — 문서 이미지 전처리 유틸
=========================================
방향 보정 / 합본분리 / 기울기 보정.
YOLO 결과에 무관하게 이미지만 있으면 동작하는 함수들.
"""

import cv2
import numpy as np
from typing import Optional


# ──────────────────────────────────────────
# 방향 보정 (Orientation)
# ──────────────────────────────────────────

ROTATE_MAP = {
    90:  cv2.ROTATE_90_COUNTERCLOCKWISE,  # PaddleOCR: "90도 돌아있다" → 반시계로 되돌림
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,         # PaddleOCR: "270도 돌아있다" → 시계로 되돌림
}


def correct_orientation(img: np.ndarray, angle: int) -> np.ndarray:
    """감지된 각도만큼 역방향으로 회전해 정방향으로 복원."""
    if angle == 0 or angle not in ROTATE_MAP:
        return img
    return cv2.rotate(img, ROTATE_MAP[angle])


# ──────────────────────────────────────────
# 합본분리 (Split Combined Scan)
# ──────────────────────────────────────────

_SPLIT_KEYWORDS = [
    # 정확한 제목
    "자동차등록원부(갑)", "자동차등록원부(갑) 등본", "자동차등록원부(갑) 등본·초본",
    # 띄어쓰기 오인식
    "자동차 등록원부(갑)", "자동차등록 원부(갑)", "자동차 등록 원부",
    # 괄호 오인식
    "자동차등록원부갑", "자동차등록원부 갑",
    # 뒷부분만 읽힌 경우
    "자동차등록원부", "등록원부(갑)", "등록원부",
    # 등본·초본 오인식
    "등본·초본", "등본초본", "등본 초본", "등본", "초본",
]

_FOOTER_EXCLUDE = ["이등본은", "이 등본은"]

_MAX_OCR_SIDE = 4000


def split_combined_scan(
    img:          np.ndarray,
    ocr_instance: Optional[object] = None,
    verbose:      bool = False,
) -> list[np.ndarray]:
    """
    스캔 합본을 페이지 단위로 분리.
    - aspect ratio(h/w) < 2.0 → 단일 페이지, 그대로 반환
    - aspect ratio >= 2.0 → OCR로 헤더 탐지 후 슬라이싱

    Args:
        img:          입력 이미지 (BGR)
        ocr_instance: PaddleOCR 인스턴스 (None이면 내부에서 초기화)
        verbose:      디버그 로그 출력 여부

    Returns:
        페이지별 이미지 리스트 (분리 안 되면 [img])
    """
    h, w = img.shape[:2]
    if h / w < 2.0:
        return [img]

    if verbose:
        print(f"  [Multi-Page Detected] aspect={h/w:.2f} → attempting page split")

    # OCR용 이미지 resize (PaddleOCR max_side_limit 대응)
    img_h, img_w = img.shape[:2]
    ocr_scale    = 1.0
    if max(img_h, img_w) > _MAX_OCR_SIDE:
        ocr_scale = _MAX_OCR_SIDE / max(img_h, img_w)
        img_ocr   = cv2.resize(img, (int(img_w * ocr_scale), int(img_h * ocr_scale)))
        if verbose:
            print(f"  [Multi-Page] OCR resize: {img_h}→{int(img_h*ocr_scale)}px (scale={ocr_scale:.3f})")
    else:
        img_ocr = img

    try:
        if ocr_instance is None:
            from paddleocr import PaddleOCR as _PaddleOCR
            ocr_instance = _PaddleOCR(
                use_textline_orientation=False,
                lang="korean",
                device="cpu",
            )
        result = ocr_instance.predict(img_ocr)
    except Exception as e:
        if verbose:
            print(f"  [Multi-Page] OCR failed: {e} → treating as single page")
        return [img]

    if not result or not result[0]:
        if verbose:
            print("  [Multi-Page] No OCR result → treating as single page")
        return [img]

    header_ys = _extract_header_ys(result, ocr_scale, verbose)

    if not header_ys:
        if verbose:
            print("  [Multi-Page] No header found → treating as single page")
        return [img]

    return _slice_pages(img, header_ys, h, verbose)


def _extract_header_ys(result, ocr_scale: float, verbose: bool) -> list[float]:
    """OCR 결과에서 헤더 키워드 y좌표 추출 (역보정 포함)."""
    header_ys = []
    try:
        data   = result[0] if isinstance(result[0], dict) else {}
        texts  = data.get("rec_texts",  [])
        scores = data.get("rec_scores", [])
        polys  = data.get("rec_polys",  [])

        if verbose:
            print(f"  [Multi-Page] Detected text count: {len(texts)}")
            for t, c in zip(texts[:10], scores[:10]):
                print(f"  [Multi-Page] OCR: '{t}'  conf={c:.2f}")

        for text, conf, bbox in zip(texts, scores, polys):
            if conf <= 0.3:
                continue
            if not any(kw in text for kw in _SPLIT_KEYWORDS):
                continue
            if any(ex in text for ex in _FOOTER_EXCLUDE):
                continue
            if len(text) > 25:
                if verbose:
                    print(f"  [Multi-Page] Excluded (likely footer): '{text[:20]}...'  len={len(text)}")
                continue
            y_raw = float(min(pt[1] for pt in bbox))
            y_top = y_raw / ocr_scale  # 원본 좌표로 역보정
            header_ys.append(y_top)
            if verbose:
                print(f"  [Multi-Page] Header: '{text}'  y_ocr={y_raw:.0f} → y_orig={y_top:.0f}  conf={conf:.2f}")

    except Exception as e:
        if verbose:
            print(f"  [Multi-Page] Result parse failed: {e}")

    return header_ys


def _slice_pages(img: np.ndarray, header_ys: list[float], h: int, verbose: bool) -> list[np.ndarray]:
    """헤더 y좌표 기반으로 페이지 슬라이싱."""
    header_ys  = sorted(int(y) for y in header_ys)
    min_page_h = max(h // 5, 300)

    # 너무 가까운 헤더 병합
    merged = [header_ys[0]]
    for y in header_ys[1:]:
        if y - merged[-1] >= min_page_h:
            merged.append(y)

    MARGIN       = max(int(h * 0.02), 50)
    split_points = [max(0, int(y) - MARGIN) for y in merged] + [h]

    pages = [img[split_points[i]:split_points[i + 1], :] for i in range(len(split_points) - 1)]

    if verbose:
        print(f"  [Multi-Page] Separated {len(pages)} pages")
    return pages


# ──────────────────────────────────────────
# 기울기 보정 (Deskew)
# ──────────────────────────────────────────

def deskew(img: np.ndarray) -> np.ndarray:
    """
    Hough line으로 수평선 각도를 측정해 미세 기울기 보정.
    스캔 경로 전용 (camera는 perspective warp로 이미 정렬됨).
    0.5도 미만이면 보정 없이 원본 반환.
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100, minLineLength=100, maxLineGap=10,
    )
    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:   # 수평선만 (수직선 노이즈 제거)
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:   # 미미한 기울기는 무시
        return img

    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
