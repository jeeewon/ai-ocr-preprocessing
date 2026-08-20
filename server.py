"""
server.py — AI-OCR 파이프라인 데모 서버
==========================================
실행:
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload

접속:
  http://localhost:8000
"""

import io
import uuid
import time
import base64
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from classify   import ScanCameraClassifier, ClassificationResult
from segment    import DocumentSegmentor, visualize
from preprocess import correct_orientation, split_combined_scan, deskew
from layout     import DocLayoutAnalyzer, draw_layout_boxes

# ── 설정 ──────────────────────────────────────────────
YOLO_MODEL_PATH = "weights/best.pt"
DEVICE          = "cuda"

# ── 모델 초기화 (서버 시작 시 1회) ────────────────────
print("[Server] Loading models...")

source_clf = ScanCameraClassifier(score_threshold=2)
segmentor  = DocumentSegmentor(YOLO_MODEL_PATH, conf=0.25, device=DEVICE)

try:
    from paddleocr import DocImgOrientationClassification, PaddleOCR
    ocr_orient = DocImgOrientationClassification(device="cpu")
    ocr_text   = PaddleOCR(use_textline_orientation=False, lang="korean", device="cpu")
    PADDLE_OK  = True
    print("[Server] PaddleOCR loaded")
except Exception as e:
    ocr_orient = ocr_text = None
    PADDLE_OK  = False
    print(f"[Server] PaddleOCR failed: {e}")

try:
    layout_analyzer = DocLayoutAnalyzer(conf=0.0, device=DEVICE)
    LAYOUT_OK = True
    print("[Server] LayoutAnalyzer loaded")
except Exception as e:
    layout_analyzer = None
    LAYOUT_OK = False
    print(f"[Server] LayoutAnalyzer failed: {e}")

print("[Server] All models ready ✓")

# ── FastAPI 앱 ────────────────────────────────────────
app = FastAPI(title="AI-OCR 파이프라인 데모")


def _get_orientation(img: np.ndarray) -> tuple[int | None, float]:
    if not PADDLE_OK or ocr_orient is None:
        return None, 0.0
    try:
        res   = list(ocr_orient.predict(img))[0]
        angle = int(res["label_names"][0])
        conf  = float(res["scores"][0])
        return angle, conf
    except Exception:
        return None, 0.0


def _img_to_b64(img: np.ndarray) -> str:
    """numpy BGR 이미지 → base64 JPEG 문자열"""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode()


def _apply_layout(final: np.ndarray, boxes: list) -> dict:
    """레이아웃 분석 결과 → overlay + crops dict 생성"""
    overlay = draw_layout_boxes(final, boxes)
    img_h, img_w = final.shape[:2]
    crops = []
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_w, x2); y2 = min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = final[y1:y2, x1:x2]
        crops.append({
            "label":    b["label"],
            "conf":     round(b["conf"], 2),
            "box":      [x1, y1, x2, y2],
            "crop_b64": _img_to_b64(crop),
        })
    return {
        "layout_b64":   _img_to_b64(overlay),
        "layout_crops": crops,
    }


def _apply_search_ocr(final: np.ndarray) -> list[dict]:
    """
    PaddleOCR로 전체 페이지 OCR 실행 (합본분리용 ocr_text 재활용).
    Returns: [{"text": str, "box": [x1,y1,x2,y2], "conf": float}, ...]
    """
    if not PADDLE_OK or ocr_text is None:
        return []
    try:
        result = ocr_text.predict(final)
        if not result or not result[0]:
            return []
        data    = result[0]
        texts   = data.get("rec_texts",  [])
        scores  = data.get("rec_scores", [])
        polys   = data.get("rec_polys",  [])
        ocr_data = []
        for text, conf, poly in zip(texts, scores, polys):
            if conf < 0.3:
                continue
            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            ocr_data.append({
                "text": text,
                "box":  [min(xs), min(ys), max(xs), max(ys)],
                "conf": round(float(conf), 2),
            })
        return ocr_data
    except Exception as e:
        print(f"[SearchOCR] 오류: {e}")
        return []


def run_pipeline(img: np.ndarray, use_layout: bool = False,
                 use_search: bool = False,
                 force_source: str | None = None) -> list[dict]:
    """
    파이프라인 실행 후 결과 리스트 반환.
    force_source: "scan" | "camera" | "screenshot" — 분류기 결과 무시하고 강제 지정
    use_search: True면 EasyOCR로 전체 텍스트 + bbox 추출
    """
    t0 = time.time()
    sep = "─" * 44

    # 1. Source classification
    if force_source in ("scan", "camera", "screenshot"):
        source      = force_source
        source_conf = 1.0
        print(f"\n{sep}")
        print(f"[Classify   ]   forced  → {source}")
    else:
        t = time.time()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, img)
            clf_result: ClassificationResult = source_clf.classify(f.name)
        Path(f.name).unlink(missing_ok=True)
        source      = clf_result.label
        source_conf = clf_result.confidence
        print(f"\n{sep}")
        print(f"[Classify   ] {(time.time()-t)*1000:6.0f}ms  → {source} ({source_conf:.0%})")

    ROTATE_MAP = {
        90:  cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }

    results = []

    if source == "scan":
        t = time.time()
        angle, conf = _get_orientation(img)
        img_proc = correct_orientation(img, angle) if angle else img
        print(f"[Orient     ] {(time.time()-t)*1000:6.0f}ms  → {angle}deg ({conf:.2f})")

        t = time.time()
        pages = split_combined_scan(img_proc, ocr_instance=ocr_text)
        print(f"[SplitOCR   ] {(time.time()-t)*1000:6.0f}ms  → {len(pages)} pages")

        for page in pages:
            t = time.time()
            final = deskew(page)
            print(f"[Deskew     ] {(time.time()-t)*1000:6.0f}ms")

            entry = {
                "source":      source,
                "source_conf": source_conf,
                "orientation": angle,
                "orient_conf": conf,
                "final_b64":   _img_to_b64(final),
                "elapsed_ms":  max(0.0, (time.time() - t0) * 1000),
            }
            if use_layout and LAYOUT_OK and layout_analyzer:
                t = time.time()
                boxes = layout_analyzer.predict(final)
                print(f"[Layout     ] {(time.time()-t)*1000:6.0f}ms  → {len(boxes)} boxes")
                entry.update(_apply_layout(final, boxes))
            if use_search:
                t = time.time()
                entry["ocr_data"] = _apply_search_ocr(final)
                print(f"[SearchOCR  ] {(time.time()-t)*1000:6.0f}ms  → {len(entry['ocr_data'])} texts")
            results.append(entry)

    else:
        t = time.time()
        detections = segmentor.predict(img)
        print(f"[YOLO       ] {(time.time()-t)*1000:6.0f}ms  → {len(detections)} objects")

        if not detections:
            detections = [{"warped": None, "mask_crop": None, "confidence": 0.0, "mask": None, "corners": None}]

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
                t = time.time()
                pages = split_combined_scan(warped, ocr_instance=ocr_text)
                print(f"[SplitOCR   ] {(time.time()-t)*1000:6.0f}ms  → {len(pages)} pages")
                if len(pages) > 1:
                    for p in pages:
                        expanded.append({"warped": p, "mask_crop": None,
                                         "confidence": det["confidence"],
                                         "mask": None, "corners": None})
                    continue
            expanded.append(det)

        for det in expanded:
            target = det.get("warped")
            if target is None:
                target = det.get("mask_crop")
            if target is None:
                target = img

            t = time.time()
            angle, conf = _get_orientation(target)
            final = correct_orientation(target, angle) if angle else target
            print(f"[Orient     ] {(time.time()-t)*1000:6.0f}ms  → {angle}deg ({conf:.2f})")

            entry = {
                "source":      source,
                "source_conf": source_conf,
                "orientation": angle,
                "orient_conf": conf,
                "final_b64":   _img_to_b64(final),
                "elapsed_ms":  max(0.0, (time.time() - t0) * 1000),
            }
            if use_layout and LAYOUT_OK and layout_analyzer:
                t = time.time()
                boxes = layout_analyzer.predict(final)
                print(f"[Layout     ] {(time.time()-t)*1000:6.0f}ms  → {len(boxes)} boxes")
                entry.update(_apply_layout(final, boxes))
            if use_search:
                t = time.time()
                entry["ocr_data"] = _apply_search_ocr(final)
                print(f"[SearchOCR  ] {(time.time()-t)*1000:6.0f}ms  → {len(entry['ocr_data'])} texts")
            results.append(entry)

    print(f"[Total      ] {(time.time()-t0)*1000:6.0f}ms")
    print(sep)
    return results


# ── 라우터 ────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/process")
async def process(file: UploadFile = File(...),
                  layout_analysis: bool = False,
                  text_search: bool = False,
                  force_source: str = ""):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패.")

    try:
        results = run_pipeline(img,
                               use_layout=layout_analysis,
                               use_search=text_search,
                               force_source=force_source or None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"results": results, "filename": file.filename})


@app.post("/api/search")
async def search(file: UploadFile = File(...)):
    """전체 페이지 이미지 OCR 실행 → 텍스트 + bbox 반환"""
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패.")
    try:
        t = time.time()
        ocr_data = _apply_search_ocr(img)
        print(f"[SearchOCR  ] {(time.time()-t)*1000:6.0f}ms  → {len(ocr_data)}texts")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"ocr_data": ocr_data})


# ── 정적 파일 ─────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")