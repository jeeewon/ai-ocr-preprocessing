import { useState, useRef } from 'react';

const SOURCE_LABEL = { scan: 'SCAN', camera: 'CAMERA', screenshot: 'SCREENSHOT' };

export default function ResultCard({ r, idx, total, query, ocr_data }) {
  const [tab, setTab]           = useState('final');
  const [cropOpen, setCropOpen] = useState(false);
  const [lightbox, setLightbox] = useState(null);
  const [imgSize, setImgSize]   = useState(null);
  const imgRef                  = useRef(null);

  const hasLayout = !!r.layout_b64;
  const crops     = r.layout_crops || [];

  const imgSrc = hasLayout && tab === 'layout'
    ? `data:image/jpeg;base64,${r.layout_b64}`
    : `data:image/jpeg;base64,${r.final_b64}`;

  const onImgLoad = (e) => {
    const el = e.target;
    setImgSize({ w: el.naturalWidth, h: el.naturalHeight });
  };

  // 픽셀 좌표 → CSS % 좌표 변환 (bbox-overlay가 img 위에 absolute로 올라가므로 % 단위 필요)
  const toPercent = (box) => {
    if (!imgSize) return null;
    const [x1, y1, x2, y2] = box;
    return {
      left:   `${(x1 / imgSize.w * 100).toFixed(2)}%`,
      top:    `${(y1 / imgSize.h * 100).toFixed(2)}%`,
      width:  `${((x2 - x1) / imgSize.w * 100).toFixed(2)}%`,
      height: `${((y2 - y1) / imgSize.h * 100).toFixed(2)}%`,
    };
  };

  const matches = (query?.trim() && ocr_data)
    ? ocr_data.filter(d => d.text.toLowerCase().includes(query.trim().toLowerCase()))
    : [];

  return (
    <div className="result-card">
      {hasLayout && (
        <div className="card-tabs">
          <button className={`card-tab${tab === 'final' ? ' active' : ''}`} onClick={() => setTab('final')}>
            Output
          </button>
          <button className={`card-tab${tab === 'layout' ? ' active' : ''}`} onClick={() => setTab('layout')}>
            Layout Analysis
          </button>
        </div>
      )}

      <div className="img-wrap">
        <img
          ref={imgRef}
          className="card-img"
          src={imgSrc}
          alt={`result ${idx + 1}`}
          onLoad={onImgLoad}
          onClick={() => setLightbox(imgSrc)}
          style={{ cursor: tab === 'layout' ? 'default' : 'zoom-in' }}
        />

        {/* Layout bbox 오버레이 */}
        {tab === 'layout' && imgSize && crops.length > 0 && (
          <div className="bbox-overlay">
            {crops.map((c, i) => {
              const style = toPercent(c.box);
              if (!style) return null;
              return (
                <div
                  key={i}
                  className="bbox-hit"
                  style={style}
                  title={`${c.label} ${(c.conf * 100).toFixed(0)}%`}
                  onClick={() => setLightbox(`data:image/jpeg;base64,${c.crop_b64}`)}
                />
              );
            })}
          </div>
        )}

        {/* 검색 하이라이트 오버레이 */}
        {tab === 'final' && imgSize && matches.length > 0 && (
          <div className="bbox-overlay">
            {matches.map((d, i) => {
              const style = toPercent(d.box);
              if (!style) return null;
              return (
                <div key={i} className="search-hit" style={style} title={d.text} />
              );
            })}
          </div>
        )}
      </div>

      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <img src={lightbox} alt="enlarged" />
        </div>
      )}

      <div className="card-info">
        <div className="card-meta">
          {total > 1 && <span className="card-page">Page {idx + 1} / {total}</span>}
          {matches.length > 0 && (
            <span style={{ fontSize: '10px', fontFamily: 'var(--mono)', color: 'var(--accent)' }}>
              {matches.length} match{matches.length > 1 ? 'es' : ''}
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: 16 }}>
          <div className="stat-row">
            <div className="stat-label">Source</div>
            <div className="stat-value">{SOURCE_LABEL[r.source]}</div>
          </div>
          <div className="stat-row">
            <div className="stat-label">Orientation</div>
            <div className="stat-value">{r.orientation != null ? `${r.orientation}°` : '—'}</div>
          </div>
          <div className="stat-row">
            <div className="stat-label">Confidence</div>
            <div className="stat-value">{(r.source_conf * 100).toFixed(0)}%</div>
          </div>
        </div>

        {hasLayout && crops.length > 0 && tab === 'layout' && (
          <>
            <div className="stat-row">
              <div className="stat-label">Detected Regions</div>
              <div className="layout-tags">
                {crops.map((c, i) => (
                  <span className="layout-tag" key={i}>
                    {c.label} {(c.conf * 100).toFixed(0)}%
                  </span>
                ))}
              </div>
            </div>

            <button
              className={`accordion-btn${cropOpen ? ' open' : ''}`}
              onClick={() => setCropOpen(o => !o)}
            >
              <span>Regions ({crops.length})</span>
              <span className="chevron">▼</span>
            </button>

            {cropOpen && (
              <div className="crop-grid">
                {crops.map((c, i) => (
                  <div className="crop-item" key={i}>
                    <img
                      src={`data:image/jpeg;base64,${c.crop_b64}`}
                      alt={c.label}
                      onClick={() => setLightbox(`data:image/jpeg;base64,${c.crop_b64}`)}
                      style={{ cursor: 'zoom-in' }}
                    />
                    <div className="crop-label">{c.label} {(c.conf * 100).toFixed(0)}%</div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        <div className="elapsed">{r.elapsed_ms.toFixed(0)} ms</div>
      </div>
    </div>
  );
}