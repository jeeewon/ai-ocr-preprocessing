import { useRef } from 'react';

export default function Sidebar({
  drag, setDrag,
  filename, preview,
  useLayout, setUseLayout,
  file, loading, results,
  onFile, onProcess, onReset,
}) {
  const inputRef = useRef();

  const handleDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    onFile(e.dataTransfer.files[0]);
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-scroll">
        <div className="sidebar-title">Upload</div>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={e => onFile(e.target.files[0])}
        />
        <div
          className={`dropzone${drag ? ' drag' : ''}`}
          onDragOver={e => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={handleDrop}
          onClick={(e) => {
            e.stopPropagation();
            inputRef.current.value = '';  // 같은 파일 재선택 허용
            inputRef.current.click();
          }}
        >
          <div className="dz-icon">📄</div>
          <div className="dz-text">
            Drag & drop or <strong>click to select</strong><br />
            JPG · PNG · JPEG
          </div>
        </div>

        {filename && (
          <div style={{
            fontSize: '11px', fontFamily: 'var(--mono)', color: 'var(--muted)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            paddingLeft: '4px',
          }}>
            📎 {filename}
          </div>
        )}

        <div className="preview-wrap">
          {preview
            ? <img src={preview} alt="preview" />
            : <span className="preview-placeholder">preview</span>
          }
        </div>
      </div>

      <div className="sidebar-footer">
        <div className="toggle-row">
          <div className="toggle-label">Layout Analysis</div>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={useLayout}
              onChange={e => setUseLayout(e.target.checked)}
            />
            <span className="toggle-track" />
          </label>
        </div>

        <button
          className="btn btn-primary"
          onClick={() => onProcess()}
          disabled={!file || loading}
        >
          {loading
            ? <><div className="spinner" />Processing...</>
            : '▶  Run'
          }
        </button>

        {(file || results) &&
          <button className="btn btn-ghost" onClick={onReset}>Reset</button>
        }
      </div>
    </aside>
  );
}