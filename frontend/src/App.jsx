import { useState, useCallback } from 'react';
import Header     from './components/Header';
import Sidebar    from './components/Sidebar';
import ResultCard from './components/ResultCard';

export default function App() {
  const [file, setFile]               = useState(null);
  const [preview, setPreview]         = useState(null);
  const [filename, setFilename]       = useState('');
  const [loading, setLoading]         = useState(false);
  const [results, setResults]         = useState(null);
  const [error, setError]             = useState(null);
  const [drag, setDrag]               = useState(false);
  const [useLayout, setUseLayout]     = useState(false);
  const [forceSource, setForceSource] = useState('');

  const [query, setQuery]           = useState('');
  const [ocrLoading, setOcrLoading] = useState(false);
  const [ocrDataMap, setOcrDataMap] = useState({});

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) return;
    setFile(f);
    setFilename(f.name);
    setResults(null);
    setError(null);
    setForceSource('');
    setQuery('');
    setOcrDataMap({});
    const reader = new FileReader();
    reader.onload = e => setPreview(e.target.result);
    reader.readAsDataURL(f);
  }, []);

  const handleProcess = async (fs = forceSource) => {
    if (!file) return;
    setLoading(true); setError(null); setResults(null);
    setQuery(''); setOcrDataMap({});
    try {
      const form   = new FormData();
      form.append('file', file);
      const params = new URLSearchParams({ layout_analysis: useLayout });
      if (fs) params.set('force_source', fs);
      const res  = await fetch(`/api/process?${params}`, { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Server error');
      setResults(data.results);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSourceChange = (newSource) => {
    setForceSource(newSource);
    handleProcess(newSource);
  };

  // ocrDataMap: { [pageIndex]: [{text, box, conf}, ...] }
  const handleEnableSearch = async () => {
    if (!results) return;
    setOcrLoading(true);
    try {
      const newMap = {};
      await Promise.all(results.map(async (r, i) => {
        // /api/search는 UploadFile을 요구하므로 base64 → Uint8Array → Blob → FormData로 변환
        const byteStr = atob(r.final_b64);
        const arr     = new Uint8Array(byteStr.length);
        for (let j = 0; j < byteStr.length; j++) arr[j] = byteStr.charCodeAt(j);
        const blob = new Blob([arr], { type: 'image/jpeg' });
        const form = new FormData();
        form.append('file', blob, `page_${i}.jpg`);
        const res  = await fetch('/api/search', { method: 'POST', body: form });
        const data = await res.json();
        newMap[i]  = data.ocr_data || [];
      }));
      setOcrDataMap(newMap);
    } catch (e) {
      console.error('OCR error:', e);
    } finally {
      setOcrLoading(false);
    }
  };

  const reset = () => {
    setFile(null); setPreview(null); setResults(null);
    setError(null); setForceSource('');
    setQuery(''); setOcrDataMap({});
  };

  const source        = results?.[0]?.source;
  const searchEnabled = Object.keys(ocrDataMap).length > 0;

  return (
    <>
      <Header />
      <main>
        <Sidebar
          drag={drag} setDrag={setDrag}
          filename={filename} preview={preview}
          useLayout={useLayout} setUseLayout={setUseLayout}
          file={file} loading={loading} results={results}
          onFile={handleFile}
          onProcess={handleProcess}
          onReset={reset}
        />
        <section className="result-area">
          {error && <div className="error-box">⚠ {error}</div>}
          {!results && !error && (
            <div className="empty-state">
              <div className="empty-icon">🔍</div>
              <div className="empty-text">
                Upload an image and click Run<br />to see results here
              </div>
            </div>
          )}
          {results && (
            <>
              <div className="result-header">
                <select
                  className={`badge-select ${source}`}
                  value={forceSource || source}
                  onChange={e => handleSourceChange(e.target.value)}
                  disabled={loading}
                >
                  <option value="scan">SCAN</option>
                  <option value="camera">CAMERA</option>
                  <option value="screenshot">SCREENSHOT</option>
                </select>
                <span className="result-filename">{filename}</span>
                <span className="result-count">
                  {results.length} page{results.length > 1 ? 's' : ''}
                </span>
              </div>

              <div className="search-bar">
                {!searchEnabled ? (
                  <button
                    className="btn-enable-search"
                    onClick={handleEnableSearch}
                    disabled={ocrLoading}
                  >
                    {ocrLoading ? '⏳  Indexing...' : '🔍  Enable Search'}
                  </button>
                ) : (
                  <div className="search-bar-inner">
                    <span className="search-bar-icon">🔍</span>
                    <input
                      className="search-bar-input"
                      type="text"
                      placeholder="Search text across all pages..."
                      value={query}
                      onChange={e => setQuery(e.target.value)}
                      autoFocus
                    />
                    {query.trim() && (
                      <span className="search-bar-count">
                        {Object.values(ocrDataMap).flat()
                          .filter(d => d.text.toLowerCase().includes(query.trim().toLowerCase()))
                          .length
                        } matches
                      </span>
                    )}
                    <button
                      className="search-bar-clear"
                      onClick={() => setQuery('')}
                    >
                      ✕
                    </button>
                  </div>
                )}
              </div>

              <div className="result-grid">
                {results.map((r, i) => (
                  <ResultCard
                    key={i} r={r} idx={i} total={results.length}
                    query={query}
                    ocr_data={ocrDataMap[i] || null}
                  />
                ))}
              </div>
            </>
          )}
        </section>
      </main>
    </>
  );
}