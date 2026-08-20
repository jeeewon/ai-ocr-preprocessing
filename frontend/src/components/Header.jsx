export default function Header() {
  return (
    <header>
      <div className="logo">
        <div className="logo-mark">AI</div>
        <div>
          <div className="logo-text">DocLens</div>
          <div className="logo-sub">Document AI Playground</div>
        </div>
      </div>
      <div className="status-dot">
        <div className="dot" />
        Server running
      </div>
    </header>
  );
}