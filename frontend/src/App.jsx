import React, { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";

// ─── Constants ───────────────────────────────────────────────────────────────

const API = "";

const MONTHS = [
  "January","February","March","April","May","June",
  "July","August","September","October","November","December",
];

const KNOWN_REGIONS = [
  "Phoenix, AZ","Tucson, AZ","Scottsdale, AZ","Mesa, AZ",
  "Tempe, AZ","Chandler, AZ","Gilbert, AZ","Glendale, AZ",
  "United States","National",
];

// ─── Styles ──────────────────────────────────────────────────────────────────

const css = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0a0c0f;
    --surface:  #111418;
    --border:   #1e2328;
    --accent:   #c8f542;
    --accent2:  #42c8f5;
    --muted:    #4a5260;
    --text:     #e8eaed;
    --text2:    #8a94a0;
    --red:      #f54242;
    --radius:   4px;
    --mono:     'DM Mono', monospace;
    --sans:     'Syne', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }

  .app {
    max-width: 1200px;
    margin: 0 auto;
    padding: 48px 24px;
  }

  /* Header */
  .header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 48px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 24px;
  }
  .header-left h1 {
    font-family: var(--sans);
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1;
  }
  .header-left h1 span { color: var(--accent); }
  .header-left p {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    margin-top: 8px;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .status-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text2);
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 999px;
  }
  .status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--muted);
  }
  .status-dot.online { background: var(--accent); box-shadow: 0 0 6px var(--accent); }
  .status-dot.error  { background: var(--red);    box-shadow: 0 0 6px var(--red); }

  /* Metrics strip */
  .metrics-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    margin-bottom: 32px;
    border-radius: var(--radius);
    overflow: hidden;
  }
  .metric-cell {
    background: var(--surface);
    padding: 20px 24px;
  }
  .metric-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .metric-value {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 500;
    color: var(--text);
  }
  .metric-value.good { color: var(--accent); }
  .metric-value.loading { color: var(--muted); }

  /* Main grid */
  .main-grid {
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 16px;
    align-items: start;
  }

  /* Panel */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .panel-header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .panel-title {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text2);
  }
  .panel-body { padding: 20px; }

  /* Form */
  .field { margin-bottom: 16px; }
  .field label {
    display: block;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .field input,
  .field select {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    padding: 10px 12px;
    border-radius: var(--radius);
    outline: none;
    transition: border-color 0.15s;
    appearance: none;
  }
  .field input:focus,
  .field select:focus { border-color: var(--accent); }

  .field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }

  .predict-btn {
    width: 100%;
    padding: 14px;
    background: var(--accent);
    color: #0a0c0f;
    border: none;
    border-radius: var(--radius);
    font-family: var(--sans);
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
    margin-top: 8px;
  }
  .predict-btn:hover   { opacity: 0.88; }
  .predict-btn:active  { transform: scale(0.98); }
  .predict-btn:disabled { opacity: 0.3; cursor: not-allowed; }

  /* Result */
  .result-box {
    margin-top: 16px;
    padding: 20px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    text-align: center;
    opacity: 0;
    transform: translateY(8px);
    transition: all 0.3s ease;
  }
  .result-box.visible { opacity: 1; transform: translateY(0); }
  .result-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
  }
  .result-price {
    font-family: var(--mono);
    font-size: 36px;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: -1px;
  }
  .result-error {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--red);
  }

  /* Chart */
  .chart-panel { height: 100%; }
  .chart-wrap { padding: 20px; }
  .no-data {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 280px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
  }

  /* Tooltip */
  .custom-tooltip {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 10px 14px;
    border-radius: var(--radius);
    font-family: var(--mono);
    font-size: 12px;
  }
  .custom-tooltip .label { color: var(--muted); margin-bottom: 4px; font-size: 10px; }
  .custom-tooltip .val   { color: var(--accent); font-size: 15px; }

  @media (max-width: 860px) {
    .main-grid { grid-template-columns: 1fr; }
    .metrics-strip { grid-template-columns: 1fr 1fr; }
  }
`;

// ─── Custom tooltip ───────────────────────────────────────────────────────────

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tooltip">
      <div className="label">{label}</div>
      <div className="val">${Number(payload[0].value).toLocaleString()}</div>
    </div>
  );
};

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [apiStatus, setApiStatus] = useState("loading"); // loading | online | error

  // Form state — core fields always present
  const [year, setYear]       = useState(new Date().getFullYear());
  const [month, setMonth]     = useState(new Date().getMonth() + 1);
  const [region, setRegion]   = useState("");
  const [extraFields, setExtraFields] = useState({}); // dynamic extras

  const [prediction, setPrediction] = useState(null);
  const [predError, setPredError]   = useState(null);
  const [loading, setLoading]       = useState(false);

  // History of predictions for chart
  const [history, setHistory] = useState([]);

  // ── Fetch model info on mount ──────────────────────────────────────────────

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/model-info`);
        if (!res.ok) throw new Error();
        const data = await res.json();
        setModelInfo(data);
        setApiStatus("online");

        // Seed extra fields from feature_columns (exclude year/month/region)
        const coreKeys = new Set(["year", "month", "region"]);
        const extras = {};
        (data.feature_columns || []).forEach((col) => {
          if (!coreKeys.has(col)) extras[col] = "";
        });
        setExtraFields(extras);
      } catch {
        setApiStatus("error");
      }
    })();
  }, []);

  // ── Predict ───────────────────────────────────────────────────────────────

  const handlePredict = useCallback(async () => {
    setLoading(true);
    setPredError(null);
    setPrediction(null);

    const features = { year: Number(year), month: Number(month) };
    if (region) features.region = region;
    Object.entries(extraFields).forEach(([k, v]) => {
      if (v !== "") features[k] = Number(v);
    });

    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }
      const data = await res.json();
      setPrediction(data);
      setHistory((prev) => [
        ...prev,
        {
          label: `${MONTHS[month - 1].slice(0, 3)} ${year}`,
          price: data.predicted_median_sale_price,
        },
      ]);
    } catch (e) {
      setPredError(e.message);
    } finally {
      setLoading(false);
    }
  }, [year, month, region, extraFields]);

  // ── Render ────────────────────────────────────────────────────────────────

  const metrics = modelInfo?.metrics;
  const extraKeys = Object.keys(extraFields);

  return (
    <>
      <style>{css}</style>
      <div className="app">

        {/* Header */}
        <div className="header">
          <div className="header-left">
            <h1>Housing<span>Price</span></h1>
            <p>Redfin · RandomForest · Median Sale Price</p>
          </div>
          <div className="status-pill">
            <div className={`status-dot ${apiStatus === "online" ? "online" : apiStatus === "error" ? "error" : ""}`} />
            {apiStatus === "loading" ? "connecting..." : apiStatus === "online" ? "model online" : "api offline"}
          </div>
        </div>

        {/* Metrics strip */}
        <div className="metrics-strip">
          {[
            { label: "MAE", key: "mae", fmt: (v) => `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
            { label: "RMSE", key: "rmse", fmt: (v) => `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
            { label: "MAPE", key: "mape", fmt: (v) => `${(v * 100).toFixed(1)}%` },
            { label: "R²", key: "r2", fmt: (v) => Number(v).toFixed(3) },
          ].map(({ label, key, fmt }) => (
            <div className="metric-cell" key={key}>
              <div className="metric-label">{label}</div>
              <div className={`metric-value ${metrics ? (key === "r2" ? "good" : key === "mape" ? "good" : "") : "loading"}`}>
                {metrics ? fmt(metrics[key]) : "—"}
              </div>
            </div>
          ))}
        </div>

        {/* Main grid */}
        <div className="main-grid">

          {/* Left: form */}
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Run Prediction</span>
            </div>
            <div className="panel-body">

              <div className="field-row">
                <div className="field">
                  <label>Year</label>
                  <input
                    type="number"
                    value={year}
                    min="2000" max="2035"
                    onChange={(e) => setYear(e.target.value)}
                  />
                </div>
                <div className="field">
                  <label>Month</label>
                  <select value={month} onChange={(e) => setMonth(Number(e.target.value))}>
                    {MONTHS.map((m, i) => (
                      <option key={m} value={i + 1}>{m}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="field">
                <label>Region</label>
                <input
                  list="region-list"
                  value={region}
                  onChange={(e) => setRegion(e.target.value)}
                  placeholder="e.g. Phoenix, AZ"
                />
                <datalist id="region-list">
                  {KNOWN_REGIONS.map((r) => <option key={r} value={r} />)}
                </datalist>
              </div>

              {/* Dynamic extra feature fields */}
              {extraKeys.map((col) => (
                <div className="field" key={col}>
                  <label>{col}</label>
                  <input
                    type="number"
                    placeholder="optional"
                    value={extraFields[col]}
                    onChange={(e) =>
                      setExtraFields((prev) => ({ ...prev, [col]: e.target.value }))
                    }
                  />
                </div>
              ))}

              <button
                className="predict-btn"
                onClick={handlePredict}
                disabled={loading || apiStatus !== "online"}
              >
                {loading ? "Predicting..." : "Predict Price →"}
              </button>

              {/* Result */}
              <div className={`result-box ${prediction || predError ? "visible" : ""}`}>
                {predError ? (
                  <div className="result-error">{predError}</div>
                ) : prediction ? (
                  <>
                    <div className="result-label">Predicted Median Sale Price</div>
                    <div className="result-price">{prediction.formatted}</div>
                  </>
                ) : null}
              </div>
            </div>
          </div>

          {/* Right: chart */}
          <div className="panel chart-panel">
            <div className="panel-header">
              <span className="panel-title">Prediction History</span>
            </div>
            <div className="chart-wrap">
              {history.length === 0 ? (
                <div className="no-data">Run predictions to see them plotted here</div>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={history} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
                    <CartesianGrid stroke="#1e2328" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="label"
                      tick={{ fontFamily: "DM Mono", fontSize: 11, fill: "#4a5260" }}
                      axisLine={false} tickLine={false}
                    />
                    <YAxis
                      tick={{ fontFamily: "DM Mono", fontSize: 11, fill: "#4a5260" }}
                      axisLine={false} tickLine={false}
                      tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                      width={55}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    {prediction && (
                      <ReferenceLine
                        y={prediction.predicted_median_sale_price}
                        stroke="#c8f542"
                        strokeDasharray="4 4"
                        strokeOpacity={0.4}
                      />
                    )}
                    <Line
                      type="monotone"
                      dataKey="price"
                      stroke="#c8f542"
                      strokeWidth={2}
                      dot={{ r: 4, fill: "#c8f542", strokeWidth: 0 }}
                      activeDot={{ r: 6, fill: "#c8f542" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

        </div>
      </div>
    </>
  );
}
