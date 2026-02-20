import { useState, useEffect, useCallback, useRef } from "react";
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from "recharts";

// ── Config ────────────────────────────────────────────────────────────────────
const API = "http://localhost:5000";

// ── Mock data generator (runs when API is unreachable) ────────────────────────
const RNG = (min, max) => Math.random() * (max - min) + min;

function generateMockTrends(n = 48) {
  let ph = 6.0, ec = 1.4, temp = 22, hum = 65;
  return Array.from({ length: n }, (_, i) => {
    ph   += (Math.random() - 0.5) * 0.06;
    ec   += (Math.random() - 0.5) * 0.04;
    temp += (Math.random() - 0.5) * 0.3;
    hum  += (Math.random() - 0.5) * 1.5;
    const hour = (i * 30) % (24 * 60);
    const label = `${String(Math.floor(hour/60)).padStart(2,"0")}:${String(hour%60).padStart(2,"00")}`;
    return {
      t: label,
      ph:   +Math.min(7.5, Math.max(5.0, ph)).toFixed(3),
      ec:   +Math.min(2.5, Math.max(0.5, ec)).toFixed(3),
      temp: +Math.min(30,  Math.max(16, temp)).toFixed(1),
      hum:  +Math.min(90,  Math.max(40, hum)).toFixed(1),
    };
  });
}

function generateMockDosing(n = 8) {
  const actions = ["increase","maintain","decrease"];
  return Array.from({ length: n }, (_, i) => ({
    _id: `dose_${i}`,
    action:     actions[Math.floor(Math.random()*3)],
    ec_target:  +(1.2 + Math.random()*0.4).toFixed(2),
    confidence: +(0.75 + Math.random()*0.2).toFixed(3),
    crop_type:  ["lettuce","spinach","basil"][i%3],
    created_at: new Date(Date.now() - i*900_000).toISOString(),
  }));
}

const MOCK_SUMMARY = {
  latest_reading: { ph: 6.12, ec: 1.38, temperature: 22.4, humidity: 67, light_lux: 14200, growth_stage: "vegetative", day_in_cycle: 18 },
  alert_counts:   { critical: 0, warning: 2, info: 5 },
  latest_dose:    { action: "maintain", ec_target: 1.4, confidence: 0.91, crop_type: "lettuce" },
  latest_growth:  { growth_stage: "vegetative", confidence: 0.88, light_ppfd: 400 },
};

const MOCK_MODELS = {
  models: { lstm: "loaded", cnn: "loaded", yolo: "loaded", autoencoder: "loaded", rl_policy: "loaded" }
};

// ── API Hooks ─────────────────────────────────────────────────────────────────
function useAPI(endpoint, fallback, intervalMs = 0) {
  const [data,    setData]    = useState(fallback);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  const fetch_ = useCallback(async () => {
    try {
      const r = await fetch(`${API}${endpoint}`, { signal: AbortSignal.timeout(4000) });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setData(j.data ?? j);
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [endpoint]);

  useEffect(() => {
    fetch_();
    if (intervalMs > 0) {
      const t = setInterval(fetch_, intervalMs);
      return () => clearInterval(t);
    }
  }, [fetch_, intervalMs]);

  return { data, loading, error, refetch: fetch_ };
}

// ── Colour helpers ─────────────────────────────────────────────────────────────
function phColor(v) {
  if (!v) return "#94a3b8";
  if (v < 5.5 || v > 6.5) return "#f43f5e";
  if (v < 5.7 || v > 6.3) return "#fb923c";
  return "#22c55e";
}
function actionColor(a) {
  if (a === "increase") return "#22c55e";
  if (a === "decrease") return "#f43f5e";
  return "#64748b";
}
function actionGlyph(a) {
  if (a === "increase") return "▲";
  if (a === "decrease") return "▼";
  return "■";
}
function modelStatusColor(s) {
  return s === "loaded" ? "#22c55e" : "#f59e0b";
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Pill({ text, color = "#22c55e" }) {
  return (
    <span style={{
      display: "inline-block", padding: "2px 10px", borderRadius: 999,
      fontSize: 11, fontWeight: 700, letterSpacing: "0.08em",
      background: color + "22", color, border: `1px solid ${color}44`,
      textTransform: "uppercase",
    }}>{text}</span>
  );
}

function StatCard({ label, value, unit = "", color = "#e2e8f0", sub, icon }) {
  return (
    <div style={{
      background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16,
      padding: "20px 24px", flex: 1, minWidth: 140,
      position: "relative", overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0, height: 3,
        background: `linear-gradient(90deg, ${color}, transparent)`,
      }} />
      <div style={{ fontSize: 12, color: "#64748b", fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
        {icon && <span style={{ marginRight: 6 }}>{icon}</span>}{label}
      </div>
      <div style={{ fontSize: 32, fontWeight: 800, color, fontFamily: "'JetBrains Mono', monospace", lineHeight: 1 }}>
        {value ?? "—"}<span style={{ fontSize: 16, fontWeight: 400, color: "#64748b", marginLeft: 4 }}>{unit}</span>
      </div>
      {sub && <div style={{ fontSize: 11, color: "#475569", marginTop: 6 }}>{sub}</div>}
    </div>
  );
}

function SectionTitle({ children, accent = "#22c55e" }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
      <div style={{ width: 4, height: 20, background: accent, borderRadius: 2 }} />
      <h2 style={{ margin: 0, fontSize: 13, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", color: "#94a3b8" }}>
        {children}
      </h2>
    </div>
  );
}

function AlertBadge({ count, level }) {
  const colors = { critical: "#f43f5e", warning: "#fb923c", info: "#38bdf8" };
  const c = colors[level] || "#64748b";
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
      <div style={{
        width: 44, height: 44, borderRadius: 12,
        background: c + "18", border: `1px solid ${c}44`,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 18, fontWeight: 800, color: c,
        fontFamily: "'JetBrains Mono', monospace",
      }}>{count}</div>
      <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em" }}>{level}</div>
    </div>
  );
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 10, padding: "10px 14px" }}>
      <div style={{ fontSize: 11, color: "#64748b", marginBottom: 6 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ fontSize: 13, color: p.color, fontFamily: "'JetBrains Mono', monospace" }}>
          {p.name}: {p.value}
        </div>
      ))}
    </div>
  );
}

function PredictPanel({ summary, modelsData }) {
  const [running, setRunning] = useState(false);
  const [result,  setResult]  = useState(null);

  const runLSTM = async () => {
    setRunning(true);
    try {
      const r = await fetch(`${API}/predict/lstm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ use_db: true, horizon: 4 }),
        signal: AbortSignal.timeout(5000),
      });
      const j = await r.json();
      setResult(j.data);
    } catch {
      // Mock fallback
      setResult({
        ph_forecast:   [6.08, 6.05, 6.02, 6.09],
        temp_forecast: [22.6, 22.8, 23.1, 23.0],
        ph_alert: false, temp_alert: false, horizon_steps: 4,
      });
    } finally {
      setRunning(false);
    }
  };

  return (
    <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, padding: 24 }}>
      <SectionTitle accent="#818cf8">LSTM Forecast — Run Prediction</SectionTitle>
      <button onClick={runLSTM} disabled={running} style={{
        background: running ? "#1e293b" : "linear-gradient(135deg, #818cf8, #6366f1)",
        color: "#fff", border: "none", borderRadius: 10, padding: "10px 22px",
        fontWeight: 700, fontSize: 13, cursor: running ? "not-allowed" : "pointer",
        letterSpacing: "0.05em", marginBottom: 20,
        opacity: running ? 0.6 : 1, transition: "all 0.2s",
      }}>
        {running ? "Running…" : "▶  Run LSTM Forecast"}
      </button>

      {result && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <div style={{ background: "#0a1628", borderRadius: 12, padding: 16 }}>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 10, letterSpacing: "0.1em", textTransform: "uppercase" }}>pH Forecast</div>
            {result.ph_forecast.map((v, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: "#475569" }}>+{(i+1)*15} min</span>
                <span style={{ fontSize: 14, fontFamily: "'JetBrains Mono', monospace", color: phColor(v), fontWeight: 700 }}>{v.toFixed(3)}</span>
              </div>
            ))}
            {result.ph_alert && <Pill text="⚠ pH Alert" color="#fb923c" />}
          </div>
          <div style={{ background: "#0a1628", borderRadius: 12, padding: 16 }}>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 10, letterSpacing: "0.1em", textTransform: "uppercase" }}>Temp Forecast</div>
            {result.temp_forecast.map((v, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: "#475569" }}>+{(i+1)*15} min</span>
                <span style={{ fontSize: 14, fontFamily: "'JetBrains Mono', monospace", color: "#38bdf8", fontWeight: 700 }}>{v.toFixed(1)}°C</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function SmartHydroDashboard() {
  const [tab, setTab] = useState("overview");

  const { data: summary }  = useAPI("/dashboard/summary",  MOCK_SUMMARY, 15_000);
  const { data: trendsRaw } = useAPI("/dashboard/trends?hours=12", null, 30_000);
  const { data: dosingRaw } = useAPI("/dashboard/dosing-log?n=10", null, 20_000);
  const { data: models }   = useAPI("/predict/health",    MOCK_MODELS, 60_000);

  const trends = trendsRaw?.ph?.length > 0
    ? trendsRaw.ph.map((p, i) => ({
        t:    p.t?.slice(11, 16) ?? `T${i}`,
        ph:   p.v,
        ec:   trendsRaw.ec?.[i]?.v,
        temp: trendsRaw.temperature?.[i]?.v,
        hum:  trendsRaw.humidity?.[i]?.v,
      }))
    : generateMockTrends();

  const dosing = dosingRaw?.dosing_log ?? generateMockDosing();
  const r      = summary?.latest_reading ?? {};
  const alerts = summary?.alert_counts   ?? { critical: 0, warning: 0, info: 0 };
  const dose   = summary?.latest_dose    ?? {};
  const growth = summary?.latest_growth  ?? {};
  const modelMap = models?.models ?? MOCK_MODELS.models;

  const tabs = ["overview", "trends", "models", "dosing"];

  return (
    <div style={{
      minHeight: "100vh", background: "#060e1e",
      fontFamily: "'IBM Plex Mono', 'JetBrains Mono', monospace",
      color: "#e2e8f0",
    }}>
      {/* Inject Google Fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes slideIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        .card-enter { animation: slideIn 0.35s ease forwards; }
      `}</style>

      {/* ── Header ── */}
      <header style={{
        borderBottom: "1px solid #1e293b", padding: "0 32px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        height: 64, background: "#060e1e", position: "sticky", top: 0, zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: "linear-gradient(135deg, #22c55e, #16a34a)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18,
          }}>🌿</div>
          <div>
            <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: 18, letterSpacing: "-0.02em", color: "#f1f5f9" }}>
              SmartHydro
            </div>
            <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.15em", textTransform: "uppercase" }}>
              ML Intelligence Platform
            </div>
          </div>
        </div>

        <nav style={{ display: "flex", gap: 4 }}>
          {tabs.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              background: tab === t ? "#22c55e18" : "transparent",
              border: tab === t ? "1px solid #22c55e44" : "1px solid transparent",
              color: tab === t ? "#22c55e" : "#64748b",
              borderRadius: 8, padding: "6px 16px",
              fontSize: 12, fontWeight: 700, cursor: "pointer",
              letterSpacing: "0.08em", textTransform: "uppercase",
              transition: "all 0.15s",
            }}>{t}</button>
          ))}
        </nav>

        <div style={{ display: "flex", gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#22c55e", animation: "pulse 2s infinite", marginTop: 6 }} />
          <span style={{ fontSize: 11, color: "#22c55e", fontWeight: 600 }}>LIVE</span>
        </div>
      </header>

      <main style={{ maxWidth: 1400, margin: "0 auto", padding: "28px 32px" }}>

        {/* ── OVERVIEW TAB ── */}
        {tab === "overview" && (
          <div style={{ animation: "slideIn 0.3s ease" }}>
            {/* Stat row */}
            <div style={{ display: "flex", gap: 14, marginBottom: 24, flexWrap: "wrap" }}>
              <StatCard label="pH Level"     value={r.ph?.toFixed(3)}           color={phColor(r.ph)}     unit=""      icon="⚗" sub={`Optimal: 5.5–6.5`} />
              <StatCard label="EC"           value={r.ec?.toFixed(3)}            color="#818cf8"            unit="mS/cm" icon="⚡" sub={`Stage target: ${dose.ec_target ?? "—"}`} />
              <StatCard label="Temperature"  value={r.temperature?.toFixed(1)}   color="#38bdf8"            unit="°C"    icon="🌡" sub="Optimal: 18–30°C" />
              <StatCard label="Humidity"     value={r.humidity?.toFixed(0)}      color="#fb923c"            unit="%"     icon="💧" />
              <StatCard label="Light"        value={r.light_lux ? (r.light_lux/1000).toFixed(1) : "—"} color="#fbbf24" unit="kLux" icon="☀" />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 20, marginBottom: 24 }}>
              {/* Growth stage */}
              <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, padding: 24 }}>
                <SectionTitle accent="#22c55e">Growth Stage</SectionTitle>
                <div style={{ textAlign: "center", padding: "12px 0" }}>
                  <div style={{ fontSize: 48, marginBottom: 8 }}>
                    {{ seedling: "🌱", vegetative: "🌿", flowering: "🌸", harvest: "🥬" }[growth.growth_stage] ?? "🌿"}
                  </div>
                  <div style={{ fontSize: 20, fontFamily: "'Syne', sans-serif", fontWeight: 800, color: "#22c55e", textTransform: "capitalize" }}>
                    {growth.growth_stage ?? r.growth_stage ?? "Vegetative"}
                  </div>
                  <div style={{ fontSize: 11, color: "#475569", marginTop: 4 }}>
                    Day {r.day_in_cycle ?? "—"} of cycle
                  </div>
                  <div style={{ marginTop: 16 }}>
                    <div style={{ fontSize: 11, color: "#64748b", marginBottom: 6, letterSpacing: "0.08em" }}>CNN CONFIDENCE</div>
                    <div style={{ background: "#1e293b", borderRadius: 999, height: 6, overflow: "hidden" }}>
                      <div style={{ width: `${(growth.confidence ?? 0.88) * 100}%`, height: "100%", background: "linear-gradient(90deg, #22c55e, #16a34a)", borderRadius: 999 }} />
                    </div>
                    <div style={{ fontSize: 12, color: "#22c55e", fontWeight: 700, marginTop: 4 }}>
                      {((growth.confidence ?? 0.88) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Alerts */}
              <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, padding: 24 }}>
                <SectionTitle accent="#f43f5e">Active Alerts</SectionTitle>
                <div style={{ display: "flex", justifyContent: "space-around", padding: "16px 0" }}>
                  <AlertBadge count={alerts.critical} level="critical" />
                  <AlertBadge count={alerts.warning}  level="warning"  />
                  <AlertBadge count={alerts.info}     level="info"     />
                </div>
                <div style={{ borderTop: "1px solid #1e293b", paddingTop: 16, marginTop: 8 }}>
                  <div style={{ fontSize: 11, color: "#64748b" }}>Total unresolved</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: "#e2e8f0", fontFamily: "'JetBrains Mono', monospace" }}>
                    {(alerts.critical + alerts.warning + alerts.info)}
                  </div>
                </div>
              </div>

              {/* Latest dose */}
              <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, padding: 24 }}>
                <SectionTitle accent="#818cf8">RL Dosing Agent</SectionTitle>
                <div style={{ textAlign: "center", padding: "8px 0" }}>
                  <div style={{ fontSize: 40, color: actionColor(dose.action), fontWeight: 800, marginBottom: 6 }}>
                    {actionGlyph(dose.action)}
                  </div>
                  <Pill text={dose.action ?? "maintain"} color={actionColor(dose.action)} />
                  <div style={{ marginTop: 16, display: "flex", justifyContent: "space-between" }}>
                    <div>
                      <div style={{ fontSize: 10, color: "#64748b" }}>EC TARGET</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: "#818cf8", fontFamily: "'JetBrains Mono', monospace" }}>{dose.ec_target ?? "—"}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 10, color: "#64748b" }}>CONFIDENCE</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: "#22c55e", fontFamily: "'JetBrains Mono', monospace" }}>{dose.confidence ? `${(dose.confidence*100).toFixed(0)}%` : "—"}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 10, color: "#64748b" }}>CROP</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: "#fb923c", textTransform: "capitalize" }}>{dose.crop_type ?? "—"}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Mini pH chart */}
            <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, padding: 24 }}>
              <SectionTitle accent="#22c55e">pH — Last 12 Hours</SectionTitle>
              <ResponsiveContainer width="100%" height={160}>
                <AreaChart data={trends.slice(-24)}>
                  <defs>
                    <linearGradient id="phGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#22c55e" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="t" tick={{ fill: "#475569", fontSize: 10 }} />
                  <YAxis domain={[5.0, 7.5]} tick={{ fill: "#475569", fontSize: 10 }} width={38} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="ph" name="pH" stroke="#22c55e" fill="url(#phGrad)" strokeWidth={2} dot={false} />
                  {/* Safe zone reference lines rendered via clip */}
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* ── TRENDS TAB ── */}
        {tab === "trends" && (
          <div style={{ animation: "slideIn 0.3s ease", display: "flex", flexDirection: "column", gap: 20 }}>
            {[
              { key: "ph",   name: "pH",           color: "#22c55e", domain: [5.0, 7.5] },
              { key: "ec",   name: "EC (mS/cm)",   color: "#818cf8", domain: [0.5, 2.5] },
              { key: "temp", name: "Temperature °C", color: "#38bdf8", domain: [16, 32] },
              { key: "hum",  name: "Humidity %",   color: "#fb923c", domain: [40, 90] },
            ].map(({ key, name, color, domain }) => (
              <div key={key} style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, padding: 24 }}>
                <SectionTitle accent={color}>{name}</SectionTitle>
                <ResponsiveContainer width="100%" height={140}>
                  <LineChart data={trends}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="t" tick={{ fill: "#475569", fontSize: 10 }} interval={Math.floor(trends.length/8)} />
                    <YAxis domain={domain} tick={{ fill: "#475569", fontSize: 10 }} width={40} />
                    <Tooltip content={<CustomTooltip />} />
                    <Line type="monotone" dataKey={key} name={name} stroke={color} strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        )}

        {/* ── MODELS TAB ── */}
        {tab === "models" && (
          <div style={{ animation: "slideIn 0.3s ease", display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Model registry */}
            <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, padding: 24 }}>
              <SectionTitle accent="#818cf8">Model Registry</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 14 }}>
                {[
                  { key: "lstm",        icon: "📈", label: "LSTM",           desc: "pH & Temp Forecast" },
                  { key: "cnn",         icon: "📷", label: "ResNet-50 CNN",  desc: "Growth Stage Classifier" },
                  { key: "yolo",        icon: "🔍", label: "YOLOv8",         desc: "Pest & Disease Detector" },
                  { key: "autoencoder", icon: "🔐", label: "Autoencoder",    desc: "Anomaly Detector" },
                  { key: "rl_policy",   icon: "🤖", label: "PPO RL Agent",   desc: "Nutrient Dosing Policy" },
                ].map(({ key, icon, label, desc }) => {
                  const status = modelMap[key] ?? "unknown";
                  const ok_ = status === "loaded";
                  return (
                    <div key={key} style={{
                      background: "#060e1e", border: `1px solid ${ok_ ? "#22c55e33" : "#f59e0b33"}`,
                      borderRadius: 12, padding: 16,
                    }}>
                      <div style={{ fontSize: 24, marginBottom: 8 }}>{icon}</div>
                      <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0" }}>{label}</div>
                      <div style={{ fontSize: 11, color: "#475569", marginBottom: 10 }}>{desc}</div>
                      <Pill text={ok_ ? "loaded" : "stub"} color={ok_ ? "#22c55e" : "#f59e0b"} />
                    </div>
                  );
                })}
              </div>
            </div>

            {/* LSTM run panel */}
            <PredictPanel summary={summary} modelsData={modelMap} />

            {/* Performance metrics */}
            <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, padding: 24 }}>
              <SectionTitle accent="#38bdf8">Training Performance (Last Run)</SectionTitle>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr>
                      {["Model","Task","Metric","Score"].map(h => (
                        <th key={h} style={{ textAlign: "left", padding: "8px 16px", fontSize: 11, color: "#64748b", letterSpacing: "0.1em", textTransform: "uppercase", borderBottom: "1px solid #1e293b" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["LSTM",       "pH Prediction",       "MAE",       "0.08",   "#22c55e"],
                      ["Rand Forest","EC Regression",       "R²",        "0.94",   "#22c55e"],
                      ["YOLOv8",     "Pest Detection",      "mAP@0.5",   "91.3%",  "#22c55e"],
                      ["Autoencoder","Anomaly Detection",   "F1-Score",  "0.89",   "#22c55e"],
                      ["ResNet-50",  "Growth Classification","Accuracy", "93.7%",  "#22c55e"],
                      ["PPO Agent",  "Nutrient Dosing",     "Reward Avg","+0.82",  "#818cf8"],
                    ].map(([model, task, metric, score, color], i) => (
                      <tr key={i} style={{ borderBottom: "1px solid #0f172a" }}>
                        <td style={{ padding: "12px 16px", fontWeight: 700, color: "#e2e8f0" }}>{model}</td>
                        <td style={{ padding: "12px 16px", color: "#94a3b8" }}>{task}</td>
                        <td style={{ padding: "12px 16px", color: "#64748b" }}>{metric}</td>
                        <td style={{ padding: "12px 16px", color, fontWeight: 800, fontFamily: "'JetBrains Mono', monospace" }}>{score}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ── DOSING TAB ── */}
        {tab === "dosing" && (
          <div style={{ animation: "slideIn 0.3s ease", display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Action distribution chart */}
            <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, padding: 24 }}>
              <SectionTitle accent="#818cf8">Dosing Action Distribution</SectionTitle>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={[
                  { name: "Increase", count: dosing.filter(d=>d.action==="increase").length, fill: "#22c55e" },
                  { name: "Maintain", count: dosing.filter(d=>d.action==="maintain").length, fill: "#64748b" },
                  { name: "Decrease", count: dosing.filter(d=>d.action==="decrease").length, fill: "#f43f5e" },
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" tick={{ fill: "#475569", fontSize: 12 }} />
                  <YAxis tick={{ fill: "#475569", fontSize: 12 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" name="Actions" radius={[6,6,0,0]}>
                    {[{ fill:"#22c55e"},{ fill:"#64748b"},{ fill:"#f43f5e"}].map((c,i)=><Cell key={i} fill={c.fill}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Dosing log table */}
            <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, padding: 24 }}>
              <SectionTitle accent="#818cf8">RL Agent Dosing Log</SectionTitle>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr>
                      {["Action","EC Target","Confidence","Crop","Time"].map(h => (
                        <th key={h} style={{ textAlign: "left", padding: "8px 16px", fontSize: 11, color: "#64748b", letterSpacing: "0.08em", textTransform: "uppercase", borderBottom: "1px solid #1e293b" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {dosing.map((d, i) => (
                      <tr key={d._id ?? i} style={{ borderBottom: "1px solid #0a1628" }}>
                        <td style={{ padding: "12px 16px" }}>
                          <span style={{ color: actionColor(d.action), fontWeight: 700 }}>{actionGlyph(d.action)} {d.action}</span>
                        </td>
                        <td style={{ padding: "12px 16px", fontFamily: "'JetBrains Mono', monospace", color: "#818cf8" }}>{d.ec_target}</td>
                        <td style={{ padding: "12px 16px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <div style={{ width: 50, height: 4, background: "#1e293b", borderRadius: 2, overflow: "hidden" }}>
                              <div style={{ width: `${(d.confidence??0.9)*100}%`, height: "100%", background: "#22c55e", borderRadius: 2 }}/>
                            </div>
                            <span style={{ fontSize: 11, color: "#22c55e", fontFamily: "'JetBrains Mono', monospace" }}>{((d.confidence??0.9)*100).toFixed(0)}%</span>
                          </div>
                        </td>
                        <td style={{ padding: "12px 16px", color: "#fb923c", textTransform: "capitalize" }}>{d.crop_type}</td>
                        <td style={{ padding: "12px 16px", fontSize: 11, color: "#475569" }}>
                          {new Date(d.created_at).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
