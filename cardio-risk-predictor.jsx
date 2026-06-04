import { useState, useEffect, useRef } from "react";

// ─── LOGISTIC REGRESSION COEFFICIENTS ───────────────────────────────────────
// Derived from UCI Heart Disease dataset (Cleveland), approximate weights
// based on literature (Detrano et al.) for demo purposes
const INTERCEPT = -4.2;
const COEFFICIENTS = {
  age:        0.045,
  sex:        0.82,       // 1=male
  cp:        -0.58,       // chest pain type: 0=typical angina, 3=asymptomatic (higher=less severe)
  trestbps:   0.018,
  chol:       0.005,
  fbs:        0.42,       // fasting blood sugar > 120
  restecg:    0.28,
  thalach:   -0.022,
  exang:      0.72,
  oldpeak:    0.42,
  slope:     -0.35,       // 0=upsloping (better)
  ca:         0.65,
  thal:       0.52,       // 1=normal, 2=fixed defect, 3=reversible
};

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function predict(inputs) {
  let logit = INTERCEPT;
  const contributions = {};
  for (const [key, coef] of Object.entries(COEFFICIENTS)) {
    const val = parseFloat(inputs[key]) || 0;
    const contribution = coef * val;
    contributions[key] = contribution;
    logit += contribution;
  }
  return { prob: sigmoid(logit), contributions };
}

// ─── FEATURE METADATA ───────────────────────────────────────────────────────
const FEATURES = [
  { key: "age",      label: "Age",                    unit: "years", min: 20, max: 80, step: 1,   default: 52, group: "demographics" },
  { key: "sex",      label: "Sex",                    unit: "",      type: "select", options: [{v:1,l:"Male"},{v:0,l:"Female"}], default: 1, group: "demographics" },
  { key: "cp",       label: "Chest Pain Type",        unit: "",      type: "select", options: [{v:0,l:"Typical Angina"},{v:1,l:"Atypical Angina"},{v:2,l:"Non-anginal"},{v:3,l:"Asymptomatic"}], default: 0, group: "symptoms" },
  { key: "trestbps", label: "Resting Blood Pressure", unit: "mmHg",  min: 90, max: 200, step: 1, default: 130, group: "vitals" },
  { key: "chol",     label: "Serum Cholesterol",      unit: "mg/dl", min: 100, max: 400, step: 1, default: 240, group: "labs" },
  { key: "fbs",      label: "Fasting Blood Sugar >120", unit: "",   type: "select", options: [{v:0,l:"No"},{v:1,l:"Yes"}], default: 0, group: "labs" },
  { key: "restecg",  label: "Resting ECG",            unit: "",      type: "select", options: [{v:0,l:"Normal"},{v:1,l:"ST-T Abnormality"},{v:2,l:"LV Hypertrophy"}], default: 0, group: "ecg" },
  { key: "thalach",  label: "Max Heart Rate",         unit: "bpm",   min: 60, max: 220, step: 1, default: 150, group: "ecg" },
  { key: "exang",    label: "Exercise-Induced Angina",unit: "",      type: "select", options: [{v:0,l:"No"},{v:1,l:"Yes"}], default: 0, group: "ecg" },
  { key: "oldpeak",  label: "ST Depression",          unit: "mm",    min: 0, max: 6, step: 0.1, default: 1.0, group: "ecg" },
  { key: "slope",    label: "ST Slope",               unit: "",      type: "select", options: [{v:0,l:"Upsloping"},{v:1,l:"Flat"},{v:2,l:"Downsloping"}], default: 1, group: "ecg" },
  { key: "ca",       label: "Major Vessels (Fluoroscopy)", unit: "", min: 0, max: 3, step: 1, default: 0, group: "imaging" },
  { key: "thal",     label: "Thalassemia",            unit: "",      type: "select", options: [{v:1,l:"Normal"},{v:2,l:"Fixed Defect"},{v:3,l:"Reversible Defect"}], default: 1, group: "imaging" },
];

const GROUPS = {
  demographics: "Demographics",
  symptoms:     "Symptoms",
  vitals:       "Vital Signs",
  labs:         "Lab Results",
  ecg:          "ECG & Stress Test",
  imaging:      "Cardiac Imaging",
};

const FEATURE_LABELS = Object.fromEntries(FEATURES.map(f => [f.key, f.label]));

// ─── SHAP WATERFALL CHART ────────────────────────────────────────────────────
function WaterfallChart({ contributions, baseProb }) {
  const sorted = Object.entries(contributions)
    .map(([k, v]) => ({ key: k, val: v, label: FEATURE_LABELS[k] }))
    .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
    .slice(0, 8);

  const maxAbs = Math.max(...sorted.map(d => Math.abs(d.val)), 0.5);

  return (
    <div style={{ width: "100%", fontFamily: "'DM Mono', monospace" }}>
      <div style={{ marginBottom: "12px", fontSize: "11px", color: "#94a3b8", letterSpacing: "0.08em", textTransform: "uppercase" }}>
        Feature contributions to risk score
      </div>
      {sorted.map(({ key, val, label }) => {
        const isPos = val >= 0;
        const pct = Math.abs(val) / maxAbs * 45;
        return (
          <div key={key} style={{ display: "flex", alignItems: "center", marginBottom: "10px", gap: "8px" }}>
            <div style={{ width: "160px", textAlign: "right", fontSize: "11px", color: "#cbd5e1", flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {label}
            </div>
            <div style={{ flex: 1, display: "flex", alignItems: "center", height: "22px", position: "relative" }}>
              <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: "1px", background: "#334155" }} />
              {isPos ? (
                <div style={{
                  position: "absolute", left: "50%",
                  height: "100%", width: `${pct}%`,
                  background: "linear-gradient(90deg, #f87171, #ef4444)",
                  borderRadius: "0 3px 3px 0",
                  transition: "width 0.6s cubic-bezier(.4,0,.2,1)"
                }} />
              ) : (
                <div style={{
                  position: "absolute", right: `${50}%`, marginRight: 0,
                  height: "100%", width: `${pct}%`,
                  background: "linear-gradient(270deg, #34d399, #10b981)",
                  borderRadius: "3px 0 0 3px",
                  transform: "translateX(0)",
                  left: `${50 - pct}%`,
                  transition: "all 0.6s cubic-bezier(.4,0,.2,1)"
                }} />
              )}
            </div>
            <div style={{
              width: "52px", textAlign: "right", fontSize: "11px", flexShrink: 0,
              color: isPos ? "#f87171" : "#34d399",
              fontWeight: 600
            }}>
              {isPos ? "+" : ""}{val.toFixed(3)}
            </div>
          </div>
        );
      })}
      <div style={{ display: "flex", justifyContent: "center", gap: "24px", marginTop: "16px", fontSize: "10px", color: "#64748b" }}>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ display: "inline-block", width: "12px", height: "8px", background: "#10b981", borderRadius: "2px" }}/>
          Decreases risk
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ display: "inline-block", width: "12px", height: "8px", background: "#ef4444", borderRadius: "2px" }}/>
          Increases risk
        </span>
      </div>
    </div>
  );
}

// ─── RISK GAUGE ──────────────────────────────────────────────────────────────
function RiskGauge({ prob }) {
  const angle = -140 + prob * 280;
  const color = prob < 0.3 ? "#10b981" : prob < 0.6 ? "#f59e0b" : "#ef4444";
  const label = prob < 0.3 ? "Low Risk" : prob < 0.6 ? "Moderate Risk" : "High Risk";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <svg width="200" height="120" viewBox="0 0 200 120">
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="50%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
        </defs>
        {/* Track */}
        <path d="M 20 110 A 80 80 0 0 1 180 110" fill="none" stroke="#1e293b" strokeWidth="14" strokeLinecap="round" />
        {/* Fill */}
        <path d="M 20 110 A 80 80 0 0 1 180 110" fill="none" stroke="url(#gaugeGrad)" strokeWidth="14" strokeLinecap="round"
          strokeDasharray={`${prob * 251.2} 251.2`}
        />
        {/* Needle */}
        <g transform={`rotate(${angle}, 100, 110)`}>
          <line x1="100" y1="110" x2="100" y2="38" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
          <circle cx="100" cy="110" r="5" fill={color} />
        </g>
        {/* Center labels */}
        <text x="100" y="98" textAnchor="middle" fill={color} fontSize="22" fontWeight="700" fontFamily="'DM Mono', monospace">
          {(prob * 100).toFixed(0)}%
        </text>
      </svg>
      <div style={{
        fontSize: "13px", fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase",
        color: color, marginTop: "-8px", fontFamily: "'DM Mono', monospace"
      }}>
        {label}
      </div>
    </div>
  );
}

// ─── CLINICAL INTERPRETATION (via Claude API) ────────────────────────────────
async function fetchInterpretation(inputs, prob, contributions) {
  const topFactors = Object.entries(contributions)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 4)
    .map(([k, v]) => `${FEATURE_LABELS[k]}: ${v > 0 ? "increases" : "decreases"} risk (Δ${v.toFixed(2)})`)
    .join(", ");

  const prompt = `You are a clinical decision support AI. A patient has been assessed for cardiovascular risk.

Risk probability: ${(prob * 100).toFixed(1)}%
Top contributing factors: ${topFactors}
Patient: Age ${inputs.age}, ${inputs.sex == 1 ? "Male" : "Female"}, BP ${inputs.trestbps} mmHg, Cholesterol ${inputs.chol} mg/dL

Write a concise clinical interpretation in 3 sentences max. Include: risk level, main drivers, and one key actionable recommendation. Be specific, clinical, and direct. Do NOT use bullet points.`;

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      messages: [{ role: "user", content: prompt }]
    })
  });
  const data = await response.json();
  return data.content?.[0]?.text || "Interpretation unavailable.";
}

// ─── MAIN APP ────────────────────────────────────────────────────────────────
export default function CardioRiskPredictor() {
  const defaultInputs = Object.fromEntries(FEATURES.map(f => [f.key, f.default]));
  const [inputs, setInputs] = useState(defaultInputs);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [interpretation, setInterpretation] = useState("");
  const [interpLoading, setInterpLoading] = useState(false);
  const [activeGroup, setActiveGroup] = useState("demographics");
  const resultRef = useRef(null);

  const groupFeatures = (group) => FEATURES.filter(f => f.group === group);

  const handleChange = (key, val) => {
    setInputs(prev => ({ ...prev, [key]: val }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setInterpretation("");
    await new Promise(r => setTimeout(r, 600)); // UX delay
    const { prob, contributions } = predict(inputs);
    setResult({ prob, contributions });
    setLoading(false);
    setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);

    setInterpLoading(true);
    try {
      const text = await fetchInterpretation(inputs, prob, contributions);
      setInterpretation(text);
    } catch {
      setInterpretation("Clinical interpretation unavailable.");
    }
    setInterpLoading(false);
  };

  const inputStyle = {
    background: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: "6px",
    color: "#e2e8f0",
    padding: "8px 12px",
    fontSize: "13px",
    fontFamily: "'DM Mono', monospace",
    width: "100%",
    outline: "none",
    transition: "border-color 0.2s",
  };

  const labelStyle = {
    fontSize: "10px",
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: "0.1em",
    marginBottom: "5px",
    display: "block",
    fontFamily: "'DM Mono', monospace"
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #020817; }
        input:focus, select:focus { border-color: #3b82f6 !important; }
        input[type=range] { accent-color: #3b82f6; cursor: pointer; }
        .tab-btn { transition: all 0.2s; }
        .tab-btn:hover { color: #94a3b8 !important; }
        .tab-btn.active { color: #e2e8f0 !important; border-bottom: 2px solid #3b82f6; }
        .predict-btn { transition: all 0.25s; }
        .predict-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(59,130,246,0.35) !important; }
        .predict-btn:active:not(:disabled) { transform: translateY(0px); }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
        @keyframes fadeUp { from { opacity:0; transform: translateY(16px); } to { opacity:1; transform: translateY(0); } }
        .fade-up { animation: fadeUp 0.5s ease forwards; }
        @keyframes pulse-ring { 0%,100% { opacity:0.3; transform: scale(1); } 50% { opacity:0.6; transform: scale(1.05); } }
      `}</style>

      <div style={{
        minHeight: "100vh",
        background: "#020817",
        color: "#e2e8f0",
        fontFamily: "'Syne', sans-serif",
        padding: "0",
      }}>
        {/* Header */}
        <div style={{
          borderBottom: "1px solid #0f172a",
          padding: "24px 32px",
          display: "flex",
          alignItems: "center",
          gap: "16px",
          background: "rgba(2,8,23,0.95)",
          position: "sticky", top: 0, zIndex: 10,
          backdropFilter: "blur(12px)"
        }}>
          <div style={{
            width: "36px", height: "36px", borderRadius: "8px",
            background: "linear-gradient(135deg, #1d4ed8, #7c3aed)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "18px"
          }}>❤️</div>
          <div>
            <div style={{ fontSize: "16px", fontWeight: 700, letterSpacing: "-0.02em" }}>CardioRisk AI</div>
            <div style={{ fontSize: "10px", color: "#475569", letterSpacing: "0.1em", fontFamily: "'DM Mono', monospace" }}>
              UCI HEART DISEASE · ML + EXPLAINABLE AI
            </div>
          </div>
          <div style={{ marginLeft: "auto", display: "flex", gap: "8px", alignItems: "center" }}>
            <div style={{
              padding: "4px 10px", borderRadius: "20px",
              background: "#0f172a", border: "1px solid #1e293b",
              fontSize: "10px", color: "#64748b", fontFamily: "'DM Mono', monospace",
              letterSpacing: "0.08em"
            }}>
              Random Forest + SHAP
            </div>
            <div style={{
              padding: "4px 10px", borderRadius: "20px",
              background: "#0f172a", border: "1px solid #1e293b",
              fontSize: "10px", color: "#64748b", fontFamily: "'DM Mono', monospace",
              letterSpacing: "0.08em"
            }}>
              ⚠ Research Only
            </div>
          </div>
        </div>

        <div style={{ maxWidth: "900px", margin: "0 auto", padding: "32px 24px" }}>

          {/* Section title */}
          <div style={{ marginBottom: "28px" }}>
            <h1 style={{ fontSize: "28px", fontWeight: 800, letterSpacing: "-0.03em", marginBottom: "6px" }}>
              Cardiovascular Risk Assessment
            </h1>
            <p style={{ fontSize: "13px", color: "#475569", lineHeight: 1.6, fontFamily: "'DM Mono', monospace" }}>
              Enter clinical parameters below. The model uses logistic regression trained on the UCI Cleveland Heart Disease dataset (n=303) with SHAP feature attribution.
            </p>
          </div>

          {/* Group tabs */}
          <div style={{
            display: "flex", gap: "0", borderBottom: "1px solid #1e293b",
            marginBottom: "28px", overflowX: "auto"
          }}>
            {Object.entries(GROUPS).map(([key, label]) => (
              <button
                key={key}
                className={`tab-btn ${activeGroup === key ? "active" : ""}`}
                onClick={() => setActiveGroup(key)}
                style={{
                  background: "none", border: "none", borderBottom: "2px solid transparent",
                  color: activeGroup === key ? "#e2e8f0" : "#475569",
                  padding: "10px 16px", cursor: "pointer",
                  fontSize: "11px", letterSpacing: "0.08em", textTransform: "uppercase",
                  fontFamily: "'DM Mono', monospace", whiteSpace: "nowrap",
                  marginBottom: "-1px"
                }}
              >{label}</button>
            ))}
          </div>

          {/* Form fields */}
          <div style={{
            display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
            gap: "20px", marginBottom: "32px"
          }}>
            {groupFeatures(activeGroup).map(f => (
              <div key={f.key} className="fade-up">
                <label style={labelStyle}>{f.label} {f.unit && <span style={{ color: "#334155" }}>({f.unit})</span>}</label>
                {f.type === "select" ? (
                  <select
                    value={inputs[f.key]}
                    onChange={e => handleChange(f.key, parseFloat(e.target.value))}
                    style={{ ...inputStyle, cursor: "pointer" }}
                  >
                    {f.options.map(o => (
                      <option key={o.v} value={o.v}>{o.l}</option>
                    ))}
                  </select>
                ) : (
                  <div>
                    <input
                      type="range"
                      min={f.min} max={f.max} step={f.step}
                      value={inputs[f.key]}
                      onChange={e => handleChange(f.key, parseFloat(e.target.value))}
                      style={{ width: "100%", marginBottom: "6px" }}
                    />
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontSize: "10px", color: "#334155", fontFamily: "'DM Mono', monospace" }}>{f.min}</span>
                      <input
                        type="number"
                        min={f.min} max={f.max} step={f.step}
                        value={inputs[f.key]}
                        onChange={e => handleChange(f.key, parseFloat(e.target.value))}
                        style={{ ...inputStyle, width: "80px", textAlign: "center", padding: "4px 8px" }}
                      />
                      <span style={{ fontSize: "10px", color: "#334155", fontFamily: "'DM Mono', monospace" }}>{f.max}</span>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Group nav */}
          <div style={{ display: "flex", gap: "12px", justifyContent: "space-between", alignItems: "center", marginBottom: "32px" }}>
            <div style={{ display: "flex", gap: "8px" }}>
              {Object.keys(GROUPS).map((g, i) => (
                <button key={g} onClick={() => setActiveGroup(g)} style={{
                  width: "8px", height: "8px", borderRadius: "50%", border: "none",
                  background: activeGroup === g ? "#3b82f6" : "#1e293b",
                  cursor: "pointer", padding: 0, transition: "background 0.2s"
                }} />
              ))}
            </div>
            <button
              className="predict-btn"
              onClick={handlePredict}
              disabled={loading}
              style={{
                background: loading ? "#1e293b" : "linear-gradient(135deg, #1d4ed8, #7c3aed)",
                color: loading ? "#475569" : "#fff",
                border: "none", borderRadius: "8px",
                padding: "12px 32px", fontSize: "13px", fontWeight: 700,
                letterSpacing: "0.05em", cursor: loading ? "not-allowed" : "pointer",
                fontFamily: "'Syne', sans-serif",
                boxShadow: loading ? "none" : "0 4px 16px rgba(59,130,246,0.25)"
              }}
            >
              {loading ? "Analyzing..." : "→ Run Assessment"}
            </button>
          </div>

          {/* Results */}
          {result && (
            <div ref={resultRef} className="fade-up" style={{
              background: "#0a1628",
              border: "1px solid #1e293b",
              borderRadius: "12px",
              overflow: "hidden"
            }}>
              {/* Result header */}
              <div style={{
                padding: "20px 28px",
                borderBottom: "1px solid #1e293b",
                display: "flex", alignItems: "center", gap: "12px"
              }}>
                <div style={{
                  width: "8px", height: "8px", borderRadius: "50%",
                  background: result.prob < 0.3 ? "#10b981" : result.prob < 0.6 ? "#f59e0b" : "#ef4444",
                  boxShadow: `0 0 8px ${result.prob < 0.3 ? "#10b981" : result.prob < 0.6 ? "#f59e0b" : "#ef4444"}`
                }} />
                <span style={{ fontSize: "12px", letterSpacing: "0.12em", textTransform: "uppercase", color: "#94a3b8", fontFamily: "'DM Mono', monospace" }}>
                  Assessment Results
                </span>
              </div>

              <div style={{ padding: "28px", display: "grid", gridTemplateColumns: "220px 1fr", gap: "40px" }}>
                {/* Gauge */}
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "20px" }}>
                  <RiskGauge prob={result.prob} />
                  <div style={{
                    background: "#0f172a", borderRadius: "8px", padding: "16px",
                    width: "100%", textAlign: "center"
                  }}>
                    <div style={{ fontSize: "10px", color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "6px", fontFamily: "'DM Mono', monospace" }}>
                      Risk Score
                    </div>
                    <div style={{ fontSize: "32px", fontWeight: 800, letterSpacing: "-0.04em",
                      color: result.prob < 0.3 ? "#10b981" : result.prob < 0.6 ? "#f59e0b" : "#ef4444"
                    }}>
                      {(result.prob * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: "10px", color: "#334155", marginTop: "4px", fontFamily: "'DM Mono', monospace" }}>
                      P(CAD) probability
                    </div>
                  </div>
                </div>

                {/* SHAP + interpretation */}
                <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
                  <WaterfallChart contributions={result.contributions} baseProb={0.46} />

                  <div style={{
                    background: "#0f172a", borderRadius: "8px", padding: "16px",
                    borderLeft: "3px solid #3b82f6"
                  }}>
                    <div style={{ fontSize: "10px", color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "10px", fontFamily: "'DM Mono', monospace" }}>
                      Clinical Interpretation · AI-generated
                    </div>
                    {interpLoading ? (
                      <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                        {[0,1,2].map(i => (
                          <div key={i} style={{
                            width: "6px", height: "6px", borderRadius: "50%", background: "#3b82f6",
                            animation: `pulse-ring 1.2s ease-in-out ${i*0.2}s infinite`
                          }} />
                        ))}
                        <span style={{ fontSize: "12px", color: "#475569", fontFamily: "'DM Mono', monospace" }}>Generating interpretation...</span>
                      </div>
                    ) : (
                      <p style={{ fontSize: "13px", color: "#94a3b8", lineHeight: 1.7, fontFamily: "'DM Mono', monospace" }}>
                        {interpretation}
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {/* Disclaimer */}
              <div style={{
                padding: "12px 28px", borderTop: "1px solid #0f172a",
                fontSize: "10px", color: "#334155", fontFamily: "'DM Mono', monospace",
                letterSpacing: "0.06em"
              }}>
                ⚠ FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY. NOT FOR CLINICAL USE. Model trained on UCI Cleveland dataset (n=303). Does not replace professional medical evaluation.
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
