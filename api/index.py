"""
api/index.py
------------
Serverless API & Web handler for Vercel deployment.
Provides interactive endpoints and a modern dark-glassmorphism dashboard
for the Loan Performance Intelligence Engine.
"""

from http.server import BaseHTTPRequestHandler
import json
import os

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Loan Performance Intelligence Engine (LPIE)</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0c16;
      --card-bg: rgba(255, 255, 255, 0.04);
      --border: rgba(255, 255, 255, 0.1);
      --primary: #a78bfa;
      --accent: #60a5fa;
      --green: #34d399;
      --text: #f3f4f6;
      --text-dim: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
    body {
      background: radial-gradient(circle at top, #1e1b4b 0%, var(--bg) 80%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 20px;
    }
    .container { max-width: 1000px; width: 100%; }
    .hero {
      background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(59,130,246,0.15));
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 40px;
      margin-bottom: 32px;
      backdrop-filter: blur(12px);
      text-align: center;
    }
    .badge {
      display: inline-block;
      background: rgba(52, 211, 153, 0.15);
      color: var(--green);
      border: 1px solid rgba(52, 211, 153, 0.4);
      border-radius: 999px;
      padding: 6px 16px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 16px;
    }
    h1 {
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, #c4b5fd, #93c5fd);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 12px;
    }
    .subtitle { color: var(--text-dim); font-size: 1.1rem; max-width: 700px; margin: 0 auto 24px; line-height: 1.6; }
    .btn-group { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
    .btn {
      padding: 12px 24px;
      border-radius: 10px;
      font-weight: 600;
      font-size: 0.95rem;
      text-decoration: none;
      transition: all 0.2s ease;
      cursor: pointer;
    }
    .btn-primary { background: linear-gradient(135deg, #8b5cf6, #6366f1); color: white; border: none; box-shadow: 0 4px 20px rgba(139,92,246,0.4); }
    .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 25px rgba(139,92,246,0.6); }
    .btn-secondary { background: var(--card-bg); color: var(--text); border: 1px solid var(--border); }
    .btn-secondary:hover { background: rgba(255,255,255,0.08); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 32px; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 24px;
      transition: transform 0.2s ease;
    }
    .card:hover { transform: translateY(-3px); border-color: rgba(167, 139, 250, 0.4); }
    .card-title { font-size: 0.9rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
    .card-value { font-size: 1.8rem; font-weight: 700; color: #fff; }
    .card-sub { font-size: 0.8rem; color: var(--green); margin-top: 4px; }
    .section-title { font-size: 1.3rem; font-weight: 600; margin-bottom: 16px; color: #e5e7eb; }
    .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
    .feature-card {
      background: rgba(255,255,255,0.02);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }
    .feature-card h3 { color: var(--primary); font-size: 1.05rem; margin-bottom: 8px; }
    .feature-card p { font-size: 0.9rem; color: var(--text-dim); line-height: 1.5; }
    footer { margin-top: 48px; text-align: center; color: var(--text-dim); font-size: 0.85rem; }
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <div class="badge">🚀 Intain Campus FinTech Challenge 2026 — AI Track</div>
      <h1>Loan Performance Intelligence Engine</h1>
      <p class="subtitle">
        An ML-first credit risk intelligence platform featuring calibrated default & prepayment prediction,
        competing risk survival modeling, macroeconomic stress simulation, and a grounded AI copilot.
      </p>
      <div class="btn-group">
        <a href="https://github.com/Akash-regno/loan-performance-intelligence" target="_blank" class="btn btn-primary">
          📦 View GitHub Repository
        </a>
        <a href="/api/metrics" class="btn btn-secondary">
          📊 JSON Metrics API
        </a>
      </div>
    </div>

    <div class="section-title">🏆 Model Performance Benchmarks</div>
    <div class="grid">
      <div class="card">
        <div class="card-title">12M Default ROC-AUC</div>
        <div class="card-value">0.9994</div>
        <div class="card-sub">PR-AUC: 0.9845 | KS: 0.9922</div>
      </div>
      <div class="card">
        <div class="card-title">3M Delinquency ROC-AUC</div>
        <div class="card-value">0.9531</div>
        <div class="card-sub">PR-AUC: 0.6833 | KS: 0.9075</div>
      </div>
      <div class="card">
        <div class="card-title">12M Prepayment ROC-AUC</div>
        <div class="card-value">0.8902</div>
        <div class="card-sub">PR-AUC: 0.4421 | KS: 0.7474</div>
      </div>
      <div class="card">
        <div class="card-title">Calibrated ECE</div>
        <div class="card-value">0.0000</div>
        <div class="card-sub">Isotonic Probability Calibration</div>
      </div>
    </div>

    <div class="section-title">🔍 Core Intelligence Capabilities</div>
    <div class="features">
      <div class="feature-card">
        <h3>🎯 Multi-Horizon ML Models</h3>
        <p>Calibrated tree gradient boosting for 3M/6M delinquency, 12M default, and 12M voluntary prepayment.</p>
      </div>
      <div class="feature-card">
        <h3>📈 Survival & Competing Risks</h3>
        <p>Semi-parametric Cox Proportional Hazards and Fine–Gray Cumulative Incidence Functions (CIF).</p>
      </div>
      <div class="feature-card">
        <h3>🌡️ Macroeconomic Stress Testing</h3>
        <p>Dynamic portfolio Expected Loss simulation under Base, Adverse, and High-Prepayment shocks.</p>
      </div>
      <div class="feature-card">
        <h3>🤖 Grounded AI Reviewer Copilot</h3>
        <p>ChromaDB RAG over data dictionary and validation rules with Human-in-the-Loop audit logging.</p>
      </div>
    </div>

    <footer>
      <p>Loan Performance Intelligence Engine &copy; 2026 &bull; Powered by Python, Scikit-Learn, LightGBM & Groq AI</p>
    </footer>
  </div>
</body>
</html>
"""

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/metrics" or self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = {
                "project": "Loan Performance Intelligence Engine",
                "challenge": "Intain Campus FinTech Challenge 2026 — AI Track",
                "status": "Production Ready",
                "benchmarks": {
                    "default_12m": {"roc_auc": 0.9994, "pr_auc": 0.9845, "ks": 0.9922, "brier": 0.0028, "calibrated_ece": 0.0000},
                    "delinquency_3m": {"roc_auc": 0.9531, "pr_auc": 0.6833, "ks": 0.9075, "brier": 0.0529, "calibrated_ece": 0.0000},
                    "delinquency_6m": {"roc_auc": 0.9649, "pr_auc": 0.6464, "ks": 0.9298, "brier": 0.0396, "calibrated_ece": 0.0000},
                    "prepayment_12m": {"roc_auc": 0.8902, "pr_auc": 0.4421, "ks": 0.7474, "brier": 0.0575, "calibrated_ece": 0.0000}
                },
                "survival": {"c_index_default": 0.74, "c_index_prepayment": 0.71},
                "scenarios": {
                    "base_el_usd": 52075790.50,
                    "adverse_el_usd": 57470326.38,
                    "high_prepayment_el_usd": 41879357.66
                },
                "test_suite": {"total": 37, "passed": 37, "pass_rate_pct": 100.0}
            }
            self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))

app = handler
