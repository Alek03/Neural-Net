# NeuralNet - Simple Feedforward Neural Network for Image Classification

This project implements a fully-connected feedforward neural network in Python using **NumPy**. It supports mini-batch stochastic gradient descent, computes categorical cross-entropy loss, and can generate predictions for Kaggle submissions.

---

## Network Architecture
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Neural Network Data Flow</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f1117;
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh;
    padding: 30px 20px;
  }
  h1 {
    text-align: center;
    font-size: 1.5rem;
    color: #a78bfa;
    margin-bottom: 6px;
    letter-spacing: 0.05em;
  }
  .subtitle {
    text-align: center;
    font-size: 0.8rem;
    color: #64748b;
    margin-bottom: 40px;
  }

  /* ── Main flow ── */
  .flow {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    flex-wrap: nowrap;
    overflow-x: auto;
    padding-bottom: 10px;
  }

  .block {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 110px;
  }

  .node {
    border-radius: 12px;
    padding: 14px 10px;
    width: 110px;
    text-align: center;
    position: relative;
    cursor: default;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .node:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.5); }

  .node .label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.75; margin-bottom: 4px; }
  .node .name  { font-size: 0.95rem; font-weight: 700; margin-bottom: 4px; }
  .node .dim   { font-size: 0.72rem; font-family: monospace; opacity: 0.85; }
  .node .desc  { font-size: 0.65rem; margin-top: 6px; opacity: 0.65; line-height: 1.4; }

  /* colour themes */
  .input-node  { background: linear-gradient(135deg,#1e293b,#0f2744); border: 1px solid #3b82f6; }
  .input-node .name { color: #60a5fa; }
  .hidden-node { background: linear-gradient(135deg,#1e1b3a,#0f1a2e); border: 1px solid #8b5cf6; }
  .hidden-node .name { color: #a78bfa; }
  .output-node { background: linear-gradient(135deg,#1a2e1a,#0f1f0f); border: 1px solid #22c55e; }
  .output-node .name { color: #4ade80; }
  .act-node    { background: linear-gradient(135deg,#2a1a10,#1a0f08); border: 1px solid #f97316; }
  .act-node .name { color: #fb923c; }
  .loss-node   { background: linear-gradient(135deg,#2a1020,#1a0818); border: 1px solid #ec4899; }
  .loss-node .name { color: #f472b6; }
  .update-node { background: linear-gradient(135deg,#0f2420,#081a10); border: 1px solid #14b8a6; }
  .update-node .name { color: #2dd4bf; }

  /* arrows */
  .arrow {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-width: 52px;
    gap: 2px;
  }
  .arrow .op {
    font-size: 0.62rem;
    font-family: monospace;
    color: #94a3b8;
    white-space: nowrap;
    text-align: center;
    line-height: 1.5;
  }
  .arrow svg { flex-shrink: 0; }

  /* backward pass row */
  .backward-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin-top: 16px;
    flex-wrap: nowrap;
    overflow-x: auto;
  }

  /* section labels */
  .section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 30px 0 12px;
    text-align: center;
  }
  .fwd-label  { color: #60a5fa; }
  .bwd-label  { color: #f472b6; }

  /* variable table */
  .var-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 10px;
    max-width: 900px;
    margin: 0 auto;
  }
  .var-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 14px;
  }
  .var-card .var-name { font-family: monospace; color: #a78bfa; font-size: 0.85rem; font-weight: 700; }
  .var-card .var-shape { font-family: monospace; color: #4ade80; font-size: 0.75rem; margin: 2px 0; }
  .var-card .var-desc  { font-size: 0.68rem; color: #94a3b8; }

  /* batch note */
  .note {
    max-width: 700px;
    margin: 20px auto 0;
    background: #1e293b;
    border-left: 3px solid #f97316;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 0.75rem;
    color: #94a3b8;
    line-height: 1.7;
  }
  .note strong { color: #fb923c; }
</style>
</head>
<body>

<h1>Neural Network — Data Flow</h1>
<p class="subtitle">MNIST digit classifier · 784 → 10 → 10 neurons · Mini-batch SGD</p>

<!-- ══════════════════ FORWARD PASS ══════════════════ -->
<p class="section-label fwd-label">▶ Forward Pass</p>

<div class="flow">

  <!-- Input -->
  <div class="block">
    <div class="node input-node">
      <div class="label">Input</div>
      <div class="name">X batch</div>
      <div class="dim">(500 × 784)</div>
      <div class="desc">500 images,<br>784 pixels each</div>
    </div>
  </div>

  <!-- Arrow: @ W1 + b1 -->
  <div class="arrow">
    <svg width="48" height="16" viewBox="0 0 48 16"><defs><marker id="ah1" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#3b82f6"/></marker></defs><line x1="2" y1="8" x2="42" y2="8" stroke="#3b82f6" stroke-width="1.5" marker-end="url(#ah1)"/></svg>
    <div class="op">@ W1 + b1<br>Z1 = X·W1+b1</div>
  </div>

  <!-- Z1 -->
  <div class="block">
    <div class="node hidden-node">
      <div class="label">Pre-act</div>
      <div class="name">Z1</div>
      <div class="dim">(500 × 10)</div>
      <div class="desc">Hidden layer<br>linear output</div>
    </div>
  </div>

  <!-- Arrow: ReLU -->
  <div class="arrow">
    <svg width="48" height="16" viewBox="0 0 48 16"><defs><marker id="ah2" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#f97316"/></marker></defs><line x1="2" y1="8" x2="42" y2="8" stroke="#f97316" stroke-width="1.5" marker-end="url(#ah2)"/></svg>
    <div class="op">ReLU(Z1)<br>max(0, x)</div>
  </div>

  <!-- A1 -->
  <div class="block">
    <div class="node act-node">
      <div class="label">Activation</div>
      <div class="name">A1</div>
      <div class="dim">(500 × 10)</div>
      <div class="desc">Hidden layer<br>activated output</div>
    </div>
  </div>

  <!-- Arrow: @ W2 + b2 -->
  <div class="arrow">
    <svg width="48" height="16" viewBox="0 0 48 16"><defs><marker id="ah3" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#8b5cf6"/></marker></defs><line x1="2" y1="8" x2="42" y2="8" stroke="#8b5cf6" stroke-width="1.5" marker-end="url(#ah3)"/></svg>
    <div class="op">@ W2 + b2<br>Z2 = A1·W2+b2</div>
  </div>

  <!-- Z2 -->
  <div class="block">
    <div class="node hidden-node">
      <div class="label">Pre-act</div>
      <div class="name">Z2</div>
      <div class="dim">(500 × 10)</div>
      <div class="desc">Output layer<br>linear output</div>
    </div>
  </div>

  <!-- Arrow: softmax -->
  <div class="arrow">
    <svg width="48" height="16" viewBox="0 0 48 16"><defs><marker id="ah4" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#22c55e"/></marker></defs><line x1="2" y1="8" x2="42" y2="8" stroke="#22c55e" stroke-width="1.5" marker-end="url(#ah4)"/></svg>
    <div class="op">softmax(Z2)<br>eˣ / Σeˣ</div>
  </div>

  <!-- y -->
  <div class="block">
    <div class="node output-node">
      <div class="label">Output</div>
      <div class="name">ŷ (y)</div>
      <div class="dim">(500 × 10)</div>
      <div class="desc">Class probabilities<br>digits 0–9</div>
    </div>
  </div>

  <!-- Arrow: CE loss -->
  <div class="arrow">
    <svg width="48" height="16" viewBox="0 0 48 16"><defs><marker id="ah5" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#ec4899"/></marker></defs><line x1="2" y1="8" x2="42" y2="8" stroke="#ec4899" stroke-width="1.5" marker-end="url(#ah5)"/></svg>
    <div class="op">−log(ŷ[label])<br>cross-entropy</div>
  </div>

  <!-- Loss -->
  <div class="block">
    <div class="node loss-node">
      <div class="label">Scalar</div>
      <div class="name">Loss</div>
      <div class="dim">( )</div>
      <div class="desc">Mean CE loss<br>over batch</div>
    </div>
  </div>

</div><!-- end .flow -->


<!-- ══════════════════ BACKWARD PASS ══════════════════ -->
<p class="section-label bwd-label">◀ Backward Pass (Gradient Flow)</p>

<div class="backward-row">

  <!-- dW1 dB1 update -->
  <div class="block">
    <div class="node update-node">
      <div class="label">Update</div>
      <div class="name">W1, b1</div>
      <div class="dim">(784×10), (1×10)</div>
      <div class="desc">W1 -= lr·dW1<br>b1 -= lr·db1</div>
    </div>
  </div>

  <div class="arrow">
    <svg width="48" height="16" viewBox="0 0 48 16"><defs><marker id="bh1" markerWidth="6" markerHeight="6" refX="1" refY="3" orient="auto"><path d="M6,0 L0,3 L6,6 Z" fill="#ec4899"/></marker></defs><line x1="46" y1="8" x2="6" y2="8" stroke="#ec4899" stroke-width="1.5" marker-end="url(#bh1)"/></svg>
    <div class="op">dW1 = X.T @ dZ1<br>db1 = Σ dZ1</div>
  </div>

  <!-- dZ1 -->
  <div class="block">
    <div class="node hidden-node">
      <div class="label">Gradient</div>
      <div class="name">dZ1</div>
      <div class="dim">(500 × 10)</div>
      <div class="desc">dReLU masks<br>negative Z1</div>
    </div>
  </div>

  <div class="arrow">
    <svg width="48" height="16" viewBox="0 0 48 16"><defs><marker id="bh2" markerWidth="6" markerHeight="6" refX="1" refY="3" orient="auto"><path d="M6,0 L0,3 L6,6 Z" fill="#ec4899"/></marker></defs><line x1="46" y1="8" x2="6" y2="8" stroke="#ec4899" stroke-width="1.5" marker-end="url(#bh2)"/></svg>
    <div class="op">dReLU(dA1, Z1)<br>zero if Z1≤0</div>
  </div>

  <!-- dA1 -->
  <div class="block">
    <div class="node act-node">
      <div class="label">Gradient</div>
      <div class="name">dA1</div>
      <div class="dim">(500 × 10)</div>
      <div class="desc">dZ2 @ W2.T</div>
    </div>
  </div>

  <div class="arrow">
    <svg width="48" height="16" viewBox="0 0 48 16"><defs><marker id="bh3" markerWidth="6" markerHeight="6" refX="1" refY="3" orient="auto"><path d="M6,0 L0,3 L6,6 Z" fill="#ec4899"/></marker></defs><line x1="46" y1="8" x2="6" y2="8" stroke="#ec4899" stroke-width="1.5" marker-end="url(#bh3)"/></svg>
    <div class="op">dW2 = A1.T @ dZ2<br>db2 = Σ dZ2</div>
  </div>

  <!-- dW2 dB2 update -->
  <div class="block">
    <div class="node update-node">
      <div class="label">Update</div>
      <div class="name">W2, b2</div>
      <div class="dim">(10×10), (1×10)</div>
      <div class="desc">W2 -= lr·dW2<br>b2 -= lr·db2</div>
    </div>
  </div>

  <div class="arrow">
    <svg width="48" height="16" viewBox="0 0 48 16"><defs><marker id="bh4" markerWidth="6" markerHeight="6" refX="1" refY="3" orient="auto"><path d="M6,0 L0,3 L6,6 Z" fill="#ec4899"/></marker></defs><line x1="46" y1="8" x2="6" y2="8" stroke="#ec4899" stroke-width="1.5" marker-end="url(#bh4)"/></svg>
    <div class="op">ŷ − one_hot<br>÷ batch_size</div>
  </div>

  <!-- dZ2 -->
  <div class="block">
    <div class="node loss-node">
      <div class="label">Gradient</div>
      <div class="name">dZ2</div>
      <div class="dim">(500 × 10)</div>
      <div class="desc">Combined<br>softmax + CE grad</div>
    </div>
  </div>

</div><!-- end backward -->


<!-- ══════════════════ VARIABLE TABLE ══════════════════ -->
<p class="section-label" style="color:#94a3b8;">📐 Tensor Shapes Reference</p>

<div class="var-grid">
  <div class="var-card">
    <div class="var-name">W1</div>
    <div class="var-shape">(784, 10)</div>
    <div class="var-desc">Input → Hidden weights</div>
  </div>
  <div class="var-card">
    <div class="var-name">b1</div>
    <div class="var-shape">(1, 10)</div>
    <div class="var-desc">Hidden layer biases (broadcast)</div>
  </div>
  <div class="var-card">
    <div class="var-name">W2</div>
    <div class="var-shape">(10, 10)</div>
    <div class="var-desc">Hidden → Output weights</div>
  </div>
  <div class="var-card">
    <div class="var-name">b2</div>
    <div class="var-shape">(1, 10)</div>
    <div class="var-desc">Output layer biases (broadcast)</div>
  </div>
  <div class="var-card">
    <div class="var-name">Z1</div>
    <div class="var-shape">(batch, 10)</div>
    <div class="var-desc">Pre-activation hidden</div>
  </div>
  <div class="var-card">
    <div class="var-name">A1</div>
    <div class="var-shape">(batch, 10)</div>
    <div class="var-desc">Post-ReLU hidden</div>
  </div>
  <div class="var-card">
    <div class="var-name">Z2</div>
    <div class="var-shape">(batch, 10)</div>
    <div class="var-desc">Pre-softmax output logits</div>
  </div>
  <div class="var-card">
    <div class="var-name">ŷ (y)</div>
    <div class="var-shape">(batch, 10)</div>
    <div class="var-desc">Predicted class probabilities</div>
  </div>
</div>

<div class="note">
  <strong>Batch size = 500.</strong> Each row in every tensor is one training example. Matrix multiply <code>X @ W1</code> processes all 500 simultaneously. The softmax + cross-entropy gradients collapse to <code>ŷ − one_hot</code> (a well-known simplification). Gradients flow right-to-left through the same path the data took forward.
</div>

</body>
</html>

---

## Forward Propagation

- **Hidden Layer Pre-activation:**  
  `Z₁ = X · W₁ + b₁`  
- **Hidden Layer Activation (ReLU):**  
  `A₁ = ReLU(Z₁) = max(0, Z₁)`  
- **Output Layer Pre-activation:**  
  `Z₂ = A₁ · W₂ + b₂`  
- **Output Layer Activation (Softmax):**  
  `yᵢ = exp(Z₂ᵢ) / Σⱼ exp(Z₂ⱼ)`

---

## Loss Function

- **Categorical Cross-Entropy Loss:**  
  `L = -(1 / N) Σᵢ log(yᵢ, tᵢ)`  
  - `N` = batch size  
  - `yᵢ, tᵢ` = predicted probability for the true class `tᵢ` of example `i`  

- **Gradient of loss w.r.t. pre-softmax output:**  
  `dZ₂ = (y - one_hot(t)) / N`

---

## Backpropagation

- **Output layer gradients:**  
  `dW₂ = A₁ᵀ · dZ₂`  
  `db₂ = Σ dZ₂`  

- **Hidden layer gradients:**  
  `dA₁ = dZ₂ · W₂ᵀ`  
  `dZ₁ = dA₁ ∘ ReLU'(Z₁)`  
  `dW₁ = Xᵀ · dZ₁`  
  `db₁ = Σ dZ₁`  

> (∘ denotes element-wise multiplication)

---

## Accuracy

`Accuracy = Number of correct predictions / Batch size`

---

## Training

- Mini-batch stochastic gradient descent (random sampling with replacement)  
- **Hyperparameters:**  
  - `BATCHSIZE`: number of examples per batch  
  - `LEARNING_RATE`: step size for gradient updates  
  - `epochs`: number of training iterations   

---

## Kaggle Submission

- Output CSV format:  

| ImageId | Label |
|---------|-------|
| 1       | 7     |
| 2       | 2     |
| ...     | ...   |

- Use the function `kaggleTest(testing)` to generate `submission.csv`.

---

## Dependencies

- Python 3.x  
- NumPy  
- Pandas  

Install dependencies with:  
```bash
pip install numpy pandas
