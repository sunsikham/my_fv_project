---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: "Aptos", "Segoe UI", sans-serif;
    padding: 34px 48px 26px 48px;
    color: #111;
    background: #fff;
  }
  h1 {
    font-size: 32px;
    margin: 0 0 10px 0;
    color: #0f172a;
  }
  p {
    font-size: 17px;
    color: #334155;
    margin: 0 0 12px 0;
  }
  .takeaway {
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1.5px solid #cbd5e1;
    font-size: 18px;
    color: #0f172a;
  }
  img.main {
    width: 100%;
    max-height: 68vh;
    object-fit: contain;
    display: block;
    margin: 4px auto 0 auto;
  }
  .cols {
    display: grid;
    grid-template-columns: 20% 80%;
    gap: 14px;
    align-items: start;
    margin-top: 6px;
  }
  .legend-col {
    padding-top: 28px;
  }
  .legend-row {
    display: grid;
    grid-template-columns: 22px auto;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
    font-size: 15px;
    color: #334155;
  }
  .swatch {
    width: 16px;
    height: 12px;
    border-radius: 2px;
    display: inline-block;
  }
  .plotbox img {
    width: 100%;
    max-height: 60vh;
    object-fit: contain;
    display: block;
    margin: 0 auto;
  }
  .legend-inline {
    margin: 8px 0 10px 0;
    font-size: 14px;
    color: #334155;
  }
  .legend-inline .item {
    display: inline-flex;
    align-items: center;
    margin-right: 16px;
    white-space: nowrap;
  }
  .legend-inline .label {
    margin-left: 6px;
    font-weight: 700;
  }
  .shep-grid {
    display: grid;
    grid-template-columns: 46% 54%;
    gap: 26px;
    align-items: center;
    margin-top: 12px;
  }
  .shep-copy {
    font-size: 19px;
    line-height: 1.45;
    color: #0f172a;
  }
  .shep-copy p {
    font-size: 18px;
    line-height: 1.5;
    color: #334155;
    margin: 0 0 14px 0;
  }
  .shep-card {
    background: #f8fafc;
    border: 1.5px solid #dbe4ee;
    border-radius: 14px;
    padding: 16px 18px 12px 18px;
  }
  .shep-caption {
    margin-top: 8px;
    font-size: 15px;
    color: #475569;
    text-align: center;
  }
---

# Which Inside-A Features Change Over Steps?

Inside-A carries most of the stepwise change. The next question is which A-features are being reweighted across matched A slots.

![w:1500 class:main](./q1_inside_a_feature_changes.png)

<div class="takeaway"><strong>Takeaway.</strong> Inside A, the same role-defined features do not stay fixed: <code>g2/g3/g4</code> are amplified along the B branch, while the D branch strengthens <code>g0</code> and suppresses anti-D axes such as <code>g1</code> and <code>g3</code>.</div>

---

# BABABA: Which Features Drive the B-Directed Drift?

Contribution asks not just which features changed, but which of those changes actually pushed the state toward B. Each colored segment shows how much one inside-A feature contributes to that B-directed drift, relative to the AAA baseline. Segments above zero support B-directed drift; segments below zero oppose it.

<div class="legend-inline">
<span class="item"><span class="swatch" style="background:#D9893D"></span><span class="label" style="color:#D9893D">g0</span></span>
<span class="item"><span class="swatch" style="background:#7A6E9C"></span><span class="label" style="color:#7A6E9C">g1</span></span>
<span class="item"><span class="swatch" style="background:#2F6BDA"></span><span class="label" style="color:#2F6BDA">g2</span></span>
<span class="item"><span class="swatch" style="background:#0F9D8B"></span><span class="label" style="color:#0F9D8B">g3</span></span>
<span class="item"><span class="swatch" style="background:#7EB6FF"></span><span class="label" style="color:#7EB6FF">g4</span></span>
<span class="item"><span class="swatch" style="background:#000000"></span><span class="label" style="color:#000000">total</span></span>
</div>

![w:1450 class:main](./q1_inside_a_feature_contributions_bab.png)

---

# DADADA: Which Features Drive the D-Directed Drift?

Contribution asks not just which features changed, but which of those changes actually pushed the state toward D. Each colored segment shows how much one inside-A feature contributes to that D-directed drift, relative to the AAA baseline. Segments above zero support D-directed drift; segments below zero oppose it.

<div class="legend-inline">
<span class="item"><span class="swatch" style="background:#D9893D"></span><span class="label" style="color:#D9893D">g0</span></span>
<span class="item"><span class="swatch" style="background:#7A6E9C"></span><span class="label" style="color:#7A6E9C">g1</span></span>
<span class="item"><span class="swatch" style="background:#2F6BDA"></span><span class="label" style="color:#2F6BDA">g2</span></span>
<span class="item"><span class="swatch" style="background:#0F9D8B"></span><span class="label" style="color:#0F9D8B">g3</span></span>
<span class="item"><span class="swatch" style="background:#7EB6FF"></span><span class="label" style="color:#7EB6FF">g4</span></span>
<span class="item"><span class="swatch" style="background:#000000"></span><span class="label" style="color:#000000">total</span></span>
</div>

![w:1450 class:main](./q1_inside_a_feature_contributions_dad.png)

---

# A Shepard-Like Geometric Substrate

<div class="shep-grid">

<div class="shep-copy">
<p><strong>Relational states are organized in a geometric state space.</strong></p>

<p>Their differences define meaningful directions, allowing us to measure movement toward endpoints such as <code>A→B</code> and <code>A→D</code>.</p>

<p>This is why progress, alignment, and endpoint-directed movement are meaningful quantities in our analysis.</p>
</div>

<div class="shep-card">
<img src="./shepard_geometry_cartoon.png" style="width:100%; display:block;" />

<div class="shep-caption">States occupy positions; differences define directions.</div>
</div>

</div>
