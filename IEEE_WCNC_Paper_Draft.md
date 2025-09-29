# Intelligent Multi-Agent RIS Optimization Framework for 6G Wireless Networks

## Abstract

This paper presents an intelligent multi-agent framework for Reconfigurable Intelligent Surface (RIS) optimization in multi-user 6G wireless systems. The proposed framework integrates a Coordinator and an Evaluator agent, supported by a lightweight RAG-style memory, to make power- and phase-aware decisions that maximize a rank-weighted satisfaction metric under an energy budget. We formalize two complementary, publication-ready metrics—Rank-Weighted Satisfaction per Watt (RWS/W) and Power–Satisfaction Dominance Part 2 (PSD-2)—and report averaged visuals across diverse scenarios. Results show that the Agent consistently achieves higher satisfaction-per-watt and requires equal or lower power to reach a common satisfaction target than classical baselines (GD, AO, Manifold) under the tested conditions.

---

## II. System Model and Problem Formulation

### A. Network Architecture and Geometry

We consider a RIS-assisted downlink system with one base station (BS), one RIS with M reflective elements, and K single-antenna users:

- BS position: (0, 0, 10) m
- RIS position: (50, 0, 10) m
- RIS elements: M = 128
- Users: K ∈ {3, 4, 5, 6, 7} depending on scenario
- Noise power: σ² = −94 dBm (from config)
- BS power operating range: P ∈ [10, 45] dBm (with a nominal comparison at 35 dBm)
- Default starting power: 20 dBm
- Path loss model parameters: PL₀ = 100 dB, exponent γ = 3.5

Users are placed in a 2D region consistent with the scenarios in `scenario.py` (3U, 4U, 5U_A, 5U_B, 5U_C) with LoS/NLoS and fading specified per-user.

### B. Channel and Signal Models

1) Large- and small-scale effects

- Path-loss (in dB) as in code:
  $$
  \beta^{(\mathrm{dB})}(d) = PL_0 + 10\,\gamma\,\log_{10}(d).
  $$

- Rician/Rayleigh small-scale fading:
  $$
  g_{d,k} =
  \begin{cases}
    \sqrt{\tfrac{K_{d,k}}{K_{d,k}+1}} + \sqrt{\tfrac{1}{K_{d,k}+1}}\,\tilde g_{d,k}, & \text{LoS},\\
    \tilde g_{d,k}, & \text{NLoS},
  \end{cases}
  $$
  with $\tilde g_{d,k} \sim \mathcal{CN}(0,1)$.

2) RIS reflection model

The RIS applies phase shifts $\{\theta_m\}_{m=1}^M$ with unit-amplitude reflection:
$$
\mathbf{\Theta} = \operatorname{diag}\big(e^{j\theta_1},\dots,e^{j\theta_M}\big).
$$

3) Composite channel and received signal

Let $\mathbf{h}_{\mathrm{BR}}$ denote the BS→RIS channel, and $\mathbf{h}_{\mathrm{RU},k}$ the RIS→user-k channel. The effective channel is
$$
 h_{\mathrm{eff},k} = h_{d,k} + \mathbf{h}_{\mathrm{RU},k}^{\mathrm H}\,\mathbf{\Theta}\,\mathbf{h}_{\mathrm{BR}},
$$
with received signal
$$
 y_k = \sqrt{P_k}\, h_{\mathrm{eff},k}\, s + n_k, \quad n_k \sim \mathcal{CN}(0, \sigma_n^2),
$$
and SNR
$$
 \mathrm{SNR}_k = \frac{P_k \lvert h_{\mathrm{eff},k}\rvert^2}{\sigma_n^2}.
$$

4) Interference note

When multiple users share time–frequency resources, the SINR is
$$
 \mathrm{SINR}_k = \frac{P_k |h_{\mathrm{eff},k}|^2}{\sum_{j\neq k} P_j |h_{\mathrm{eff},k,j}|^2 + \sigma_n^2}.
$$
In this work we treat per-iteration algorithm evaluations at a scenario power level and report final per-user SNRs (or margins) as produced by the implemented algorithms for fair comparison.

### C. Optimization Objective and Constraints

Define the per-user satisfaction margin $\Delta_k = \mathrm{SNR}_k^{\text{achieved}} - \mathrm{SNR}_k^{\text{req}}$ with required SNRs fixed per scenario (see Table I). We combine per-user margins via an asymmetric, rank-weighted core $\tilde S$ (Section III) and use it in an energy-aware objective:
$$
 \max_{\mathbf{\Theta},\,\mathbf{P}}\; \mathrm{RSWS}(\mathbf{\Theta}, \mathbf{P}) 
 = \frac{\tilde S(\mathbf{\\Delta})}{P_{\mathrm{tot}}},\quad P_{\mathrm{tot}}\in[10,45]\,\mathrm{dBm}.
$$
Subject to per-user QoS thresholds $\mathrm{SNR}_k\ge \mathrm{SNR}_k^{\text{req}}$ and physically valid RIS phases $\theta_m\in[0,2\pi)$. Fairness is encouraged implicitly by rank weighting (no explicit spread constraint enforced in our runs).

---

## III. Intelligent Multi-Agent RIS Framework

### A. Agents and Roles

- Coordinator Agent: selects users, algorithm (GD/AO/Manifold), and power adjustments. It follows a dual-phase strategy: (i) recover satisfaction for all users, then (ii) minimize power while keeping all users satisfied.
- Evaluator Agent: scores outcomes, labels power usage as wasteful/appropriate/efficient/minimal, and recommends power steps. It enforces strict “no convergence until all users are satisfied and power is near-minimal.”
- RAG-style Memory: stores iteration histories and successful patterns (including metric values) to bias future decisions.

### B. Iteration Flow

1) Scenario analysis → 2) Coordinator proposal → 3) Algorithm execution → 4) Evaluator scoring and guidance → 5) Optional power change → 6) Memory update. Stopping requires: all users satisfied and power minimized within a safety margin.

### C. Metric-Aware Decision Logic

The Evaluator computes RSWS and related indicators from per-user $\Delta_k$ and the current power, then: (i) requests power increase if any $\Delta_k<0$; (ii) gradually reduces power if all users are satisfied with ample margin; (iii) halts reduction when the worst $\Delta_k$ approaches the safety buffer (≈0.5–1 dB).

---

## IV. Performance Evaluation

### A. Simulation Setup and Scenarios

- BS/RIS coordinates: (0,0,10) m and (50,0,10) m
- RIS elements: M = 128
- Noise power: −94 dBm
- BS power: operating range [10,45] dBm; nominal comparison at 35 dBm; default start 20 dBm
- Path loss: PL₀ = 100 dB, exponent γ = 3.5
- Scenarios: 3U, 4U, 5U_A, 5U_B, 5U_C from `scenario.py` with per-user apps, LoS/NLoS, fading and K-factors
- Metrics runs: deterministic per-iteration seeds, scenario list ['5U_A','3U','4U','5U_B','5U_C']

Required SNRs are fixed per scenario (Table I in this draft mirrors `scenario.py`). A sample, mobility-tinged 5U_B layout is used for scenario visuals; final metrics are averaged across the scenario list.

### B. Metrics Used and Final Visuals

We report two visuals in the final metrics set:

1) RWS/W (higher is better). Define weights and transform with parameters from `config.py`:
$$
 w_k = \frac{e^{-\gamma\,\mathrm{rank}_k}}{\sum_i e^{-\gamma\,\mathrm{rank}_i}},\quad
 \phi(\Delta) = \begin{cases}-\alpha\,\log(1+\kappa|\Delta|), & \Delta<0,\\
 \beta\,\log(1+\kappa\Delta), & \Delta\ge 0,\end{cases}
$$
with $(\alpha,\beta,\kappa,\gamma)=(1.0,0.2,1.0,\ln 2)$, and
$$
 \tilde S = \sum_k w_k\,\phi(\Delta_k),\qquad \mathrm{RSWS} = \frac{\tilde S}{\max(P_{\mathrm{tot}},\varepsilon)},\;\;\varepsilon=10^{-3}\,\mathrm{W}.
$$

2) PSD-2: Required power (dBm) to reach a common target $S^*$. We set $S^*$ to the best method’s $\tilde S$ at nominal power $P_0$ and estimate required power over a local grid around $P_0$ with offsets {−6, −3, 0, +3} dB.

Figures: “RWS/W (averaged across scenarios)” and “PSD Part 2: Power to reach $S^*$ (lower is better)”.

### C. Methods Compared and Reporting Policy

- Methods: Agent (ours), Gradient Descent (GD), Alternating Optimization (AO), Manifold Optimization; a Random baseline may be used only for context and is excluded from final averaged bars.
- Reporting: We provide qualitative statements consistent with the averaged figures without over-claiming fixed percentage gaps, as these depend on random seeds and per-iteration scenario realizations.

### D. Results Summary (Qualitative)

- RWS/W: The Agent consistently yields higher satisfaction-per-watt due to rank-aware selection and evaluator-guided power control.
- PSD-2: For the same target satisfaction $S^*$, the Agent requires equal or lower power than GD/AO/Manifold across the tested scenarios, evidencing improved energy efficiency at equivalent QoS.
- Robustness: Gains persist across user counts (3U/4U/5U) and mixed LoS/NLoS fading; improvements are most pronounced in heterogeneous placements with cell-edge users.

---

## V. Conclusion and Future Work

We presented an intelligent multi-agent framework for RIS optimization with explicit energy-satisfaction trade-off modeling via RWS/W and PSD-2. The framework operationalizes a dual-phase policy—first recover satisfaction for all users, then minimize power while preserving satisfaction—monitored by an Evaluator enforcing strict convergence. Averaged final visuals show higher RWS/W and lower required power to hit a common satisfaction target relative to classical methods under the tested settings.

Future work: multi-RIS extensions; reinforcement learning for policy refinement; tighter physical channel modeling; and over-the-air validation with hardware-in-the-loop.

---

## References

[1] M. Giordani et al., “Toward 6G Networks: Use Cases and Technologies,” IEEE Communications Magazine, 2020.

[2] C. Huang et al., “Reconfigurable Intelligent Surfaces for 6G: Nine Fundamental Issues and One Critical Problem,” Tsinghua Science and Technology, 2023.

[3] Q. Wu et al., “Intelligent Reflecting Surface Enhanced Wireless Network via Joint Active and Passive Beamforming,” IEEE TWC, 2019.

[4] E. Basar et al., “Wireless Communications Through Reconfigurable Intelligent Surfaces,” IEEE Access, 2019.

[5] M. Di Renzo et al., “Smart Radio Environments Empowered by Reconfigurable AI Meta-Surfaces,” EURASIP JWCN, 2019.