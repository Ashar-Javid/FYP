# Detailed Documentation: PSD and RWS/W Metrics

## Table of Contents
- [Part 1: Metric Definitions and Explanations](#part-1-metric-definitions-and-explanations)
  - [1.1 Power-Satisfaction Dominance (PSD)](#11-power-satisfaction-dominance-psd)
  - [1.2 Rank-Weighted Satisfaction per Watt (RWS/W)](#12-rank-weighted-satisfaction-per-watt-rwsw)
- [Part 2: Mathematical Formulations and Significance](#part-2-mathematical-formulations-and-significance)
  - [2.1 PSD Mathematical Framework](#21-psd-mathematical-framework)
  - [2.2 RWS/W Mathematical Framework](#22-rwsw-mathematical-framework)
  - [2.3 Significance and Applications](#23-significance-and-applications)

---

## Part 1: Metric Definitions and Explanations

### 1.1 Power-Satisfaction Dominance (PSD)

**Power-Satisfaction Dominance (PSD)** is a two-part composite metric designed to evaluate the trade-off between power efficiency and user quality of service (QoS) satisfaction in multi-user RIS-assisted 6G communication systems.

#### Definition Overview
PSD provides a comprehensive assessment framework that addresses two fundamental questions in wireless system optimization:
1. **Part 1 (Fixed-Power Satisfaction)**: Given the same power budget, which algorithm achieves higher user satisfaction?
2. **Part 2 (Fixed-Satisfaction Power Saving)**: To achieve the same satisfaction level, which algorithm requires less power?

#### Key Characteristics
- **Dual Perspective**: Evaluates both satisfaction-centric and power-centric performance
- **Comparative Nature**: Designed for algorithm comparison rather than absolute assessment
- **QoS-Aware**: Incorporates user satisfaction as the primary performance indicator
- **Power-Conscious**: Recognizes power efficiency as a critical constraint in modern wireless systems

#### Practical Applications
- **Algorithm Selection**: Helps identify the most suitable optimization algorithm for different scenarios
- **System Design**: Guides power allocation strategies in multi-user environments
- **Performance Benchmarking**: Provides standardized comparison framework across different RIS optimization approaches
- **Energy-Efficient Design**: Supports sustainable wireless communication system development

### 1.2 Rank-Weighted Satisfaction per Watt (RWS/W)

**Rank-Weighted Satisfaction per Watt (RWS/W)** is a unified energy efficiency metric that combines user satisfaction assessment with power consumption considerations through a rank-weighted scoring mechanism.

#### Definition Overview
RWS/W quantifies the energy efficiency of wireless systems by measuring how much user satisfaction is achieved per unit of consumed power, with special emphasis on prioritizing underperforming users through rank-based weighting.

#### Key Characteristics
- **Energy Efficiency Focus**: Direct measure of satisfaction per watt of consumed power
- **Fairness-Aware**: Rank-weighted mechanism prioritizes improvement of worst-performing users
- **Asymmetric Scoring**: Different penalties for unsatisfied users versus rewards for over-satisfied users
- **Scalable**: Applicable across different system sizes and user counts

#### Conceptual Foundation
The metric is built on the principle that:
- **Worst users matter most**: Users with the poorest QoS receive highest weighting
- **Diminishing returns**: Excessive over-satisfaction provides minimal benefit
- **Power awareness**: Every improvement must be justified by its power cost
- **System-wide perspective**: Individual user performance contributes to overall system efficiency

#### Practical Applications
- **Resource Allocation**: Optimizes power distribution across users for maximum efficiency
- **Algorithm Tuning**: Provides objective function for power-aware optimization algorithms
- **System Monitoring**: Real-time assessment of energy efficiency in operational networks
- **Green Communications**: Supports environmentally conscious network design and operation

---

## Part 2: Mathematical Formulations and Significance

### 2.1 PSD Mathematical Framework

#### 2.1.1 Satisfaction Core Function

The foundation of PSD is the **Satisfaction Core** function S̃, which aggregates individual user delta SNR values (Δ_i) into a system-wide satisfaction score:

```
S̃ = Σ(i=1 to N) w_i · φ(Δ_i)
```

Where:
- **N**: Total number of users in the system
- **Δ_i**: Delta SNR for user i (achieved SNR - required SNR) in dB
- **w_i**: Rank-based weight for user i
- **φ(·)**: Asymmetric transformation function

#### 2.1.2 Rank-Based Weighting

Users are ranked by their delta SNR values (worst first), and weights are assigned exponentially:

```
w_i = exp(-γ · rank_i) / Σ(j=1 to N) exp(-γ · rank_j)
```

Where:
- **rank_i**: Rank of user i (1 = worst performance, N = best performance)
- **γ**: Weight decay parameter (default: ln(2) ≈ 0.693, halving weight per rank)

#### 2.1.3 Asymmetric Transformation Function

**Linear Form:**
```
φ_linear(Δ) = -α · max(0, -Δ) + β · max(0, Δ)
```

**Logarithmic Form (Default):**
```
φ_log(Δ) = -α · log₁(1 + κ · max(0, -Δ)) + β · log₁(1 + κ · max(0, Δ))
```

Where:
- **α**: Penalty weight for negative delta SNR (default: 1.0)
- **β**: Reward weight for positive delta SNR (default: 0.01, much smaller than α)
- **κ**: Logarithmic scaling factor (default: 1.0)

#### 2.1.4 PSD Part 1: Fixed-Power Satisfaction Dominance

For algorithms A and B operating at the same power level P:

```
PSD₁(A,B) = S̃_A(P) - S̃_B(P)
```

**Interpretation:**
- **PSD₁ > 0**: Algorithm A achieves higher satisfaction than B at same power
- **PSD₁ < 0**: Algorithm B achieves higher satisfaction than A at same power
- **PSD₁ = 0**: Both algorithms achieve equivalent satisfaction at same power

#### 2.1.5 PSD Part 2: Fixed-Satisfaction Power Dominance

For a target satisfaction level S*:

```
PSD₂(A,B) = P_B(S*) - P_A(S*)
```

Where P_X(S*) is the minimum power required by algorithm X to achieve satisfaction S*.

**Interpretation:**
- **PSD₂ > 0**: Algorithm A requires less power than B to achieve target satisfaction
- **PSD₂ < 0**: Algorithm B requires less power than A to achieve target satisfaction
- **PSD₂ = 0**: Both algorithms require equal power for target satisfaction

### 2.2 RWS/W Mathematical Framework

#### 2.2.1 Core RWS/W Formula

```
RWS/W = S̃ / max(P_watts, ε_watts)
```

Where:
- **S̃**: Satisfaction core (as defined in PSD framework)
- **P_watts**: Power consumption in watts
- **ε_watts**: Small epsilon to prevent division by zero (default: 1e-3 W)

#### 2.2.2 Power Conversion

Power is converted from dBm to watts using:

```
P_watts = 10^((P_dBm - 30) / 10)
```

#### 2.2.3 Component Analysis

**Satisfaction Component (S̃):**
- Captures user QoS through rank-weighted delta SNR aggregation
- Emphasizes improvement of worst-performing users
- Provides asymmetric penalties for unsatisfied vs. over-satisfied users

**Power Component (P_watts):**
- Represents actual energy consumption
- Denominator position incentivizes power reduction
- Epsilon floor prevents mathematical instabilities

### 2.3 Significance and Applications

#### 2.3.1 Theoretical Significance

**Multi-Objective Optimization:**
Both metrics address the fundamental trade-off in wireless systems between QoS and energy efficiency, providing quantitative frameworks for:
- **Pareto Frontier Analysis**: Identifying optimal operating points in satisfaction-power space
- **Algorithm Comparison**: Objective evaluation of different optimization approaches
- **System Design**: Guiding architectural decisions in RIS-assisted networks

**Fairness Integration:**
The rank-weighting mechanism ensures that:
- **Worst users receive priority**: Addressing the weakest link improves overall system robustness
- **Diminishing returns**: Prevents over-optimization of already satisfied users
- **Balanced improvement**: Encourages system-wide performance enhancement

#### 2.3.2 Practical Applications

**Network Planning:**
- **Base Station Deployment**: Optimize BS placement considering both coverage and power efficiency
- **RIS Positioning**: Determine optimal RIS locations for maximum RWS/W
- **Power Budget Allocation**: Distribute available power resources for optimal PSD performance

**Real-Time Operation:**
- **Dynamic Resource Allocation**: Adjust power and RIS configurations based on changing user requirements
- **Load Balancing**: Redistribute users across serving stations to maximize RWS/W
- **Admission Control**: Make informed decisions about new user acceptance based on PSD impact

**Algorithm Development:**
- **Objective Function Design**: Use RWS/W as optimization target in algorithm development
- **Performance Benchmarking**: Standardized comparison framework for new algorithms
- **Hyperparameter Tuning**: Optimize algorithm parameters for maximum PSD performance

#### 2.3.3 Implementation Considerations

**Computational Complexity:**
- **O(N log N)** for sorting users by delta SNR
- **O(N)** for weight calculation and satisfaction core computation
- **Suitable for real-time implementation** in moderate-sized networks

**Parameter Sensitivity:**
- **α, β**: Control asymmetry between penalties and rewards
- **γ**: Determines rank-weight decay rate
- **κ**: Affects logarithmic transformation sensitivity
- **Robust to parameter variations** within reasonable ranges

**Scalability:**
- **User Count**: Metrics scale well with increasing number of users
- **Scenario Diversity**: Applicable across different propagation environments
- **Algorithm Agnostic**: Compatible with various RIS optimization approaches

#### 2.3.4 Visualization and Interpretation Guidelines

**Advanced Visualization Features:**
- **Best Performer Highlighting**: Only the best-performing method in each metric receives numeric value annotations
- **Smart Color Coding**: Agent-based methods highlighted in orange, with traditional algorithms in blue/green/gray
- **Performance Emphasis**: Bar thickness and border colors indicate relative performance levels
- **Combined Views**: Integrated plots showing all metrics simultaneously for comprehensive comparison

**RWS/W Values:**
- **Higher values indicate better energy efficiency**
- **Best performer**: Highlighted with gold annotation box
- **Typical range**: Varies significantly based on scenario complexity and power levels
- **Relative comparison more meaningful** than absolute values

**PSD Values:**
- **Part 1**: Satisfaction difference at fixed power (arbitrary units) - higher is better
- **Part 2**: Power difference at fixed satisfaction (dBm or watts) - lower is better
- **Best performers**: Part 1 uses gold highlighting, Part 2 uses green highlighting
- **Combined analysis** provides comprehensive algorithm assessment

**Visual Interpretation Framework:**
- **Gold annotations**: Identify the single best performer in each metric
- **Thick red borders**: Emphasize superior performance methods
- **Color consistency**: Agent/Our methods always in orange across all plots
- **Satisfaction threshold**: Red dashed line at 0 dB in algorithm comparisons

**Decision Framework:**
- **Multiple gold annotations**: Algorithm consistently excels across metrics
- **High RWS/W + High PSD1 + Low PSD2**: Optimal algorithm for deployment
- **Agent prominence**: Orange highlighting indicates intelligent optimization capability
- **User satisfaction focus**: Above-threshold performance prioritized in visualizations

---

## Advanced Visualization System

### Intelligent Performance Highlighting

The framework now includes an **Advanced Visualization Engine** that automatically identifies and highlights the best-performing methods across all metrics:

**Key Features:**
- **Best-Only Annotations**: Only the top performer in each metric receives value annotations
- **Smart Emphasis**: Bar appearance (thickness, borders, transparency) reflects performance ranking
- **Consistent Color Scheme**: Agent-based methods always highlighted in distinctive orange
- **Multi-Metric Integration**: Combined plots show all three metrics with unified highlighting

**Technical Implementation:**
- **RISVisualizationEngine Class**: Centralized plotting system with advanced highlighting logic
- **Performance Detection**: Automatic identification of best performers (highest for RWS/W & PSD1, lowest for PSD2)
- **Visual Hierarchy**: Progressive emphasis levels based on algorithm type and performance
- **Gold Standard Marking**: Best performers receive distinctive gold annotation boxes

**Benefits:**
- **Immediate Clarity**: Instantly identify which method performs best in each metric
- **Reduced Visual Clutter**: Only essential performance values are annotated
- **Enhanced Comparison**: Emphasis levels make performance differences immediately apparent
- **Professional Presentation**: Publication-ready visualizations with consistent styling

---

## Conclusion

The PSD and RWS/W metrics, combined with the advanced visualization system, provide a comprehensive framework for RIS-assisted communication system evaluation:

- **PSD** offers detailed analysis of satisfaction-power trade-offs through its two-part structure
- **RWS/W** provides a unified energy efficiency measure with built-in fairness considerations
- **Advanced Visualization** ensures optimal methods are immediately identifiable through intelligent highlighting

Together, these components enable comprehensive evaluation and optimization of multi-user RIS systems, supporting the development of next-generation wireless networks that balance user satisfaction with energy sustainability while providing clear visual guidance for algorithm selection and system design decisions.