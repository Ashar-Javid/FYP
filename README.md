# RIS Multi-User Optimization Framework

A sophisticated framework for optimizing Quality of Service (QoS) in multi-user RIS-assisted 6G systems using intelligent coordination and pattern learning.

## 🎯 Overview

This framework implements an intelligent optimization system with:

- **Coordinator Agent**: Makes decisions about user selection, algorithm choice, and power adjustments using RAG-based memory
- **Evaluator Agent**: Assesses optimization performance, power efficiency, and fairness
- **Scenario Tool**: Manages test scenarios and visualizations
- **RAG Memory System**: Learns patterns from historical data to improve future decisions
- **Modular Tools**: Algorithm execution, power control, memory management, and visualization

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  RIS Framework Main Loop                   │
├─────────────────────────────────────────────────────────────┤
│  1. Scenario Selection (5U_B)                             │
│  2. Coordinator Decision (RAG-based)                      │
│  3. Algorithm Execution (GD/Manifold/AO)                  │
│  4. Evaluator Assessment                                   │
│  5. Power Control & Memory Update                         │
└─────────────────────────────────────────────────────────────┘

┌───────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Coordinator   │  │ RAG Memory      │  │ Evaluator       │
│ Agent         │◄─┤ System          │─►│ Agent           │
│               │  │                 │  │                 │
│ • User        │  │ • Pattern       │  │ • Performance   │
│   Selection   │  │   Learning      │  │   Assessment    │
│ • Algorithm   │  │ • Similarity    │  │ • Power         │
│   Choice      │  │   Search        │  │   Efficiency    │
│ • Power       │  │ • History       │  │ • Fairness      │
│   Control     │  │   Storage       │  │   Analysis      │
└───────────────┘  └─────────────────┘  └─────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install required packages
pip install -r requirements.txt

# Or install basic dependencies manually
pip install numpy matplotlib
```

### 2. Test Framework

```bash
# Run component tests
python test_framework.py
```

### 3. Run Optimization

```bash
# Run full optimization session
python ris_framework.py
```

## 📁 Project Structure

```
FYP/
├── config.py                  # Framework configuration
├── tools.py                   # Modular tool implementations
├── agents.py                  # Coordinator and Evaluator agents
├── rag_memory.py              # RAG-based memory system
├── ris_framework.py           # Main framework loop
├── test_framework.py          # Test suite
├── scenario.py                # Scenario definitions
├── multiuserdatasetgenerator.py # Algorithm implementations
├── ris_user_algorithm_dataset.json # Training data
├── requirements.txt           # Dependencies
├── plots/                     # Generated visualizations
├── results/                   # Session results
└── framework_memory.json      # Persistent memory
```

## 🔧 Configuration

The framework uses several configuration files:

### API Configuration (`config.py`)
- Cerebras API key and model settings
- Framework parameters (max iterations, power range)
- Agent system prompts

### RAG Override Control
The framework supports complete RAG bypass for pure LLM-based decisions:

```python
# In config.py - disable RAG completely
"rag_override_disable": True   # Forces LLM-only decisions

# When enabled:
# - RAG memory system is bypassed
# - No similar scenario lookup
# - Direct LLM decision-making
# - Reasoning includes "LLM-only mode" indicator
```

### Scenario Settings (`scenario.py`)
- Predefined scenarios (3U, 4U, 5U_A, 5U_B, 5U_C)
- System parameters (BS/RIS coordinates, power settings)
- Channel models and user configurations

## 🎮 Usage Examples

### Basic Usage

```python
from ris_framework import RISOptimizationFramework

# Initialize framework
framework = RISOptimizationFramework()

# Run optimization session
results = framework.run_optimization_session()

# Display results
framework.print_session_summary(results)
```

### Component Usage

```python
# Use individual components
from tools import ScenarioTool, AlgorithmTool
from agents import CoordinatorAgent

# Load scenario
scenario_tool = ScenarioTool()
scenario = scenario_tool.get_scenario_5ub()

# Make coordinator decision
coordinator = CoordinatorAgent()
decision = coordinator.analyze_scenario_and_decide(scenario)

# Execute algorithm
algorithm_tool = AlgorithmTool()
results = algorithm_tool.execute_algorithm(
    decision["selected_algorithm"],
    scenario,
    decision["selected_users"],
    25.0  # power in dBm
)
```

## 📊 Features

### Intelligent Decision Making
- **Pattern Learning**: RAG system learns from historical scenarios
- **Adaptive User Selection**: Learns optimal user selection strategies
- **Algorithm Choice**: Recommends best algorithm based on scenario characteristics
- **Power Control**: Adaptive power adjustment within 20-45 dB range

### Performance Assessment
- **SNR Analysis**: Before/after comparison of user SNR values
- **Power Efficiency**: Detects wasteful power usage
- **Fairness**: Evaluates distribution of improvements across users
- **Convergence**: Tracks algorithm performance and convergence

### Visualization and Reporting
- **Scenario Plots**: System layout with user positions and channel conditions
- **Progress Tracking**: Iteration-by-iteration performance visualization
- **Comprehensive Reports**: Detailed session summaries with metrics

## 🔬 Algorithm Support

The framework supports multiple RIS optimization algorithms:

1. **Gradient Descent (GD)**: Adam-based gradient descent optimization
2. **Manifold Optimization**: Riemannian manifold-based approach
3. **Alternating Optimization (AO)**: Element-wise optimization
4. **Analytical**: Closed-form solutions (when available)

### (Experimental) Multi-Run Algorithm Comparison (3 Runs)

An upcoming enhancement allows running each optimization algorithm multiple times (default: 3 runs) to compute mean and standard deviation of per-user ΔSNR (final SNR − required SNR). This improves statistical reliability over a single run.

Status: Disabled by default (no behavior change yet). You can prepare for it now by configuring the flag below; when the feature is activated in code, it will automatically use these settings.

Configuration (in `config.py`):

```python
FRAMEWORK_CONFIG["algorithm_comparison_multi_run"] = {
        "enabled": False,  # set to True once feature is active
        "runs": 3          # number of repetitions per algorithm
}
```

Planned Behavior When Enabled:
- Each algorithm (GD, Manifold, AO, Random baseline) will execute 3 independent runs
- Per-user ΔSNR values aggregated: mean ± std
- Bar chart will display mean ΔSNR with error bars (std)
- A CSV `algorithm_comparison_<session>.csv` will be generated with columns:
    `session_id, algorithm, user_id, mean_delta_snr, std_delta_snr, runs, type`

Why 3 Runs? Balances statistical signal with runtime overhead; adjust `runs` to 5+ for research-grade reproducibility if needed.

Until the feature is switched on (set `enabled` to True and supporting code merged), the framework continues to run a single-pass comparison as currently implemented.

## 📈 Stopping Criteria

The framework stops optimization when:
- All users meet SNR requirements AND power usage is efficient
- Maximum iterations (10) reached
- Overall success score > 0.8

## 🎯 Customization

### Adding New Scenarios
```python
# In scenario.py
NEW_SCENARIO = {
    "num_users": 3,
    "users": [
        {
            "id": 1,
            "coord": (x, y, z),
            "req_snr_dB": target_snr,
            "app": "application_type",
            "csi": {"los": True/False, "fading": "Rician/Rayleigh", ...}
        },
        # ... more users
    ]
}
```

### Modifying Agent Behavior
Edit the system prompts in `config.py`:
- `COORDINATOR_CONFIG["system_prompt"]`
- `EVALUATOR_CONFIG["system_prompt"]`

### Adding New Algorithms
Implement in `tools.py` -> `AlgorithmTool.execute_algorithm()`

## 🐛 Troubleshooting

### Common Issues

1. **API Errors**: Check Cerebras API key in `config.py`
2. **Import Errors**: Ensure all dependencies are installed
3. **Memory Issues**: Large datasets may require more RAM
4. **Plot Errors**: Ensure matplotlib backend is properly configured

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Output Files

### Results Directory
- `session_results_YYYYMMDD_HHMMSS.json`: Complete session data
- `framework_memory.json`: Persistent learning memory

### Plots Directory
- `scenario_*.png`: Initial scenario visualization
- `session_progress_*.png`: Performance progression plots
- Algorithm-specific convergence plots

## 🔮 Future Enhancements

- [ ] Advanced embedding models for better similarity search
- [ ] Real-time scenario adaptation
- [ ] Multi-objective optimization
- [ ] Distributed processing support
- [ ] Web-based dashboard

## 📚 References

- Base algorithms from `multiuserdatasetgenerator.py`
- Scenario configurations from `scenario.py`
- Training data from `ris_user_algorithm_dataset.json`

---

🎉 **Happy Optimizing!** 

For questions or issues, check the test suite first: `python test_framework.py`