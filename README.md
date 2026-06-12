# ASTRIA-CAT: Physics-Informed Edge AI

**Physics-Informed Spatiotemporal Architecture for Clear-Air Turbulence (CAT) Prediction.**

![Status: V2 Research Phase](https://img.shields.io/badge/Status-V2_Research_Phase-blue.svg)
![Platform: Edge AI](https://img.shields.io/badge/Platform-ARM_Cortex_A-darkblue.svg)
![Latency: Bounded WCET](https://img.shields.io/badge/Latency-Hard_Real_Time-brightgreen.svg)

---

## 📋 System Overview
**ASTRIA-CAT V2** is an embedded aerospace system that replaces traditional signal classification with a **physics-informed regression pipeline**. By utilizing dimensionless aerodynamic invariants ($C_p$, $R_i$), the system achieves stability-aware, turbulence-intensity prediction (EDR) capable of running on constrained ARM hardware.

## 📐 V2 Architectural Roadmap
- **Input Transformation:** Nondimensionalization of raw pressure into Pressure Coefficient ($C_p$) and Richardson Number ($R_i$) to ensure flight-regime invariance.
- **Inference Core:** **ST-GNN + TCN** (Spatiotemporal Graph Neural Network coupled with Dilated Temporal Convolutional Network).
- **Execution:** Ahead-of-Time (AOT) compiled binaries via **Apache TVM**, ensuring deterministic Worst-Case Execution Time (WCET).
- **Output:** Regression of ICAO-standard **Eddy Dissipation Rate (EDR)**.

## 📄 Documentation
For the complete engineering derivation, fluid dynamics consistency regularization, and the deployment roadmap:
> **[Read the Technical Whitepaper (V2.0)](./papers/ASTRIA-CAT_Technical_Whitepaper.pdf)**

## 📊 Performance Summary
| Metric | Specification |
| :--- | :--- |
| **Inference Model** | Quantized 1D-CNN / ST-GNN |
| **Quantization** | INT8 (Quantization-Aware Training) |
| **Target Hardware** | ARM Cortex-A (Embedded Flight Computer) |
| **Compute Bound** | Deterministic (Static Memory Arena) |

## 🛠️ Repository Structure
```bash
ASTRIA-CAT/
├── core_cpp/          # Native C++ / Rust kernels
├── data/              # Atmospheric physics synthetic datasets
├── papers/            # Technical Whitepaper & Architectural Docs
├── src/               # ST-GNN + TCN Implementation (PyTorch/TVM)
└── requirements.txt   # Embedded inference dependencies
Author: Houssam Rharbi - Independent Systems Researcher
