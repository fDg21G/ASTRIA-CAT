# ASTRIA-CAT
Predictive detection of clear-air turbulence using MEMS sensors and edge AI
# ASTRIA-CAT

## Predictive Detection of Clear-Air Turbulence (CAT)

**Author:** Houssam Rharbi  
**Field:** Civil Aviation – Aeronautical Safety  

### Overview
Clear-Air Turbulence (CAT) represents one of the most dangerous and unpredictable hazards in civil aviation, as it cannot be detected by conventional onboard radar systems.  
This project proposes a predictive approach based on distributed pressure sensing and edge artificial intelligence to identify aerodynamic precursors of CAT in real time.

### Project Objective
The objective of this work is to demonstrate, through experimental and synthetic data, that high-frequency pressure fluctuations measured on the aircraft surface can be used to anticipate clear-air turbulence before it is encountered.

### Methodology
- Distributed MEMS pressure sensors simulating a "smart skin"
- Physics-informed synthetic turbulence generation
- Wind-tunnel experimental setup
- Signal processing and feature extraction
- Lightweight 1D Convolutional Neural Network (1D-CNN)
- Edge deployment on Arduino-class hardware

### Results
- Prediction accuracy: **88.5%**
- Recall (turbulence detection): **93%**
- Inference latency: **< 60 ms**
- Demonstrated early-warning capability under experimental conditions

### Repository Content
- `Dissertation.pdf` – Full research dissertation
- `src/` – Signal processing and machine learning code
- `data/` – Sample datasets (synthetic and experimental)
- `results/` – Figures and performance metrics

### Research Context
This project represents an applied experimental study in aeronautical safety and predictive sensing.  
It is developed
