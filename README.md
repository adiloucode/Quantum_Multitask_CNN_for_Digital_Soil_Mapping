# Quantum Multi-Task CNN for Digital Soil Mapping

This repository presents a **Quantum Multi-Task Convolutional Neural Network (QMT-CNN)** for **Digital Soil Mapping (DSM)**, where **each model is dedicated to a single soil chemical component**, and **multi-task learning is applied across soil depths**.

Instead of predicting multiple soil properties with one model, **each chemical component (e.g. SOC, pH, nutrients) has its own model**, and **each task corresponds to a specific depth interval**.

---

## 🚀 Project Overview

Soil chemical properties vary significantly with depth, and depth-wise dependencies are often ignored in classical Digital Soil Mapping approaches.

This project proposes a **depth-aware multi-task learning framework**, enhanced with **quantum-inspired convolutional layers (Quanvolution)**, to jointly predict a **single soil component at multiple depth intervals**.

Key ideas:
- One model per chemical component  
- Shared representation across depths  
- Task-specific heads for each depth  
- Quantum-enhanced local feature extraction  

---

## 🧠 Core Concepts

- **Quantum Machine Learning**
  - Quanvolutional layers
  - Quantum data encoding
  - Parameterized quantum circuits

- **Deep Learning**
  - Convolutional Neural Networks (CNNs)
  - Multi-task learning across depths
  - Shared backbone with depth-specific outputs

- **Digital Soil Mapping**
  - Depth-dependent soil variation
  - Remote sensing and environmental covariates
  - Multi-depth prediction strategy

---

## 🏗️ Architecture Overview

For **each soil chemical component**, a separate model is trained with the following structure:

1. **Input Features**
   - Multi-channel spatial data
   - Selected spectral bands and environmental covariates

2. **Quantum Feature Extraction**
   - Quanvolutional layer applied to local patches
   - Pixel values encoded into quantum circuits

3. **Shared CNN Backbone**
   - Classical convolutional layers
   - Shared latent representation across depths

4. **Depth-Specific Heads (Multi-Task)**
   - One output head per depth interval
   - Each head predicts the component concentration at its corresponding depth

---

## 📏 Depth as Tasks

Multi-task learning is defined **across depth layers**, for example:

- 0–5 cm  
- 5–15 cm  
- 15–30 cm  
- 30–60 cm  
- 60–100 cm  

Each depth interval is treated as an **independent but related task**, allowing the model to:
- Learn vertical soil correlations  
- Share spatial features across depths  
- Improve generalization for deeper layers with sparse data  

---

## 📁 Repository Structure

Data_CNN_Basic/
│
├── data/ # Depth-wise datasets per component
├── models/ # CNN & Quantum CNN architectures
├── quantum/ # Quantum circuits and encoders
├── training/ # Training and evaluation loops
├── utils/ # Helper functions
└── main.ipynb / main.py # Experiments for one soil component


Each execution typically targets **one chemical component**, with **multiple depth outputs**.

---

## ⚙️ Technologies & Tools

- **Python**
- **PyTorch**
- **PennyLane** (quantum circuits)
- **NumPy / Pandas**
- **Optuna** (architecture & hyperparameter optimization)
- **JAX / Flax** (experimental implementations)

---

## 🧪 Experiments

- Comparison between:
  - Classical CNN (depth-wise multi-task)
  - Quantum-enhanced CNN
  - Quantum Multi-Task CNN

- Evaluation across multiple depths for the same component
- Study of quantum feature extraction impact per depth
- Analysis of shallow vs deep soil prediction performance

---

## 📊 Key Observations

- Shared learning across depths improves stability
- Deeper layers benefit from information learned at shallow depths
- Quantum-enhanced layers introduce richer spatial representations
- Performance gains vary depending on depth and component

(See notebooks for detailed metrics and visualizations.)

---

## 🎓 Academic Context

This work was developed as a **Master’s graduation project in Artificial Intelligence**, exploring the intersection of:

- Quantum Computing  
- Multi-Task Deep Learning  
- Digital Soil Mapping  

The project contributes a **depth-aware quantum-enhanced modeling strategy** for soil chemical component prediction.

---

## 🔮 Future Directions

- Cross-depth attention mechanisms
- Larger and adaptive quantum circuits
- Transfer learning between chemical components
- Integration with transformer-based spatial encoders

---

## 👤 Author

**Abdelkader Saidi**  
Master’s Student in Artificial Intelligence  
USTHB – University of Bab Ezzouar  
Interests: Quantum ML, LLMs, Computer Vision, Data Science

---

## 📜 License
Research paper will be shared 
