# Stellar Luminosity Regression (Linear & Polynomial Models)

Predicting stellar luminosity from physical features using regression models implemented from first principles without machine learning libraries.

---

## Table of Contents

- [About the Project](#about-the-project)     
- [Data](#data)  
- [Methods](#methods)  
- [Usage](#usage)  
- [Results](#results)    
- [AWS SageMaker Execution](#aws-sagemaker-execution)  
- [Conclusions](#conclusions)  
- [Authors](#authors)  
- [License](#license) 

---

## About the Project

This repository explores the relationship between **stellar mass**, **temperature**, and **luminosity** through manually coding regression models using only NumPy and Matplotlib.

The goal is not only to fit a model, but also to grasp how a machine learning model works from the inside out by manually coding:

- The hypothesis function  
- The loss function  
- The gradients  
- The optimization algorithm (gradient descent)

---

## Data

All datasets are defined directly inside the notebooks using hard-coded NumPy arrays.

| Variable | Meaning |
|---------|---------|
| M | Stellar Mass (in solar masses) |
| T | Effective Temperature (Kelvin) |
| L | Stellar Luminosity (in solar luminosities) |

---

## Methods

### Notebook 1 — Linear Regression

Model:

`L_hat = w * M + b`

Steps performed:

- Dataset visualization (Mass vs Luminosity)
- Implementation of MSE loss
- Cost surface visualization J(w, b)
- Manual gradient computation
- Gradient descent:
  - Non-vectorized
  - Vectorized
- Convergence plots
- Learning rate experiments
- Final regression line and discussion

---

### Notebook 2 — Polynomial Regression

Model:
```python
L_hat = X @ w + b
```

Feature Matrix:
```python
X = [M, T, M^2, M*T]
```

Steps Performed:
- Visualization of luminosity vs mass with temperature encoded by color
- Feature Engineering using NumPy
- Feature Normalization for Stable Training
- Vectorized Loss and Gradients
- Gradient Descent Training and Convergence Plots
- Feature Selection Experiment:

| Model | Features |
|------|----------|
| M1 | [M, T] |
| M2 | [M, T, M^2] |
| M3 | [M, T, M^2, M*T] |
- Predicted vs Actual Plots for each Model
- Cost vs Interaction Coefficient Analysis
- Inference Demo for a New Star (M = 1.3, T = 6600)
---

## Usage

1. Open both notebooks:
   - `01_part1_linreg_1feature.ipynb`
   - `02_part2_polyreg.ipynb`
2. Run all cells from top to bottom.
3. Observe plots, losses, and model comparisons.

Required libraries:

```
pip install numpy matplotlib
```

---

## Results
First of all we have to make sure that our SageMaker machine is on so we open

### Linear Regression

The linear model captures the overall increasing trend between mass and luminosity but cannot model the strong nonlinear growth. This produces systematic errors at higher masses.

### Polynomial Regression

Adding the nonlinear (`M^2`) and interaction (`M*T`) features improves the performance significantly. The full model (M3) has the lowest loss, showing the importance of the nonlinear and interaction effects in stellar physics.

---

## AWS SageMaker Execution

Both notebooks were uploaded and executed in **AWS SageMaker Studio**.

Steps followed:

1. Opened SageMaker Studio  
2. Uploaded the `.ipynb` files  
3. Opened each notebook  
4. Ran all cells successfully  
If you want to see the results they are on the section results

### Local vs SageMaker

The numerical results and plots matched local execution. SageMaker required slightly more time to start the environment, but the outputs and visualizations were identical.

---

## Conclusions

- Linear regression is an initial approximation, but it cannot deal with the nonlinear nature of stellar data.
- Polynomial regression with interaction terms greatly improves prediction accuracy.
- Feature scaling is critical for gradient descent.
- The interaction term `M*T` plays an important role in modeling luminosity.

---

## Author

Camilo Fernandez  
Systems engenieering \
Escuela Colombiana de Ingenieria de sistemas