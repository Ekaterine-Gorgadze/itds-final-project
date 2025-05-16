# ITDS Final Project – Regression & Forecasting Lab

**Course:** Introduction to Data Science
**Name:** Ekaterine Gorgadze
**Submission Date:** May 16, 2025

---

## Project Overview

This project serves as the final assignment for the Introduction to Data Science course. It integrates the regression techniques, synthetic data generation, and forecasting methods covered throughout the semester. The project is implemented in Python and divided into three distinct parts:

---

## Project Components

### Part 1 – Univariate Regression

* Generated synthetic datasets from the following mathematical functions:

  * $f_1(x) = x \cdot \sin(x) + 2x$
  * $f_2(x) = 10 \cdot \sin(x) + x^2$
  * $f_3(x) = \text{sign}(x)(x^2 + 300) + 20 \cdot \sin(x)$
* Applied multiple regression models:

  * Linear Regression
  * Ridge Regression
  * Multi-Layer Perceptron Regressor
  * Random Forest Regressor
* Evaluated using Mean Squared Error (MSE) and R² metrics
* Included additional experiments:

  * Noise injection from a normal distribution
  * Polynomial and trigonometric feature engineering

### Part 2 – Multivariate Regression

* Generated a synthetic dataset using `make_regression` with 2000 samples and 10 features (5 informative)
* Trained and evaluated a regression model on multivariate data
* Printed model coefficients to assess feature importance, especially for non-informative features

### Part 3 – Time Series Forecasting (WWII Weather Data)

* Worked with real-world temperature data from station ID 22508 (Honolulu)
* Filtered and processed the data to build a rolling window structure for one-day-ahead forecasting
* Applied Random Forest Regressor and used TimeSeriesSplit for cross-validation
* Evaluated forecasting performance using R² and MSE
* Plotted both actual and predicted temperatures for visual comparison

---

## Files Included

* `itds_final_flawed.py` – Main project script with all implementations
* `SummaryOfWeather.csv` – Dataset used for forecasting temperature (must be uploaded manually)
* `README.md` – Project description and instructions

---

## How to Run

1. Open the script in a Python 3.11+ environment or Google Colab
2. Ensure `SummaryOfWeather.csv` is in the root directory (or upload it manually in Colab)
3. Run the entire script to view results and visualizations

---

## GitHub Repository

This project is also available on GitHub:
https://github.com/Ekaterine-Gorgadze/itds-final-project

---

## Completion Checklist

* [x] Univariate regression with 3 functions
* [x] Multiple regression models evaluated
* [x] Feature engineering and noise robustness tested
* [x] Multivariate regression with synthetic data
* [x] Time series forecasting on real dataset
* [x] TimeSeriesSplit cross-validation implemented

---

## Final Remarks

The final project reflects the core techniques and analytical practices learned throughout the course. All exercises, including optional components, have been implemented and verified. Thank you for a rewarding and insightful semester.

*Submitted by:* Ekaterine Gorgadze
