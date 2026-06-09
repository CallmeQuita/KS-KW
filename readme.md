# Statistical Testing System Project

This repository contains a comprehensive automated statistical testing system that implements non-parametric statistical algorithms manually and compares them against the standard implementations provided by `scipy`.

## Implemented Algorithms

### 1. Kruskal-Wallis Test

Used for comparing multiple independent groups to determine whether they originate from the same distribution.

### 2. Kolmogorov-Smirnov Test (K-S Test)

* **One-Sample K-S Test**: Used for good-ness of-fit testing.
* **Two-Sample K-S Test**: Used for comparing the distributions of two independent samples.
* **Two-Dimensional K-S Test**: Used for comparing spatial distributions in two-dimensional datasets.

### 3. Dunn's Post-hoc Test

Used for pairwise comparisons when the Kruskal-Wallis test detects statistically significant differences among groups.

## How to Run the System

### Prerequisites

* Python 3.12 or later
* Required libraries listed in `requirements.txt`

### Installation

Navigate to the project root directory and install the required dependencies:

```bash
pip install -r requirements.txt
```

### Algorithm Verification

Run the test suite to validate the custom implementations against the corresponding SciPy implementations:

```bash
python test.py
```

This process performs consistency checks, absolute deviation analysis, and logical validation for all implemented Kruskal-Wallis and Kolmogorov-Smirnov algorithms.

### Launching the Dashboard

Run the interactive Streamlit application:

```bash
streamlit run app.py
```

The dashboard provides visualization of statistical results and post-hoc analysis.



