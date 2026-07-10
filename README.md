# SVM Robust Against Outliers and Noise

This repository contains the project developed for the **Machine Learning I** course, as part of the **Bachelors Degree in Bioinformatics** at the Faculty of Sciences, University of Porto (FCUP)

**Authors:** Carlos Pereira & Maria Santos (PL1)

---

## Project Overview
The core objective of this project was to analyze the impact of noise and outliers on Support Vector Machines (SVM) and implement an updated Sequential Minimal Optimization (SMO) algorithm incorporating **adaptive regularization** to mitigate these vulnerabilities[cite: 1].

Standard SVM hyperplanes can be significantly warped by outliers (anomalies that deviate from their class distribution) and noise (points that fall inside the soft margin), causing a narrowed margin and subpar classification performance[cite: 1].

## Generalized Pre-processing & Metafeatures Pipeline
To benchmark the models across various datasets, we built an automated pipeline using the **PMFE API** to extract structural metafeatures, including[cite: 1]:
* Number of outliers (`nr_outliers`)
* High noise ratio (`high_ns_ratio`)
* Data separability (`linear_discr`) and class imbalance (e.g., high kurtosis/skewness)

### Core Functions:
* `metafeatures_extraction()`: Dynamically extracts structural indicators from each dataset.
* `best_kernel()`: Dynamically selects the ideal kernel type based on a weighted normalization of the dataset's metafeatures (tailored for high dimensionality or outlier density)
* `calc_correlation()`: Filters out feature columns with low target correlation or high multi-collinearity.

---

## Proposed Modifications (Adaptive SMO)

Our implementation builds upon a baseline SMO algorithm by adjusting the soft-margin regularization parameter $C$ at both a global and individual sample level.

### 1. Global $C$ Adjustment
Based on the extracted `noise_level` argument, the global regularization factor $C$ adapts dynamically to relax the margin weight when severe noise is present: 
* **Moderate Noise** (5% - 15%): $C$ is decreased by 20%.
* **High Noise** (> 15%): $C$ is decreased by 50%.
* **Low Noise**: $C$ retains its baseline value (1.0).

### 2. Transversal Sample-Level Adaptive $C$
To handle specific misclassifications and outliers, we implemented an instance-level adaptive weight calculation based on the instance error ($Error[i]$):

$$C_{\text{individual}} = \frac{C}{1 + |Error[i]|}$$

* **Outliers** generate a high classification error, lowering their individual $C_{\text{individual}}$ weight, which reduces their overall influence on defining the separating hyperplane.
* **Stability via Backtracking:** If an update degrades the error metric, a backtracking mechanism reverts the iteration to its baseline state.
* **Efficiency:** Error calculation was vectorized to decrease computational overhead.
* **KKT Heuristics:** Only points violating the Karush-Kuhn-Tucker conditions are chosen for active hyperplane refinement.

### 3. Evaluated Variants
* **SVM_modified_3 (Modified 1):** Uses a greedy heuristic to pick the optimization pair $(i, j)$ that maximizes the learning step magnitude ($|E_i - E_j|$).
* **SVM_modified_3.1 (Modified 2):** Incorporates a stochastic approach, picking index $i$ and $j$ at random to avoid getting trapped in local minima caused by noisy data points.

---

## Empirical Evaluation & Results
Performance was validated through multiple hypothesis testing using a non-parametric **Friedman Test** followed by a **Bonferroni-Dunn post-hoc test** to avoid cumulative Type I errors.

### Key Findings:
* **SVM_modified_3 (Modified 1)** successfully out-performed the standard SVM baseline across **19 individual datasets**, showing strong tolerance to overlapping classes, high noise, and imbalance.
* According to the **Critical Difference Diagram (95% Confidence Level)** generated via `scikit_posthocs`, **SVM_modified_3.1 (Modified 2)** and `SVM_STANDARD` were not systematically distinct from each other. Overall, the baseline standard model maintained structural superiority regarding average ranking metrics.

### Relative Dataset Win Frequency:
* **SVM_STANDARD:** 45.2%[cite: 1]
* **SVM_modified_3.1 (Modified 2):** 45.2%[cite: 1]
* **SVM_modified_3 (Modified 1):** 9.5%[cite: 1]

Thank You !
