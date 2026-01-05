# 🎯 Lead Scoring Case Study: Predictive Conversion Model

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Logistic--Regression-orange.svg)
![Data Analysis](https://img.shields.io/badge/Analysis-EDA-green.svg)

## 📌 Abstract
This project focuses on developing a **Lead Scoring Engine** for an education company to identify potential customers with a high probability of conversion. By assigning a lead score between **0 and 100**, the model allows the sales team to prioritize "Hot Leads," effectively increasing the conversion rate and streamlining the sales funnel.



## 🎯 Objectives
* **Lead Prioritization:** Assign scores to identify leads with a higher probability of converting into paying customers.
* **Efficiency:** Enhance lead conversion efficiency by focusing sales efforts on high-potential targets.
* **Growth:** Contribute to revenue growth through data-driven decision-making.

---

## 🛠️ Methodology

### 1. Data Collection & Exploration
* Analyzing demographic data, website engagement, and past interactions from the `Lead.csv` dataset.
* Identifying patterns and correlations between user behavior and conversion.

### 2. Data Preprocessing
* **Cleaning:** Handling missing values and removing redundant variables (e.g., Prospect ID).
* **Optimization:** Handling outliers and standardizing categorical labels.
* **Missing Value Treatment:** Dropping columns with >40% missing data and imputing where necessary.



### 3. Feature Engineering
* Converting categorical variables into numerical representations using **One-Hot Encoding**.
* Creating dummy variables to capture the impact of features like `Lead Source` and `Specialization`.

### 4. Model Development
* **Algorithm:** Logistic Regression was utilized for its interpretability and efficiency in binary classification.
* **Feature Selection:** Employed **Recursive Feature Elimination (RFE)** to select the top 15 predictive features.
* **Validation:** Data was split into 75% training and 25% testing sets.



### 5. Model Evaluation
We used multiple metrics to ensure the model's robustness:
* **Accuracy:** Overall correctness of the model.
* **Sensitivity (Recall):** Ensuring we identify as many potential converters as possible.
* **Precision:** Accuracy of the positive predictions.
* **ROC-AUC Score:** Measuring the model's ability to distinguish between classes.



---

## 🚀 How to Use

### Installation
1. Clone the repository:
   
   ```bash
   git clone [https://github.com/Saravanan-S-315/Lead-Scoring-Case-Study.git]
   cd Lead-Scoring-Case-Study
