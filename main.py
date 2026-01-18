# ======================================
# 1. IMPORT LIBRARIES
# ======================================
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ======================================
# 2. LOAD DATA
# ======================================
leads = pd.read_csv("Lead.csv")

print(leads.shape)
print(leads.info())
print(leads.isnull().sum() * 100 / leads.shape[0])


# ======================================
# 3. DATA CLEANING
# ======================================

# Drop ID columns
leads.drop(columns=['Prospect ID', 'Lead Number'], inplace=True)

# Replace "Select" with NaN
cat_cols = [
    "Specialization",
    "How did you hear about X Education",
    "Lead Profile",
    "City"
]
leads[cat_cols] = leads[cat_cols].replace("Select", np.nan)

# Drop columns with >40% missing
cols_to_drop = [
    col for col in leads.columns
    if leads[col].isnull().mean() > 0.40
]
leads.drop(columns=cols_to_drop, inplace=True)

# Drop object columns with >15% missing
cols_to_drop_2 = [
    col for col in leads.columns
    if leads[col].isnull().mean() > 0.15 and leads[col].dtype == 'object'
]
leads.drop(columns=cols_to_drop_2, inplace=True)

# Drop rows with missing critical numerical values
leads = leads[~pd.isnull(leads["TotalVisits"])]
leads = leads[~pd.isnull(leads["Page Views Per Visit"])]


# ======================================
# 4. EDA
# ======================================
cat_cols = list(leads.select_dtypes(include='object'))
for col in cat_cols:
    plt.figure(figsize=(18, 6))
    sns.countplot(x=col, hue="Converted", data=leads)
    plt.xticks(rotation=90)
    plt.show()

num_cols = [
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit"
]

sns.pairplot(leads[num_cols])
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(leads[num_cols].corr(), annot=True, cmap="BuPu")
plt.show()


# ======================================
# 5. DATA PREPARATION
# ======================================

# Binary encoding
binary_cols = [
    "Do Not Email", "Do Not Call", "Search", "Magazine",
    "Newspaper Article", "X Education Forums", "Newspaper",
    "Digital Advertisement", "Get updates on DM Content"
]

leads[binary_cols] = leads[binary_cols].apply(
    lambda x: x.map({"Yes": 1, "No": 0})
)

# One-hot encoding
cat_cols = list(leads.select_dtypes(include='object'))
leads_final = pd.get_dummies(leads, columns=cat_cols, drop_first=True)

# Split X and y
X = leads_final.drop("Converted", axis=1)
y = leads_final["Converted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10
)


# ======================================
# 6. FEATURE SCALING
# ======================================
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])   # IMPORTANT


# ======================================
# 7. FEATURE SELECTION USING RFE
# ======================================
lr = LogisticRegression(max_iter=1000)
rfe = RFE(lr, n_features_to_select=15)
rfe.fit(X_train, y_train)

selected_cols = X_train.columns[rfe.support_]
X_train_rfe = X_train[selected_cols]


# ======================================
# 8. STATSMODELS LOGISTIC REGRESSION
# ======================================
X_train_sm = sm.add_constant(X_train_rfe)
logm1 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
res = logm1.fit()
print(res.summary())


# ======================================
# 9. VIF CHECK
# ======================================
vif = pd.DataFrame()
vif["Feature"] = X_train_rfe.columns
vif["VIF"] = [
    variance_inflation_factor(X_train_rfe.values, i)
    for i in range(X_train_rfe.shape[1])
]
print(vif.sort_values(by="VIF", ascending=False))

# Drop high VIF feature (example)
X_train_rfe = X_train_rfe.drop('Tags_Interested in Next batch', axis=1)

# Refit model
X_train_sm = sm.add_constant(X_train_rfe)
logm2 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
res = logm2.fit()
print(res.summary())


# ======================================
# 10. TRAINING SET PREDICTION
# ======================================
y_train_prob = res.predict(X_train_sm)

train_df = pd.DataFrame({
    "Converted": y_train,
    "Converted_prob": y_train_prob
})


# ======================================
# 11. ROC & AUC
# ======================================
fpr, tpr, thresholds = roc_curve(train_df["Converted"], train_df["Converted_prob"])
auc_score = roc_auc_score(train_df["Converted"], train_df["Converted_prob"])

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# ======================================
# 12. CUTOFF SELECTION
# ======================================
cutoffs = np.arange(0.0, 1.0, 0.1)
cutoff_df = pd.DataFrame(columns=['cutoff', 'accuracy', 'recall', 'specificity'])

for c in cutoffs:
    y_pred = (train_df["Converted_prob"] >= c).astype(int)
    cm = metrics.confusion_matrix(train_df["Converted"], y_pred)

    TN, FP, FN, TP = cm.ravel()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)

    cutoff_df.loc[len(cutoff_df)] = [c, accuracy, recall, specificity]

print(cutoff_df)

cutoff_df.plot(x='cutoff', y=['accuracy', 'recall', 'specificity'])
plt.show()


# ======================================
# 13. FINAL CUTOFF & LEAD SCORE
# ======================================
final_cutoff = 0.3

train_df["final_predicted"] = (
    train_df["Converted_prob"] >= final_cutoff
).astype(int)

train_df["Lead_Score"] = (train_df["Converted_prob"] * 100).round()


# ======================================
# 14. TEST SET EVALUATION
# ======================================
X_test_final = X_test[X_train_rfe.columns]
X_test_sm = sm.add_constant(X_test_final)

y_test_prob = res.predict(X_test_sm)
y_test_pred = (y_test_prob >= final_cutoff).astype(int)

print("Accuracy:", metrics.accuracy_score(y_test, y_test_pred))
print("Precision:", metrics.precision_score(y_test, y_test_pred))
print("Recall:", metrics.recall_score(y_test, y_test_pred))

confusion = metrics.confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:\n", confusion)
