#!/usr/bin/env python3
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data", "crocodile_dataset.csv")
FIG  = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

TARGET = "Conservation Status"
SEED = 42

df = pd.read_csv(DATA).drop_duplicates().reset_index(drop=True)
df = df.dropna(subset=[TARGET])

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols and c != TARGET]

X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True))])

transformers = []
if len(num_cols) > 0: transformers.append(("num", numeric, num_cols))
if len(cat_cols) > 0: transformers.append(("cat", categorical, cat_cols))
pre = ColumnTransformer(transformers=transformers)

logreg = Pipeline([("preprocessor", pre), ("model", LogisticRegression(solver="liblinear", max_iter=1000))])
pre_rf = ColumnTransformer(transformers=transformers)
rf = Pipeline([("preprocessor", pre_rf), ("model", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1))])

logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

def eval(model):
    yp = model.predict(X_test)
    return yp, accuracy_score(y_test, yp), f1_score(y_test, yp, average="macro")

pred_l, acc_l, f1_l = eval(logreg)
pred_r, acc_r, f1_r = eval(rf)

rows = [
    {"Model":"LogisticRegression","Accuracy":round(acc_l,4),"F1_macro":round(f1_l,4)},
    {"Model":"RandomForest","Accuracy":round(acc_r,4),"F1_macro":round(f1_r,4)},
]
metrics_df = pd.DataFrame(rows).sort_values(["F1_macro","Accuracy"], ascending=False).reset_index(drop=True)
print("=== TEST METRICS ===")
print(metrics_df.to_string(index=False))

best = metrics_df.loc[0,"Model"]
pred_best = pred_l if best=="LogisticRegression" else pred_r
classes = sorted(pd.Series(y_test).unique())
cm = confusion_matrix(y_test, pred_best, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(values_format="d")
plt.title(f"Confusion Matrix — {best}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIG,"confusion_matrix_best_model.png"), dpi=160, bbox_inches="tight")
plt.close()

pre_best = logreg.named_steps["preprocessor"] if best=="LogisticRegression" else rf.named_steps["preprocessor"]
model_best = logreg.named_steps["model"] if best=="LogisticRegression" else rf.named_steps["model"]

feat_names = []
try:
    feat_names = pre_best.get_feature_names_out().tolist()
except Exception:
    for name, trans, cols in pre_best.transformers_:
        if name=="num" and len(cols)>0: feat_names += list(cols)
        elif name=="cat" and len(cols)>0:
            try:
                oh = pre_best.named_transformers_["cat"].named_steps["onehot"]
                feat_names += oh.get_feature_names_out(cols).tolist()
            except Exception:
                feat_names += [f"{c}_encoded" for c in cols]

importances = None
if best=="LogisticRegression":
    coef = model_best.coef_
    importances = pd.Series(np.mean(np.abs(coef), axis=0))
else:
    importances = pd.Series(model_best.feature_importances_)

if len(importances) != len(feat_names):
    feat_names = [f"feat_{i}" for i in range(len(importances))]

fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(10)
plt.figure(figsize=(8,5))
plt.barh(fi["feature"][::-1], fi["importance"][::-1])
plt.title(f"Top-10 Feature Importance — {best}")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(FIG,"top10_feature_importance.png"), dpi=160, bbox_inches="tight")
plt.close()

with open(os.path.join(BASE, "classification_report.txt"),"w") as f:
    f.write(classification_report(y_test, pred_best, digits=4, zero_division=0))

metrics_df.to_csv(os.path.join(BASE, "TEST_METRICS.csv"), index=False)
with open(os.path.join(BASE, "metrics.json"),"w") as f:
    json.dump({"best_model":best, "metrics":rows}, f, indent=2)

print("\nArtifacts written to:", BASE)
