# LR_6_task_4.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from io import StringIO
import matplotlib
matplotlib.use('TkAgg')

input_file = r'D:\users\Документы\СШІ\LAB 6\renfe_small.csv'
df = pd.read_csv(input_file)
df = df.dropna()
print("=" * 55)
print("Task 4 — Байєсівський аналіз: ціни на квитки")
print("=" * 55)
print(f"\nDataset size: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst rows:")
print(df.head(8).to_string(index=False))
print(f"\nPrice statistics:\n{df['price'].describe().round(2)}")

# --- Цільова змінна ---
df['price_category'] = pd.cut(df['price'], bins=3, labels=['Low', 'Medium', 'High'])

print(f"\nDistribution of price categories:")
for cat, cnt in df['price_category'].value_counts().items():
    print(f"  {cat}: {cnt} ({cnt/len(df)*100:.1f}%)")

# --- Кодування ---
features = ['train_type', 'train_class', 'fare']
le_dict = {}
X = pd.DataFrame()
for col in features:
    le = LabelEncoder()
    X[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

le_y = LabelEncoder()
y = le_y.fit_transform(df['price_category'].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# --- Модель ---
model = CategoricalNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of model: {acc:.4f} ({acc*100:.1f}%)")
print(f"\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))
# --- Графіки ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Task 4 — Байєсівський аналіз: ціни на квитки', fontsize=13, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=le_y.classes_, yticklabels=le_y.classes_, ax=axes[0])
axes[0].set_title('Confusion matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

colors = ['#4CAF50', '#FFC107', '#F44336']
for cat, color in zip(['Low', 'Medium', 'High'], colors):
    subset = df[df['price_category'] == cat]['price']
    axes[1].hist(subset, bins=20, alpha=0.65, label=cat, color=color)
axes[1].set_title('Distribution of prices by categories')
axes[1].set_xlabel('Price (EUR)')
axes[1].set_ylabel('Quantity')
axes[1].legend()

plt.tight_layout()
plt.savefig('task4_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGraphic saved as task4_plots.png")

# --- Демо прогнозу ---
print("\n--- Prediction of ticket price categories ---")
test_cases = [
    {'train_type': 'REGIONAL', 'train_class': 'Turista', 'fare': 'Promo'},
    {'train_type': 'AVE', 'train_class': 'Club', 'fare': 'Flexible'},
    {'train_type': 'INTERCITY', 'train_class': 'Preferente', 'fare': 'Promo +'},
    {'train_type': 'AVE-TGV', 'train_class': 'Turista Plus', 'fare': 'Mesa'}
]
for case in test_cases:
    row = {col: le_dict[col].transform([val])[0] for col, val in case.items()}
    pred_cat = le_y.inverse_transform(model.predict(pd.DataFrame([row])))[0]
    proba    = model.predict_proba(pd.DataFrame([row]))[0]
    pb = " | ".join([f"{c}:{p:.2f}" for c, p in zip(le_y.classes_, proba)])
    print("\n-----------------------------------")
    print(f"Train type: {case['train_type']}")
    print(f"Class: {case['train_class']}")
    print(f"Fare type: {case['fare']}")
    print(f"Predicted category: {pred_cat}")
    print(f"Probabilities: {pb}")
