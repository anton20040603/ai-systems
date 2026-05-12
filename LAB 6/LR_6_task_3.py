# LR_6_task_3.py
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
# ============================================================
# Task 3 — Варіант 6 (Overcast, High, Weak)
# ============================================================
data = {
    'Day':      ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14'],
    'Outlook':  ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind':     ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play':     ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

print("=" * 55)
print("Task 3 — Варіант 6")
print("=" * 55)
print(df.to_string(index=False))

# --- Умови для прогнозу ---
outlook_val = 'Overcast'
humidity_val = 'High'
wind_val = 'Weak'

yes_rows = df[df['Play'] == 'Yes']
no_rows = df[df['Play'] == 'No']
total = len(df)
n_yes = len(yes_rows)
n_no = len(no_rows)

p_yes = n_yes / total
p_no = n_no / total

p_out_yes = len(yes_rows[yes_rows['Outlook'] == outlook_val]) / n_yes
p_hum_yes = len(yes_rows[yes_rows['Humidity'] == humidity_val]) / n_yes
p_wind_yes = len(yes_rows[yes_rows['Wind'] == wind_val]) / n_yes

p_out_no = len(no_rows[no_rows['Outlook'] == outlook_val]) / n_no
p_hum_no = len(no_rows[no_rows['Humidity'] == humidity_val]) / n_no
p_wind_no = len(no_rows[no_rows['Wind'] == wind_val]) / n_no

prob_yes_raw = p_out_yes * p_hum_yes * p_wind_yes * p_yes
prob_no_raw = p_out_no * p_hum_no * p_wind_no * p_no
total_prob = prob_yes_raw + prob_no_raw

prob_yes = prob_yes_raw / total_prob
prob_no = prob_no_raw / total_prob

print(f"\nConditions: Outlook={outlook_val}, Humidity={humidity_val}, Wind={wind_val}")
print("\n--- Manual Calculations ---")
print(f"P(Yes) = {n_yes}/{total} = {p_yes:.4f}")
print(f"P(No)  = {n_no}/{total}  = {p_no:.4f}")
print(f"P(Overcast|Yes)  = {int(p_out_yes*n_yes)}/{n_yes} = {p_out_yes:.4f}")
print(f"P(High|Yes) = {int(p_hum_yes*n_yes)}/{n_yes} = {p_hum_yes:.4f}")
print(f"P(Weak|Yes) = {int(p_wind_yes*n_yes)}/{n_yes} = {p_wind_yes:.4f}")
print(f"P(Overcast|No)   = {int(p_out_no*n_no)}/{n_no}  = {p_out_no:.4f}")
print(f"P(High|No)  = {int(p_hum_no*n_no)}/{n_no}  = {p_hum_no:.4f}")
print(f"P(Weak|No)  = {int(p_wind_no*n_no)}/{n_no}  = {p_wind_no:.4f}")
print(f"\nP(Yes|conditions) not normalized = {p_out_yes:.4f} × {p_hum_yes:.4f} × {p_wind_yes:.4f} × {p_yes:.4f} = {prob_yes_raw:.6f}")
print(f"P(No|conditions) not normalized = {p_out_no:.4f} × {p_hum_no:.4f} × {p_wind_no:.4f} × {p_no:.4f}  = {prob_no_raw:.6f}")
print(f"\nP(Yes) normalized = {prob_yes:.4f} ({prob_yes*100:.1f}%)")
print(f"P(No) normalized = {prob_no:.4f}  ({prob_no*100:.1f}%)")

verdict = "За таких умов матчу бути! ✓" if prob_yes > prob_no else "Матч не може відбутися за таких умов ✗"
print(f"\n>>> Conclusion: {verdict}")

# --- Перевірка через sklearn ---
le_out = LabelEncoder()
le_hum = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()

X = pd.DataFrame({
    'Outlook':  le_out.fit_transform(df['Outlook']),
    'Humidity': le_hum.fit_transform(df['Humidity']),
    'Wind':     le_wind.fit_transform(df['Wind']),
})
y = le_play.fit_transform(df['Play'])

model = CategoricalNB()
model.fit(X, y)

test = pd.DataFrame({
    'Outlook':  le_out.transform([outlook_val]),
    'Humidity': le_hum.transform([humidity_val]),
    'Wind':     le_wind.transform([wind_val]),
})
pred = le_play.inverse_transform(model.predict(test))[0]
proba = model.predict_proba(test)[0]

print(f"\n--- sklearn CategoricalNB checking ---")
print(f"Prediction: {pred}")
print(f"P(No)={proba[0]:.4f}  P(Yes)={proba[1]:.4f}")
