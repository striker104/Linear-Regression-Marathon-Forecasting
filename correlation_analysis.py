import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model


df = pd.read_csv("Male.csv")
print("Dataset shape:", df.shape)
print("Columns:", df.columns)
print(df.head())

def time_to_minutes(t):
    h, m, s = map(int, t.split(':'))
    return h*60 + m + s/60

df['race_time'] = df['time'].apply(time_to_minutes)


df_numeric = df[['race_time', 'Age', 'VO2 Max', 'BMI', 'trained_10_week', 'trained_im', 'Cadence', 'has_trainer','n Marathons run']]

print(df_numeric.head())


corr_matrix = df_numeric.corr()
print("Full Correlation Matrix:")
print(corr_matrix)


corr_with_time = corr_matrix['race_time'].sort_values(ascending=False)
print("\nCorrelation with Race Time:")
print(corr_with_time)
corr_matrix_abs = corr_matrix.abs()


plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix_abs, annot=True, cmap='Reds', vmin=0, vmax=1, fmt=".2f", cbar=True)
plt.title("Absolute Correlation Heatmap (Red = Strong)")
plt.tight_layout()
plt.show()


