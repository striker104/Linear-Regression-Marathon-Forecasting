# linear_regression_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
# =======================================================
# CHANGE here depending on what gender you are predicting
# =======================================================
df = pd.read_csv("Female.csv")

# Convert race time to minutes
def time_to_minutes(t):
    try:
        h, m, s = map(int, t.split(':'))
        return h * 60 + m + s / 60
    except:
        return None  # handle missing/invalid times safely

df['race_time'] = df['time'].apply(time_to_minutes)

# Select relevant columns
features = ['club', 'gender', 'trained_10_week', 'has_trainer',
            'VO2 Max', 'BMI', 'n Marathons run', 'Cadence']
target = 'race_time'

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['club', 'gender'], drop_first=True)

# Drop unnecessary or non-numeric columns
df_encoded = df_encoded.drop(columns=['time', 'first_name', 'id'], errors='ignore')

# Drop rows with missing data
df_encoded = df_encoded.dropna(subset=[target])

# Split data
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nLinear Regression Model Results")
print("--------------------------------")
print("Intercept:", model.intercept_)
print("R² Score:", r2)
print("Mean Squared Error:", mse)
print("\nCoefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col:25} {coef:.4f}")


# Create a DataFrame for the new runner
# =======================================================
# CHANGE here depending on what the runners features are
# =======================================================
new_runner = pd.DataFrame([{
    'club': 'None',           # if unknown, default to 'None'
    'gender': 'female',
    'trained_10_week': 2,
    'has_trainer': 1,
    'VO2 Max': 12.75,
    'BMI': 19,
    'n Marathons run': 0,
    'Cadence': 155
}])

# Encode categorical variables to match training columns
new_runner_encoded = pd.get_dummies(new_runner, columns=['club', 'gender'], drop_first=True)

# Add missing columns (that existed in training)
for col in X.columns:
    if col not in new_runner_encoded.columns:
        new_runner_encoded[col] = 0  # fill absent dummies with 0

# Reorder columns to match training set
new_runner_encoded = new_runner_encoded[X.columns]

# Predict race time (in minutes)
predicted_time = model.predict(new_runner_encoded)[0]

# Convert minutes back to hh:mm:ss format
hours = int(predicted_time // 60)
minutes = int(predicted_time % 60)
seconds = int((predicted_time * 60) % 60)

print("\nPredicted Race Time for New Runner:")
print("------------------------------------")
print(f"Estimated race time: {hours:02d}:{minutes:02d}:{seconds:02d} (hh:mm:ss)")


#  Visualize results
plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Race Time (minutes)")
plt.ylabel("Predicted Race Time (minutes)")
plt.title("Predicted vs Actual Race Time")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', lw=2)
plt.tight_layout()
plt.show()
