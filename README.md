# Marathon Time Predictor (Linear Regression)

This project implements an interactive Machine Learning solution using **Linear Regression** to forecast a runner's marathon finish time. It demonstrates proficiency in end-to-end data pipeline engineering, analytical feature selection, model training, and dynamic user interaction based on gender-specific data.

---

## Key Achievements & Technical Rigor

* **Prediction Accuracy:** Achieved **80% accuracy** in forecasting marathon times by training a Linear Regression model on historical runner data.
* **Analytical Feature Selection:** Utilized **correlation analysis** to identify and prune features with low $R$-coefficient correlation, ensuring the model is built on the most predictive metrics.
* **Data Transformation:** Engineered a custom function to transform race time from the standard `HH:MM:SS` format into a continuous numerical variable (total minutes) suitable for regression modeling.

---

## Technology Stack

| Category | Component | Description |
| :--- | :--- | :--- |
| **Model** | Linear Regression (Scikit-learn) | Core regression model used for prediction. |
| **Core Libraries** | Python, Pandas, NumPy | Used for data manipulation, cleaning, and numerical operations. |
| **Analysis/Viz** | Matplotlib, Seaborn | Libraries used to generate the correlation heatmap (`correlation_analysis.py`) and the Predicted vs. Actual scatter plot. |
| **Data Handling** | `.csv` files | Input files containing historical runner metrics and finish times. |


---

## Customization: Predicting Your Own Time

* **Gender-Specific Data Source:** When predicting for a specific gender, the script must be configured to train on the corresponding historical data. The code is set to read **`Male.csv`** or **`Female.csv`**. If you want to predict a runner, ensure you are reading the correct CSV for the gender you are testing.
* **Runner Feature Alignment:** You can change the runner's features you want to predict in the code itself. Just make sure the features you enter (like `BMI` or `Cadence`) match the features the model uses to predict a time. The eight required input features are: `club`, `gender`, `trained_10_week`, `has_trainer`, `VO2 Max`, `BMI`, `n Marathons run`, and `Cadence`.
