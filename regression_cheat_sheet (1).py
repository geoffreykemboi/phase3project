# REGRESSION CHEAT SHEET FOR EXAMS/ASSIGNMENTS
# ==============================================

## THE 8-STEP PROCESS (Memorize this!)
## =====================================

# 1. IMPORTS (copy-paste this every time)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 2. LOAD DATA
df = pd.read_csv("data.csv")  # or sns.load_dataset("name")
df = df.dropna()  # remove missing values

# 3. DEFINE X AND Y
X = df[['feature1', 'feature2', 'feature3']]  # predictors
y = df['target']  # what you're predicting

# 4. ENCODE CATEGORICALS (if you have species, sex, country, etc.)
X = pd.get_dummies(X, drop_first=True)

# 5. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. FIT MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# 7. PREDICT
y_pred = model.predict(X_test)

# 8. EVALUATE
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


## QUICK FORMULAS TO REMEMBER
## ===========================

# Adjusted R²:
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
# where: n = number of samples, p = number of predictors

# RMSE:
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# OR: rmse = np.sqrt(np.mean((y_test - y_pred)**2))


## WHEN TO USE WHAT ENCODING
## ==========================

# ONE-HOT ENCODING (most common):
# Use for: species, country, occupation, color (no order)
X = pd.get_dummies(X, drop_first=True)

# LABEL ENCODING (rare in MLR):
# Use for: days of week, education level, size (has order)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X['column'] = encoder.fit_transform(X['column'])


## INTERPRETING METRICS
## =====================

# R² = 0.85 means:
# → Model explains 85% of variation
# → Higher is better (max = 1.0)

# Adjusted R² penalizes extra predictors
# → Use this when comparing different models

# MAE = 150 means:
# → Average error is 150 units
# → In same units as target variable

# RMSE = 200 means:
# → Typical error is 200 units
# → Penalizes large errors more than MAE


## COMMON EXAM QUESTIONS & ANSWERS
## ================================

Q: "What does R² of 0.90 mean?"
A: The model explains 90% of the variation in [target variable].

Q: "Why use Adjusted R² instead of R²?"
A: Because R² always increases when we add predictors, even useless ones. 
   Adjusted R² penalizes unnecessary predictors.

Q: "What's the difference between MAE and RMSE?"
A: MAE treats all errors equally. RMSE penalizes large errors more.
   If RMSE >> MAE, you have some big outlier errors.

Q: "Why drop_first=True in one-hot encoding?"
A: To avoid multicollinearity (the dummy variable trap).
   If penguin is not Chinstrap and not Gentoo, it must be Adelie!

Q: "What does a negative coefficient mean?"
A: That feature decreases the target value.
   Positive coefficient = feature increases target.


## TYPICAL ASSIGNMENT STRUCTURE
## =============================

1. Question asks: "Predict [Y] using [X1, X2, X3]"

2. You write:
   - Load data
   - Handle categoricals
   - Split data
   - Fit model
   - Calculate all 4 metrics (R², Adj R², MAE, RMSE)
   
3. Interpret in 2-3 sentences:
   "The model achieves R² of [value], explaining [%] of variation.
    The average prediction error (MAE) is [value] [units].
    This suggests the model performs [well/poorly] for this task."


## DIAGNOSTICS REMINDER
## =====================

If asked to check assumptions:

1. Linearity: scatter plot of X vs Y (should be roughly linear)
2. Homoscedasticity: residual plot (should be random scatter)
3. Normality: histogram of residuals (should be bell-shaped)

Fixes:
- Non-linear? → Add polynomial terms or log transform
- Errors growing? → Log transform
- Variables interact? → Add interaction terms (X1 * X2)


## SPEED TIPS FOR EXAMS
## =====================

1. Copy the template first, then fill in your specifics
2. Always use random_state=42 for reproducibility
3. test_size=0.2 is standard (80% train, 20% test)
4. Remember: R² and Adjusted R² are percentages (multiply by 100)
5. MAE and RMSE are in same units as your target variable
6. If you forget Adjusted R² formula, it's in the code above!
