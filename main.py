import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Function to generate synthetic data with outliers
def generate_data(n, outliers_fraction=0.1):
    np.random.seed(0)
    x = np.linspace(0, 100, n)
    y = 0.5 * x + 10 + np.random.normal(size=n)  # Linear relation with noise
    num_outliers = int(outliers_fraction * n)
    indices = np.random.choice(np.arange(n), num_outliers, replace=False)
    y[indices] += np.random.normal(50, 10, size=num_outliers)  # Add large deviations to create outliers
    return x, y

# Function to perform robust polynomial regression using IRLS
def robust_polynomial_regression(x, y, degree):
    X = np.vander(x, degree + 1)
    rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()
    return rlm_results

# Function to calculate Median Absolute Deviation (MAD)
def mad(data, axis=None):
    return np.median(np.abs(data - np.median(data, axis)), axis)

# Function to detect outliers using robust 3-sigma rule
def detect_outliers(y, y_fitted, scale):
    residuals = y - y_fitted
    mad_scale = mad(residuals)
    threshold = 3 * mad_scale
    outliers = np.abs(residuals) > threshold
    return outliers

# Generate synthetic data
n = 100
x, y = generate_data(n)

# Perform robust polynomial regression
degree = 4
rlm_results = robust_polynomial_regression(x, y, degree)

# Detect outliers
outliers = detect_outliers(y, rlm_results.fittedvalues, rlm_results.scale)

# Plot the data, robust polynomial fit, and outliers
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data', color='lightgray')
plt.plot(x, rlm_results.fittedvalues, label='Robust Polynomial Fit', color='red')
plt.scatter(x[outliers], y[outliers], label='Outliers', color='blue', edgecolor='k', s=100)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Print the summary of the robust regression model
summary = rlm_results.summary()
summary.tables[0], summary.tables[1]
