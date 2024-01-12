import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data from the CSV file without column names
data = pd.read_csv('Food-Truck-LineReg.csv')

# Assign generic column names
data.columns = ['X', 'Y']

# Extract the X and Y variables from the dataset
X = data[['X']].values
Y = data[['Y']].values

# Create and fit the LinearRegression model
model = LinearRegression().fit(X, Y)

# Scatter plot
plt.scatter(X, Y)
plt.plot(X, model.predict(X), color='red') # Plot the regression line
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot with Regression Line")
plt.show()

# Get regression parameters
intercept, slope = model.intercept_[0], model.coef_[0][0]

# Calculate Cost, SSE, SSR, SST, and R-squared
Y_pred = model.predict(X)
Cost = np.mean((Y_pred - Y) ** 2)
SSE = np.sum((Y - Y_pred) ** 2)
SSR = np.sum((Y_pred - np.mean(Y)) ** 2)
SST = np.sum((Y - np.mean(Y)) ** 2)
R_squared = SSR / SST
correlation_matrix = data.corr()

print("Regression Parameters:")
print(f"Intercept (b0): {intercept}")
print(f"Slope (b1): {slope}")

print("Cost:", Cost)
print("SSE:", SSE)
print("SSR:", SSR)
print("SST:", SST)
print("R-squared:", R_squared)
print("Correlation Matrix: ",correlation_matrix)
