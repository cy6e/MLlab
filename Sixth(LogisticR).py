import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load the local dataset
df = pd.read_csv('Student-University.csv')

#Adding columns
df.columns=['X','y','Target']

# Assuming 'X' and 'y' are the feature and target columns
X = df[['X', 'y']].values
y = df['Target'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Train the model on the training data
model.fit(X_train, y_train)

# Predict classes on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy}")

# Gradient Descent for Logistic Regression
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate x values
x = np.linspace(-7, 7, 200)

# Calculate corresponding y values using the sigmoid function
y = sigmoid(x)

# Plot the sigmoid function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Sigmoid Function', color='b')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.title('Sigmoid Function')
plt.grid(True)
plt.legend()
plt.show()

# Initialize parameters
theta = np.zeros(X_train.shape[1]) # Initialize parameters to zeros
alpha = 0.01 # Learning rate
num_iterations = 1000

# Perform gradient descent
for _ in range(num_iterations):
    z = np.dot(X_train, theta)
    h = sigmoid(z)
    error = h - y_train
    gradient = np.dot(X_train.T, error) / len(y_train)
    theta -= alpha * gradient

# Print the computed regression parameters
print(f'Regression Parameters using Gradient Descent: {theta}')
print("Original Data:")
print(df)

# Step 1: Removing Noise
# In this example, we'll consider any value above 10 as noise and remove it.
df = df[df['y'] <= 10]

# Step 2: Handling Missing Values (NaNs)
# In this example, we'll use mean imputation to fill missing values.
df['X'].fillna(df['X'].mean(), inplace=True)
df['y'].fillna(df['y'].mean(), inplace=True)

# Display the cleaned dataset
print("\nCleaned Data:")
print(df)
