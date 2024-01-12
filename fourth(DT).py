import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report


data = pd.read_csv('zoo_data.csv')

# Split the dataset into features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a Decision Tree Classifier object
clf = DecisionTreeClassifier()

# Train the model using the training sets
clf.fit(X_train, y_train)


# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Print the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:")
print(cr)

# Visualize the Decision Tree with a simplified text representation
from sklearn.tree import export_text
tree_text = export_text(clf, feature_names=list(X.columns))
print("Decision Tree:")
print(tree_text)
