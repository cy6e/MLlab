from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.impute import SimpleImputer

dataset = pd.read_csv("dataset.csv")
dataset.dropna()

s1_mapping = {"itching":1 , "skin_rash":2 , " continuous_sneezing":3 , "shivering":4 , " stomach_pain":5 , "acidity":6 , "vomiting":7 , 
              "indigestion":8 , "fatigue":9}
s2_mapping = {"itching":1 , "skin_rash":2 , " continuous_sneezing":3 , "shivering":4 , " stomach_pain":5 , "acidity":6 , "vomiting":7 , 
              "indigestion":8 , "fatigue":9}
s3_mapping = {"itching":1 , "skin_rash":2 , " continuous_sneezing":3 , "shivering":4 , " stomach_pain":5 , "acidity":6 , "vomiting":7 , 
              "indigestion":8 , "fatigue":9}
dataset["Symptom_1"] = dataset["Symptom_1"].map(s1_mapping)
dataset["Symptom_2"] = dataset["Symptom_2"].map(s2_mapping)
dataset["Symptom_3"] = dataset["Symptom_3"].map(s3_mapping)

cols_to_drop = ["Disease"]
X = dataset.drop(cols_to_drop,axis=1)
y = dataset["Disease"]

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

dt_predictions = dt_classifier.predict(X_test)

print("Decision Tree Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print("Classification Report:\n", classification_report(y_test, dt_predictions))

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

rf_predictions = rf_classifier.predict(X_test)

print("\nRandom Forest Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))
