# Heart-disease-prediction-
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("heart.csv")

# Split data into features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict for a new patient
new_data = [[52, 1, 140, 250, 0, 1, 150, 0, 2.3, 2, 0, 2, 3]]
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")
