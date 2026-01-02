# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 2: Create dataset
data = {
    'Income': [50000, 30000, 40000, 80000, 20000],
    'CreditScore': [700, 600, 650, 750, 580],
    'Employed': [1, 1, 0, 1, 0],
    'LoanApproved': [1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

# Step 3: Separate features (X) and target (y)
X = df[['Income', 'CreditScore', 'Employed']]
y = df['LoanApproved']

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Logistic Regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Step 6: Train Decision Tree model
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

# Step 7: Make predictions
log_pred = log_model.predict(X_test)
tree_pred = tree_model.predict(X_test)

# Step 8: Calculate accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_pred))
