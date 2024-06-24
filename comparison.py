import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_regression

# Generate synthetic classification dataset
X_class, y_class = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Generate synthetic regression dataset
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split classification dataset into training and testing sets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Split regression dataset into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Initialize models
regression_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
}

classification_models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
}

# Function to generate classification report
def generate_classification_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

# Function to generate regression report
def generate_regression_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    return {'Mean Squared Error': mse, 'Root Mean Squared Error': rmse}

# Train and evaluate regression models
regression_reports = {}
for name, model in regression_models.items():
    model.fit(X_train_reg, y_train_reg)
    regression_reports[name] = generate_regression_report(model, X_test_reg, y_test_reg)

# Train and evaluate classification models
classification_reports = {}
for name, model in classification_models.items():
    model.fit(X_train_class, y_train_class)
    classification_reports[name] = generate_classification_report(model, X_test_class, y_test_class)

# Convert reports to DataFrame for comparison
regression_df = pd.DataFrame(regression_reports).T
classification_df = pd.DataFrame({model_name: pd.Series(report) for model_name, report in classification_reports.items()})

# Print reports
print("Regression Reports:")
print(regression_df)
print("\nClassification Reports:")
print(classification_df)

classification_df.to_json('classification_reports.json')

# Plotting
models = list(regression_reports.keys()) + list(classification_reports.keys())
mse_values = [report['Mean Squared Error'] if 'Mean Squared Error' in report else np.nan for report in regression_reports.values()]
rmse_values = [report['Root Mean Squared Error'] if 'Root Mean Squared Error' in report else np.nan for report in regression_reports.values()]
accuracy_values = [accuracy_score(y_test_class, model.predict(X_test_class)) for model in classification_models.values()]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].barh(models[:len(regression_models)], mse_values, color='skyblue', label='MSE')
ax[0].barh(models[:len(regression_models)], rmse_values, color='orange', label='RMSE')
ax[0].set_xlabel('Error')
ax[0].set_title('Regression Model Performance')
ax[0].legend()

ax[1].barh(models[len(regression_models):], accuracy_values, color='lightgreen')
ax[1].set_xlabel('Accuracy')
ax[1].set_title('Classification Model Performance')

plt.tight_layout()
plt.show()
