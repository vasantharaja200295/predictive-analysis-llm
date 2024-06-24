import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load historical market data
data = pd.read_csv('market_data.csv')

# Preprocess data
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Select features and target variable
features = ['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Volume']
target = 'Close'

X = data[features]
y = data[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train predictive model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate model performance
accuracy = model.score(X_test, y_test)
print(f'Model accuracy: {accuracy}')

# Prepare new data for predictions
new_data = pd.DataFrame({
    'Year': [2023, 2023, 2023],
    'Month': [1, 2, 3],
    'Day': [1, 2, 3],
    'Open': [135.12, 136.45, 137.89],
    'High': [137.23, 138.67, 139.45],
    'Low': [134.45, 135.78, 136.23],
    'Volume': [7890123, 8901234, 9012345]
})

# Generate predictions
future_predictions = model.predict(new_data[features])
print(future_predictions)