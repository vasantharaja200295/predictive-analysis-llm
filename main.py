import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from openai import OpenAI

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

# Initialize LM Studio client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# User interaction
while True:
    user_input = input('Enter your query or type "exit" to quit: ')
    if user_input.lower() == 'exit':
        break
    elif user_input.lower() == 'predict':
        # Prepare new data for predictions (using the last 30 days of the existing data)
        new_data = data.iloc[-30:][features]
        future_predictions = model.predict(new_data)

        # Analyze predictions
        future_predictions_df = pd.DataFrame({'Date': data.iloc[-30:]['Date'], 'Predicted Stock Price': future_predictions})
        future_predictions_df['Trend'] = future_predictions_df['Predicted Stock Price'].diff().apply(lambda x: 'Increasing' if x > 0 else 'Decreasing')
        increasing_predictions = future_predictions_df[future_predictions_df['Trend'] == 'Increasing']
        decreasing_predictions = future_predictions_df[future_predictions_df['Trend'] == 'Decreasing']

        # Generate text response using LM Studio
        system_prompt = f"Here are the insights about the predicted stock prices for the next 30 days:\n\nPredicted Stock Prices:\n{future_predictions_df.to_string(index=False)}\n\nIncreasing Predictions:\n{increasing_predictions.to_string(index=False)}\n\nDecreasing Predictions:\n{decreasing_predictions.to_string(index=False)}"
        text_prompt = f"Based on the provided insights, can you analyze and summarize the key trends in the predicted stock prices?"
        completion = client.chat.completions.create(
            model='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text_prompt}],
            temperature=0.7,
        )
        text_response = completion.choices[0].message.content
        print(text_response)

        # Generate visual response using generative AI
        plt.figure(figsize=(10, 6))
        plt.plot(future_predictions_df['Date'], future_predictions_df['Predicted Stock Price'])
        plt.xlabel('Date')
        plt.ylabel('Predicted Stock Price')
        plt.title('Predicted Stock Prices')
        plt.show()
    else:
        # Generate text response using LM Studio
        new_data = data.iloc[-30:][features]
        future_predictions = model.predict(new_data)

        # Analyze predictions
        future_predictions_df = pd.DataFrame({'Date': data.iloc[-30:]['Date'], 'Predicted Stock Price': future_predictions})
        future_predictions_df['Trend'] = future_predictions_df['Predicted Stock Price'].diff().apply(lambda x: 'Increasing' if x > 0 else 'Decreasing')
        increasing_predictions = future_predictions_df[future_predictions_df['Trend'] == 'Increasing']
        decreasing_predictions = future_predictions_df[future_predictions_df['Trend'] == 'Decreasing']

        # Generate text response using LM Studio
        system_prompt = f"Here are the insights about the predicted stock prices for the next 30 days:\n\nPredicted Stock Prices:\n{future_predictions_df.to_string(index=False)}\n\nIncreasing Predictions:\n{increasing_predictions.to_string(index=False)}\n\nDecreasing Predictions:\n{decreasing_predictions.to_string(index=False)}"
        text_prompt = input("Enter your queries: ")
        completion = client.chat.completions.create(
            model='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text_prompt}],
            temperature=0.7,
        )
        text_response = completion.choices[0].message.content
        print(text_response)