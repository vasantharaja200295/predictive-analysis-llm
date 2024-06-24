import csv
import random
from datetime import datetime, timedelta

# Define the start and end dates
start_date = datetime(2018, 1, 2)
end_date = datetime(2022, 12, 31)

# Define the range for stock prices and volume
open_price_range = (100, 135)
price_range = (95, 140)
volume_range = (5000000, 10000000)

# Function to generate a random stock price within the specified range
def generate_price(price_range):
    return round(random.uniform(price_range[0], price_range[1]), 2)

# Function to generate a random trading volume within the specified range
def generate_volume(volume_range):
    return random.randint(volume_range[0], volume_range[1])

# Create an empty list to store the data
data = []

# Generate data for each trading day
current_date = start_date
while current_date <= end_date:
    # Skip weekends
    if current_date.weekday() >= 5:
        current_date += timedelta(days=1)
        continue

    # Generate stock prices and volume for the current date
    open_price = generate_price(open_price_range)
    high_price = generate_price(price_range)
    low_price = generate_price(price_range)
    close_price = generate_price(price_range)
    volume = generate_volume(volume_range)

    # Ensure that the open price is within the high-low range
    open_price = max(min(open_price, high_price), low_price)

    # Ensure that the close price is within the high-low range
    close_price = max(min(close_price, high_price), low_price)

    # Add the data to the list
    data.append([current_date.strftime('%Y-%m-%d'), open_price, high_price, low_price, close_price, volume])

    # Move to the next trading day
    current_date += timedelta(days=1)

# Write the data to a CSV file
with open('market_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    writer.writerows(data)

print('Dataset generated successfully!')