import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
fish_df = pd.read_csv('Fish.csv')

# Define features and target variable
X = fish_df.drop(['Species', 'Weight'], axis=1)
y = fish_df['Weight']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Save the model to a file using pickle
model_filename = 'linear_regression_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(linear_model, file)

print(f"Model saved to {model_filename}")
