from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model_path = 'models/linear_regression_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')  # This will render a template from the templates folder

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    input_features = [float(x) for x in request.form.values()]
    features_value = np.array(input_features).reshape(1, -1)
    
    # Use model to predict
    prediction = model.predict(features_value)
    
    # Render the result
    return render_template('result.html', prediction_text='The predicted weight of the fish is: {:.2f} grams'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
