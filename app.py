from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

# Load the dataset and model
df = pd.read_csv('Cleaned_Car_data.csv')
model = pickle.load(open('model_car.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    companies = sorted(df['company'].unique())
    years = sorted(df['year'].unique(), reverse=True)
    fuel_types = sorted(df['fuel_type'].unique())
    return render_template('index.html', companies=companies, years=years, fuel_types=fuel_types)

@app.route('/get_models', methods=['POST'])
def get_models():
    selected_company = request.json.get('company')
    if not selected_company:
        return jsonify([])  # Return an empty list if no company is selected
    models = sorted(df[df['company'] == selected_company]['name'].unique())
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        company = data['company']
        model_name = data['name']
        year = int(data['year'])
        fuel_type = data['fuel_type']
        kms_driven = int(data['km'])  # Updated to match the model's column name

        # Perform prediction using the model
        input_data = pd.DataFrame([[company, model_name, year, fuel_type, kms_driven]],
                                  columns=['company', 'name', 'year', 'fuel_type', 'kms_driven'])
        prediction = model.predict(input_data)[0]
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return f"Error: {e}", 400  # For debugging errors in the form or model

if __name__ == '__main__':
    app.run(debug=True)
