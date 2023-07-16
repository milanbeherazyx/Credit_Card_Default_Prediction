from flask import Flask, request, render_template
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    """
    Render the home page.
    """
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handle the prediction request.
    """
    if request.method == 'GET':
        return render_template('home.html')
    # Get the input data from the form
    try:
        data = CustomData(
            LIMIT_BAL=float(request.form.get('LIMIT_BAL')),
            SEX=int(request.form.get('SEX')),
            EDUCATION=int(request.form.get('EDUCATION')),
            MARRIAGE=int(request.form.get('MARRIAGE')),
            AGE=int(request.form.get('AGE')),
            PAY_0=int(request.form.get('PAY_0')),
            PAY_2=int(request.form.get('PAY_2')),
            PAY_3=int(request.form.get('PAY_3')),
            PAY_4=int(request.form.get('PAY_4')),
            PAY_5=int(request.form.get('PAY_5')),
            PAY_6=int(request.form.get('PAY_6')),
            BILL_AMT1=float(request.form.get('BILL_AMT1')),
            BILL_AMT2=float(request.form.get('BILL_AMT2')),
            BILL_AMT3=float(request.form.get('BILL_AMT3')),
            BILL_AMT4=float(request.form.get('BILL_AMT4')),
            BILL_AMT5=float(request.form.get('BILL_AMT5')),
            BILL_AMT6=float(request.form.get('BILL_AMT6')),
            PAY_AMT1=float(request.form.get('PAY_AMT1')),
            PAY_AMT2=float(request.form.get('PAY_AMT2')),
            PAY_AMT3=float(request.form.get('PAY_AMT3')),
            PAY_AMT4=float(request.form.get('PAY_AMT4')),
            PAY_AMT5=float(request.form.get('PAY_AMT5')),
            PAY_AMT6=float(request.form.get('PAY_AMT6'))
        )
    except (TypeError, ValueError):
        # Invalid input data, handle the error
        return render_template('home.html', error='Invalid input data')

    pred_df = data.get_data_as_data_frame()
    print(pred_df)

    predict_pipeline = PredictPipeline()

    try:
        results = predict_pipeline.predict(pred_df)
        result_value = "Defaulter" if results[0] == 1 else "Non-Defaulter"
    except Exception as e:
        # Handle the prediction error
        return render_template('home.html', error='Prediction failed')

    return render_template('home.html', results=result_value)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
