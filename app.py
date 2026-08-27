from flask import Flask, request, render_template
import pandas as pd
from src.Airbnb.pipeline.prediction_pipline import PredictPipeline

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_datapoint():
    if request.method == "POST":
        # 1. Collect inputs from HTML form
        data = {
            "property_type": [request.form.get("property_type")],
            "room_type": [request.form.get("room_type")],
            "amenities": [int(request.form.get("amenities", 0))],
            "accommodates": [int(request.form.get("accommodates", 1))],
            "bathrooms": [float(request.form.get("bathrooms", 1.0))],
            "bed_type": [request.form.get("bed_type")],
            "cancellation_policy": [request.form.get("cancellation_policy")],
            "cleaning_fee": [request.form.get("cleaning_fee")],
            "city": [request.form.get("city")],
            "host_has_profile_pic": [request.form.get("host_has_profile_pic")],
            "host_identity_verified": [request.form.get("host_identity_verified")],
            "host_response_rate": [float(request.form.get("host_response_rate", 100))],
            "instant_bookable": [request.form.get("instant_bookable")],
            "bedrooms": [float(request.form.get("bedrooms", 1))],
            "beds": [float(request.form.get("beds", 1))],
            "latitude": [float(request.form.get("latitude", 40.7128))],
            "longitude": [float(request.form.get("longitude", -74.0060))],
            "number_of_reviews": [int(request.form.get("number_of_reviews", 0))],
            "review_scores_rating": [
                float(request.form.get("review_scores_rating", 95.0))
            ],
        }

        # 2. Convert to DataFrame
        pred_df = pd.DataFrame(data)

        # 3. Predict price
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template("index.html", results=round(results[0], 2))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
