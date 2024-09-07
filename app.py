from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# تعریف DataFrameSelector
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values

# تعریف CombinedAttributesAdder
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, 3] / X[:, 6]  # rooms_ix = 3, household_ix = 6
        population_per_household = X[:, 5] / X[:, 6]  # population_ix = 5, household_ix = 6
        bedrooms_per_room = X[:, 4] / X[:, 3]  # bedrooms_ix = 4, rooms_ix = 3
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

app = Flask(__name__)

# بارگذاری مدل ذخیره‌شده
model = joblib.load('linear_regression_model.pkl')

# بارگذاری pipeline پیش‌پردازش داده‌ها
pipeline = joblib.load('data_preprocessing_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict_home_price():
    try:
        # دریافت داده‌های ورودی از درخواست
        data = request.json

        # تبدیل داده‌های ورودی به DataFrame
        input_data = pd.DataFrame({
            'longitude': [data['longitude']],
            'latitude': [data['latitude']],
            'housing_median_age': [30],  # مقدار فرضی معقول
            'total_rooms': [3000],       # مقدار فرضی معقول
            'total_bedrooms': [500],     # مقدار فرضی معقول
            'population': [1500],        # مقدار فرضی معقول
            'households': [500],         # مقدار فرضی معقول
            'median_income': [6],        # مقدار فرضی معقول
            'ocean_proximity': ['NEAR BAY']  # مقدار دسته‌ای فرضی
        })

        # پیش‌پردازش داده‌های جدید با استفاده از pipeline
        input_data_prepared = pipeline.transform(input_data)

        # پیش‌بینی قیمت با استفاده از مدل
        predicted_price = model.predict(input_data_prepared)

        # برگرداندن نتیجه به صورت JSON
        return jsonify({'predicted_price': predicted_price[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
