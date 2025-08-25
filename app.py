from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import datetime
import pickle
import joblib


location_map = {
    "24": "Adelaide",
    "7": "Albany",
    "30": "Albury",
    "46": "AliceSprings",
    "33": "BadgerysCreek",
    "14": "Ballarat",
    "36": "Bendigo",
    "21": "Brisbane",
    "2": "Cairns",
    "43": "Cobar",
    "9": "CoffsHarbour",
    "4": "Dartmoor",
    "11": "Darwin",
    "15": "GoldCoast",
    "17": "Hobart",
    "45": "Katherine",
    "23": "Launceston",
    "28": "Melbourne",
    "25": "Melbourne Airport",
    "44": "Mildura",
    "42": "Moree",
    "5": "MountGambier",
    "12": "MountGinini",
    "19": "Newcastle",
    "47": "Nhil",
    "13": "NorahHead",
    "6": "NorfolkIsland",
    "32": "Nuriootpa",
    "40": "PearceRAAF",
    "31": "Penrith",
    "26": "Perth",
    "35": "Perth Airport",
    "1": "Portland",
    "37": "Richmond",
    "27": "Sale",
    "41": "Salmon Gums",
    "10": "Sydney",
    "16": "Sydney Airport",
    "39": "Townsville",
    "34": "Tuggeranong",
    "49": "Uluru",
    "38": "WaggaWagga",
    "3": "Walpole",
    "18": "Watsonia",
    "22": "William Town",
    "8": "Witchcliffe",
    "20": "Wollongong",
    "48": "Woomera"
}



with open("forecast_models.pkl", "rb") as f:
    forecast_models = pickle.load(f)
print("Forecast models loaded.")
print(forecast_models.keys())

clf = pickle.load(open("rain_classifier.pkl", "rb"))


with open("xgb_columns.pkl", "rb") as f:
    X_columns = pickle.load(f)


app = Flask(__name__, template_folder="template")
model = pickle.load(open("./cat.pkl", "rb"))
print("Model Loaded")



def forecast_weather(location, future_date):
    predicted = {}
    for feature, model in forecast_models[location].items():
        future_df = pd.DataFrame({'ds': [future_date]})
        forecast = model.predict(future_df)
        predicted[feature] = forecast['yhat'].iloc[0]
    return predicted

def predict_future_rain(location, future_date):
    # Step 1: Forecast weather features
    predicted_weather = forecast_weather(location, future_date)
    
    # Step 2: Create input row for classifier
    input_df = pd.DataFrame([predicted_weather])
    
    # Add missing categorical dummy columns (set them to 0)
    for col in X_columns :
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Activate the location column
    loc_col = f'Location_{location}'
    if loc_col in X_columns :
        input_df[loc_col] = 1
    
    # ⚡ Key Fix: Reindex input_df to match training columns
    input_df = input_df.reindex(columns=X_columns , fill_value=0)
    
    # Step 3: Predict rain probability
    rain_prob = clf.predict_proba(input_df)[0][1]
    return rain_prob



@app.route("/",methods=['GET'])
@cross_origin()
def home():
	return render_template("index.html")

@app.route("/predictorloc", methods=['GET', 'POST'])
def predictorloc():
    if request.method == "POST":
        date = request.form['date']
        location_id = request.form['location']  # This is "30", "10", etc.
        
        # Map ID → Name
        location_name = location_map.get(location_id)
        if location_name is None:
            return "Invalid location selected!"
        
        prob = predict_future_rain(location_name, date)
        
        if prob >= 0.5:
            return render_template("after_rainy.html", probability=prob)
        else:
            return render_template("after_sunny.html", probability=prob)
    
    return render_template("predictorloc.html")




@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
	if request.method == "POST":
		# DATE
		date = request.form['date']
		day = float(pd.to_datetime(date, format="%Y-%m-%d").day)
		month = float(pd.to_datetime(date, format="%Y-%m-%d").month)
		year = float(pd.to_datetime(date, format="%Y-%m-%d").year)
		# MinTemp
		minTemp = float(request.form['mintemp'])
		# MaxTemp
		maxTemp = float(request.form['maxtemp'])
		# Rainfall
		rainfall = float(request.form['rainfall'])
		# Evaporation
		evaporation = float(request.form['evaporation'])
		# Sunshine
		sunshine = float(request.form['sunshine'])
		# Wind Gust Speed
		windGustSpeed = float(request.form['windgustspeed'])
		# Wind Speed 9am
		windSpeed9am = float(request.form['windspeed9am'])
		# Wind Speed 3pm
		windSpeed3pm = float(request.form['windspeed3pm'])
		# Humidity 9am
		humidity9am = float(request.form['humidity9am'])
		# Humidity 3pm
		humidity3pm = float(request.form['humidity3pm'])
		# Pressure 9am
		pressure9am = float(request.form['pressure9am'])
		# Pressure 3pm
		pressure3pm = float(request.form['pressure3pm'])
		# Temperature 9am
		temp9am = float(request.form['temp9am'])
		# Temperature 3pm
		temp3pm = float(request.form['temp3pm'])
		# Cloud 9am
		cloud9am = float(request.form['cloud9am'])
		# Cloud 3pm
		cloud3pm = float(request.form['cloud3pm'])
		# Cloud 3pm
		location = float(request.form['location'])
		# Wind Dir 9am
		winddDir9am = float(request.form['winddir9am'])
		# Wind Dir 3pm
		winddDir3pm = float(request.form['winddir3pm'])
		# Wind Gust Dir
		windGustDir = float(request.form['windgustdir'])
		# Rain Today
		rainToday = float(request.form['raintoday'])

		input_lst = [location , minTemp , maxTemp , rainfall , evaporation , sunshine ,
					 windGustDir , windGustSpeed , winddDir9am , winddDir3pm , windSpeed9am , windSpeed3pm ,
					 humidity9am , humidity3pm , pressure9am , pressure3pm , cloud9am , cloud3pm , temp9am , temp3pm ,
					 rainToday , month , day , year]
		pred = model.predict(input_lst)
		output = pred
		if output == 0:
			return render_template("after_sunny.html")
		else:
			return render_template("after_rainy.html")
	return render_template("predictor.html")

if __name__=='__main__':

	app.run(debug=True)
