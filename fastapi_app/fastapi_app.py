from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

import os
import pickle

MODEL_FILE = os.getenv('MODEL_FILE', 'trees_regression.bin')

with open(MODEL_FILE, 'rb') as f_in:
    model = pickle.load(f_in)

rides = {
	0 : {
	'PULocationID' : '43',
	'DOLocationID' : '151',
	'passenger_count' : '1.0',
	'trip_distance' : '1.01',
	'fare_forecast' : None
	}
}

class Ride(BaseModel):
	PULocationID: int
	DOLocationID: int
	passenger_count: int
	trip_distance: float
	fare_forecast: Optional[float] = None

	def to_list(self):
		return [Ride.PULocationID, Ride.DOLocationID, Ride.passenger_count, Ride.trip_distance]

app = FastAPI()

@app.get("/get-ride/{ride_id}")
def get_ride(ride_id : int):
	if ride_id in rides.keys():
		return rides[ride_id]
	else: 
		return "Ride is not found"

@app.post("/add-ride/{ride_id}")
def add_ride(ride_id: int, ride: Ride):
	if ride_id in rides.keys():
		return {"Error":"The ride is already exits"}

	rides[ride_id] = ride
	return rides[ride_id]


@app.post("/add-and-predict-ride/{ride_id}")
def add_predict_ride(ride_id: int, ride: Ride):
	if ride_id in rides.keys():
		return {"Error":"The ride is already exist"}

	result = model.predict([[ride.PULocationID, ride.DOLocationID, ride.passenger_count, ride.trip_distance]])[0]

	rides[ride_id] = {
		"PULocationID": ride.PULocationID, 
		"DOLocationID": ride.DOLocationID, 
		"passenger_count": ride.passenger_count, 
		"trip_distance": ride.trip_distance,
		"fare_forecast": result
		}

	return rides[ride_id]

#@app.post("/predict-ride/{ride_id}")
#def predict_ride(ride_id: int):
#	if ride_id not in rides.keys():
#		return {"Error":"The ride does not exist"}
#
#	result = model.predict([rides[ride_id][PULocationID", "DOLocationID", "passenger_count", "trip_distance"]])[0]
#	rides[ride_id].fare_forecast = result
#	return rides[ride_id]







