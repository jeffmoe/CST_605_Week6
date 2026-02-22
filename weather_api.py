from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
class WeatherForecast(BaseModel):
    temperature: float
    precipitation: float
    humidity: int
    wind_speed: float

@app.get("/forecast")
async def get_forecast(latitude: float, longitude: float):
    forecast = WeatherForecast(temperature=25.5, precipitation=0.2, humidity=78, wind_speed=5.5) 
    return forecast

@app.get("/city")
async def get_city(temp: float, precip: float):
    city = WeatherForecast(temperature=25.5, 
                precipitation=0.2, 
                humidity=78, 
                wind_speed=5.5)
    return city