from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
class WeatherForecast(BaseModel):
    temperature: float
    precipitation: float
    humidity: int
    wind_speed: float

