from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Tuple
from app.router import SafePathRouter
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",  
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

csv_file_name = "NYPD.csv"

current_file_dir = Path(__file__).parent

crime_data_path = current_file_dir / "app" / csv_file_name

# Load your crime data once at startup
router = SafePathRouter(crime_data_path=crime_data_path)
#router = SafePathRouter(crime_data_path="app/NYPD.csv")

class CoordinateTuple(BaseModel):
    latitude: float
    longitude: float

class RouteRequest(BaseModel):
    start: Tuple[float, float]  # (latitude, longitude)
    end: Tuple[float, float]
    time: Optional[str] = None  # Format: "HH:MM"

@app.post("/route")
def get_safe_route(req: RouteRequest):
    try:
        time = datetime.now()
        if req.time:
            try:
                hour, minute = map(int, req.time.split(":"))
                time = time.replace(hour=hour, minute=minute)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM.")
        
        result = router.generate_safe_route(req.start, req.end, time)
        result['request_time'] = time.strftime('%H:%M')
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating route: {e}")
