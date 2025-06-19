from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Tuple
from app.router import SafePathRouter

app = FastAPI()

# Load your crime data once at startup
router = SafePathRouter(crime_data_path="app/NYPD.csv")

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
