from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import numpy as np
import torch
import threading
from contextlib import asynccontextmanager

# Import your database components
from database import SessionLocal, ClickedPoint
# Import your models (assuming they are in models.py or defined above)
from models import KDEModel, NNModel  

# Lock to prevent concurrent training issues with global models
model_lock = threading.Lock()

# Initialize global instances of your models
kde_model = KDEModel(bandwidth=0.04)
nn_model = NNModel(bandwidth=0.03, grid_res=40)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load old datapoints from the database on startup
    db = SessionLocal()
    try:
        points = db.query(ClickedPoint).all()
        if len(points) >= 2:
            coords_np = np.array([[p.x, p.y] for p in points], dtype=np.float32)
            coords_torch = torch.tensor(coords_np, dtype=torch.float32)
            with model_lock:
                kde_model(coords_np)
                nn_model.train(coords_torch, epochs=20)
    finally:
        db.close()
    yield

app = FastAPI(title="Spatial Density API", lifespan=lifespan)

# Enable CORS so your Django frontend can communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your Django domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas for data validation
class PointCreate(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)

class PointResponse(BaseModel):
    id: int
    x: float
    y: float

    class Config:
        from_attributes = True

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/points/", response_model=PointResponse)
def add_point(point: PointCreate, db: Session = Depends(get_db)):
    """Registers a new clicked point coordinate."""
    db_point = ClickedPoint(x=point.x, y=point.y)
    db.add(db_point)
    db.commit()
    db.refresh(db_point)
    return db_point


@app.get("/points/", response_model=list[PointResponse])
def get_points(db: Session = Depends(get_db)):
    """Returns all registered points."""
    return db.query(ClickedPoint).all()


@app.post("/train-and-evaluate/")
def train_and_evaluate(grid_size: int = 100, db: Session = Depends(get_db)):
    points = db.query(ClickedPoint).all()
    
    if len(points) < 2:
        return {"status": "insufficient_data", "kde": None, "nn": None}

    coords_np = np.array([[p.x, p.y] for p in points], dtype=np.float32)
    coords_torch = torch.tensor(coords_np, dtype=torch.float32)

    with model_lock:
        # 1. Process KDE
        kde_model(coords_np)
        kde_X, kde_Y, kde_Z = kde_model.visualize(grid_size=grid_size)
    
        # 2. Process NN
        nn_model.train(coords_torch, epochs=5) 
        nn_X, nn_Y, nn_Z = nn_model.visualize(grid_size=grid_size)

    # Send clean 1D axis arrays along with the 2D Z density matrix
    return {
        "status": "success",
        "total_points": len(points),
        "axis_range": np.linspace(0, 1, grid_size).tolist(),  # Clean [0, 1] 1D array
        "kde_Z": kde_Z.tolist(),
        "nn_Z": nn_Z.tolist()
    }


@app.delete("/points/clear/")
def clear_points(db: Session = Depends(get_db)):
    """Clears the database to restart the experiment."""
    db.query(ClickedPoint).delete()
    db.commit()
    return {"message": "All points cleared successfully."}


if __name__ == "__main__":
    import uvicorn
    # This matches the terminal execution format
    uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=True)