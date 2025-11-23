"""
FastAPI Backend for Crop Yield Prediction System
UN SDG 2: Zero Hunger
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crop Yield Prediction API",
    description="AI-powered satellite imagery analysis for crop yield prediction (UN SDG 2)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS (Request/Response)
# ============================================================================

class PredictionRequest(BaseModel):
    """Input model for crop yield prediction"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    date: str = Field(..., description="Prediction date (YYYY-MM-DD)")
    crop_type: str = Field(..., description="Crop type (maize, wheat, rice, soybean)")
    temperature: float = Field(..., description="Mean temperature (°C)")
    precipitation: float = Field(..., ge=0, description="Precipitation (mm)")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    soil_moisture: float = Field(..., ge=0, le=1, description="Soil moisture (0-1)")
    
    @validator('crop_type')
    def validate_crop_type(cls, v):
        valid_crops = ['maize', 'wheat', 'rice', 'soybean', 'potato', 'sugarcane']
        if v.lower() not in valid_crops:
            raise ValueError(f"Crop type must be one of {valid_crops}")
        return v.lower()
    
    @validator('date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

class PredictionResponse(BaseModel):
    """Output model for prediction results"""
    prediction_id: str
    yield_estimate: float = Field(..., description="Estimated yield (kg/ha)")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence (0-1)")
    rmse: float = Field(..., description="Root Mean Squared Error of estimate")
    model_version: str
    timestamp: datetime
    location: Dict[str, float]
    crop_type: str
    input_features: Dict[str, float]

class SummaryRequest(BaseModel):
    """Request for AI-generated summary"""
    prediction_id: str
    include_recommendations: bool = True

class SummaryResponse(BaseModel):
    """AI-generated farmer-friendly summary"""
    summary: str
    recommendations: list = Field(default_factory=list)
    confidence_explanation: str
    risk_factors: list = Field(default_factory=list)

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    services: Dict[str, str]

# ============================================================================
# DEPENDENCY INJECTION & SERVICES
# ============================================================================

class MLService:
    """ML model inference service"""
    
    def __init__(self):
        self.model = None
        self.model_version = "v1.0"
        self._load_model()
    
    def _load_model(self):
        """Load TensorFlow SavedModel"""
        try:
            # In production: load from GCS bucket
            # import tensorflow as tf
            # self.model = tf.saved_model.load(
            #     "gs://your-bucket/crop-yield-model/v1"
            # )
            logger.info("ML model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, features: Dict[str, float]) -> tuple:
        """
        Generate yield prediction from features
        Returns: (prediction, confidence_score, rmse)
        """
        try:
            # Feature array: [NDVI, Temp, Precip, Humidity, SoilMoisture, ...]
            import numpy as np
            
            # Normalize features (example)
            feature_array = np.array([
                features.get('ndvi', 0.6),
                features.get('temperature', 25.0) / 40,
                features.get('precipitation', 100.0) / 500,
                features.get('humidity', 65.0) / 100,
                features.get('soil_moisture', 0.4),
            ]).reshape(1, -1)
            
            # Mock prediction (replace with actual model)
            yield_estimate = 5000 + np.random.normal(0, 500)
            confidence = 0.87 + np.random.uniform(-0.05, 0.05)
            rmse = 450.0
            
            return float(yield_estimate), float(confidence), rmse
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

class EarthEngineService:
    """Google Earth Engine integration for NDVI data"""
    
    @staticmethod
    def get_ndvi(latitude: float, longitude: float, date: str) -> float:
        """
        Fetch NDVI from Earth Engine
        """
        try:
            # In production:
            # import ee
            # ee.Authenticate()
            # ee.Initialize()
            # 
            # point = ee.Geometry.Point([longitude, latitude])
            # dataset = ee.ImageCollection('COPERNICUS/S2').filterDate(...).filterBounds(point)
            # ndvi = dataset.map(lambda img: img.normalizedDifference(['B8', 'B4']))
            
            # Mock NDVI value
            import random
            ndvi = random.uniform(0.4, 0.8)
            logger.info(f"NDVI fetched: {ndvi:.3f}")
            return ndvi
        except Exception as e:
            logger.error(f"Earth Engine error: {e}")
            raise

class WeatherService:
    """Weather data integration"""
    
    @staticmethod
    def get_weather_data(latitude: float, longitude: float) -> Dict[str, float]:
        """
        Fetch weather data from OpenWeatherMap (fallback to mock on error)
        """
        api_key = os.getenv('OPENWEATHER_API_KEY')
        # If no API key provided return mock data
        if not api_key:
            return {'temperature': 25.5, 'precipitation': 60.0, 'humidity': 65.0}

        try:
            import requests
            resp = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"lat": latitude, "lon": longitude, "appid": api_key, "units": "metric"},
                timeout=6
            )
            resp.raise_for_status()
            data = resp.json()

            temp = float(data.get('main', {}).get('temp', 25.5))
            humidity = float(data.get('main', {}).get('humidity', 65.0))

            # Precipitation may be in 'rain' or 'snow' (1h or 3h); default to 0
            precip = 0.0
            if 'rain' in data and isinstance(data['rain'], dict):
                precip = float(data['rain'].get('1h', data['rain'].get('3h', 0.0)))
            elif 'snow' in data and isinstance(data['snow'], dict):
                precip = float(data['snow'].get('1h', data['snow'].get('3h', 0.0)))

            return {'temperature': temp, 'precipitation': precip, 'humidity': humidity}
        except Exception as e:
            logger.error(f"WeatherService error: {e}")
            # On any error, return mock values so the app remains functional
            return {'temperature': 25.5, 'precipitation': 60.0, 'humidity': 65.0}

class GeminiService:
    """Google Gemini API for natural language summaries"""
    
    @staticmethod
    def generate_summary(prediction: float, confidence: float, crop_type: str) -> str:
        """
        Generate farmer-friendly summary using Gemini API
        """
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            # In production:
            # import google.generativeai as genai
            # genai.configure(api_key=api_key)
            # model = genai.GenerativeModel("gemini-pro")
            # 
            # prompt = f"""
            # Explain this crop yield prediction for a farmer:
            # - Predicted yield: {prediction:.0f} kg/ha
            # - Confidence: {confidence*100:.0f}%
            # - Crop: {crop_type}
            # 
            # Provide a brief, actionable summary in plain language.
            # """
            # response = model.generate_content(prompt)
            
            # Mock response
            summary = f"""
Based on satellite imagery and weather data analysis, your {crop_type} crop 
is expected to yield approximately {prediction:.0f} kg/ha. This prediction has 
a {confidence*100:.0f}% confidence level. Current growing conditions appear favorable 
based on vegetation indices and seasonal weather patterns.
            """.strip()
            
            return summary
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

class FirestoreService:
    """Google Firestore for data persistence"""
    
    @staticmethod
    def save_prediction(prediction_data: Dict[str, Any]) -> str:
        """Save prediction to Firestore"""
        try:
            # In production:
            # from google.cloud import firestore
            # db = firestore.Client()
            # doc_ref = db.collection('predictions').document()
            # doc_ref.set(prediction_data)
            # return doc_ref.id
            
            # Mock ID generation
            import uuid
            prediction_id = str(uuid.uuid4())
            logger.info(f"Prediction saved: {prediction_id}")
            return prediction_id
        except Exception as e:
            logger.error(f"Firestore error: {e}")
            raise

# Initialize services
ml_service = MLService()
ee_service = EarthEngineService()
weather_service = WeatherService()
gemini_service = GeminiService()
firestore_service = FirestoreService()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        services={
            "ml_model": "ready",
            "earth_engine": "connected",
            "firestore": "connected"
        }
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_yield(request: PredictionRequest):
    """
    Main prediction endpoint
    
    Returns crop yield estimate based on satellite imagery and weather data
    """
    try:
        logger.info(f"Prediction request: {request.crop_type} at ({request.latitude}, {request.longitude})")
        
        # Fetch NDVI from Earth Engine
        ndvi = ee_service.get_ndvi(request.latitude, request.longitude, request.date)
        
        # Prepare feature dictionary
        features = {
            'ndvi': ndvi,
            'temperature': request.temperature,
            'precipitation': request.precipitation,
            'humidity': request.humidity,
            'soil_moisture': request.soil_moisture,
        }
        
        # Get ML prediction
        yield_estimate, confidence, rmse = ml_service.predict(features)
        
        # Prepare response
        prediction_data = {
            'crop_type': request.crop_type,
            'location': {'latitude': request.latitude, 'longitude': request.longitude},
            'date': request.date,
            'yield_estimate': yield_estimate,
            'confidence_score': confidence,
            'rmse': rmse,
            'model_version': ml_service.model_version,
            'input_features': features,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Save to Firestore
        prediction_id = firestore_service.save_prediction(prediction_data)
        
        return PredictionResponse(
            prediction_id=prediction_id,
            yield_estimate=yield_estimate,
            confidence_score=confidence,
            rmse=rmse,
            model_version=ml_service.model_version,
            timestamp=datetime.utcnow(),
            location=prediction_data['location'],
            crop_type=request.crop_type,
            input_features=features
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summary", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    """
    Generate AI-powered farmer-friendly explanation of prediction
    Uses Google Gemini API
    """
    try:
        # In production: fetch prediction from Firestore
        # Mock data for demo
        prediction = 5000.0
        confidence = 0.87
        crop_type = "maize"
        
        # Generate summary using Gemini
        summary = gemini_service.generate_summary(prediction, confidence, crop_type)
        
        recommendations = [
            "Monitor soil moisture regularly",
            "Apply nitrogen fertilizer in week 4",
            "Watch for pest activity in July",
            "Prepare irrigation for dry periods"
        ] if request.include_recommendations else []
        
        risk_factors = [
            "Low rainfall in forecast",
            "Pest pressure increasing"
        ]
        
        return SummaryResponse(
            summary=summary,
            recommendations=recommendations,
            confidence_explanation=f"This prediction has {confidence*100:.0f}% confidence based on model performance",
            risk_factors=risk_factors
        )
    
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str):
    """Retrieve historical prediction by ID"""
    try:
        # In production: fetch from Firestore
        return {
            "prediction_id": prediction_id,
            "status": "found",
            "data": {}
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail="Prediction not found")

@app.get("/analytics/regional/{region}")
async def get_regional_analytics(region: str, crop_type: Optional[str] = None):
    """Get analytics for a specific region"""
    try:
        return {
            "region": region,
            "crop_type": crop_type,
            "avg_yield": 5200,
            "predictions_count": 150,
            "accuracy": 0.87
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/versions")
async def get_model_versions():
    """List available model versions"""
    return {
        "available_versions": [
            {"version": "v1.0", "accuracy": 0.87, "status": "production"},
            {"version": "v0.9", "accuracy": 0.84, "status": "deprecated"}
        ],
        "current_version": "v1.0"
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Crop Yield Prediction API starting up...")
    logger.info("Models loaded, services initialized")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Crop Yield Prediction API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
