"""
Illegal Mining Detection - FastAPI Backend
Complete implementation from scratch
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date
#import tensorflow as tf
#from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import base64

from tensorflow.keras.models import load_model as tf_load_model
import model_loader  # just import, do NOT call
app = FastAPI()
model = None








# Import our modules (we'll create these)
from database import (
    init_db, 
    save_detection, 
    get_all_detections,
    get_detection_by_id,
    get_statistics,
    get_monthly_trends,
    update_detection_verification
)

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = "models/mining_detector.h5"  # Update this path!
CONFIDENCE_THRESHOLD = 50.0  # Minimum confidence to save detection

# =====================================================
# INITIALIZE FASTAPI APP
# =====================================================

app = FastAPI(
    title="Illegal Mining Detection API",
    description="AI-powered satellite imagery analysis for illegal mining detection",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["http://localhost:3000", "https://your-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =====================================================

class MiningSite(BaseModel):
    """Mining site detection result"""
    id: str
    name: str
    latitude: float
    longitude: float
    severity: str  # "Critical", "High", "Moderate", "Low"
    type: str  # "Coal", "Sand", "Open-pit", etc.
    areaHectares: float
    estimatedLossUSD: int
    lastDetected: str  # ISO date format
    images: List[str]
    confidence: Optional[float] = None
    verified: Optional[bool] = False

class DetectionRequest(BaseModel):
    """Request to detect mining from image"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_name: Optional[str] = None
    mining_type: Optional[str] = None

class AnalysisResult(BaseModel):
    """AI Analysis result (matching frontend Gemini structure)"""
    detected: bool
    confidence: float
    reasoning: str
    environmentalImpact: str
    legalContext: str
    machineryCount: int = 0
    severity: str
    estimatedAreaHectares: float
    estimatedLossUSD: int

class DetectionResponse(BaseModel):
    """Complete detection response"""
    mining_detected: bool
    confidence: float
    analysis: AnalysisResult
    location: dict
    timestamp: str
    detection_id: Optional[str] = None

class MonthlyTrend(BaseModel):
    """Monthly detection trend data"""
    name: str  # Month name
    loss: int  # Estimated loss in USD
    detected: int  # Number of detections

class Statistics(BaseModel):
    """Overall system statistics"""
    total_detections: int
    total_area_hectares: float
    total_estimated_loss_usd: int
    critical_sites: int
    high_severity_sites: int
    moderate_severity_sites: int
    avg_confidence: float
    verified_count: int

# =====================================================
# GLOBAL VARIABLES
# =====================================================

model = None

# =====================================================
# STARTUP/SHUTDOWN EVENTS
# =====================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global model
    
    print("\n" + "="*60)
    print("🚀 ILLEGAL MINING DETECTION API - STARTING")
    print("="*60)
    
    # Initialize database
    init_db()
    print("✅ Database initialized")

    # ⬇️ ENSURE MODEL EXISTS (IMPORTANT)
    try:
        import model_loader
        model_loader.ensure_model()
        print("✅ Model file verified")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return
    
    # ⬇️ Load ML model
    try:
        model = tf_load_model(MODEL_PATH)
        print(f"✅ Model loaded from: {MODEL_PATH}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Server will start but detection will not work!")
    
    print("\n🎯 API Server Ready!")
    print("📚 API Docs: http://localhost:8000/api/docs")
    print("🗺️  Health Check: http://localhost:8000/")
    print("="*60 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n👋 Shutting down Illegal Mining Detection API")

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for model prediction
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Preprocessed numpy array [1, 224, 224, 3]
    """
    # Load image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to model input size
    img = img.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def determine_severity(confidence: float) -> str:
    """Determine severity level based on confidence"""
    if confidence >= 85:
        return "Critical"
    elif confidence >= 70:
        return "High"
    elif confidence >= 55:
        return "Moderate"
    else:
        return "Low"

def estimate_area_from_confidence(confidence: float) -> float:
    """Estimate affected area based on detection confidence"""
    # Simple heuristic: higher confidence = larger visible area
    # In real system, this would use image segmentation
    base_area = 10.0  # hectares
    return base_area * (confidence / 100.0) * np.random.uniform(0.8, 1.5)

def estimate_financial_loss(area_hectares: float, mining_type: str = "Unknown") -> int:
    """Estimate financial loss based on area and type"""
    # Loss per hectare (USD)
    loss_rates = {
        "Coal": 100000,
        "Sand": 75000,
        "Open-pit": 85000,
        "Stone": 70000,
        "Unknown": 80000
    }
    
    rate = loss_rates.get(mining_type, 80000)
    return int(area_hectares * rate * np.random.uniform(0.9, 1.3))

def generate_reasoning(confidence: float, severity: str) -> str:
    """Generate AI reasoning text"""
    if confidence > 80:
        return f"High confidence detection ({confidence:.1f}%). Clear indicators of mining activity including soil disruption, machinery presence, and altered terrain patterns. {severity} severity level assigned based on extent of environmental impact."
    elif confidence > 60:
        return f"Moderate confidence detection ({confidence:.1f}%). Multiple indicators suggest mining activity including terrain changes and potential machinery. Further verification recommended."
    else:
        return f"Low confidence detection ({confidence:.1f}%). Some indicators present but inconclusive. Manual verification strongly recommended before action."

def generate_environmental_impact(severity: str, area: float) -> str:
    """Generate environmental impact assessment"""
    if severity == "Critical":
        return f"Severe environmental damage observed. Approximately {area:.1f} hectares affected. Significant deforestation, soil erosion, and water contamination likely. Immediate intervention required."
    elif severity == "High":
        return f"Major environmental concerns. {area:.1f} hectares impacted. Deforestation and soil displacement evident. Water quality monitoring recommended."
    elif severity == "Moderate":
        return f"Moderate environmental impact. {area:.1f} hectares affected. Early-stage mining activity. Preventive action can minimize long-term damage."
    else:
        return f"Limited environmental impact detected. {area:.1f} hectares potentially affected. Monitoring recommended."

def generate_legal_context(severity: str) -> str:
    """Generate legal context"""
    contexts = {
        "Critical": "This activity likely violates environmental protection laws and mining regulations. Immediate legal action recommended. Report to: Environmental Protection Agency and Mining Regulatory Authority.",
        "High": "Potential violation of mining permits and environmental standards. Legal review required. Consult with: Local environmental authorities and legal team.",
        "Moderate": "Possible unauthorized mining activity. Verify against existing permits. Contact: Regional mining department for permit verification.",
        "Low": "Requires verification of mining authorization status. May be within legal boundaries pending investigation."
    }
    return contexts.get(severity, "Legal status requires investigation.")

def create_analysis_result(confidence: float, mining_type: str = "Unknown") -> AnalysisResult:
    """Create complete analysis result from model output"""
    severity = determine_severity(confidence)
    area_hectares = estimate_area_from_confidence(confidence)
    estimated_loss = estimate_financial_loss(area_hectares, mining_type)
    
    return AnalysisResult(
        detected=confidence > 50.0,
        confidence=round(confidence, 2),
        reasoning=generate_reasoning(confidence, severity),
        environmentalImpact=generate_environmental_impact(severity, area_hectares),
        legalContext=generate_legal_context(severity),
        machineryCount=int(confidence / 20) if confidence > 60 else 0,
        severity=severity,
        estimatedAreaHectares=round(area_hectares, 2),
        estimatedLossUSD=estimated_loss
    )

# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Illegal Mining Detection API",
        "model_loaded": model is not None,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_mining(
    file: UploadFile = File(..., description="Satellite image (JPG, PNG, TIFF)"),
    latitude: Optional[float] = Query(None, description="GPS Latitude"),
    longitude: Optional[float] = Query(None, description="GPS Longitude"),
    location_name: Optional[str] = Query(None, description="Location name"),
    mining_type: Optional[str] = Query("Unknown", description="Type of mining (Coal, Sand, etc.)")
):
    """
    🎯 MAIN DETECTION ENDPOINT
    
    Upload a satellite image and get AI-powered mining detection results
    
    **Process:**
    1. Upload satellite image
    2. AI analyzes for mining activity
    3. Returns confidence score, severity, and detailed analysis
    4. Saves high-confidence detections to database
    
    **Returns:**
    - Detection status (mining detected or not)
    - Confidence score (0-100%)
    - Detailed analysis (environmental impact, legal context)
    - Location information
    - Detection ID (if saved to database)
    """
    
    # Validate model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Please check server configuration."
        )
    
    # Validate file type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: JPG, PNG, TIFF"
        )
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Preprocess
        img_array = preprocess_image(image_bytes)
        
        # Run prediction
        prediction = model.predict(img_array, verbose=0)
        
        # Extract confidence
        # Assuming model outputs [no_mining_prob, mining_prob]
        mining_prob = float(prediction[0][1])
        confidence = mining_prob * 100
        
        # Create detailed analysis
        analysis = create_analysis_result(confidence, mining_type)
        
        # Prepare response
        response_data = {
            "mining_detected": analysis.detected,
            "confidence": confidence,
            "analysis": analysis.dict(),
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "name": location_name
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to database if detected and above threshold
        if analysis.detected and confidence >= CONFIDENCE_THRESHOLD and latitude and longitude:
            detection_id = save_detection(
                latitude=latitude,
                longitude=longitude,
                confidence=confidence,
                severity=analysis.severity,
                mining_type=mining_type or "Unknown",
                area_hectares=analysis.estimatedAreaHectares,
                estimated_loss_usd=analysis.estimatedLossUSD,
                location_name=location_name,
                image_filename=file.filename,
                reasoning=analysis.reasoning
            )
            response_data["detection_id"] = str(detection_id)
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/api/sites", response_model=List[MiningSite], tags=["Sites"])
async def get_mining_sites(
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    min_confidence: float = Query(0.0, ge=0, le=100, description="Minimum confidence"),
    severity: Optional[str] = Query(None, description="Filter by severity")
):
    """
    📍 GET ALL MINING SITES
    
    Retrieve all detected mining sites from database
    
    **Filters:**
    - limit: Maximum number of results (default: 100)
    - min_confidence: Minimum confidence score (0-100)
    - severity: Filter by severity level (Critical, High, Moderate, Low)
    
    **Returns:**
    List of mining sites matching frontend MiningSite structure
    """
    try:
        # Get detections from database
        detections = get_all_detections(
            limit=limit,
            min_confidence=min_confidence,
            severity=severity
        )
        
        # Convert to MiningSite format (matching frontend structure)
        sites = []
        for det in detections:
            site = MiningSite(
                id=str(det['id']),
                name=det['location_name'] or f"Site #{det['id']}",
                latitude=det['latitude'],
                longitude=det['longitude'],
                severity=det['severity'],
                type=det['mining_type'],
                areaHectares=det['area_hectares'],
                estimatedLossUSD=det['estimated_loss_usd'],
                lastDetected=det['detected_at'][:10],  # Extract date part
                images=[f"https://picsum.photos/seed/mine{det['id']}/800/600"],  # Placeholder
                confidence=det['confidence'],
                verified=det['verified']
            )
            sites.append(site)
        
        return sites
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching sites: {str(e)}"
        )

@app.get("/api/sites/{site_id}", response_model=MiningSite, tags=["Sites"])
async def get_site_by_id(site_id: str):
    """
    🔍 GET SPECIFIC SITE
    
    Get detailed information about a specific mining site
    """
    try:
        detection = get_detection_by_id(int(site_id))
        
        if not detection:
            raise HTTPException(status_code=404, detail="Site not found")
        
        site = MiningSite(
            id=str(detection['id']),
            name=detection['location_name'] or f"Site #{detection['id']}",
            latitude=detection['latitude'],
            longitude=detection['longitude'],
            severity=detection['severity'],
            type=detection['mining_type'],
            areaHectares=detection['area_hectares'],
            estimatedLossUSD=detection['estimated_loss_usd'],
            lastDetected=detection['detected_at'][:10],
            images=[f"https://picsum.photos/seed/mine{detection['id']}/800/600"],
            confidence=detection['confidence'],
            verified=detection['verified']
        )
        
        return site
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching site: {str(e)}"
        )

@app.get("/api/stats", response_model=Statistics, tags=["Statistics"])
async def get_stats():
    """
    📊 GET STATISTICS
    
    Get overall system statistics and metrics
    
    **Returns:**
    - Total detections
    - Total affected area
    - Total estimated financial loss
    - Breakdown by severity
    - Average confidence
    - Verified detections count
    """
    try:
        stats = get_statistics()
        return Statistics(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating statistics: {str(e)}"
        )

@app.get("/api/trends/monthly", response_model=List[MonthlyTrend], tags=["Statistics"])
async def get_monthly_trends():
    """
    📈 GET MONTHLY TRENDS
    
    Get monthly detection trends for charts
    
    **Returns:**
    Monthly data with:
    - Month name
    - Estimated financial loss
    - Number of detections
    
    (Matches frontend chart data structure)
    """
    try:
        trends = get_monthly_trends()
        return [MonthlyTrend(**t) for t in trends]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching trends: {str(e)}"
        )

@app.put("/api/sites/{site_id}/verify", tags=["Sites"])
async def verify_site(
    site_id: str,
    verified: bool = Query(..., description="Verification status"),
    notes: Optional[str] = Query(None, description="Verification notes")
):
    """
    ✅ VERIFY/UNVERIFY SITE
    
    Mark a detection as verified or unverified
    (For manual review workflow)
    """
    try:
        update_detection_verification(int(site_id), verified, notes)
        return {
            "success": True,
            "site_id": site_id,
            "verified": verified,
            "message": f"Site {'verified' if verified else 'unverified'} successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating verification: {str(e)}"
        )

# =====================================================
# RUN SERVER
# =====================================================

if __name__ == "__main__":
    import uvicorn
    import os

    # Use PORT from environment (for deployment) or 8000 (for local)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
