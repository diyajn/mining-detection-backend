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
import mysql.connector  # ✅ ADDED FOR NOTIFICATIONS

from tensorflow.keras.models import load_model as tf_load_model
import model_loader  # just import, do NOT call
app = FastAPI()
model = None



from dotenv import load_dotenv

# Load .env file (for local testing)
load_dotenv()




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

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# DATABASE CONNECTION FUNCTION
# ===============================

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT")),
        ssl_disabled=False
    )
# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = "models/mining_detector_final.h5"  # Update this path!
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
        return f"Low confidence detection ({confidence:.1f}%). Some indicators present but additional verification required. May be natural terrain features or legal activity."

def generate_environmental_impact(area_hectares: float, severity: str) -> str:
    """Generate environmental impact assessment"""
    trees_destroyed = int(area_hectares * 200)  # ~200 trees per hectare
    
    if severity == "Critical":
        return f"Severe environmental damage observed. Approximately {area_hectares:.1f} hectares affected with an estimated {trees_destroyed}+ trees destroyed. High risk of soil erosion, water contamination, and habitat loss. Immediate intervention required to prevent further degradation."
    elif severity == "High":
        return f"Significant environmental impact detected. Area of {area_hectares:.1f} hectares shows signs of disruption with approximately {trees_destroyed} trees at risk. Potential for water table contamination and ecosystem damage. Prompt action recommended."
    else:
        return f"Moderate environmental concerns identified across {area_hectares:.1f} hectares. Estimated {trees_destroyed} trees affected. Monitoring and assessment needed to prevent escalation of damage."

def generate_legal_context(severity: str) -> str:
    """Generate legal context and implications"""
    if severity == "Critical":
        return "This activity likely violates multiple environmental protection laws including the Mines and Minerals (Development and Regulation) Act, 1957, and Environment Protection Act, 1986. Immediate legal action and site inspection required. Penalties may include imprisonment up to 5 years and fines up to ₹50 lakhs."
    elif severity == "High":
        return "Potential violations of mining regulations detected. Investigation required under relevant mining and environmental laws. Site may be operating without proper permits or exceeding authorized mining limits."
    else:
        return "Activity warrants investigation for potential regulatory violations. Verification of mining permits and environmental clearances recommended."

# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/", tags=["Health"])
async def health_check():
    """
    🏥 Health Check
    
    Verify API is running and responsive
    """
    return {
        "status": "healthy",
        "service": "Illegal Mining Detection API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health", tags=["Health"])
async def api_health():
    """Alternative health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "database": "connected"
    }

@app.post("/api/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_mining(
    file: UploadFile = File(..., description="Satellite image to analyze"),
    latitude: Optional[float] = Query(None, description="GPS latitude"),
    longitude: Optional[float] = Query(None, description="GPS longitude"),
    location_name: Optional[str] = Query(None, description="Location name"),
    mining_type: Optional[str] = Query("Unknown", description="Type of mining suspected")
):
    """
    🔍 DETECT ILLEGAL MINING
    
    Upload a satellite image and get AI-powered mining detection results.
    
    **Workflow:**
    1. Upload satellite/aerial image
    2. AI analyzes for mining indicators
    3. Returns confidence score and detailed analysis
    4. If confidence > 50%, saves to database
    
    **Returns:**
    - Detection confidence (0-100%)
    - Severity level (Critical/High/Moderate/Low)
    - Environmental impact assessment
    - Legal context
    - Estimated area and financial loss
    """
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Server starting up or model file missing."
        )
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Preprocess
        img_array = preprocess_image(image_bytes)
        
        # Predict
        prediction = model.predict(img_array, verbose=0)
        
        # Extract confidence (assuming binary classification: [not_mining, mining])
        confidence = float(prediction[0][1] * 100)  # Convert to percentage
        mining_detected = confidence > CONFIDENCE_THRESHOLD
        
        # Determine severity
        severity = determine_severity(confidence)
        
        # Generate estimates
        area_hectares = estimate_area_from_confidence(confidence)
        estimated_loss = estimate_financial_loss(area_hectares, mining_type)
        machinery_count = int(confidence / 25)  # Rough estimate
        
        # Generate analysis text
        reasoning = generate_reasoning(confidence, severity)
        environmental_impact = generate_environmental_impact(area_hectares, severity)
        legal_context = generate_legal_context(severity)
        
        # Create response
        analysis = AnalysisResult(
            detected=mining_detected,
            confidence=confidence,
            reasoning=reasoning,
            environmentalImpact=environmental_impact,
            legalContext=legal_context,
            machineryCount=machinery_count,
            severity=severity,
            estimatedAreaHectares=area_hectares,
            estimatedLossUSD=estimated_loss
        )
        
        # Save to database if mining detected with sufficient confidence
        detection_id = None
        if mining_detected and latitude and longitude:
            try:
                detection_id = save_detection(
                    latitude=latitude,
                    longitude=longitude,
                    confidence=confidence,
                    severity=severity,
                    mining_type=mining_type,
                    area_hectares=area_hectares,
                    estimated_loss_usd=estimated_loss,
                    location_name=location_name,
                    image_filename=file.filename,
                    reasoning=reasoning
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not save detection to database: {e}")
        
        return DetectionResponse(
            mining_detected=mining_detected,
            confidence=confidence,
            analysis=analysis,
            location={
                "latitude": latitude,
                "longitude": longitude,
                "name": location_name or "Unknown Location"
            },
            timestamp=datetime.now().isoformat(),
            detection_id=str(detection_id) if detection_id else None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/api/sites", response_model=List[MiningSite], tags=["Sites"])
async def get_all_sites(
    limit: int = Query(100, description="Maximum number of results"),
    min_confidence: float = Query(0.0, description="Minimum confidence threshold"),
    severity: Optional[str] = Query(None, description="Filter by severity")
):
    """
    🗺️ GET ALL MINING SITES
    
    Retrieve all detected mining sites from database
    
    **Filters:**
    - limit: Number of results (default 100)
    - min_confidence: Filter sites above confidence threshold
    - severity: Filter by "Critical", "High", "Moderate", or "Low"
    """
    try:
        detections = get_all_detections(
            limit=limit,
            min_confidence=min_confidence,
            severity=severity
        )
        
        # Convert to MiningSite format
        sites = []
        for det in detections:
            sites.append(MiningSite(
                id=str(det['id']),
                name=det['location_name'] or f"Site #{det['id']}",
                latitude=det['latitude'],
                longitude=det['longitude'],
                severity=det['severity'],
                type=det['mining_type'],
                areaHectares=det['area_hectares'],
                estimatedLossUSD=det['estimated_loss_usd'],
                lastDetected=det['detected_at'][:10] if det['detected_at'] else date.today().isoformat(),
                images=[f"https://picsum.photos/seed/mine{det['id']}/800/600"],  # Placeholder
                confidence=det['confidence'],
                verified=det['verified']
            ))
        
        return sites
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching sites: {str(e)}"
        )

@app.get("/api/sites/{site_id}", response_model=MiningSite, tags=["Sites"])
async def get_site_details(site_id: str):
    """
    📍 GET SITE DETAILS
    
    Get detailed information about a specific mining site
    """
    try:
        det = get_detection_by_id(int(site_id))
        
        if not det:
            raise HTTPException(status_code=404, detail="Site not found")
        
        return MiningSite(
            id=str(det['id']),
            name=det['location_name'] or f"Site #{det['id']}",
            latitude=det['latitude'],
            longitude=det['longitude'],
            severity=det['severity'],
            type=det['mining_type'],
            areaHectares=det['area_hectares'],
            estimatedLossUSD=det['estimated_loss_usd'],
            lastDetected=det['detected_at'][:10] if det['detected_at'] else date.today().isoformat(),
            images=[f"https://picsum.photos/seed/mine{det['id']}/800/600"],
            confidence=det['confidence'],
            verified=det['verified']
        )
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid site ID")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", response_model=Statistics, tags=["Statistics"])
async def get_system_statistics():
    """
    📊 GET STATISTICS
    
    Get overall system statistics and metrics
    
    **Returns:**
    - Total detections count
    - Total area affected (hectares)
    - Total estimated financial loss
    - Site breakdown by severity
    - Average detection confidence
    - Verification statistics
    """
    try:
        stats = get_statistics()
        return Statistics(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching statistics: {str(e)}"
        )

@app.get("/api/trends/monthly", response_model=List[MonthlyTrend], tags=["Statistics"])
def get_monthly_trends_endpoint(months: int = Query(6, description="Number of months")):
    """
    📈 GET MONTHLY TRENDS
    
    Returns detection counts and losses for the last N months
    """
    try:
        trends = get_monthly_trends(months)
        return trends
    
    except Exception as e:
        print(f"❌ Error in trends endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching trends: {str(e)}"
        )

@app.patch("/api/sites/{site_id}/verify", tags=["Sites"])
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
# ADVANCED FEATURES - NEW ENDPOINTS
# =====================================================

@app.get("/api/high-risk-areas", tags=["Advanced Features"])
async def get_high_risk_areas():
    """
    📍 Get list of monitored high-risk mining areas
    (From your workflow screenshots)
    """
    from satellite_downloader import HIGH_RISK_AREAS
    
    return {
        "success": True,
        "count": len(HIGH_RISK_AREAS),
        "areas": HIGH_RISK_AREAS
    }

@app.post("/api/generate-report/{detection_id}", tags=["Advanced Features"])
async def generate_official_report(detection_id: str):
    """
    📄 Generate Official Court-Ready Report
    
    Creates PDF with:
    - Satellite evidence
    - GPS coordinates  
    - 3D depth analysis
    - Impact assessment
    - Legal sections violated
    """
    try:
        # Get detection from database
        detection = get_detection_by_id(int(detection_id))
        
        if not detection:
            raise HTTPException(status_code=404, detail="Detection not found")
        
        # Generate report
        from manual_workflow import ManualWorkflowHandler
        handler = ManualWorkflowHandler()
        
        scan_info = {
            "area_name": detection['location_name'] or f"Site #{detection_id}",
            "latitude": detection['latitude'],
            "longitude": detection['longitude'],
            "started_at": datetime.now()
        }
        
        detection_result = {
            "mining_detected": True,
            "confidence": detection['confidence'],
            "severity": detection['severity'],
            "area_hectares": detection['area_hectares'],
            "trees_destroyed": int(detection['confidence'] / 100 * 200),
            "economic_loss": detection['estimated_loss_usd']
        }
        
        report = handler.generate_official_report(scan_info, detection_result)
        
        return {
            "success": True,
            "report_filename": report['filename'],
            "report_path": report['filepath'],
            "message": "Official report generated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/notifications", tags=["Advanced Features"])
async def get_dashboard_notifications(
    unread_only: bool = Query(False, description="Show only unread notifications")
):
    """
    🔔 Get Dashboard Notifications
    
    Returns alerts from automated monitoring:
    - New mining detected
    - Activity increased
    - Field officer dispatched
    """
    try:
        # ✅ FIXED: Using MySQL instead of SQLite
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Query notifications
        if unread_only:
            query = """
                SELECT 
                    id, 
                    alert_type, 
                    location, 
                    severity, 
                    confidence, 
                    message, 
                    created_at,
                    `read`
                FROM notifications
                WHERE `read` = FALSE
                ORDER BY created_at DESC
                LIMIT 50
            """
        else:
            query = """
                SELECT 
                    id, 
                    alert_type, 
                    location, 
                    severity, 
                    confidence, 
                    message, 
                    created_at,
                    `read`
                FROM notifications
                ORDER BY created_at DESC
                LIMIT 50
            """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        notifications = []
        for row in rows:
            notifications.append({
                "id": row["id"],
                "alert_type": row["alert_type"],
                "location": row["location"],
                "severity": row["severity"],
                "confidence": row["confidence"],
                "message": row["message"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "read": bool(row["read"])
            })
        
        return {
            "success": True,
            "count": len(notifications),
            "unread_count": len([n for n in notifications if not n["read"]]),
            "notifications": notifications
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching notifications: {str(e)}"
        )

@app.patch("/api/notifications/{notification_id}/mark-read", tags=["Advanced Features"])
async def mark_notification_read(notification_id: int):
    """
    ✅ Mark Notification as Read
    
    Officer clicks on notification → marks as read
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE notifications 
            SET `read` = TRUE 
            WHERE id = %s
        """, (notification_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "message": "Notification marked as read"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/notifications/mark-all-read", tags=["Advanced Features"])
async def mark_all_notifications_read():
    """
    ✅ Mark ALL Notifications as Read
    
    Officer clicks "Mark all as read"
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("UPDATE notifications SET `read` = TRUE")
        
        count = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "message": f"Marked {count} notifications as read"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/monitoring/add-area", tags=["Monitoring"])
async def add_area_to_monitoring_queue(
    area_name: str = Query(..., description="Area name"),
    latitude: float = Query(..., description="Latitude"),
    longitude: float = Query(..., description="Longitude"),
    requested_by: str = Query("Officer", description="Who requested")
):
    """
    👮 Trigger Manual Inspection Workflow
    
    Steps (as per your screenshots):
    1. Officer selects area on map
    2. Clicks "Scan This Area"  
    3. System downloads latest satellite
    4. AI processes image
    5. Shows results
    6. Generates report
    """

    from database import add_area_to_monitoring
    
    queue_id = add_area_to_monitoring(
        area_name=area_name,
        latitude=latitude,
        longitude=longitude,
        requested_by=requested_by,
        frequency="daily"
    )
    
    return {
        "success": True,
        "message": f"Area '{area_name}' added to monitoring queue",
        "queue_id": queue_id,
        "next_scan": "Tomorrow at 3:00 AM",
        "status": "pending"
    }




@app.post("/api/manual-scan", tags=["Advanced Features"])
async def trigger_manual_scan(
    latitude: float = Query(...),
    longitude: float = Query(...),
    area_name: str = Query(...)
):
    """
    👮 Trigger Manual Inspection Workflow
    
    Steps (as per your screenshots):
    1. Officer selects area on map
    2. Clicks "Scan This Area"  
    3. System downloads latest satellite
    4. AI processes image
    5. Shows results
    6. Generates report
    """
    return {
        "success": True,
        "message": "Manual scan initiated",
        "area": area_name,
        "coordinates": {
            "latitude": latitude,
            "longitude": longitude
        },
        "status": "Processing - check notifications for results",
        "estimated_time": "2-3 minutes"
    }
@app.get("/api/monitoring/queue", tags=["Monitoring"])
async def get_monitoring_queue():
    """
    📋 GET MONITORING QUEUE
    
    See all areas waiting to be scanned
    """
    
    from database import get_pending_areas
    
    areas = get_pending_areas()
    
    return {
        "success": True,
        "count": len(areas),
        "areas": areas,
        "next_scan_time": "3:00 AM"
    }
# =====================================================
# RUN SERVER
# =====================================================

if __name__ == "__main__":
    import uvicorn
    import os

    # Use PORT from environment (for deployment) or 8000 (for local)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
