"""
Database Module - SQLite/PostgreSQL operations
Handles all data persistence for mining detections
"""

import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from calendar import month_abbr

# Database configuration
DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(__file__), "mining_detections.db")
)

# =====================================================
# DATABASE INITIALIZATION
# =====================================================

def init_db():
    """Initialize database and create tables"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create detections table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            mining_type TEXT NOT NULL,
            area_hectares REAL NOT NULL,
            estimated_loss_usd INTEGER NOT NULL,
            location_name TEXT,
            image_filename TEXT,
            reasoning TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            verified BOOLEAN DEFAULT 0,
            verification_notes TEXT,
            verified_at TIMESTAMP
        )
    """)
    
    # Create indexes for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_coordinates 
        ON detections(latitude, longitude)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_confidence 
        ON detections(confidence DESC)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_severity 
        ON detections(severity)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detected_at 
        ON detections(detected_at DESC)
    """)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database initialized: {os.path.abspath(DB_PATH)}")

# =====================================================
# CREATE OPERATIONS
# =====================================================

def save_detection(
    latitude: float,
    longitude: float,
    confidence: float,
    severity: str,
    mining_type: str,
    area_hectares: float,
    estimated_loss_usd: int,
    location_name: Optional[str] = None,
    image_filename: Optional[str] = None,
    reasoning: Optional[str] = None
) -> int:
    """
    Save a new mining detection
    
    Args:
        latitude: GPS latitude
        longitude: GPS longitude
        confidence: Detection confidence (0-100)
        severity: Severity level (Critical/High/Moderate/Low)
        mining_type: Type of mining (Coal/Sand/Open-pit/etc)
        area_hectares: Estimated affected area
        estimated_loss_usd: Estimated financial loss
        location_name: Optional location name
        image_filename: Original image filename
        reasoning: AI reasoning text
        
    Returns:
        ID of inserted record
    """
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO detections (
            latitude, longitude, confidence, severity, mining_type,
            area_hectares, estimated_loss_usd, location_name, 
            image_filename, reasoning
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        latitude, longitude, confidence, severity, mining_type,
        area_hectares, estimated_loss_usd, location_name,
        image_filename, reasoning
    ))
    
    detection_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    print(f"💾 Saved detection #{detection_id}: {severity} severity at ({latitude}, {longitude})")
    
    return detection_id

# =====================================================
# READ OPERATIONS
# =====================================================

def get_all_detections(
    limit: int = 100,
    min_confidence: float = 0.0,
    severity: Optional[str] = None
) -> List[Dict]:
    """
    Get all detections with optional filters
    
    Args:
        limit: Maximum results
        min_confidence: Minimum confidence threshold
        severity: Filter by severity level
        
    Returns:
        List of detection dictionaries
    """
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Build query with filters
    query = """
        SELECT * FROM detections
        WHERE confidence >= ?
    """
    params = [min_confidence]
    
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    
    query += " ORDER BY detected_at DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to list of dicts
    detections = []
    for row in rows:
        detections.append({
            "id": row["id"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "confidence": row["confidence"],
            "severity": row["severity"],
            "mining_type": row["mining_type"],
            "area_hectares": row["area_hectares"],
            "estimated_loss_usd": row["estimated_loss_usd"],
            "location_name": row["location_name"],
            "image_filename": row["image_filename"],
            "reasoning": row["reasoning"],
            "detected_at": row["detected_at"],
            "verified": bool(row["verified"]),
            "verification_notes": row["verification_notes"]
        })
    
    return detections

def get_detection_by_id(detection_id: int) -> Optional[Dict]:
    """Get a specific detection by ID"""
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM detections WHERE id = ?", (detection_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row["id"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "confidence": row["confidence"],
            "severity": row["severity"],
            "mining_type": row["mining_type"],
            "area_hectares": row["area_hectares"],
            "estimated_loss_usd": row["estimated_loss_usd"],
            "location_name": row["location_name"],
            "image_filename": row["image_filename"],
            "reasoning": row["reasoning"],
            "detected_at": row["detected_at"],
            "verified": bool(row["verified"]),
            "verification_notes": row["verification_notes"]
        }
    return None

def get_detections_in_area(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float
) -> List[Dict]:
    """Get all detections within a geographic bounding box"""
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM detections
        WHERE latitude BETWEEN ? AND ?
        AND longitude BETWEEN ? AND ?
        ORDER BY confidence DESC
    """, (min_lat, max_lat, min_lon, max_lon))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

# =====================================================
# UPDATE OPERATIONS
# =====================================================

def update_detection_verification(
    detection_id: int,
    verified: bool,
    notes: Optional[str] = None
):
    """Mark a detection as verified/unverified"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE detections 
        SET verified = ?,
            verification_notes = ?,
            verified_at = ?
        WHERE id = ?
    """, (verified, notes, datetime.now().isoformat() if verified else None, detection_id))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Detection #{detection_id} {'verified' if verified else 'unverified'}")

# =====================================================
# DELETE OPERATIONS
# =====================================================

def delete_detection(detection_id: int):
    """Delete a detection"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
    
    conn.commit()
    conn.close()
    
    print(f"🗑️ Deleted detection #{detection_id}")

# =====================================================
# STATISTICS & ANALYTICS
# =====================================================

def get_statistics() -> Dict:
    """Get overall system statistics"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total detections
    cursor.execute("SELECT COUNT(*) FROM detections")
    total_detections = cursor.fetchone()[0]
    
    # Total area
    cursor.execute("SELECT SUM(area_hectares) FROM detections")
    total_area = cursor.fetchone()[0] or 0.0
    
    # Total estimated loss
    cursor.execute("SELECT SUM(estimated_loss_usd) FROM detections")
    total_loss = cursor.fetchone()[0] or 0
    
    # Severity breakdown
    cursor.execute("SELECT COUNT(*) FROM detections WHERE severity = 'Critical'")
    critical_sites = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM detections WHERE severity = 'High'")
    high_severity = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM detections WHERE severity = 'Moderate'")
    moderate_severity = cursor.fetchone()[0]
    
    # Average confidence
    cursor.execute("SELECT AVG(confidence) FROM detections")
    avg_confidence = cursor.fetchone()[0] or 0.0
    
    # Verified count
    cursor.execute("SELECT COUNT(*) FROM detections WHERE verified = 1")
    verified_count = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_detections": total_detections,
        "total_area_hectares": round(total_area, 2),
        "total_estimated_loss_usd": int(total_loss),
        "critical_sites": critical_sites,
        "high_severity_sites": high_severity,
        "moderate_severity_sites": moderate_severity,
        "avg_confidence": round(avg_confidence, 2),
        "verified_count": verified_count
    }

def get_monthly_trends(months: int = 6) -> List[Dict]:
    """
    Get monthly detection trends
    
    Args:
        months: Number of months to retrieve
        
    Returns:
        List of monthly data matching frontend chart structure
    """
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get data for last N months
    trends = []
    
    for i in range(months - 1, -1, -1):
        # Calculate date range for this month
        target_date = datetime.now() - timedelta(days=30 * i)
        month_start = target_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Next month start
        if month_start.month == 12:
            next_month = month_start.replace(year=month_start.year + 1, month=1)
        else:
            next_month = month_start.replace(month=month_start.month + 1)
        
        # Get count and total loss for this month
        cursor.execute("""
            SELECT 
                COUNT(*) as detected,
                COALESCE(SUM(estimated_loss_usd), 0) as loss
            FROM detections
            WHERE detected_at >= ? AND detected_at < ?
        """, (month_start.isoformat(), next_month.isoformat()))
        
        row = cursor.fetchone()
        
        trends.append({
            "name": month_abbr[month_start.month],  # 'Jan', 'Feb', etc.
            "detected": row[0],
            "loss": row[1]
        })
    
    conn.close()
    
    return trends

def get_detection_count_by_type() -> Dict[str, int]:
    """Get detection count grouped by mining type"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT mining_type, COUNT(*) as count
        FROM detections
        GROUP BY mining_type
        ORDER BY count DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return {row[0]: row[1] for row in rows}

# =====================================================
# TESTING
# =====================================================

if __name__ == "__main__":
    """Test database operations"""
    
    print("\n🧪 Testing Database Module...\n")
    
    # Initialize
    init_db()
    
    # Add test data
    test_id = save_detection(
        latitude=23.74,
        longitude=86.41,
        confidence=87.5,
        severity="Critical",
        mining_type="Coal",
        area_hectares=145.5,
        estimated_loss_usd=12400000,
        location_name="Jharia Coal Sector 4",
        image_filename="test.jpg",
        reasoning="High confidence detection. Clear indicators of mining activity."
    )
    
    print(f"\n✅ Test detection created: ID = {test_id}")
    
    # Retrieve
    detections = get_all_detections(limit=10)
    print(f"✅ Retrieved {len(detections)} detections")
    
    # Statistics
    stats = get_statistics()
    print(f"\n📊 Statistics:")
    print(f"   Total detections: {stats['total_detections']}")
    print(f"   Total area: {stats['total_area_hectares']} hectares")
    print(f"   Total loss: ${stats['total_estimated_loss_usd']:,}")
    
    # Trends
    trends = get_monthly_trends(months=6)
    print(f"\n📈 Monthly Trends:")
    for trend in trends:
        print(f"   {trend['name']}: {trend['detected']} detections, ${trend['loss']:,} loss")
    
    print("\n✅ All tests passed!\n")
