"""
Quick Database Viewer
Run this to see all stored detections
"""

import sqlite3
from datetime import datetime

DB_PATH = "mining_detections.db"

print("\n" + "="*80)
print("📊 MINING DETECTIONS DATABASE VIEWER")
print("="*80 + "\n")

# Connect to database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get total count
cursor.execute("SELECT COUNT(*) FROM detections")
total = cursor.fetchone()[0]

print(f"📍 Total Detections: {total}\n")

if total == 0:
    print("No detections yet. Upload some images first!")
    exit()

# Get all detections
cursor.execute("""
    SELECT 
        id, 
        location_name, 
        latitude, 
        longitude, 
        confidence, 
        severity,
        mining_type,
        area_hectares,
        estimated_loss_usd,
        detected_at,
        verified
    FROM detections 
    ORDER BY detected_at DESC
""")

rows = cursor.fetchall()

print("="*80)
print(f"{'ID':<5} {'Location':<25} {'Lat':<10} {'Lon':<10} {'Conf%':<8} {'Severity':<10}")
print("="*80)

for row in rows:
    id_, name, lat, lon, conf, sev, mtype, area, loss, date, verified = row
    
    # Format name
    name = name or f"Site #{id_}"
    if len(name) > 24:
        name = name[:21] + "..."
    
    # Format coordinates
    lat_str = f"{lat:.2f}"
    lon_str = f"{lon:.2f}"
    
    # Format confidence
    conf_str = f"{conf:.1f}%"
    
    # Print row
    print(f"{id_:<5} {name:<25} {lat_str:<10} {lon_str:<10} {conf_str:<8} {sev:<10}")

print("="*80 + "\n")

# Statistics
cursor.execute("SELECT SUM(area_hectares) FROM detections")
total_area = cursor.fetchone()[0] or 0

cursor.execute("SELECT SUM(estimated_loss_usd) FROM detections")
total_loss = cursor.fetchone()[0] or 0

cursor.execute("SELECT COUNT(*) FROM detections WHERE severity = 'Critical'")
critical = cursor.fetchone()[0]

cursor.execute("SELECT AVG(confidence) FROM detections")
avg_conf = cursor.fetchone()[0] or 0

print("📊 STATISTICS:")
print(f"   Total Area Affected: {total_area:,.1f} hectares")
print(f"   Total Estimated Loss: ${total_loss:,}")
print(f"   Critical Sites: {critical}")
print(f"   Average Confidence: {avg_conf:.1f}%")

# Recent detections
print("\n🕒 RECENT DETECTIONS (Last 5):")
cursor.execute("""
    SELECT location_name, confidence, severity, detected_at 
    FROM detections 
    ORDER BY detected_at DESC 
    LIMIT 5
""")

for row in cursor.fetchall():
    name, conf, sev, date = row
    name = name or "Unnamed Site"
    print(f"   • {name} - {conf:.1f}% ({sev}) - {date}")

conn.close()

print("\n" + "="*80 + "\n")