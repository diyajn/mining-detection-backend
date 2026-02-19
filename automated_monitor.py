"""
Automated Daily Monitoring System - FIXED VERSION
Runs at 3:00 AM daily to scan queued areas
"""

import os
from datetime import datetime
from typing import List, Dict
import numpy as np
from PIL import Image
import mysql.connector
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
# AUTOMATED MONITOR CLASS
# =====================================================

class AutomatedMonitor:
    """
    Handles automated daily monitoring workflow
    Complete workflow from screenshots
    """
    
    def __init__(self):
        self.model = None
        self.download_folder = "auto_downloads"
        os.makedirs(self.download_folder, exist_ok=True)
    
    def load_model(self):
        """Load ML model for detection"""
        try:
            from tensorflow.keras.models import load_model
            
            MODEL_PATH = "models/mining_detector_final.h5"
            
            if not os.path.exists(MODEL_PATH):
                print(f"   ⚠️  Model not found at {MODEL_PATH}")
                print(f"   Using simulated results for demo")
                return False
            
            print(f"   Loading model from {MODEL_PATH}...")
            self.model = load_model(MODEL_PATH)
            print(f"   ✅ Model loaded successfully")
            print(f"      Input shape: {self.model.input_shape}")
            print(f"      Output shape: {self.model.output_shape}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            print(f"   Using simulated results")
            return False
    
    def run_detection_on_images(self, image_paths: List[str]) -> List[Dict]:
        """
        Run ML detection on downloaded images
        """
        results = []
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n   Processing image {i}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            try:
                if self.model is not None:
                    # Use actual model
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((224, 224))
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    prediction = self.model.predict(img_array, verbose=0)
                    confidence = float(prediction[0][1] * 100)
                else:
                    # Simulated results
                    confidence = np.random.uniform(60, 95)
                
                # Determine severity
                if confidence >= 85:
                    severity = "Critical"
                elif confidence >= 70:
                    severity = "High"
                elif confidence >= 55:
                    severity = "Moderate"
                else:
                    severity = "Low"
                
                result = {
                    "image_path": img_path,
                    "confidence": confidence,
                    "severity": severity,
                    "mining_detected": confidence > 50
                }
                
                results.append(result)
                print(f"      Confidence: {confidence:.1f}% - {severity}")
                
            except Exception as e:
                print(f"   ❌ Error processing {img_path}: {e}")
        
        return results
    
    def compare_with_yesterday(self, detection_results: List[Dict]) -> List[Dict]:
        """
        Compare today's detections with yesterday
        Generate alerts for NEW or WORSENED mining
        """
        print("\n🔍 Checking if mining is NEW or WORSENED...")
        
       
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            alerts = []
            
            # For each detection, check if it's new
            for result in detection_results:
                # Extract area name from image path
                filename = os.path.basename(result['image_path'])
                area_name = filename.replace('_', ' ').replace('.jpg', '').replace('.png', '')
                
                # Check if we have recent detections for this area
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM detections 
                    WHERE location_name LIKE %s
                    AND detected_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
                """, (f"%{area_name.split()[0]}%",))
                
                count = cursor.fetchone()[0]
                
                if count == 0:
                    print(f"   ⚠️  NEW mining detected at {area_name}")
                    alerts.append({
                        "type": "new_mining_detected",
                        "location": area_name,
                        "severity": result['severity'],
                        "confidence": result['confidence'],
                        "message": f"⚠️ NEW MINING DETECTED at {area_name} with {result['confidence']:.1f}% confidence"
                    })
                else:
                    print(f"   ℹ️  Mining already known at {area_name}")
            
            cursor.close()
            conn.close()
            
            return alerts
            
        except Exception as e:
            print(f"   ❌ Error comparing with history: {e}")
            return []
    
    def send_email_alert(self, alert: Dict):
        """Send email alert to district collector"""
        print(f"\n📧 Sending email alert to: collector@example.com")
        print(f"   Subject: 🚨 URGENT: New Illegal Mining Detected - {alert['location']}")
        print(f"   ✅ Email sent (simulation)")
    
    def send_sms_alert(self, alert: Dict):
        """Send SMS to field officer"""
        print(f"\n📱 Sending SMS to: +919876543210")
        print(f"   Message: ALERT: New mining detected at {alert['location']} ({alert['confidence']:.1f}% confidence)")
        print(f"   ✅ SMS sent (simulation)")
    
    def send_dashboard_notification(self, alert: Dict):
        """Create notification in dashboard"""
        print(f"\n📊 Creating dashboard notification...")
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Insert notification
            cursor.execute("""
                INSERT INTO notifications (alert_type, location, severity, confidence, message)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                alert['type'],
                alert['location'],
                alert['severity'],
                alert['confidence'],
                alert['message']
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"   ✅ Dashboard notification created")
            
        except Exception as e:
            print(f"   ❌ Error creating notification: {e}")
    
    def run_daily_monitoring(self):
        """
        Complete daily monitoring workflow
        Scans areas from monitoring queue
        """
        
        print("\n" + "="*70)
        print("🌅 DAILY AUTOMATIC MONITORING - START")
        print("="*70)
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        # Step 1: Get areas from queue
        from database import get_pending_areas, mark_area_scanned, save_detection
        
        print("STEP 1: Checking monitoring queue...")
        pending_areas = get_pending_areas()
        
        if not pending_areas:
            print("✅ No areas in queue. Nothing to scan today.")
            return
        
        print(f"📍 Found {len(pending_areas)} areas in queue:\n")
        for area in pending_areas:
            print(f"   • {area['area_name']} ({area['latitude']}, {area['longitude']})")
        
        # Step 2: Download satellite images
        print("\nSTEP 2: Downloading satellite images for queued areas...")
        
        from satellite_downloader import download_sentinel_image, get_sentinel_token
        
        token = get_sentinel_token()
        downloaded_images = []
        
        for area in pending_areas:
            filepath = download_sentinel_image(
                area={
                    "name": area['area_name'],
                    "latitude": area['latitude'],
                    "longitude": area['longitude']
                },
                token=token
            )
            
            if filepath:
                downloaded_images.append({
                    "queue_id": area['id'],
                    "area": area,
                    "filepath": filepath
                })
        
        if not downloaded_images:
            print("❌ No images downloaded. Aborting.")
            return
        
        # Step 3: Load ML model
        print("\nSTEP 3: Loading ML model...")
        model_loaded = self.load_model()
        
        # Step 4: Run detection
        print("\nSTEP 4: Running ML detection...")
        image_paths = [img['filepath'] for img in downloaded_images]
        detection_results = self.run_detection_on_images(image_paths)
        
        # Save detections to database AND mark areas as scanned
        for i, result in enumerate(detection_results):
            area = downloaded_images[i]['area']
            queue_id = downloaded_images[i]['queue_id']
            
            # Save ONLY if mining detected with confidence > 50%
            if result['mining_detected'] and result['confidence'] > 50:
                # Calculate estimates
                area_hectares = round(result['confidence'] / 100 * 2.5, 1)
                estimated_loss = int(result['confidence'] / 100 * 3000000)
                
                # Save to database
                detection_id = save_detection(
                    latitude=area['latitude'],
                    longitude=area['longitude'],
                    confidence=result['confidence'],
                    severity=result['severity'],
                    mining_type="Unknown",
                    area_hectares=area_hectares,
                    estimated_loss_usd=estimated_loss,
                    location_name=area['area_name'],
                    image_filename=os.path.basename(result['image_path']),
                    reasoning=f"Automated scan detected mining with {result['confidence']:.1f}% confidence"
                )
            
            # ✅ CRITICAL FIX: Mark as scanned REGARDLESS of detection result
            # This ensures status changes from 'pending' to 'scanned' for ALL areas
            mark_area_scanned(queue_id)
            print(f"   ✅ Marked area '{area['area_name']}' as scanned (queue_id: {queue_id})")
        
        # Step 5: Compare with yesterday
        print("\nSTEP 5: Comparing with yesterday...")
        alerts = self.compare_with_yesterday(detection_results)
        
        # Step 6: Send alerts
        if alerts:
            print(f"\nSTEP 6: Sending {len(alerts)} alerts...")
            
            for alert in alerts:
                self.send_email_alert(alert)
                self.send_sms_alert(alert)
                self.send_dashboard_notification(alert)
        else:
            print("\nSTEP 6: No alerts to send (all clear ✅)")
        
        # Summary
        print("\n" + "="*70)
        print("📊 DAILY MONITORING - SUMMARY")
        print("="*70)
        print(f"✅ Areas scanned: {len(pending_areas)}")
        print(f"✅ Images downloaded: {len(downloaded_images)}")
        print(f"✅ Detections processed: {len(detection_results)}")
        print(f"🚨 Alerts generated: {len(alerts)}")
        print(f"⏰ Completed at: {datetime.now().strftime('%H:%M:%S')}")
        print("="*70 + "\n")


# =====================================================
# TESTING
# =====================================================

if __name__ == "__main__":
    print("\n🧪 AUTOMATED MONITOR - TEST MODE\n")
    
    monitor = AutomatedMonitor()
    monitor.run_daily_monitoring()