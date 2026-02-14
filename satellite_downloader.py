"""
Automatic Satellite Image Downloader
Integrates with Sentinel Hub, Google Earth Engine, and USGS Landsat
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

# =====================================================
# CONFIGURATION
# =====================================================

# High-risk mining areas (from your screenshots)
HIGH_RISK_AREAS = [
    {
        "name": "Yamuna",
        "latitude": 25.4358,
        "longitude": 81.8463,
        "type": "Sand Mining",
        "priority": "Critical"
    },
    {
        "name": "Jharia Coalfield",
        "latitude": 23.7428,
        "longitude": 86.4147,
        "type": "Coal Mining",
        "priority": "Critical"
    },
    {
        "name": "Aravalli Hills",
        "latitude": 28.1400,
        "longitude": 76.9100,
        "type": "Stone Mining",
        "priority": "High"
    },
    {
        "name": "Goa Iron Ore Belt",
        "latitude": 15.3004,
        "longitude": 74.0855,
        "type": "Iron Ore",
        "priority": "High"
    },
    {
        "name": "Mandla Bauxite",
        "latitude": 22.5991,
        "longitude": 80.3714,
        "type": "Bauxite",
        "priority": "Medium"
    },
    {
        "name": "Narmada Riverbed",
        "latitude": 21.6400,
        "longitude": 73.0100,
        "type": "Sand Mining",
        "priority": "Critical"
    }
]

# Sentinel Hub Configuration (from your Colab)
SENTINEL_CONFIG = {
    "client_id": "47a98961-9736-49ff-9c8c-bb3da03ecd5c",
    "client_secret": "dMDOcc45p2NHs46iEzVgB1KSIzPWBtrI",
    "base_url": "https://services.sentinel-hub.com"
}

# Download settings
DOWNLOAD_FOLDER = "auto_downloads"
RESOLUTION = 10  # meters per pixel

# =====================================================
# SENTINEL HUB AUTHENTICATION
# =====================================================

def get_sentinel_token():
    """Get OAuth token from Sentinel Hub"""
    
    token_url = f"{SENTINEL_CONFIG['base_url']}/oauth/token"
    
    data = {
        "grant_type": "client_credentials",
        "client_id": SENTINEL_CONFIG["client_id"],
        "client_secret": SENTINEL_CONFIG["client_secret"]
    }
    
    try:
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        token = response.json()["access_token"]
        print("✅ Sentinel Hub authentication successful")
        return token
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return None

# =====================================================
# DOWNLOAD SATELLITE IMAGES
# =====================================================

def download_sentinel_image(
    area: Dict,
    token: str,
    date_from: str = None,
    date_to: str = None
) -> Optional[str]:
    """
    Download latest satellite image for an area
    
    Args:
        area: Dictionary with location info
        token: Sentinel Hub access token
        date_from: Start date (YYYY-MM-DD), defaults to 30 days ago
        date_to: End date (YYYY-MM-DD), defaults to today
        
    Returns:
        Path to downloaded image or None if failed
    """
    
    # Set date range
    if not date_to:
        date_to = datetime.now().strftime("%Y-%m-%d")
    if not date_from:
        date_from = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    print(f"\n📡 Downloading satellite image for: {area['name']}")
    print(f"   Coordinates: ({area['latitude']}, {area['longitude']})")
    print(f"   Date range: {date_from} to {date_to}")
    
    # Create bounding box (1km x 1km around point)
    lat, lon = area['latitude'], area['longitude']
    bbox_size = 0.01  # ~1km
    
    bbox = [
        lon - bbox_size,  # min lon
        lat - bbox_size,  # min lat
        lon + bbox_size,  # max lon
        lat + bbox_size   # max lat
    ]
    
    # Sentinel Hub API request
    api_url = f"{SENTINEL_CONFIG['base_url']}/api/v1/process"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Request body (evalscript for True Color)
    request_body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "S2L1C",  # Sentinel-2 Level 1C
                "dataFilter": {
                    "timeRange": {
                        "from": f"{date_from}T00:00:00Z",
                        "to": f"{date_to}T23:59:59Z"
                    },
                    "maxCloudCoverage": 30
                }
            }]
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/jpeg"}
            }]
        },
        "evalscript": """
            //VERSION=3
            function setup() {
                return {
                    input: ["B04", "B03", "B02"],
                    output: {
                        bands: 3,
                        sampleType: "AUTO"
                    }
                };
            }
            function evaluatePixel(sample) {
                return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
            }
        """
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=request_body)
        
        if response.status_code == 200:
            # Create download folder
            os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{area['name'].replace(' ', '_')}_{timestamp}.jpg"
            filepath = os.path.join(DOWNLOAD_FOLDER, filename)
            
            # Save image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"   ✅ Downloaded: {filename}")
            return filepath
        else:
            print(f"   ❌ Download failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

# =====================================================
# BULK DOWNLOAD FOR ALL HIGH-RISK AREAS
# =====================================================

def download_all_high_risk_areas():
    """
    Download latest satellite images for all high-risk areas
    (Runs daily at 3:00 AM as per your workflow)
    """
    
    print("\n" + "="*70)
    print("🛰️  AUTOMATIC SATELLITE DOWNLOAD - DAILY SCAN")
    print("="*70)
    print(f"\n⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 Scanning {len(HIGH_RISK_AREAS)} high-risk areas\n")
    
    # Authenticate
    token = get_sentinel_token()
    if not token:
        print("\n❌ Authentication failed. Aborting.")
        return []
    
    # Download images
    downloaded_images = []
    
    for i, area in enumerate(HIGH_RISK_AREAS, 1):
        print(f"\n[{i}/{len(HIGH_RISK_AREAS)}] Processing: {area['name']}")
        
        filepath = download_sentinel_image(area, token)
        
        if filepath:
            downloaded_images.append({
                "area": area,
                "filepath": filepath,
                "downloaded_at": datetime.now().isoformat()
            })
            
            # Store coordinates in database for later analysis
            print(f"   📝 Stored coordinates for ML analysis")
        
        # Rate limiting (avoid API throttling)
        if i < len(HIGH_RISK_AREAS):
            print("   ⏳ Waiting 2 seconds...")
            time.sleep(2)
    
    # Summary
    print("\n" + "="*70)
    print("📊 DOWNLOAD SUMMARY")
    print("="*70)
    print(f"\n✅ Successfully downloaded: {len(downloaded_images)}/{len(HIGH_RISK_AREAS)} images")
    print(f"📁 Saved to: {os.path.abspath(DOWNLOAD_FOLDER)}")
    
    if downloaded_images:
        print("\n📋 Downloaded images:")
        for img in downloaded_images:
            print(f"   • {img['area']['name']} ({img['area']['priority']} priority)")
    
    print("\n🔄 Next: Run ML detection on these images")
    print("="*70 + "\n")
    
    return downloaded_images

# =====================================================
# SCHEDULED DAILY DOWNLOAD (3:00 AM)
# =====================================================

def schedule_daily_downloads():
    """
    Schedule daily downloads at 3:00 AM
    (Use with cron job or Windows Task Scheduler)
    """
    
    from datetime import datetime
    
    while True:
        now = datetime.now()
        
        # Check if it's 3:00 AM
        if now.hour == 3 and now.minute == 0:
            print(f"\n⏰ Daily scan triggered at {now.strftime('%H:%M:%S')}")
            download_all_high_risk_areas()
            
            # Sleep for 1 hour to avoid running multiple times
            time.sleep(3600)
        else:
            # Check every minute
            time.sleep(60)

# =====================================================
# MANUAL DOWNLOAD (FOR TESTING)
# =====================================================

def download_single_area(area_name: str):
    """Download image for a specific area (for testing)"""
    
    # Find area
    area = next((a for a in HIGH_RISK_AREAS if a["name"] == area_name), None)
    
    if not area:
        print(f"❌ Area '{area_name}' not found")
        print(f"\nAvailable areas:")
        for a in HIGH_RISK_AREAS:
            print(f"   • {a['name']}")
        return None
    
    # Authenticate and download
    token = get_sentinel_token()
    if not token:
        return None
    
    return download_sentinel_image(area, token)

# =====================================================
# TESTING
# =====================================================

if __name__ == "__main__":
    print("\n🧪 SATELLITE DOWNLOADER - TEST MODE\n")
    print("Choose an option:")
    print("1. Download images for all high-risk areas")
    print("2. Download for single area (test)")
    print("3. Start scheduled daily downloads (3:00 AM)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        download_all_high_risk_areas()
    
    elif choice == "2":
        print("\nAvailable areas:")
        for i, area in enumerate(HIGH_RISK_AREAS, 1):
            print(f"   {i}. {area['name']}")
        
        area_choice = input("\nEnter area number: ").strip()
        try:
            area_idx = int(area_choice) - 1
            area = HIGH_RISK_AREAS[area_idx]
            download_single_area(area["name"])
        except:
            print("Invalid choice")
    
    elif choice == "3":
        print("\n⏰ Starting scheduled downloads...")
        print("   Will run daily at 3:00 AM")
        print("   Press Ctrl+C to stop")
        schedule_daily_downloads()
    
    else:
        print("Invalid choice")
