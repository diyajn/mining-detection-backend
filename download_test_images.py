# Run this Python script to download test images

import requests
import os

# Create folder
os.makedirs('test-images', exist_ok=True)

# Test image URLs (satellite-like images)
test_images = [
    {
        'url': 'https://picsum.photos/seed/mining1/800/600',
        'filename': 'test_mining_1.jpg',
        'type': 'Should detect mining (brown terrain)'
    },
    {
        'url': 'https://picsum.photos/seed/forest1/800/600', 
        'filename': 'test_forest_1.jpg',
        'type': 'Should NOT detect mining (green forest)'
    },
    {
        'url': 'https://picsum.photos/seed/mining2/800/600',
        'filename': 'test_mining_2.jpg',
        'type': 'Should detect mining'
    }
]

for img in test_images:
    print(f"Downloading: {img['filename']}")
    response = requests.get(img['url'])
    with open(f"test-images/{img['filename']}", 'wb') as f:
        f.write(response.content)
    print(f"  ✅ {img['type']}")

print("\n✅ All test images downloaded to test-images/ folder")