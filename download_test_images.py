# Run this Python script to download test images

import requests
import os

# Create folder
os.makedirs('test-images', exist_ok=True)

# Test image URLs (satellite-like images)
test_images = [

    {
        'url': 'https://picsum.photos/id/1003/800/600',
        'filename': 'brown_rocky.jpg'
    },
    {
        'url': 'https://picsum.photos/id/102/800/600',
        'filename': 'green_forest.jpg'
    },
    {
        'url': 'https://picsum.photos/id/1021/800/600',
        'filename': 'gray_rocky.jpg'
    },
    {
        'url': 'https://picsum.photos/id/1002/800/600',
        'filename': 'red_brown.jpg'
    },
    {
        'url': 'https://picsum.photos/id/1036/800/600',
        'filename': 'sandy_beige.jpg'
    },
    {
        'url': 'https://picsum.photos/id/1040/800/600',
        'filename': 'dark_industrial.jpg'
    },
    {
        'url': 'https://picsum.photos/id/1062/800/600',
        'filename': 'copper_orange.jpg'
    },
    {
        'url': 'https://picsum.photos/id/1015/800/600',
        'filename': 'river_water.jpg'
    },
    {
        'url': 'https://picsum.photos/id/1050/800/600',
        'filename': 'coal_black.jpg'
    },
    {
        'url': 'https://picsum.photos/id/1074/800/600',
        'filename': 'red_clay.jpg'
    }

]

for img in test_images:
    print(f"Downloading: {img['filename']}")
    response = requests.get(img['url'])
    with open(f"test-images/{img['filename']}", 'wb') as f:
        f.write(response.content)
    print(f"  ✅ Done")

print("\n✅ All test images downloaded to test-images/ folder")