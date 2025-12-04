from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Create a simple test image
width, height = 800, 600
img = Image.new('RGB', (width, height), color=(73, 109, 137))

# Add some text
draw = ImageDraw.Draw(img)
text = "Test Image for Verity"
# Use default font
try:
    font = ImageFont.truetype("arial.ttf", 40)
except:
    font = ImageFont.load_default()

# Get text bbox and center it
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
position = ((width - text_width) // 2, (height - text_height) // 2)

draw.text(position, text, fill=(255, 255, 255), font=font)

# Save the image
img.save('test_image1.png')
print("Created test_image1.png")
