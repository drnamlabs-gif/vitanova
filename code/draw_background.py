from PIL import Image, ImageDraw
import random

# Settings
width, height = 800, 600
grass_color = (50, 220, 70)
water_color = (70, 160, 220)
rock_color = (130, 130, 130)

# Create a new image with a green base
image = Image.new("RGB", (width, height), grass_color)
draw = ImageDraw.Draw(image)

# Function to draw water bodies
def draw_water_bodies(draw, count=5):
    for _ in range(count):
        radius = random.randint(20, 50)
        x = random.randint(radius, width-radius)
        y = random.randint(radius, height-radius)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=water_color)

# Function to draw rocks
def draw_rocks(draw, count=10):
    for _ in range(count):
        radius = random.randint(10, 30)
        x = random.randint(radius, width-radius)
        y = random.randint(radius, height-radius)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=rock_color)

# Draw water bodies and rocks
draw_water_bodies(draw)
draw_rocks(draw)

# Save the image
image.save("ecosystem_background.png")

