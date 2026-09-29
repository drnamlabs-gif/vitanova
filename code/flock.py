import pygame
import random
import math

random.seed(2023)

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Boid parameters
BOID_COUNT = 100
SPEED = 2
SEPARATION_RADIUS = 20
ALIGNMENT_RADIUS = 40
COHESION_RADIUS = 40

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Additional constants for gridlines
GRIDLINE_COLOR = (50, 50, 50)  # Blurring black color
GRID_SPACING = 50  # Spacing between gridlines

MIN_DISTANCE = 10  # Minimum distance to maintain between boids to prevent overlap
BOID_SIZE = 5

def draw_gridlines(screen, width, height, spacing):
    # Draw vertical lines
    for x in range(0, width, spacing):
        pygame.draw.line(screen, GRIDLINE_COLOR, (x, 0), (x, height))
    # Draw horizontal lines
    for y in range(0, height, spacing):
        pygame.draw.line(screen, GRIDLINE_COLOR, (0, y), (width, y))

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-organisation")
clock = pygame.time.Clock()

class Boid:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-1, 1)
        
        self.size  = BOID_SIZE

    def update(self, boids):
        # Calculate behavior adjustments
        separation_x, separation_y = self.separation(boids)
        alignment_x, alignment_y = self.alignment(boids)
        cohesion_x, cohesion_y = self.cohesion(boids)

        # Update velocity and position
        self.dx += separation_x + alignment_x + cohesion_x
        self.dy += separation_y + alignment_y + cohesion_y
        magnitude = math.sqrt(self.dx**2 + self.dy**2)
        self.dx = (self.dx / magnitude) * SPEED
        self.dy = (self.dy / magnitude) * SPEED

        self.x += self.dx
        self.y += self.dy

        # Boundary conditions
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))

    def separation(self, boids):
        separation_x, separation_y = 0, 0
        for boid in boids:
            if boid == self:
                continue
            distance = math.sqrt((boid.x - self.x)**2 + (boid.y - self.y)**2)
            if distance < SEPARATION_RADIUS:
                # Enhanced separation: stronger push if too close
                push_strength = 1 if distance >= MIN_DISTANCE else SEPARATION_RADIUS / distance
                separation_x += (self.x - boid.x) * push_strength
                separation_y += (self.y - boid.y) * push_strength
        return separation_x, separation_y

    def alignment(self, boids):
        alignment_x, alignment_y = 0, 0
        alignment_count = 0
        for boid in boids:
            distance = math.sqrt((boid.x - self.x)**2 + (boid.y - self.y)**2)
            if distance < ALIGNMENT_RADIUS:
                alignment_x += boid.dx
                alignment_y += boid.dy
                alignment_count += 1
        if alignment_count > 0:
            alignment_x /= alignment_count
            alignment_y /= alignment_count
        return alignment_x, alignment_y

    def cohesion(self, boids):
        cohesion_x, cohesion_y = 0, 0
        cohesion_count = 0
        for boid in boids:
            distance = math.sqrt((boid.x - self.x)**2 + (boid.y - self.y)**2)
            if distance < COHESION_RADIUS:
                cohesion_x += boid.x
                cohesion_y += boid.y
                cohesion_count += 1
        if cohesion_count > 0:
            cohesion_x /= cohesion_count
            cohesion_y /= cohesion_count
            cohesion_x -= self.x
            cohesion_y -= self.y
        return cohesion_x, cohesion_y
    
    def draw1(self, screen):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.size)
    
    def draw(self, screen):
        # Define the size of the triangle
        size = BOID_SIZE

        # Calculate the angle of movement
        angle = math.atan2(self.dy, self.dx)

        # Calculate the three points of the triangle
        point1 = (self.x + math.cos(angle) * size, self.y + math.sin(angle) * size)
        point2 = (self.x + math.cos(angle + 2.5) * size, self.y + math.sin(angle + 2.5) * size)
        point3 = (self.x + math.cos(angle - 2.5) * size, self.y + math.sin(angle - 2.5) * size)

        # Draw the triangle
        pygame.draw.polygon(screen, (255, 0, 0), [point1, point2, point3])

# Initialize boids
boids = [Boid(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(BOID_COUNT)]

running = True
while running:
    screen.fill(WHITE)
    
    # Draw gridlines
    draw_gridlines(screen, WIDTH, HEIGHT, GRID_SPACING)

    for boid in boids:
        boid.update(boids)
        boid.draw(screen)

    pygame.display.flip()
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
