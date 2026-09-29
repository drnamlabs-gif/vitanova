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
MUTATION_RATE = 0.01

NUMBER_OF_FOOD_SOURCES = 20

def draw_gridlines(screen, width, height, spacing):
    # Draw vertical lines
    for x in range(0, width, spacing):
        pygame.draw.line(screen, GRIDLINE_COLOR, (x, 0), (x, height))
    # Draw horizontal lines
    for y in range(0, height, spacing):
        pygame.draw.line(screen, GRIDLINE_COLOR, (0, y), (width, y))
        
# Define roles
GUARD = "guard"
GATHERER = "gatherer"

class Boid:
    def __init__(self, x, y, genotype=None, role=None):
        self.x = x
        self.y = y
        self.genotype = genotype if genotype else self.random_genotype()
        self.role = role if role else random.choice([GUARD, GATHERER])
        
        # Role-specific attributes
        if self.role == GUARD:
            self.energy = 100  # Example attribute for guards
            # Add other guard-specific attributes or behaviors
        elif self.role == GATHERER:
            self.food_collected = 0  # Example attribute for gatherers
        
        self.speed, self.sep_tendency, self.align_tendency, self.cohesion_tendency = self.genotype
        self.dx = random.uniform(-1, 1) * self.speed
        self.dy = random.uniform(-1, 1) * self.speed
        self.size = BOID_SIZE

    def random_genotype(self):
        # Increase variability in genotype
        speed = random.uniform(1, 3)
        sep_tendency = random.uniform(-1, 1)  # Increased range for separation tendency
        align_tendency = random.uniform(-1, 1)  # Reduced range for alignment tendency
        cohesion_tendency = random.uniform(-1, 1)  # Reduced range for cohesion tendency
        return (speed, sep_tendency, align_tendency, cohesion_tendency)


    def mutate_genotype(self):
        mutated = list(self.genotype)
        for i in range(len(mutated)):
            if random.random() < MUTATION_RATE:
                mutated[i] += random.uniform(-0.1, 0.1)  # Small mutation
        self.genotype = tuple(mutated)
        self.speed, self.sep_tendency, self.align_tendency, self.cohesion_tendency = self.genotype
        
     # Add role-specific methods
    def protect(self):
        # Guard behavior for protecting gatherers
        if self.role == GUARD:
            # Implement protection logic
            pass

    def gather_food(self):
        # Gatherer behavior for collecting food
        if self.role == GATHERER:
            # Implement food gathering logic
            pass

    def update(self, boids):
        #Adjust behavior based on genotype spectrum
        separation_x, separation_y = self.separation(boids)
        alignment_x, alignment_y = self.alignment(boids)
        cohesion_x, cohesion_y = self.cohesion(boids)
        
        # Apply tendencies based on genotype
        self.dx += separation_x * self.sep_tendency
        self.dy += separation_y * self.sep_tendency
        self.dx += alignment_x * self.align_tendency
        self.dy += alignment_y * self.align_tendency
        self.dx += cohesion_x * self.cohesion_tendency
        self.dy += cohesion_y * self.cohesion_tendency
        
        # Normalize to maintain speed
        magnitude = math.sqrt(self.dx**2 + self.dy**2)
        self.dx = (self.dx / magnitude) * self.speed
        self.dy = (self.dy / magnitude) * self.speed
        
        # Update position and handle boundaries
        self.x += self.dx
        self.y += self.dy
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
                # Prevent division by zero and handle very small distances
                if distance < MIN_DISTANCE:
                    push_strength = SEPARATION_RADIUS / max(distance, 0.1)  # Use a small non-zero value like 0.1
                else:
                    push_strength = 1
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

    def draw(self, screen):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.size)

# Initialize boids
boids = [Boid(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(BOID_COUNT)]


class FoodSource:
    def __init__(self, x, y, amount):
        self.x = x
        self.y = y
        self.amount = amount  # The amount of food available at this source

    def draw(self, screen):
        # Draw the food source as a circle (or any other shape you prefer)
        pygame.draw.circle(screen, (0, 255, 0), (self.x, self.y), 10)  # Green circle

# Initialize food sources
food_sources = [FoodSource(random.randint(0, WIDTH), random.randint(0, HEIGHT), 100) for _ in range(NUMBER_OF_FOOD_SOURCES)]


class Predator:
    
    def __init__(self, x, y, speed=2, visual_range=50):
        self.x = x
        self.y = y
        self.speed = speed
        self.visual_range = visual_range
        self.catch_radius = 15  # Define catch radius here
        self.dx = random.uniform(-1, 1)  # Random initial direction
        self.dy = random.uniform(-1, 1)
        self.is_eating = False
        self.eating_timer = 0
        self.eating_duration = 60  # Number of frames for eating

    
    def detect_boids(self, boids):
        for boid in boids:
            distance = math.sqrt((boid.x - self.x) ** 2 + (boid.y - self.y) ** 2)
            angle_to_boid = math.atan2(boid.y - self.y, boid.x - self.x)
            angle_difference = abs(angle_to_boid - math.atan2(self.dy, self.dx))
            if distance <= self.visual_range and angle_difference <= math.pi / 4:
                return boid
        return None

    def move_towards_prey(self, prey):
        if prey is not None:
            self.dx, self.dy = prey.x - self.x, prey.y - self.y
            distance = math.sqrt(self.dx ** 2 + self.dy ** 2)
            self.dx, self.dy = self.dx / distance * self.speed, self.dy / distance * self.speed
            self.x += self.dx
            self.y += self.dy
            
    def eat_boid(self):
        self.is_eating = True
        self.eating_timer = self.eating_duration

    def update(self, boids):
        if self.is_eating:
            if self.eating_timer > 0:
                self.eating_timer -= 1
            else:
                self.is_eating = False
        else:
            prey = self.detect_boids(boids)
            if prey is not None:
                distance_to_prey = math.sqrt((prey.x - self.x)**2 + (prey.y - self.y)**2)
                if distance_to_prey <= self.catch_radius:
                    self.eat_boid()
                    boids.remove(prey)
                else:
                    self.move_towards_prey(prey)

    def draw(self, screen):
        # Define the size of the triangle
        size = 10

        # Calculate the angle of movement
        angle = math.atan2(self.dy, self.dx)

        # Calculate the three points of the triangle
        point1 = (self.x + math.cos(angle) * size, self.y + math.sin(angle) * size)
        point2 = (self.x + math.cos(angle + 2.5) * size, self.y + math.sin(angle + 2.5) * size)
        point3 = (self.x + math.cos(angle - 2.5) * size, self.y + math.sin(angle - 2.5) * size)

        # Draw the triangle
        pygame.draw.polygon(screen, (0, 0, 255), [point1, point2, point3])
# Spawning predators near food sources
predators = []
for food_source in food_sources:
    # Assuming predators spawn within a certain radius around food sources
    predator_x = random.randint(food_source.x - 50, food_source.x + 50)
    predator_y = random.randint(food_source.y - 50, food_source.y + 50)
    predators.append(Predator(predator_x, predator_y))


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-organisation")
clock = pygame.time.Clock()

running = True
while running:
    screen.fill(WHITE)
    
    # Draw gridlines
    draw_gridlines(screen, WIDTH, HEIGHT, GRID_SPACING)
    
    # draw the food sources
    for food_source in food_sources:
        food_source.draw(screen)

    # Update and draw boids
    for boid in boids:
        boid.update(boids)
        boid.draw(screen)
        
    # In your main game loop, update and draw predators
    for predator in predators:
        predator.update(boids)
        predator.draw(screen)

    # Display update and frame rate control
    pygame.display.flip()
    clock.tick(30)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
