import pygame
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


random.seed(2023)

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Boid parameters
BOID_COUNT = 100
SPEED = 2

VISION_RADIUS = 40  # Distance within which a boid perceives other boids and resources

SEPARATION_RADIUS = VISION_RADIUS
ALIGNMENT_RADIUS = VISION_RADIUS
COHESION_RADIUS = VISION_RADIUS
BOID_SIZE = 5

PROTECTION_RADIUS = 30  # Boids within this radius are considered close enough for mutual protection
ENERGY_SAVING_FACTOR = 0.5  # Reduction factor for energy loss when protected


# Energy parameters
INITIAL_ENERGY = 100
ENERGY_COST = 0.5  # Energy lost per cycle
REPRODUCTION_ENERGY_THRESHOLD = 200  # Energy level required to reproduce
MAX_ENERGY = 10000  # Maximum energy a boid can have

# Resource appearance
RESOURCE_SIZE = 10
NUM_RESOURCES = 10
RESOURCE_COLOR = (0, 255, 0)  # Color for resources (if needed)

# Resource interaction parameters
CONSUME_DISTANCE = (BOID_SIZE / 2 + RESOURCE_SIZE / 2) * 1.1  # 10% more than the sum of radii


ENERGY_CONSUMPTION_RATE_A = 10  # Energy consumed per cycle from Type A resources
ENERGY_CONSUMPTION_RATE_B = 20  # Energy consumed per cycle from Type B resources (higher rate)

# Resource types and regeneration
RESOURCE_A_ENERGY = 200  # Energy from Type A resources
RESOURCE_B_ENERGY = 800  # Energy from Type B resources
RESOURCE_A_ABUNDANCE = 0.8  # Proportion of Type A resources
RESOURCE_B_ABUNDANCE = 0.2  # Proportion of Type B resources
RESOURCE_A_REGEN_RATE = 1  # Regen rate for Type A resources
RESOURCE_B_REGEN_RATE = 2  # Regen rate for Type B resources
RESOURCE_A_COLOR = (100, 180, 100)  # Color for Resource A
RESOURCE_B_COLOR = (180, 100, 100)  # Color for Resource B

# Maximum energy levels for each resource type
MAX_RESOURCE_A_ENERGY = 100  # Maximum energy for Type A resources
MAX_RESOURCE_B_ENERGY = 400  # Maximum energy for Type B resources


# Area for Type B resources
TYPE_B_AREA_CENTER = (WIDTH * 3 / 4, HEIGHT * 3 / 4)
TYPE_B_AREA_RADIUS = 100
SPAWN_AREA_PADDING = 150  # Avoid spawning boids too close to Type B area

# Evolution parameters
MUTATION_RATE = 0.05  # Chance of mutation in offspring

# Colors for visualization
WHITE = (255, 255, 255)  # Background color
RED = (255, 0, 0)  # Boid color

# Gridlines for visualization
GRIDLINE_COLOR = (50, 50, 50)  # Color for gridlines
GRID_SPACING = 50  # Spacing between gridlines

def draw_gridlines(screen, width, height, spacing):
    # Draw vertical lines
    for x in range(0, width, spacing):
        pygame.draw.line(screen, GRIDLINE_COLOR, (x, 0), (x, height))
    # Draw horizontal lines
    for y in range(0, height, spacing):
        pygame.draw.line(screen, GRIDLINE_COLOR, (0, y), (width, y))

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Evolving Self-organisation")
clock = pygame.time.Clock()


def distance_to(boid, other):
        return math.sqrt((boid.x - other.x) ** 2 + (boid.y - other.y) ** 2)

class Resource:
    def __init__(self, resource_type, x, y):
        self.x = x
        self.y = y
        # self.x = random.randint(0, WIDTH)
        # self.y = random.randint(0, HEIGHT)
        self.type = resource_type
        self.energy = RESOURCE_A_ENERGY if resource_type == 'A' else RESOURCE_B_ENERGY
        
        self.regen_rate = RESOURCE_A_REGEN_RATE if resource_type == 'A' else RESOURCE_B_REGEN_RATE

        self.max_energy = MAX_RESOURCE_A_ENERGY if resource_type == 'A' else MAX_RESOURCE_B_ENERGY

    def deplete_energy(self, amount):
        self.energy -= amount
        if self.energy < 0:
            self.energy = 0

    def regenerate_energy(self):
        self.energy += self.regen_rate
        if self.energy > self.max_energy:
            self.energy = self.max_energy
            
    def draw(self, screen):
        # Draw the resource circle
        color = RESOURCE_A_COLOR if self.type == 'A' else RESOURCE_B_COLOR
        pygame.draw.circle(screen, color, (self.x, self.y), RESOURCE_SIZE)

        # Draw the letter inside the circle
        font = pygame.font.Font(None, 24)
        text = font.render(self.type, True, (0, 0, 0))  # Black text
        text_rect = text.get_rect(center=(self.x, self.y))
        screen.blit(text, text_rect)

class Boid:
    def __init__(self, x, y, genotype=None):
        self.x = x
        self.y = y
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-1, 1)
        
        self.size  = BOID_SIZE
        
        self.energy = INITIAL_ENERGY  # Starting energy level
        
        self.nearby_resources = []  # Initialize the list of nearby resources
        
        self.is_flocking = False  # Initialize the flocking status
        
        # Define genotype
        if genotype:
            self.genotype = genotype
        else:
            # Random initial genotype
            self.genotype = {
                'flocking_affinity': random.uniform(0.1, 0.9),  # Lower initial affinity
                'separation': random.uniform(0.1, 0.9),
                'alignment': random.uniform(0.1, 0.9),
                'cohesion': random.uniform(0.1, 0.9),
                
                'resource_perception': random.uniform(0.5 * VISION_RADIUS, VISION_RADIUS),
                'steer_towards_resource': random.uniform(0.1, 0.5),
                'independence_tendency': random.uniform(0, 1),  # Ranges from 0 (flock-oriented) to 1 (independent)
            }
            
    def check_for_protection(self, boids):
        nearby_boids = [boid for boid in boids if distance_to(self, boid) < PROTECTION_RADIUS]
        return len(nearby_boids) >= 1  # Returns True if at least one other boid is nearby
            
    def detect_resources(self, resources):
        self.nearby_resources = [resource for resource in resources 
                                 if math.sqrt((self.x - resource.x)**2 + (self.y - resource.y)**2) 
                                 < self.genotype['resource_perception']]
    
    def move_towards_resource(self, resource):
        # Calculate the vector pointing from the boid to the resource
        resource_direction_x = resource.x - self.x
        resource_direction_y = resource.y - self.y

        # Normalize the direction vector
        distance_to_resource = math.sqrt(resource_direction_x**2 + resource_direction_y**2)
        # distance_to_resource = distance_to(self, resource)
        if distance_to_resource > 0:
            resource_direction_x /= distance_to_resource
            resource_direction_y /= distance_to_resource

        # Steer towards the resource
        steer_strength = 1#self.genotype['steer_towards_resource']
        self.dx += resource_direction_x * steer_strength
        self.dy += resource_direction_y * steer_strength
    
    def consume_resources(self, resources):
        for resource in resources:
            # distance = math.sqrt((self.x - resource.x)**2 + (self.y - resource.y)**2)
            distance = distance_to(self, resource)
            if distance < CONSUME_DISTANCE and self.energy < MAX_ENERGY:
                # Determine the consumption rate based on resource type
                consumption_rate = ENERGY_CONSUMPTION_RATE_A if resource.type == 'A' else ENERGY_CONSUMPTION_RATE_B
                consume_amount = min(consumption_rate, resource.energy)

                self.energy += consume_amount
                resource.deplete_energy(consume_amount)

                if self.energy >= MAX_ENERGY:
                    self.energy = MAX_ENERGY  # Set energy to max limit
                    break  # Stop consuming when reaching max energy
    
    def random_movement(self):
        # Random change in direction
        angle_change = random.uniform(-0.2, 0.2)  # Adjust this range as needed
        self.dx += math.sin(angle_change)
        self.dy += math.cos(angle_change)
    
    def update(self, boids, resources):
        # Detect resources
        self.detect_resources(resources)
       
        # Decide behavior based on detected resources and boid's tendencies
        if random.random() >= self.genotype['independence_tendency']:
            # If boid prefers flocking
            self.perform_flocking_behaviors(boids)
            self.is_flocking = True  # The boid is flocking
        else:
            # If boid prefers foraging
            if self.nearby_resources:
                # Move towards the closest resource if it exists
                self.move_towards_resource(self.nearby_resources[0])
            self.is_flocking = False  # The boid is not flocking

        # Check if the boid can consume any nearby resource regardless of its primary behavior
        self.consume_resources(resources) if self.nearby_resources else None
        
        
        # Check if the boid is protected by nearby boids
        is_protected = self.check_for_protection(boids)

        # Adjust energy loss based on protection status
        energy_loss = ENERGY_COST
        if is_protected:
            energy_loss *= (1 - ENERGY_SAVING_FACTOR)

        # Apply energy loss
        self.energy -= energy_loss

        # Check energy levels to determine if the boid survives
        if self.check_energy_levels():
            return True  # Boid dies
        
        self.update_position_and_velocity()
        self.apply_boundary_conditions()
        return False # boid survives

    def perform_flocking_behaviors(self, boids):
        if not self.nearby_resources:
            separation_x, separation_y = self.separation(boids)
            alignment_x, alignment_y = self.alignment(boids)
            cohesion_x, cohesion_y = self.cohesion(boids)
            self.dx += separation_x + alignment_x + cohesion_x
            self.dy += separation_y + alignment_y + cohesion_y

    def update_position_and_velocity(self):
        magnitude = math.sqrt(self.dx**2 + self.dy**2)
        if magnitude > 0:
            self.dx = (self.dx / magnitude) * SPEED
            self.dy = (self.dy / magnitude) * SPEED
        self.x += self.dx
        self.y += self.dy

    def apply_boundary_conditions(self):
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))

    def check_energy_levels(self):
        return self.energy <= 0  # Returns True if the boid's energy is depleted

    def separation(self, boids):
        if random.random() > self.genotype['flocking_affinity']:
           return 0, 0
       
        # Enhanced separation: influenced by genotype
        separation_strength = self.genotype['separation']
        
        separation_x, separation_y = 0, 0
        for boid in boids:
            if boid == self:
                continue
            # distance = math.sqrt((boid.x - self.x)**2 + (boid.y - self.y)**2)
            distance = distance_to(self, boid)
            if distance < 1e-5:
                # Apply a default separation or just skip this boid
                separation_x += random.uniform(-1, 1) * separation_strength
                separation_y += random.uniform(-1, 1) * separation_strength
                return separation_x, separation_y
            
            if distance < SEPARATION_RADIUS:
                separation_x += (self.x - boid.x) / distance * separation_strength
                separation_y += (self.y - boid.y) / distance * separation_strength
               
                # Enhanced separation: stronger push if too close
                # push_strength = 1 if distance >= MIN_DISTANCE else SEPARATION_RADIUS / distance
                # separation_x += (self.x - boid.x) * push_strength
                # separation_y += (self.y - boid.y) * push_strength
        return separation_x, separation_y

    def alignment(self, boids):
        if random.random() > self.genotype['flocking_affinity']:
           return 0, 0
        
        alignment_x, alignment_y = 0, 0
        alignment_count = 0
        alignment_strength = self.genotype['alignment']
        for boid in boids:
            # distance = math.sqrt((boid.x - self.x)**2 + (boid.y - self.y)**2)
            distance = distance_to(self, boid)
            if distance < ALIGNMENT_RADIUS:
                alignment_x += boid.dx
                alignment_y += boid.dy
                alignment_count += 1
        if alignment_count > 0:
            alignment_x = (alignment_x / alignment_count - self.dx) * alignment_strength
            alignment_y = (alignment_y / alignment_count - self.dy) * alignment_strength
        return alignment_x, alignment_y

    def cohesion(self, boids):
        if random.random() > self.genotype['flocking_affinity']:
           return 0, 0
        
        cohesion_x, cohesion_y = 0, 0
        cohesion_count = 0
        cohesion_strength = self.genotype['cohesion']
        for boid in boids:
            # distance = math.sqrt((boid.x - self.x)**2 + (boid.y - self.y)**2)
            distance = distance_to(self, boid)
            if distance < COHESION_RADIUS:
                cohesion_x += boid.x
                cohesion_y += boid.y
                cohesion_count += 1
        if cohesion_count > 0:
            cohesion_x = ((cohesion_x / cohesion_count - self.x) * cohesion_strength)
            cohesion_y = ((cohesion_y / cohesion_count - self.y) * cohesion_strength)
        return cohesion_x, cohesion_y
    
    def reproduce(self):
        if self.energy >= REPRODUCTION_ENERGY_THRESHOLD:
            self.energy /= 2  # Split energy between parent and offspring
            offspring_genotype = self.mutate_genotype(self.genotype)
            offspring = Boid(self.x, self.y, genotype=offspring_genotype)
            return offspring
        return None
    
    def mutate_genotype(self, genotype):
        mutated_genotype = genotype.copy()
        for trait in mutated_genotype:
            if random.random() < MUTATION_RATE:
                mutated_genotype[trait] += random.uniform(-0.1, 0.1)
        mutated_genotype['independence_tendency'] = min(max(mutated_genotype['independence_tendency'], 0), 1)
        return mutated_genotype
    
    def draw(self, screen):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.size)


# Initialize resources
def create_resources():
    resources = []
    num_resources_b = int(NUM_RESOURCES * RESOURCE_B_ABUNDANCE)  # 20% of total resources

    # Create Type B resources
    for _ in range(num_resources_b):
        # Alternate between upper right and bottom right corners
        # if random.random() < 0.5:
            # Upper right corner
            # x = random.randint(WIDTH // 2, WIDTH - RESOURCE_SIZE)
            # y = random.randint(0, HEIGHT // 4 - RESOURCE_SIZE)
        # else:
        # Bottom right corner
        x = random.randint(WIDTH // 2, WIDTH - RESOURCE_SIZE)
        y = random.randint(3 * HEIGHT // 4, HEIGHT - RESOURCE_SIZE)

        resources.append(Resource('B', x, y))

    # Create Type A resources
    for _ in range(NUM_RESOURCES - num_resources_b):
        # Left half of the screen
        x = random.randint(0, WIDTH // 2 - RESOURCE_SIZE)
        y = random.randint(0, HEIGHT - RESOURCE_SIZE)
        resources.append(Resource('A', x, y))

    return resources



resources = create_resources()

def is_safe_spawn_location(x, y):
    """Check if the spawn location is safely away from Type B resource area."""
    distance_to_b_area = math.sqrt((x - TYPE_B_AREA_CENTER[0])**2 + (y - TYPE_B_AREA_CENTER[1])**2)
    return distance_to_b_area > TYPE_B_AREA_RADIUS + SPAWN_AREA_PADDING

def create_boids():
    boids = []
    while len(boids) < BOID_COUNT:
        x, y = random.randint(0, WIDTH), random.randint(0, HEIGHT)
        if is_safe_spawn_location(x, y):
            boids.append(Boid(x, y))
    return boids

boids = create_boids()


# Example data structure for tracking information
simulation_data = {
    "population_size": [],
    "average_independence_tendency": [],
    "flocking_behavior_count": [],
    "resource_consumption": {"A": 0, "B": 0},
}

time_step = 0
PRINT_INTERVAL = 100  # Interval at which to print data, e.g., every 100 time steps

# plotting
def update_plot(frame_number, x_data, y_data, line):
    line.set_xdata(x_data)
    line.set_ydata(y_data)
    plt.xlim(min(x_data), max(x_data))
    plt.ylim(min(y_data), max(y_data))
    return line,

fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot(x_data, y_data, 'r-')  # Red line plot
plt.show(block=False)

running = True
while running:
    time_step += 1
    
    screen.fill(WHITE)
    
    # Draw gridlines
    draw_gridlines(screen, WIDTH, HEIGHT, GRID_SPACING)
    
    # Update boids
    for boid in boids[:]:
        offspring = boid.reproduce()
        if offspring:
            boids.append(offspring)
        if boid.update(boids, resources):
            boids.remove(boid)  # Remove boid if it dies
        boid.draw(screen)

    # Draw resources
    for resource in resources:
        resource.draw(screen)
        
    # Regenerate energy in resources
    for resource in resources:
        resource.regenerate_energy()
        
    # Data collection
    independence_tendencies = [boid.genotype['independence_tendency'] for boid in boids]
    flocking_behavior_count = sum(1 for boid in boids if boid.is_flocking)  # Assuming 'is_flocking' is a tracked attribute

    independence_tendencies = [boid.genotype['independence_tendency'] for boid in boids]
    
    if len(boids) > 0:
        average_independence_tendency = sum(independence_tendencies) / len(boids)
    else:
        average_independence_tendency = 0  # Default value when no boids are present

    simulation_data["average_independence_tendency"].append(average_independence_tendency)


    simulation_data["population_size"].append(len(boids))
    # simulation_data["average_independence_tendency"].append(sum(independence_tendencies) / len(boids))
    simulation_data["flocking_behavior_count"].append(flocking_behavior_count)
    
    # Update plot data
    x_data.append(time_step)
    y_data.append(len(boids))
    
    
    if time_step % PRINT_INTERVAL == 0:
        plt.pause(0.01)  # Pause to update the plot
        
        print(f"Time Step: {time_step}")
        print(f"Population Size: {len(boids)}")
        
        # Get the latest average independence tendency value
        latest_avg_independence_tendency = simulation_data["average_independence_tendency"][-1] if simulation_data["average_independence_tendency"] else "N/A"
        print(f"Average Independence Tendency: {latest_avg_independence_tendency}")
        
        print(f"Flocking Behavior Count: {flocking_behavior_count}")
        
        average_energy = sum(boid.energy for boid in boids) / len(boids) if boids else 0
        print(f"Average Energy Level: {average_energy}")
    
    animation = FuncAnimation(fig, update_plot, fargs=(x_data, y_data, line), interval=100)
    plt.show()
    
    pygame.display.flip()
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
