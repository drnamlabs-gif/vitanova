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
ENERGY_COST = 0.2  # Energy lost per cycle
REPRODUCTION_ENERGY_THRESHOLD = 200  # Energy level required to reproduce
MAX_ENERGY = 201  # Maximum energy a boid can have

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
font = pygame.font.Font(None, 36)  # Choose an appropriate size
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
                'boldness': random.uniform(0, 1),  # New trait
            }
        self.escape_radius = 80  # Example escape radius
        
        self.base_boldness = self.genotype['boldness']  # Base boldness from the genotype
        self.current_boldness = self.base_boldness  # Current boldness, which can dynamically change
        self.boldness_increase = 0.1  # Increase in boldness for each nearby boid
            
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
        
        
    def adjust_boldness(self, boids):
        nearby_boids_count = sum(1 for boid in boids if distance_to(self, boid) < VISION_RADIUS and boid != self)
        if nearby_boids_count > 0:
            # Increase boldness based on the number of nearby boids
            self.current_boldness = min(self.base_boldness + self.boldness_increase * nearby_boids_count, 1)
        else:
            # Reset boldness to base level
            self.current_boldness = self.base_boldness
            
    # def update(self, boids, resources, predator):
    #     self.detect_resources(resources)
        
    #     # Initialize vectors for each behavior
    #     escape_vector = [0, 0]
    #     forage_vector = [0, 0]
    #     flocking_vector = [0, 0]
        
    #     # Escape behavior
    #     if self.is_predator_close(predator):
    #         escape_vector = self.calculate_escape_vector(predator)
            
    #     # Foraging behavior
    #     if self.nearby_resources:
    #         forage_vector = self.calculate_forage_vector(self.nearby_resources[0])
        
    #     # Flocking behavior
    #     flocking_vector = self.calculate_flocking_vector(boids)
        
    #     # Combine behaviors
    #     # Adjust these weights based on the situation or boid's genotype
    #     escape_weight = 0.3 if self.is_predator_close(predator) else 0
    #     forage_weight = 0.3 if self.nearby_resources else 0
    #     flocking_weight = 0.3  # Always apply some level of flocking behavior
        
    #     combined_dx = (escape_vector[0] * escape_weight + forage_vector[0] * forage_weight + flocking_vector[0] * flocking_weight)
    #     combined_dy = (escape_vector[1] * escape_weight + forage_vector[1] * forage_weight + flocking_vector[1] * flocking_weight)
        
    #     # Normalize the resulting vector
    #     magnitude = math.sqrt(combined_dx**2 + combined_dy**2)
    #     if magnitude > 0:
    #         self.dx = (combined_dx / magnitude) * SPEED
    #         self.dy = (combined_dy / magnitude) * SPEED

    #     # Update position and apply boundary conditions
    #     self.x += self.dx
    #     self.y += self.dy
    #     self.apply_boundary_conditions()
        
    #     # Consume resources if available
    #     self.consume_resources(resources)
        
    #     # Check energy levels
    #     return self.check_energy_levels()  # Returns True if the boid dies
    
    def calculate_escape_vector(self, predator):
        # Vector pointing away from the predator
        escape_dx = self.x - predator.x
        escape_dy = self.y - predator.y
        
        # Normalize the escape vector
        magnitude = math.sqrt(escape_dx**2 + escape_dy**2)
        if magnitude > 0:
            escape_dx /= magnitude
            escape_dy /= magnitude
        
        return [escape_dx, escape_dy]

    def calculate_forage_vector(self, resource):
        # Vector pointing towards the resource
        forage_dx = resource.x - self.x
        forage_dy = resource.y - self.y
        
        # Normalize the forage vector
        magnitude = math.sqrt(forage_dx**2 + forage_dy**2)
        if magnitude > 0:
            forage_dx /= magnitude
            forage_dy /= magnitude
        
        return [forage_dx, forage_dy]
    
    def calculate_flocking_vector(self, boids):
        # Initialize vectors for alignment, cohesion, and separation
        alignment_vector = [0, 0]
        cohesion_vector = [0, 0]
        separation_vector = [0, 0]
        
        # Calculate each vector
        alignment_vector = self.alignment(boids)
        cohesion_vector = self.cohesion(boids)
        separation_vector = self.separation(boids)
        
        # Combine vectors
        flocking_dx = alignment_vector[0] + cohesion_vector[0] + separation_vector[0]
        flocking_dy = alignment_vector[1] + cohesion_vector[1] + separation_vector[1]
        
        # Normalize the flocking vector
        magnitude = math.sqrt(flocking_dx**2 + flocking_dy**2)
        if magnitude > 0:
            flocking_dx /= magnitude
            flocking_dy /= magnitude
        
        return [flocking_dx, flocking_dy]

    
    def update(self, boids, resources, predator):
        self.detect_resources(resources)
        
        # Adjust current boldness based on nearby boids
        self.adjust_boldness(boids)
        
        if self.is_predator_close(predator) and random.random() > self.current_boldness:
            self.escape(predator)
       
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
        
        if self.is_predator_close(predator) and random.random() > self.current_boldness:
            self.escape(predator)
        elif self.nearby_resources:
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
    
    def is_predator_close(self, predator):
        return distance_to(self, predator) < self.escape_radius
    
    def escape(self, predator):
        # Calculate a vector pointing away from the predator
        escape_dx = self.x - predator.x
        escape_dy = self.y - predator.y

        # Normalize the escape direction
        magnitude = math.sqrt(escape_dx**2 + escape_dy**2)
        if magnitude > 0:
            escape_dx /= magnitude
            escape_dy /= magnitude

        # Apply escape velocity, perhaps with a boost to make it more effective
        escape_speed = SPEED * 1.5  # Escape faster than normal speed
        self.dx = escape_dx * escape_speed
        self.dy = escape_dy * escape_speed

    def perform_flocking_behaviors(self, boids):
        if not self.nearby_resources:
            separation_x, separation_y = self.separation(boids)
            alignment_x, alignment_y = self.alignment(boids)
            cohesion_x, cohesion_y = self.cohesion(boids)
            self.dx += separation_x + alignment_x + cohesion_x
            self.dy += separation_y + alignment_y + cohesion_y

    def update_position_and_velocity(self):
        # Update position
        self.x += self.dx
        self.y += self.dy

        # Check for boundary collision and reverse direction if needed
        if self.x <= 0 or self.x >= WIDTH:
            self.dx *= -1  # Reverse horizontal direction
        if self.y <= 0 or self.y >= HEIGHT:
            self.dy *= -1  # Reverse vertical direction

        # Normalize velocity to maintain constant speed
        magnitude = math.sqrt(self.dx**2 + self.dy**2)
        if magnitude > 0:
            self.dx = (self.dx / magnitude) * SPEED
            self.dy = (self.dy / magnitude) * SPEED

        # Ensure boid stays within bounds
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))

    def apply_boundary_conditions(self):
        # Check for boundary collision and reverse direction if needed
        if self.x <= 0 or self.x >= WIDTH:
            self.dx *= -1  # Reverse horizontal direction
        if self.y <= 0 or self.y >= HEIGHT:
            self.dy *= -1  # Reverse vertical direction
        
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))

    def check_energy_levels(self):
        return self.energy <= 0  # Returns True if the boid's energy is depleted

    def separation(self, boids):
        # if random.random() > self.genotype['flocking_affinity']:
           # return 0, 0
       
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
        # if random.random() > self.genotype['flocking_affinity']:
           # return 0, 0
        
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
        # if random.random() > self.genotype['flocking_affinity']:
           # return 0, 0
        
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
        mutated_genotype['boldness'] = min(max(mutated_genotype['boldness'], 0), 1)
        return mutated_genotype
    
    def draw(self, screen):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.size)


class Predator:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 3  # Faster than boids
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-1, 1)
        self.vision_radius = 100
        self.catch_radius = 15
        
        self.consuming = False  # Indicates if the predator is in the consuming state
        self.consume_time = 0  # Timer for consuming
        self.consume_duration = 100  # Number of frames to consume for (adjust as needed)

        self.flock_threshold = 10  # Threshold for number of boids to be considered a flock
        self.hesitation_duration = 50  # Frames to hesitate        

    def update_position(self):
        # Check for boundary collision and reverse direction if needed
        if self.x <= 0 or self.x >= WIDTH:
            self.dx *= -1  # Reverse horizontal direction
        if self.y <= 0 or self.y >= HEIGHT:
            self.dy *= -1  # Reverse vertical direction
        # Only update position if not consuming
        if not self.consuming:
            self.x += self.dx
            self.y += self.dy

            # Boundary conditions
            self.x = max(0, min(self.x, WIDTH))
            self.y = max(0, min(self.y, HEIGHT))

    def hunt(self, boids):
        if self.consuming:
            # Decrease consume time and return if still consuming
            self.consume_time -= 1
            if self.consume_time > 0:
                return None
            else:
                self.consuming = False  # End consuming state
                
        # Find visible boids and calculate flock sizes
        visible_boids = [boid for boid in boids if distance_to(self, boid) < self.vision_radius]
        flock_sizes = self.calculate_flock_sizes(visible_boids)

        # Hesitate or change direction if encountering a large flock
        if any(size >= self.flock_threshold for size in flock_sizes.values()):
            if self.consuming:
                return None  # Continue consuming if already doing so
            # Hesitate or move randomly
            self.hesitate_or_change_direction()
            return None

        visible_boids = [boid for boid in boids if distance_to(self, boid) < self.vision_radius]
        if visible_boids:
            closest_boid = min(visible_boids, key=lambda boid: distance_to(self, boid))
            return self.target_boid(closest_boid)
        return None
    
    def calculate_flock_sizes(self, visible_boids):
        # Define a proximity threshold to consider boids as part of the same flock
        flock_proximity_threshold = 50
        
        # Dictionary to hold flock sizes for each boid
        flock_sizes = {boid: 0 for boid in visible_boids}
        
        # Iterate over each boid to determine flock sizes
        for boid in visible_boids:
            # Initialize a counter for the number of boids in the same flock
            flock_count = 0
            for other_boid in visible_boids:
                if boid != other_boid:
                    distance = distance_to(boid, other_boid)
                    if distance < flock_proximity_threshold:
                        flock_count += 1
            # Update the flock size for this boid
            flock_sizes[boid] = flock_count

        return flock_sizes


    def hesitate_or_change_direction(self):
        # Hesitate or change direction logic
        # If hesitation timer is still active, decrease it
        if self.hesitation_duration > 0:
            self.hesitation_duration -= 1
            return  # Do not move while hesitating

        # Reset hesitation timer and change direction
        self.hesitation_duration = 50
        angle_change = random.uniform(-math.pi, math.pi)
        self.dx = math.cos(angle_change) * self.speed
        self.dy = math.sin(angle_change) * self.speed


    def target_boid(self, boid):
        direction_x = boid.x - self.x
        direction_y = boid.y - self.y
        distance = math.sqrt(direction_x**2 + direction_y**2)
        if distance < self.catch_radius:
            self.consuming = True  # Enter consuming state
            self.consume_time = self.consume_duration  # Reset consume timer
            return boid
        direction_x /= distance
        direction_y /= distance
        self.dx = direction_x * self.speed
        self.dy = direction_y * self.speed
        return None

    def draw(self, screen):
        # Calculate triangle vertices
        center = (int(self.x), int(self.y))
        angle = math.atan2(self.dy, self.dx)
        tip = (center[0] + 15 * math.cos(angle), center[1] + 15 * math.sin(angle))
        left = (center[0] + 10 * math.cos(angle + 2.5), center[1] + 10 * math.sin(angle + 2.5))
        right = (center[0] + 10 * math.cos(angle - 2.5), center[1] + 10 * math.sin(angle - 2.5))
        pygame.draw.polygon(screen, (0, 0, 139), [tip, left, right])  # Dark Blue triangle


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

# Create a predator instance
predator = Predator()


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
    
    # Predator logic
    caught_boid = predator.hunt(boids)
    if caught_boid:
        boids.remove(caught_boid)  # Remove the caught boid

    predator.update_position()
    
    # Drawing the predator
    predator.draw(screen)
    
    if predator.consuming:
        consume_text = font.render("The predator is eating a boid prey", True, (122, 122, 0))  # Red text
        screen.blit(consume_text, (50, 50))  # Position the text on the screen
    
    # Update boids
    for boid in boids[:]:
        offspring = boid.reproduce()
        if offspring:
            boids.append(offspring)
        if boid.update(boids, resources, predator):
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
