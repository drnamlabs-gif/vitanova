import pygame
import random
import math
import numpy as np
from scipy.spatial import KDTree

random.seed(2024)

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Boid parameters
BOID_COUNT = 100
SPEED = 0.5
SEPARATION_RADIUS = 50
ALIGNMENT_RADIUS = 40
COHESION_RADIUS = 40
PERCEPTION_RADIUS = 40

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)  # Color for soldier
GREEN = (0, 255, 0)  # Color for worker

# Additional constants for gridlines
GRIDLINE_COLOR = (50, 50, 50)  # Blurring black color
GRID_SPACING = 50  # Spacing between gridlines

MIN_DISTANCE = 10  # Minimum distance to maintain between boids to prevent overlap
BOID_SIZE = 5

# energy and reproduction parameters
INITIAL_ENERGY = 50  # Starting energy level for each boid
REPRODUCTION_ENERGY_THRESHOLD = 200  # Energy level needed to reproduce
REPRODUCTION_COST = 100  # Energy cost of reproduction
MUTATION_RATE = 0.1  # Variability in offspring's genotype
DEATH_ENERGY_LEVEL = 0  # Energy level at which a boid dies
BOID_COUNT = 100  # Initial boid count
PREDATOR_COUNT = 5

WORKER_ENERGY_GAIN = 1.3  # Energy gained by workers per time step
SOLDIER_ENERGY_GAIN = 0.5  # Energy gained by soldiers when near workers per time step
ENERGY_COST_PER_STEP = 1  # Basic energy cost of moving per time step
ESCAPE_ENERGY_COST = 0.2  # Additional energy cost when escaping predators
MUTATION_RATE = 0.05  # Mutation rate in offspring's genotype
INITIAL_ENERGY = 50  # Starting energy level for each boid
DEATH_ENERGY_LEVEL = 0  # Energy level at which a boid dies

PREDATOR_ESCAPE_BONUS = 5  # Energy bonus for successfully evading a predator
FLOCK_COHESION_BONUS = 2  # Energy bonus for maintaining cohesion within a flock
SOLDIER_PROTECTION_RADIUS = 75  # Radius within which soldiers can protect nearby workers
WORKER_FORAGING_RADIUS = 75  # Notional radius for workers' effective foraging area
PROXIMITY_THRESHOLD = 50
flock_threshold = 50

def draw_gridlines(screen, width, height, spacing):
    # Draw vertical lines
    for x in range(0, width, spacing):
        pygame.draw.line(screen, GRIDLINE_COLOR, (x, 0), (x, height))
    # Draw horizontal lines
    for y in range(0, height, spacing):
        pygame.draw.line(screen, GRIDLINE_COLOR, (0, y), (width, y))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, genotype):
        # Initialize weights from the genotype
        self.input_hidden_weights = np.reshape(genotype[:input_size * hidden_size], (input_size, hidden_size))
        self.hidden_output_weights = np.reshape(genotype[input_size * hidden_size:], (hidden_size, output_size))

    def feedforward(self, inputs):
        hidden = np.dot(inputs, self.input_hidden_weights)
        hidden = np.maximum(hidden, 0)  # ReLU activation
        output = np.dot(hidden, self.hidden_output_weights)
        output = np.tanh(output)  # Tanh activation
        return output

class Boid:
    boid_tree = None

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-1, 1)
        self.size = BOID_SIZE
        
        self.flock_id = None  # New attribute for tracking flock membership

        # Neural network controller
        input_size = 11  # Number of sensory inputs
        hidden_size = 10  # Number of hidden nodes
        output_size = 6   # Number of outputs
        genotype_length = (input_size * hidden_size) + (hidden_size * output_size)  # Total number of weights and biases
        self.genotype = np.random.randn(genotype_length)
        self.brain = NeuralNetwork(input_size, hidden_size, output_size, self.genotype)

        self.role = None
        
        self.energy = INITIAL_ENERGY

    @classmethod
    def build_kd_tree(cls, boids):
        boid_positions = [(boid.x, boid.y) for boid in boids]
        cls.boid_tree = KDTree(boid_positions)

    def perceive(self, boids, predators, perception_radius, max_neighbors_considered=3):
        if Boid.boid_tree is None:
            raise ValueError("KD-Tree is not built yet. Call build_kd_tree on Boid class first.")

        # Query the KD-Tree to find the closest boids within the perception radius
        close_indices = Boid.boid_tree.query_ball_point([self.x, self.y], perception_radius)
        close_boids = [boids[i] for i in close_indices if boids[i] != self]

        # Initialize sensory inputs array
        inputs = []

        # Relative position to closest boids
        for i in range(max_neighbors_considered):
            if i < len(close_boids):
                boid = close_boids[i]
                dx, dy = (boid.x - self.x) / perception_radius, (boid.y - self.y) / perception_radius
                inputs.extend([dx, dy])  # Normalized
            else:
                inputs.extend([0, 0])  # Default values

        # Filter predators within perception radius
        visible_predators = [predator for predator in predators if self.distance_to(predator) < PERCEPTION_RADIUS]

        # Find nearest predator
        nearest_predator = None
        if visible_predators:
            nearest_predator = min(visible_predators, key=lambda predator: self.distance_to(predator))

        self.nearest_predator = nearest_predator

        if nearest_predator:
            pred_dx, pred_dy = nearest_predator.x - self.x, nearest_predator.y - self.y
            pred_distance = math.sqrt(pred_dx ** 2 + pred_dy ** 2) / perception_radius  # Normalized distance
            pred_angle = math.atan2(pred_dy, pred_dx) / math.pi  # Normalized angle
            inputs.extend([pred_distance, pred_angle])
        else:
            inputs.extend([0, 0])  # Default values if no predator is close

        # Own velocity - Assuming a known max velocity for normalization
        max_velocity = SPEED
        inputs.extend([self.dx / max_velocity, self.dy / max_velocity])

        # Local density
        max_density = max_neighbors_considered
        inputs.append(len(close_boids) / max_density * 2 - 1)  # Normalized and scaled to [-1, 1]

        return inputs


    def distance_to(self, other_boid):
        return math.sqrt((other_boid.x - self.x)**2 + (other_boid.y - self.y)**2)
    
    def escape_predator(self, nearest_predator, avoidance_intensity):
        if nearest_predator is None or avoidance_intensity <= 0:
            return 0, 0  # No predator or no need to avoid

        # Calculate a vector pointing away from the predator
        escape_dx = self.x - nearest_predator.x
        escape_dy = self.y - nearest_predator.y
        distance = math.sqrt(escape_dx ** 2 + escape_dy ** 2)

        # Normalize the vector and scale by avoidance intensity
        if distance > 0:
            escape_dx = (escape_dx / distance) * avoidance_intensity
            escape_dy = (escape_dy / distance) * avoidance_intensity
        return escape_dx, escape_dy

    def update(self, boids, predators):
        # calculate inputs
        sensory_inputs = self.perceive(boids, predators, PERCEPTION_RADIUS)
        
        # Processing through neural network
        neural_outputs = self.brain.feedforward(sensory_inputs)
        
        sep_tendency, align_tendency, cohesion_tendency = neural_outputs[:3]
        
        # Calculate behavior adjustments
        separation_x, separation_y = self.separation(boids)
        alignment_x, alignment_y = self.alignment(boids)
        cohesion_x, cohesion_y = self.cohesion(boids)
        
        # Apply tendencies based on genotype
        self.dx += separation_x * sep_tendency
        self.dy += separation_y * sep_tendency
        self.dx += alignment_x * align_tendency
        self.dy += alignment_y * align_tendency
        self.dx += cohesion_x * cohesion_tendency
        self.dy += cohesion_y * cohesion_tendency
        
        # Get predator avoidance output from neural network
        # Assuming it's part of the neural_outputs
        avoidance_intensity = neural_outputs[3] 
        
        # Find nearest predator
        nearest_predator = self.nearest_predator
        
        # Calculate escape vector from predator
        escape_dx, escape_dy = self.escape_predator(nearest_predator, avoidance_intensity)

        # Add escape vector to boid's velocity
        self.dx += escape_dx
        self.dy += escape_dy
        
        # Normalize to maintain speed
        magnitude = math.sqrt(self.dx**2 + self.dy**2)
        if magnitude != 0:
            self.dx = (self.dx / magnitude) * SPEED
            self.dy = (self.dy / magnitude) * SPEED
        else:
            # Handle the case when magnitude is zero (to avoid division by zero)
            # For example, set velocity components to zero or handle it based on your requirements
            self.dx = 0
            self.dy = 0
        
        # Update position and handle boundaries
        self.x += self.dx
        self.y += self.dy
        # self.x = max(0, min(self.x, WIDTH))
        # self.y = max(0, min(self.y, HEIGHT))
        # Toroidal boundary conditions
        self.x = self.x % WIDTH
        self.y = self.y % HEIGHT
        
        # Boundary conditions
        # if self.x < 0 or self.x > WIDTH:
        #     self.dx *= -1
        #     self.x = max(0, min(self.x, WIDTH))
        
        # if self.y < 0 or self.y > HEIGHT:
        #     self.dy *= -1
        #     self.y = max(0, min(self.y, HEIGHT))
        
        # functional role outputs
        soldier_tendency, worker_tendency = neural_outputs[-2:]
        # Determine the role based on the neural network output
        if soldier_tendency > worker_tendency:
            self.role = 'Soldier'
        else:
            self.role = 'Worker'

        # Now, apply behaviors based on the role
        if self.role == 'Soldier':
            # Implement soldier-specific behavior
            pass
        elif self.role == 'Worker':
            # Implement worker-specific behavior
            pass
        
        self.adjust_energy()
        if self.energy > REPRODUCTION_ENERGY_THRESHOLD:
            self.reproduce(boids)
    
    def adjust_energy(self):
        # Adjust energy based on role, behavior, and success
        if self.role == 'Worker':
            self.energy += WORKER_ENERGY_GAIN
        elif self.role == 'Soldier' and self.near_worker():
            self.energy += SOLDIER_ENERGY_GAIN  # Assuming soldiers gain less energy or maintain energy when near workers
        
        # Energy cost for movement, more for rapid movements or escaping predators
        self.energy -= ENERGY_COST_PER_STEP
        
    def near_worker(self):
        # Simple check for proximity to workers for energy gain (optional)
        return any(boid for boid in boids if boid.role == 'Worker' and self.distance_to(boid) < PROXIMITY_THRESHOLD)

    def reproduce(self, boids):
        # Reproduction logic, creating offspring with mutated traits
        offspring_genotype = self.genotype + np.random.normal(0, MUTATION_RATE, len(self.genotype))
        offspring = Boid(self.x, self.y, offspring_genotype, self.role)  # Consider random role assignment or inheritance
        boids.append(offspring)
        self.energy -= REPRODUCTION_COST

    def die(self, boids):
        # Remove self from the boids list
        boids.remove(self)

        
    def worker_behavior(self):
        # Placeholder for future resource-oriented behavior
        # Move in a random direction for now
        self.dx += random.uniform(-1, 1)
        self.dy += random.uniform(-1, 1)
        
    def soldier_behavior(self, predators):
        # Check for nearby predators and act accordingly
        pass
            
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
        # Define the size of the triangle
        size = BOID_SIZE

        # Calculate the angle of movement
        angle = math.atan2(self.dy, self.dx)

        # Calculate the three points of the triangle
        point1 = (self.x + math.cos(angle) * size, self.y + math.sin(angle) * size)
        point2 = (self.x + math.cos(angle + 2.5) * size, self.y + math.sin(angle + 2.5) * size)
        point3 = (self.x + math.cos(angle - 2.5) * size, self.y + math.sin(angle - 2.5) * size)

        # Draw the triangle in different colors based on role
        if self.role == 'Soldier':
            color = BLUE
        elif self.role == 'Worker':
            color = GREEN
        else:
            color = RED  # Default color if no specific role is assigned

        pygame.draw.polygon(screen, color, [point1, point2, point3])
        
        
# Define a function to create and initialize boids
def create_boids():
    boids = [Boid(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)) for _ in range(BOID_COUNT)]
    Boid.build_kd_tree(boids)  # Build KD-Tree for efficient neighbor search
    return boids

class Predator:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-1, 1)
        self.speed = 1
        self.visual_range = 100  # Example value
        self.catch_radius = 15
        self.REVERSAL_DURATION = 30  # Example value
        self.eating_duration = 60
        self.field_of_view = math.pi
        self.reversal_timer = 20
        self.is_eating = False

    def detect_boids(self, boids):
        for boid in boids:
            # Calculate distance and angle to the boid
            distance = self.distance_to(boid)
            angle_to_boid = math.atan2(boid.y - self.y, boid.x - self.x)
            angle_of_heading = math.atan2(self.dy, self.dx)
            
            # Calculate the angle difference relative to predator's heading
            angle_difference = abs(angle_to_boid - angle_of_heading)
            if angle_difference > math.pi:
                angle_difference = 2 * math.pi - angle_difference

            # Check if the boid is within the visual range and field of view
            if distance <= self.visual_range and angle_difference <= self.field_of_view / 2:
                return boid
        return None

    def detect_nearby_soldiers(self, boids):
        nearby_soldiers = []
        for boid in boids:
            if boid.role != 'Soldier':
                continue

            # Calculate distance and angle to the soldier boid
            distance = self.distance_to(boid)
            angle_to_boid = math.atan2(boid.y - self.y, boid.x - self.x)
            angle_of_heading = math.atan2(self.dy, self.dx)

            # Calculate the angle difference relative to predator's heading
            angle_difference = abs(angle_to_boid - angle_of_heading)
            if angle_difference > math.pi:
                angle_difference = 2 * math.pi - angle_difference

            # Check if the soldier boid is within the visual range and field of view
            if distance <= self.visual_range and angle_difference <= self.field_of_view / 2:
                nearby_soldiers.append(boid)

        return nearby_soldiers



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
            # Detect nearby soldiers
            nearby_soldiers = self.detect_nearby_soldiers(boids)
            
            if len(nearby_soldiers) >= 3:
                # Reverse direction for a couple of time steps
                self.reverse_direction()
            elif self.reversal_timer > 0:
                self.reversal_timer -= 1  # Count down the reversal timer
            else:
                # Normal predator behavior
                prey = self.detect_boids(boids)
                if prey is not None:
                    distance_to_prey = math.sqrt((prey.x - self.x)**2 + (prey.y - self.y)**2)
                    if distance_to_prey <= self.catch_radius:
                        self.eat_boid()
                        boids.remove(prey)
                    else:
                        self.move_towards_prey(prey)
                else:
                    # Move randomly in current direction if no prey is detected
                    self.x += self.dx
                    self.y += self.dy
                    # Optionally, you can add some randomness to the direction
                    self.dx += random.uniform(-0.5, 0.5)
                    self.dy += random.uniform(-0.5, 0.5)
                    # Normalize the velocity vector to maintain constant speed
                    magnitude = math.sqrt(self.dx**2 + self.dy**2)
                    self.dx = (self.dx / magnitude) * self.speed
                    self.dy = (self.dy / magnitude) * self.speed

            # Boundary conditions
            if self.x < 0 or self.x > WIDTH:
                self.dx *= -1  # Reverse direction on X axis
                self.x = max(0, min(self.x, WIDTH))  # Keep within bounds
            
            if self.y < 0 or self.y > HEIGHT:
                self.dy *= -1  # Reverse direction on Y axis
                self.y = max(0, min(self.y, HEIGHT))  # Keep within bounds
    
    def reverse_direction(self):
        self.dx *= -1
        self.dy *= -1
        self.reversal_timer = self.REVERSAL_DURATION  # Define this duration as a class attribute

    def distance_to(self, boid):
        return math.sqrt((boid.x - self.x) ** 2 + (boid.y - self.y) ** 2)


    def draw(self, screen, font):
        # Define the size of the triangle
        size = 10

        # Calculate the angle of movement
        angle = math.atan2(self.dy, self.dx)

        # Calculate the three points of the triangle
        point1 = (self.x + math.cos(angle) * size, self.y + math.sin(angle) * size)
        point2 = (self.x + math.cos(angle + 2.5) * size, self.y + math.sin(angle + 2.5) * size)
        point3 = (self.x + math.cos(angle - 2.5) * size, self.y + math.sin(angle - 2.5) * size)

        # Draw the triangle
        pygame.draw.polygon(screen, RED, [point1, point2, point3])
        
        # Draw "Eating" text above the predator when it is eating
        if self.is_eating:
            text_surface = font.render('Eating', True, (0, 0, 0))  # White text
            screen.blit(text_surface, (self.x - 20, self.y - 20))

# Spawning predators near food sources
predators = []
for i in range(PREDATOR_COUNT):
    # Assuming predators spawn within a certain radius around food sources
    predator_x = random.randint(0, WIDTH)
    predator_y = random.randint(0, HEIGHT)
    predators.append(Predator(predator_x, predator_y))

def build_spatial_index(boids):
    positions = [(boid.x, boid.y) for boid in boids]
    return KDTree(positions)

def find_neighbors(kd_tree, boid, threshold):
    position = (boid.x, boid.y)
    indices = kd_tree.query_ball_point(position, threshold)
    return indices

def group_into_flocks(boids, neighbor_indices):
    flocks = []
    visited = set()

    for i, boid in enumerate(boids):
        if i in visited:
            continue

        # Initialize the current flock with this boid and its neighbors
        current_flock = {i}
        current_flock.update(neighbor_indices[i])
        visited.add(i)

        # Create a copy of current_flock for iteration
        for j in list(current_flock):  # Note the conversion to list for a static copy
            visited.add(j)  # Mark as visited
            for other_flock in flocks:
                if j in other_flock:
                    # Merge with existing flock
                    current_flock.update(other_flock)
                    flocks.remove(other_flock)
                    break

        flocks.append(current_flock)

    return flocks


def convert_indices_to_boids(flocks, boids):
    return [{boids[i] for i in flock} for flock in flocks]

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-organisation")
clock = pygame.time.Clock()

pygame.font.init()  # Initialize font module
my_font = pygame.font.SysFont('arial', 15)  # Create a font object


# initialise statistical track record
update_frequency_steps = 500  # How often to print statistics (every 500 steps)
steps_since_last_update = 0

# Create and initialize boids
boids = create_boids()

# Main pygame loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Efficient neighbor search with KD-Tree
    Boid.build_kd_tree(boids)
    
    kd_tree = build_spatial_index(boids)
    neighbor_indices = [find_neighbors(kd_tree, boid, flock_threshold) for boid in boids]
    flocks_indices = group_into_flocks(boids, neighbor_indices)
    flocks = convert_indices_to_boids(flocks_indices, boids)
    
    # Print statistics at specified frequency
    steps_since_last_update += 1
    if steps_since_last_update >= update_frequency_steps:
        print(f"Number of flocks: {len(flocks)}")
        if len(flocks) > 0:
            average_flock_size = sum(len(flock) for flock in flocks) / len(flocks)
            print(f"Average flock size: {average_flock_size:.2f}")
        else:
            print("No flocks identified.")
        steps_since_last_update = 0

    # Update boids, track reproduction events and predator encounters
    for boid in boids:
        initial_energy = boid.energy
        boid.update(boids, predators)
        
    for predator in predators:
        initial_count = len(boids)
        predator.update(boids)

    # Rendering
    screen.fill(WHITE)
    draw_gridlines(screen, WIDTH, HEIGHT, GRID_SPACING)
    for boid in boids:
        boid.draw(screen)
    for predator in predators:
        predator.draw(screen, my_font)

    pygame.display.flip()
    clock.tick(30)  # Control the simulation speed

# Quit pygame
pygame.quit()
