import pygame
import random
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import cProfile
import pstats

random.seed(2024)
np.random.seed(2024)

# Constants for simulation visualization
BOID_SIZE = 5 # Visual size of boids
PREDATOR_SIZE = 2 * BOID_SIZE  # Visual size of predators
MIN_DISTANCE = 2 * BOID_SIZE  # Minimum distance to prevent boid overlap


# Screen dimensions
WIDTH, HEIGHT = 160 * BOID_SIZE, 120 * BOID_SIZE

# Boid parameters
SPEED = 1
predator_speed = 1.2 * SPEED                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
SEPARATION_RADIUS = BOID_SIZE * 5
ALIGNMENT_RADIUS = BOID_SIZE * 5
COHESION_RADIUS = BOID_SIZE * 5
PERCEPTION_RADIUS = BOID_SIZE * 5

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)  # Color for soldier
GREEN = (0, 255, 0)  # Color for worker
BLACK = (0, 0, 0)

# Additional constants for gridlines
GRIDLINE_COLOR = (50, 50, 50)  # Blurring black color
GRID_SPACING = 50  # Spacing between gridlines

# Initial setup parameters
BOID_COUNT = 100  # Number of boids at the start of the simulation
PREDATOR_COUNT = 10  # Number of predators at the start

# Boid energy and reproduction parameters
INITIAL_ENERGY = 50  # Starting energy level for each boid
REPRODUCTION_ENERGY_THRESHOLD = 200  # Energy level needed for a boid to reproduce
REPRODUCTION_COST = 100  # Energy deducted from a boid upon reproduction
DEATH_ENERGY_LEVEL = 0  # Energy level at which a boid dies
MUTATION_RATE = 0.1  # Rate at which mutations occur in offspring genotypes
MUTATION_STRENGTH = 0.1
REPRODUCTION_DISTANCE = BOID_SIZE * 2

# Boid energy gain and loss parameters
WORKER_ENERGY_GAIN = 1.1  # Energy gained by worker boids per timestep
SOLDIER_ENERGY_GAIN = 0.7  # Energy gained by soldier boids when near workers
NEAR_WORKER_GAIN = 0.6  # Energy gained by workers when near soldiers
ENERGY_COST_PER_STEP = 1  # Energy cost for boids moving per timestep
WORKER_TOGETHER_ENERGY = 0.2  # Bonus/penalty energy for workers based on group size
CRITICAL_GROUP_SIZE = 20  # Group size threshold for overpopulation penalty
FLOCK_THRESHOLD = BOID_SIZE * 5
FLOCK_SIZE_THRESHOLD = 4


# Boid behavior and interaction parameters
PROXIMITY_THRESHOLD = BOID_SIZE * 3  # Distance within which soldiers gain energy from workers
SOLDIER_PROTECTION_RADIUS = BOID_SIZE * 5  # Distance within which soldiers protect workers from predators

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

def find_neighbors_within_radius(boid, boids, perception_radius):
    neighbors = []
    for other_boid in boids:
        if other_boid != boid:
            distance = boid.distance_to(other_boid)
            if distance <= perception_radius:
                neighbors.append(other_boid)
    return neighbors


def toroidal_distance(x1, y1, x2, y2, width, height):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Considering wrap-around effect for shortest distance
    dx = min(dx, width - dx)
    dy = min(dy, height - dy)

    # Calculate Euclidean distance with the shortest dx and dy
    return (dx**2 + dy**2)**0.5

class Boid:
    # boid_tree = None

    def __init__(self, x, y, genotype=None, role=None):
        self.x = x
        self.y = y
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-1, 1)
        self.size = BOID_SIZE
        
        self.flock_id = None  # New attribute for tracking flock membership

        # Initialize role
        if role:
            self.role = role
        else:
            self.role = np.random.choice(['soldier', 'worker'])
        
        # Neural network controller setup
        input_size = 11  # Number of sensory inputs
        hidden_size = 10  # Number of hidden nodes
        output_size = 6   # Number of outputs
        if genotype is not None:
            self.genotype = genotype
        else:
            genotype_length = (input_size * hidden_size) + (hidden_size * output_size)  # Total number of weights and biases
            self.genotype = np.random.randn(genotype_length)
        self.brain = NeuralNetwork(input_size, hidden_size, output_size, self.genotype)
        
        self.energy = INITIAL_ENERGY
        
        # Existing initialization code...
        self.age = 0  # Boid's current age
        self.lifespan = 1000#random.randint(500, 800)  # Boid's lifespan, introducing some variability

        # post-simulation analytics
        self.isInFlock = False  # New attribute to track flock membership
        self.survivalTime = 0  # Tracks how long the boid has survived
        self.reproductionCount = 0  # Tracks how many offspring a boid has produced
        self.reproduced_this_step = False  # Track reproduction event
        # Other attributes
        self.flock_id = None  # Track which flock this boid belongs to

    def count_nearby_workers(self, boids):
        count = 0
        for boid in boids:
            if boid != self and boid.role == 'worker' and self.distance_to(boid) < PROXIMITY_THRESHOLD:
                count += 1
        return count

    def perceive(self, boids, predators, perception_radius, max_neighbors_considered=3):
       
        close_boids = find_neighbors_within_radius(self, boids, perception_radius)


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
        # self.x = self.x % WIDTH
        # self.y = self.y % HEIGHT
        
        # Boundary conditions
        if self.x < 0 or self.x > WIDTH:
            self.dx *= -1
            self.x = max(0, min(self.x, WIDTH))
        
        if self.y < 0 or self.y > HEIGHT:
            self.dy *= -1
            self.y = max(0, min(self.y, HEIGHT))
        
        # functional role outputs
        soldier_tendency, worker_tendency = neural_outputs[-2:]
        # Determine the role based on the neural network output
        if soldier_tendency > worker_tendency:
            self.role = 'soldier'
        else:
            self.role = 'worker'

        # Now, apply behaviors based on the role
        if self.role == 'soldier':
            # Implement soldier-specific behavior
            self.soldier_behavior(boids, predators)
            # pass
        elif self.role == 'worker':
            # Implement worker-specific behavior
            pass
        
        # Increment age
        self.age += 1
        
        self.adjust_energy()
        
        
        # Reset reproduction flag at the beginning of each update
        self.reproduced_this_step = False
        if self.energy > REPRODUCTION_ENERGY_THRESHOLD:
            self.reproduce(boids)
            self.reproduced_this_step = True  # Set flag to True when reproduction occurs
            
        
        # post-simulation    
        # Update survival time
        self.survivalTime += 1
    
    def adjust_energy(self):
        # Adjust energy based on role, behavior, and success
        if self.role == 'worker':
            self.energy += WORKER_ENERGY_GAIN
            # if self.near_worker():
                # self.energy += 0.2
            count = self.count_nearby_workers(boids)
            if count > 1 and count < CRITICAL_GROUP_SIZE:
                self.energy += WORKER_TOGETHER_ENERGY
            elif count >= CRITICAL_GROUP_SIZE:
                self.energy -= WORKER_TOGETHER_ENERGY
        elif self.role == 'soldier':
            self.energy += SOLDIER_ENERGY_GAIN  # Assuming soldiers gain less energy or maintain energy when near workers
            if self.near_worker():
                self.energy += NEAR_WORKER_GAIN
            
        # Energy cost for movement, more for rapid movements or escaping predators
        self.energy -= ENERGY_COST_PER_STEP
    
    def should_die(self):
        # Determine if the boid should be removed based on age/lifespan
        return self.age >= self.lifespan
    
    def near_worker(self):
        # Simple check for proximity to workers for energy gain (optional)
        return any(boid for boid in boids if boid.role == 'worker' and self.distance_to(boid) < PROXIMITY_THRESHOLD)
    
    
    def reproduce(self, boids):
        # Reproduction logic, creating offspring with mutated traits
        # offspring_genotype = self.genotype + np.random.normal(0, MUTATION_RATE, len(self.genotype))
        # Assuming MUTATION_RATE is the probability of mutation and mutation_strength is the magnitude of mutation change
        offspring_genotype = self.genotype.copy()  # Create a copy of the parent's genotype
        
        # Iterate over each gene in the genotype
        for i in range(len(self.genotype)):
            # Check if a mutation should occur based on the mutation rate
            if np.random.random() < MUTATION_RATE:
                # Apply mutation by adding a random value drawn from a normal distribution with mean 0 and standard deviation mutation_strength
                offspring_genotype[i] += np.random.normal(0, MUTATION_STRENGTH)
        
        # Generate random direction for initial movement
        dx = random.uniform(-1, 1)
        dy = random.uniform(-1, 1)
        
        # Normalize the direction vector
        magnitude = (dx ** 2 + dy ** 2) ** 0.5
        dx /= magnitude
        dy /= magnitude

        # Move the offspring away from its mother
        new_x = self.x + dx * REPRODUCTION_DISTANCE
        new_y = self.y + dy * REPRODUCTION_DISTANCE

        # Create the offspring boid
        offspring = Boid(new_x, new_y, offspring_genotype)

        # Add offspring to the list of boids
        boids.append(offspring)

        # Deduct energy cost for reproduction from the mother boid
        self.energy -= REPRODUCTION_COST
        
        # post-simulation
        self.reproductionCount += 1  # Increment reproduction count
    
    def die(self, boids):
        # Remove self from the boids list
        boids.remove(self)
        
    def worker_behavior(self):
        # Placeholder for future resource-oriented behavior
        # Move in a random direction for now
        self.dx += random.uniform(-1, 1)
        self.dy += random.uniform(-1, 1)
        
    def soldier_behavior(self, boids, predators):
        # Initial behavior is to move randomly if no predators are nearby
        move_dx = random.uniform(-1, 1)
        move_dy = random.uniform(-1, 1)
        
        # Attempt to find the closest predator and worker
        nearest_predator = self.find_nearest(predators)
        nearest_worker = self.find_nearest([boid for boid in boids if boid.role == 'worker'])
        
        if nearest_predator and self.distance_to(nearest_predator) < SOLDIER_PROTECTION_RADIUS:
            # If a predator is too close, move to intercept or stand between predator and nearest worker
            pred_dx, pred_dy = self.vector_to(nearest_predator)
            if nearest_worker:
                # Move to a position between the worker and the predator
                worker_dx, worker_dy = self.vector_to(nearest_worker)
                move_dx = (worker_dx - pred_dx) / 2
                move_dy = (worker_dy - pred_dy) / 2
            # else:
            #     # No worker to protect, confront predator directly (or could choose to flee)
            #     move_dx = -pred_dx
            #     move_dy = -pred_dy
        elif nearest_worker:
            # No immediate predator threat, move towards the nearest worker to protect
            move_dx, move_dy = self.vector_to(nearest_worker)
    
        # Normalize the movement vector
        norm = math.sqrt(move_dx**2 + move_dy**2)
        self.dx += (move_dx / norm) * SPEED if norm != 0 else 0
        self.dy += (move_dy / norm) * SPEED if norm != 0 else 0

    def find_nearest(self, boids):
        """Find the nearest boid in a list of boids."""
        nearest = None
        min_distance = float('inf')
        for boid in boids:
            distance = self.distance_to(boid)
            if distance < min_distance:
                nearest = boid
                min_distance = distance
        return nearest

    def vector_to(self, boid):
        """Calculate the normalized vector from this boid to another."""
        dx = boid.x - self.x
        dy = boid.y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        return (dx / distance, dy / distance) if distance != 0 else (0, 0)

            
    def separation1(self, boids):
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
    
    def separation(self, boids):
        separation_x, separation_y = 0, 0
        epsilon = 1e-5  # Small value to prevent division by zero
        for boid in boids:
            if boid == self:
                continue
            dist = np.sqrt((self.x - boid.x)**2 + (self.y - boid.y)**2)
            if dist < BOID_SIZE * 5:
                # Add epsilon to dist to ensure it's never 0
                separation_x -= (boid.x - self.x) / (dist + epsilon)
                separation_y -= (boid.y - self.y) / (dist + epsilon)
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
        if self.role == 'soldier':
            color = BLUE
        elif self.role == 'worker':
            color = GREEN
        else:
            color = RED  # Default color if no specific role is assigned

        pygame.draw.polygon(screen, color, [point1, point2, point3])
        
        
# Define a function to create and initialize boids
def create_boids():
    boids = [Boid(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)) for _ in range(BOID_COUNT)]
    return boids

class Predator:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-1, 1)
        self.speed = predator_speed
        self.visual_range = PREDATOR_SIZE*5# max(WIDTH, HEIGHT)
        self.soldier_visual_range = PREDATOR_SIZE*5
        self.large_group_visual_range = PREDATOR_SIZE*5
        self.catch_radius = PREDATOR_SIZE
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
            if boid.role != 'soldier':
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
            if distance <= self.soldier_visual_range and angle_difference <= self.field_of_view / 2:
                nearby_soldiers.append(boid)

        return nearby_soldiers

    def detect_large_groups(self, boids, group_threshold=20):
        # Simple heuristic: Count all boids within a larger radius
        nearby_boids = [boid for boid in boids if self.distance_to(boid) < self.large_group_visual_range]
        return len(nearby_boids) > group_threshold

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
            # elif self.detect_large_groups(boids):
                # Implement cautious behavior around large groups
                # angle = random.uniform(0, 2 * math.pi)
                # self.dx = math.cos(angle) * self.speed
                # self.dy = math.sin(angle) * self.speed
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
        size = PREDATOR_SIZE

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


# Post simulation

def identify_flocks(boids, threshold):
    """Identify flocks based on a distance threshold, assigning a unique ID to each flock."""
    flocks = {}
    flock_id_counter = 0  # Initialize a counter to assign unique IDs
    visited = set()

    for boid in boids:
        boid.isInFlock = False
        if boid in visited:
            continue

        neighbors = find_neighbors_within_radius(boid, boids, threshold)
        if len(neighbors) < FLOCK_SIZE_THRESHOLD:
            continue

        new_flock = set(neighbors)
        new_flock.add(boid)
        visited.update(new_flock)

        # Check if this new flock overlaps with any existing flock
        overlapped_flock_ids = [flock_id for flock_id, existing_flock in flocks.items() if not new_flock.isdisjoint(existing_flock)]
        if overlapped_flock_ids:
            # Merge with the first overlapped flock and update its members
            flock_id_to_merge = overlapped_flock_ids[0]
            flocks[flock_id_to_merge] = flocks[flock_id_to_merge].union(new_flock)
            # Mark all boids in the merged flock as members of a flock
            for member in flocks[flock_id_to_merge]:
                member.isInFlock = True
            # Remove other overlapped flocks and merge their members into the first one
            for other_flock_id in overlapped_flock_ids[1:]:
                flocks[flock_id_to_merge] = flocks[flock_id_to_merge].union(flocks.pop(other_flock_id))
        else:
            # Assign a new unique flock ID
            flocks[flock_id_counter] = new_flock
            # Mark all boids in the new flock as members of a flock
            for member in new_flock:
                member.isInFlock = True
            flock_id_counter += 1  # Increment the counter for the next unique flock ID

    return flocks

# Initialize containers for metrics
# Initialize a dictionary to store energy data for each flock over time
# Key: flock_id (unique identifier for each flock)
# Value: List of dictionaries, one for each time step, with boid energies and average energy
flock_data = {}

flock_count_history = []
average_flock_size_history = []
timesteps_history = []  # This will track when each measurement was taken

def update_flock_data(flocks, current_step):
    global flock_data  # Assuming flock_data is a structured dict
    
    for flock_id, flock_members in flocks.items():
        # Initialize data storage for a new flock
        if flock_id not in flock_data:
            flock_data[flock_id] = {
                'timesteps': [],
                'individual_energies': [],  # This will store a list of lists
                'average_energies': [],
                'sizes': [],
                'soldier_proportions': [],
                'worker_proportions': [],
            }
        
        # Gather individual energies
        individual_energies = [boid.energy for boid in flock_members]
        # Calculate average energy for the current timestep
        average_energy = sum(individual_energies) / len(flock_members) if flock_members else 0
        # Record the size of the flock
        size = len(flock_members)
        
        # Count the number of soldiers and workers
        soldiers = sum(1 for boid in flock_members if boid.role == 'soldier')
        workers = sum(1 for boid in flock_members if boid.role == 'worker')
        
        # Calculate proportions
        soldier_proportion = soldiers / size if size > 0 else 0
        worker_proportion = workers / size if size > 0 else 0
        
        # Update the data structure
        flock_data[flock_id]['timesteps'].append(current_step)
        flock_data[flock_id]['individual_energies'].append(individual_energies)
        flock_data[flock_id]['average_energies'].append(average_energy)
        flock_data[flock_id]['sizes'].append(size)
        flock_data[flock_id]['soldier_proportions'].append(soldier_proportion)
        flock_data[flock_id]['worker_proportions'].append(worker_proportion)


def update_flock_metrics(flocks, step, update_frequency):
    global flock_count_history, average_flock_size_history, timesteps_history
    # Only update the metrics at specified frequency
    if step % update_frequency == 0:
        if flocks:
            flock_count = len(flocks)
            total_boids_in_flocks = sum(len(flock) for flock_id, flock in flocks.items())
            average_size = total_boids_in_flocks / flock_count
        else:
            flock_count = 0
            average_size = 0

        # Store the metrics along with the current timestep
        flock_count_history.append(flock_count)
        average_flock_size_history.append(average_size)
        timesteps_history.append(step)  # Record the current step

# Global variables to store metrics
global flock_reproduction_counts, solitary_reproduction_counts, steps_tracker
flock_reproduction_counts = []
solitary_reproduction_counts = []
steps_tracker = []

def plot_and_save_metrics():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot Flock Count
    ax1.plot(timesteps_history, flock_count_history, 'tab:red')
    ax1.set_ylabel('Flock Count', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_title('Flock Count Over Time')
    
    # Ensure the first subplot has a tight layout and shares the x-axis
    ax1.set_xlabel('Time Step')
    ax1.set_xlim([min(timesteps_history), max(timesteps_history)])  # Set limits to match the actual timestep range

    # Plot Average Flock Size
    ax2.plot(timesteps_history, average_flock_size_history, 'tab:blue')
    ax2.set_ylabel('Average Flock Size', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_title('Average Flock Size Over Time')

    # Adjust layout and x-axis label for the second subplot
    ax2.set_xlabel('Time Step')
    plt.xlim([min(timesteps_history), max(timesteps_history)])  # Ensure x-axis is consistent with the actual timesteps

    plt.tight_layout()
    plt.savefig('result/flock_size_vs_count.png')
    plt.show()

        
def plot_flock_data_over_time(flock_data, start_step=None, end_step=None):
    for flock_id, data in flock_data.items():
        # Determine the range of steps to plot
        if start_step is None:
            start_step = min(data['timesteps'])
        if end_step is None:
            end_step = max(data['timesteps'])

        # Filter the data for the specified range
        filtered_indices = [i for i, timestep in enumerate(data['timesteps']) if start_step <= timestep <= end_step]
        timesteps = [data['timesteps'][i] for i in filtered_indices]
        sizes = [data['sizes'][i] for i in filtered_indices]
        average_energies = [data['average_energies'][i] for i in filtered_indices]
        worker_soldier_ratios = [data['soldier_proportions'][i] / data['worker_proportions'][i]  if data['worker_proportions'][i]  > 0 else 0 for i in filtered_indices]

        # Start a new figure for each flock
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Plot the individual boid energies
        scatter_x = []
        scatter_y = []
        for i in filtered_indices:
            timestep = data['timesteps'][i]
            individual_energies = data['individual_energies'][i]
            scatter_x.extend([timestep] * len(individual_energies))
            scatter_y.extend(individual_energies)
        
        ax1.scatter(scatter_x, scatter_y, alpha=0.5, label='Individual Energies', s=10, marker='o')
        ax1.plot(timesteps, average_energies, label='Average Energy', color='red', linewidth=2)
        ax1.set_title(f'Flock {flock_id} Energy Dynamics')
        ax1.set_ylabel('Energy')
        ax1.legend(loc='upper left')

        ax2.plot(timesteps, sizes, label='Flock Size', color='blue', linewidth=2, linestyle='--')
        ax2.set_title(f'Flock {flock_id} Size Dynamics')
        ax2.set_ylabel('Size')
        ax2.legend(loc='upper left')

        ax3.plot(timesteps, worker_soldier_ratios, label='Soldier/Worker Ratio', color='green', linewidth=2, linestyle='-')
        ax3.set_title(f'Flock {flock_id} Role Ratio Dynamics')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Ratio')
        ax3.legend(loc='upper left')

        # Ensure that the plots do not overlap and save each flock's figure separately
        plt.tight_layout()
        plt.savefig(f'result/flock_{flock_id}_dynamics_{start_step}_to_{end_step}.png')
        plt.close(fig)  # Close the figure after saving to prevent display overlap



# Example of how to call the function for the last 300 timesteps
# plot_flock_energy_data_steps(flock_energy_data, last_n_steps=300)

group_reproduction_events = []  # Initialize outside the main loop


###############

def run_simulation(boids, predators, screen, clock):
    running = True
 
    update_frequency_steps = 100  # Update metrics every 100 steps
    current_step = 0  # Initialize a step counter
    MAX_STEPS = 4400
    
    # Initialize a font
    font_size = 24
    font = pygame.font.SysFont('arial', font_size)
    info_font = pygame.font.SysFont('arial', 12)  # Create a font object for information display
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse position
                x, y = event.pos
                # Create a new predator at this position
                new_predator = Predator(x, y)
                # Add the new predator to the list of predators
                predators.append(new_predator)
        
        if current_step >= MAX_STEPS:
            print("Maximum steps reached.")
            break
        
        # Blit the background image
        # screen.blit(background_image, (0, 0))
        # Fill the screen with black
        screen.fill(BLACK)
        draw_gridlines(screen, WIDTH, HEIGHT, GRID_SPACING)
        
        # Render the current step count as text
        text_surface = font.render(f'Steps: {current_step}', True, (255, 255, 255))  # White text
        screen.blit(text_surface, (5, 5))  # Position the text at the top left corner
    
        
        # identify flock
        flocks = identify_flocks(boids, FLOCK_THRESHOLD)
        
        if flocks:
        #     # Calculate the total number of boids across all flocks
            total_boids_in_flocks = sum(len(flock) for flock_id, flock in flocks.items())
        #     # Calculate the average flock size
            average_flock_size = total_boids_in_flocks / len(flocks)
        else:
            average_flock_size = 0
        
        # update flock metrics
        update_flock_metrics(flocks, current_step, update_frequency_steps)
        
        info_text = f"Number of flocks: {len(flocks)}, Average flock size: {average_flock_size:.2f}"
        info_surface = info_font.render(info_text, True, pygame.Color('white'))
        screen.blit(info_surface, (40, 40))  # Position the text at (10, 10) from the top left corner
        
        
        # # Right after identifying flocks, update the energy data for each flock
        if current_step >= 2000:
            # Update flock data with current information
            update_flock_data(flocks, current_step)
        
        current_step += 1  # Increment step counter

                
        # Update loop with deferred removal
        to_remove = [boid for boid in boids if boid.should_die()]
        for boid in to_remove:
            boids.remove(boid)

        for boid in boids:
            boid.update(boids, predators)
            
        for predator in predators:
            predator.update(boids)

        for boid in boids:
            boid.draw(screen)
        for predator in predators:
            predator.draw(screen, my_font)

        pygame.display.flip()
        clock.tick(30)  # Control the simulation speed

    pygame.quit()
    
    # Call the plot function for the specified timestep range
    plot_flock_data_over_time(flock_data)
    
    plot_and_save_metrics()

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("VitaNova: Computational Flocking Dynamics")
    # Load the background image
    # background_image = pygame.image.load("background.png")
    
    clock = pygame.time.Clock()
    pygame.font.init()
    my_font = pygame.font.SysFont('arial', 15)

    # Initialize boids and predators
    boids = create_boids()
    predators = [Predator(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(PREDATOR_COUNT)]

    # Call the simulation function
    # run_simulation(boids, predators, screen, clock)
    
    cProfile.run('run_simulation(boids, predators, screen, clock)', 'profile_stats')
    
    p = pstats.Stats('profile_stats')
    p.strip_dirs().sort_stats('time').print_stats(10)
