from vpython import *
import random
import math


WIDTH = 1024
HEIGHT = 768

scene.title = "Modeling boids"
scene.width = WIDTH
scene.height = HEIGHT

#==============================================================================
# Globals
#==============================================================================
SCALE = 1e3
BOX_SIZE = 1000e3 # meters

NBOIDS = 100
MAX_SPEED = 5e3 # meters/second

BOIDS = []
BOID_SIZE = 5
BOID_MASS = 1000 # kg

SPHERES = []
SPHERE_SIZE = 100e3 # m
SPHERE_MASS = 1e20  # kg


def gravity_acc(p1, p2):
	""" Returns distance between points (p1, p2) and gravity force: p1 -> p2 """
	G = 6.67e-11
	r = p1 - p2
	a = - norm(r) * G * SPHERE_MASS /mag(r)**2
	return a


def get_alive_particles():
	""" Returns number of alive (visible) particles """
	return len([x for x in BOIDS if x.figure.visible])


class Sphere:
	""" Asteroid class """
	def __init__(self, position, radius):
		self.position = position
		self.radius = radius
		self.figure = sphere(pos=self.position/SCALE, radius=self.radius/SCALE, color=color.cyan)


class Boid:
	""" Space ship class """

	cohesionRange = BOX_SIZE/SCALE/2
	cohesionStrength = 100

	separationRange = BOX_SIZE/SCALE/4
	separationStrength = 100

	alignementRange = BOX_SIZE/SCALE/2
	alignementStrength = 100

	steeringRange = 3 * SPHERE_SIZE
	steeringStrength = 500

	def __init__(self, position, velocity, radius):
		self.position = position
		self.velocity = velocity
		self.radius = radius

		# Peek a random color
		r = (random.randint(0, 1) + 1)/2
		g = (random.randint(0, 1) + 1)/2
		b = (random.randint(0, 1) + 1)/2

		# Make cone's length 4 times bigger than its radius
		axis = 4 * norm(self.velocity) * BOID_SIZE

		self.figure = cone(
			pos=self.position * SCALE,
			axis=axis,
			radius=BOID_SIZE,
			color=vector(r, g, b),
			make_trail=False,
			trail_type="curve")

	def update_neighbors(self):
		""" Find the nearest neighbors """
		global BOIDS

		maxRange = max(self.cohesionRange, self.separationRange, self.alignementRange) * SCALE

		# Particles within the maxRange distance
		self.neighbors = []

		for obj in BOIDS:
			if self == obj:
				continue

			if not obj.figure.visible:
				continue

			dx = math.fabs(self.position.x - obj.position.x)
			dy = math.fabs(self.position.y - obj.position.y)
			dz = math.fabs(self.position.z - obj.position.z)

			if dx > maxRange or dy > maxRange or dz > maxRange:
				continue

			d = (dx**2 + dy**2 + dz**2)**0.5
			if d > maxRange:
				continue

			self.neighbors.append((obj, d))

	def cohesion(self):
		""" Calculates cohesion adjustement (geometric center) """
		c = vector(0, 0, 0)
		count = 0

		if not self.figure.visible:
			return c

		for obj, d in self.neighbors:
			if d < self.cohesionRange * SCALE:
				c += obj.position
				count += 1

		if count:
			return norm(c/count - self.position) * self.cohesionStrength

		return c

	def separation(self):
		""" Calculates separation adjustement (avoidance) """
		c = vector(0, 0, 0)
		count = 0

		if not self.figure.visible:
			return c

		for obj, d in self.neighbors:
			if d < self.separationRange * SCALE:
				diff = self.position - obj.position
				k = math.pow((self.separationRange - d)/self.separationRange, 2)
				c += diff * k
				count += 1

		return norm(c) * self.separationStrength

	def alignement(self):
		""" Calculates velocity adjustement """
		c = vector(0, 0, 0)
		count = 0

		if not self.figure.visible:
			return c

		for obj, d in self.neighbors:
			if d < self.alignementRange * SCALE:
				c += obj.velocity
				count += 1

		if count:
			return norm(c - self.velocity) * self.alignementStrength

		return c

	def attraction(self):
		""" Calculates gravity forces between ship and asteroids """
		global SPHERES
	
		acc = vector(0, 0, 0)
		for obj in SPHERES:
			if not obj.figure.visible:
				continue

			d = mag(self.position - obj.position)
			if d <= obj.radius:
				self.figure.visible = False
				return vector(0, 0, 0)
			acc += gravity_acc(self.position, obj.position)
		return acc

	def steering(self):
		""" Calculates asteroids avoidance force """
		global SPHERES

		acc = vector(0, 0, 0)
		for obj in SPHERES:
			if not obj.figure.visible:
				continue

			OM = obj.position - self.position
			d = mag(cross(self.velocity, OM))/mag(self.velocity)
			if d > SPHERE_SIZE:
				obj.color = color.cyan
				continue

			obj.color = color.red
		
			if mag(OM) <= self.steeringRange:
				if d != 0:
					n = cross(cross(OM, self.velocity), self.velocity)
				else:
					n = vector(-self.velocity.y, self.velocity.x, 0)
				acc += norm(n)

		return acc * self.steeringStrength

	def check_borders(self):
		""" Cyclic conditions (hypertorus) """
		if self.position.x < -BOX_SIZE//2:
			self.figure.clear_trail()
			self.position.x += BOX_SIZE

		if self.position.x > BOX_SIZE//2:
			self.figure.clear_trail()
			self.position.x -= BOX_SIZE

		if self.position.y < -BOX_SIZE//2:
			self.figure.clear_trail()
			self.position.y += BOX_SIZE

		if self.position.y > BOX_SIZE//2:
			self.figure.clear_trail()
			self.position.y -= BOX_SIZE

		if self.position.z < -BOX_SIZE//2:
			self.figure.clear_trail()
			self.position.z += BOX_SIZE

		if self.position.z > BOX_SIZE//2:
			self.figure.clear_trail()
			self.position.z -= BOX_SIZE

	def move(self):
		self.update_neighbors()

		global dt
		self.velocity += self.cohesion() * dt
		self.velocity += self.separation() * dt
		self.velocity += self.alignement() * dt

		if mag(self.velocity) > MAX_SPEED:
			self.velocity = self.velocity/mag(self.velocity) * MAX_SPEED

		self.velocity += self.attraction() * dt
		self.velocity += self.steering() * dt
	
		self.position += self.velocity * dt

		self.check_borders()

		self.figure.pos = self.position/SCALE
		self.figure.axis = 4 * norm(self.velocity) * BOID_SIZE

		aliveParticles.text = "{}".format(get_alive_particles())

#==============================================================================
# Interface handlers
#==============================================================================
def createSphere(ev):
	loc = ev.pos
	obj = Sphere(position=loc * SCALE, radius=SPHERE_SIZE)
	SPHERES.append(obj)


def show_trail(ev):
	for obj in BOIDS:
		obj.figure.clear_trail()
		obj.figure.make_trail = ev.checked


def set_cohesionRange(ev):
	Boid.cohesionRange = ev.value
	cohesionRange_value.text = "{}".format(ev.value)


def set_cohesionStrength(ev):
	Boid.cohesionStrength = int(ev.text)

def set_separationRange(ev):
	Boid.separationRange = ev.value
	separationRange_value.text = "{}".format(ev.value)


def set_separationStrength(ev):
	Boid.separationStrength = int(ev.text)


def set_alignementRange(ev):
	Boid.alignementRange = ev.value
	alignementRange_value.text = "{}".format(ev.value)


def set_alignementStrength(ev):
	Boid.alignementStrength = int(ev.text)


def restart(ev):
	global SPHERES, BOIDS

	if ev.key != 'esc':
		return

	for obj in SPHERES:
		obj.figure.visible = False
		del obj

	SPHERES = []
	for obj in BOIDS:
		obj.figure.visible = True

#==============================================================================
# Interface handlers binding
#==============================================================================
scene.bind('click', createSphere)
scene.bind('keydown', restart)


scene.append_to_caption('\n')
cohesionRange = slider(bind=set_cohesionRange,
	vertical=False,
	min=10, max=BOX_SIZE/SCALE, step=10,
	value=Boid.cohesionRange,
	length=200)
scene.append_to_caption("Cohesion range ")
cohesionRange_value = wtext(text=" {}".format(Boid.cohesionRange))


scene.append_to_caption("\tStrength ")
cohesionStrength_value = winput(bind=set_cohesionStrength)
cohesionStrength_value.text = "{}".format(Boid.cohesionStrength)


scene.append_to_caption('\n')
separationRange = slider(bind=set_separationRange,
	vertical=False,
	min=10, max=BOX_SIZE/SCALE, step=10,
	value=Boid.separationRange,
	length=200)
scene.append_to_caption("Separation range ")
separationRange_value = wtext(text=" {}".format(Boid.separationRange))


scene.append_to_caption("\tStrength ")
separationStrength_value = winput(bind=set_separationStrength)
separationStrength_value.text = "{}".format(Boid.separationStrength)


scene.append_to_caption('\n')
alignementRange = slider(bind=set_alignementRange,
	vertical=False,
	min=10, max=BOX_SIZE/SCALE, step=10,
	value=Boid.alignementRange,
	length=200)
scene.append_to_caption("Alignement range ")
alignementRange_value = wtext(text=" {}".format(Boid.alignementRange))


scene.append_to_caption("\tStrength ")
alignementStrength_value = winput(bind=set_alignementStrength)
alignementStrength_value.text = "{}".format(Boid.alignementStrength)


scene.append_to_caption('\n')
checkbox(bind=show_trail, text="Show trail")

scene.append_to_caption("\tAlive particles =")
aliveParticles = wtext(text=" {}".format(NBOIDS))

#==============================================================================
# Generate boids
#==============================================================================

for i in range(NBOIDS):
	x = random.randint(-BOX_SIZE//2, BOX_SIZE//2)
	y = random.randint(-BOX_SIZE//2, BOX_SIZE//2)
	z = random.randint(-BOX_SIZE//2, BOX_SIZE//2)

	vx = random.randint(-MAX_SPEED, MAX_SPEED)
	vy = random.randint(-MAX_SPEED, MAX_SPEED)
	vz = random.randint(-MAX_SPEED, MAX_SPEED)

	obj = Boid(vector(x, y, z), vector(vx, vy, vz), BOID_SIZE)
	#obj = Boid(vector(-BOX_SIZE//2, y, z), vector(MAX_SPEED, 0, 0), BOID_SIZE)
	BOIDS.append(obj)

#BOIDS.append(Boid(vector(0, -BOX_SIZE/2, 0), vector(0, MAX_SPEED, 0), BOID_SIZE))
SPHERES.append(Sphere(position=vector(0, 0, 0), radius=SPHERE_SIZE))
SPHERES.append(Sphere(position=vector(BOX_SIZE/4, BOX_SIZE/4, 0), radius=SPHERE_SIZE))

#==============================================================================
# Bounding box
#==============================================================================
mybox = box(pos=vector(0, 0, 0),
	length=BOX_SIZE/SCALE, height=BOX_SIZE/SCALE, width=BOX_SIZE/SCALE, opacity=0.1) 

#==============================================================================
# Main cycle
#==============================================================================
dt = 1

while True:
	rate(100)
	for obj in BOIDS:
		obj.move()