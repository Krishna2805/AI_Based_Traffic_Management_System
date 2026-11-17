import random, time, threading, pygame, sys, os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Default values of signal times
defaultRed = 120
defaultYellow = 5
defaultGreen = 15
defaultMinimum = 10
defaultMaximum = 60

signals = []
noOfSignals = 4
simTime = 6000  
timeElapsed = 0

currentGreen = 0   # Indicates which signal is green
currentYellow = 0   # Indicates whether yellow signal is on or off 

# Average times for vehicles to pass the intersection
carTime = 2 
bikeTime = 1
rickshawTime = 2.25
busTime = 2.5
truckTime = 2.5

noOfLanes = 2

# Red signal time at which cars will be detected at a signal
detectionTime = 5

speeds = {'car':2, 'bus':1.8, 'truck':1.8, 'rickshaw':2, 'bike':2.5}  # average speeds of vehicles

# Coordinates of start
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}    
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0}, 'down': {0:[], 1:[], 2:[], 'crossed':0}, 'left': {0:[], 1:[], 2:[], 'crossed':0}, 'up': {0:[], 1:[], 2:[], 'crossed':0}}
vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'rickshaw', 4:'bike'}
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]
vehicleCountCoods = [(480,210),(880,210),(880,550),(480,550)]
vehicleCountTexts = ["0", "0", "0", "0"]

# Coordinates of stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
stops = {'right': [580,580,580], 'down': [320,320,320], 'left': [810,810,810], 'up': [545,545,545]}

mid = {'right': {'x':705, 'y':445}, 'down': {'x':695, 'y':450}, 'left': {'x':695, 'y':425}, 'up': {'x':695, 'y':400}}
rotationAngle = 3

# Gap between vehicles
gap = 25    # stopping gap
gap2 = 35   # moving gap

pygame.init() #can setup pygame modules 
simulation = pygame.sprite.Group()  #creates an empty group in Pygame to manage all vehicle sprites in the traffic simulation.

class VehicleCounter:
    def __getVehicleCount(self, direction, vehicle_type):
        count = 0
        if direction in vehicles:
            for lane in vehicles[direction]:
                if isinstance(lane, int):
                    for vehicle in vehicles[direction][lane]:
                        if vehicle.vehicleClass == vehicle_type and vehicle.crossed == 0:
                            count += 1
        return count
    
    def getBikeCount(self, direction):
        return self.__getVehicleCount(direction, 'bike')
        
    def getCarCount(self, direction):
        return self.__getVehicleCount(direction, 'car')
        
    def getRickshawCount(self, direction):
        return self.__getVehicleCount(direction, 'rickshaw')
        
    def getBusCount(self, direction):
        return self.__getVehicleCount(direction, 'bus')
        
    def getTruckCount(self, direction):
        return self.__getVehicleCount(direction, 'truck')

        
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.setup_basic_attributes(lane, vehicleClass, direction_number, direction, will_turn)
        self.load_image()
        self.setup_position()
        simulation.add(self)
        
    def setup_basic_attributes(self, lane, vehicleClass, direction_number, direction, will_turn):
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1

    def load_image(self):
        path = os.path.join(IMAGES_DIR, self.direction, f"{self.vehicleClass}.png")
        self.originalImage = pygame.image.load(path)
        self.currentImage = pygame.image.load(path)

    def setup_position(self):
        # Set stop line based on vehicles ahead
        if len(vehicles[self.direction][self.lane]) > 1 and not vehicles[self.direction][self.lane][self.index-1].crossed:
            prev_vehicle = vehicles[self.direction][self.lane][self.index-1]
            if self.direction == 'right':
                self.stop = prev_vehicle.stop - prev_vehicle.currentImage.get_rect().width - gap
            elif self.direction == 'left':
                self.stop = prev_vehicle.stop + prev_vehicle.currentImage.get_rect().width + gap
            elif self.direction == 'down':
                self.stop = prev_vehicle.stop - prev_vehicle.currentImage.get_rect().height - gap
            else:  # up
                self.stop = prev_vehicle.stop + prev_vehicle.currentImage.get_rect().height + gap
        else:
            self.stop = defaultStop[self.direction]
        
        # Adjust position based on direction
        if self.direction == 'right':
            temp = self.currentImage.get_rect().width + gap
            x[self.direction][self.lane] -= temp
            stops[self.direction][self.lane] -= temp
        elif self.direction == 'left':
            temp = self.currentImage.get_rect().width + gap
            x[self.direction][self.lane] += temp
            stops[self.direction][self.lane] += temp
        elif self.direction == 'down':
            temp = self.currentImage.get_rect().height + gap
            y[self.direction][self.lane] -= temp
            stops[self.direction][self.lane] -= temp
        elif self.direction == 'up':
            temp = self.currentImage.get_rect().height + gap
            y[self.direction][self.lane] += temp
            stops[self.direction][self.lane] += temp

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))     #blit is used to draw images in pygame

    def get_leading_vehicle_distance(self):
        """Calculate distance to vehicle ahead in same lane"""
        if self.index == 0:
            return float('inf')
        leading = vehicles[self.direction][self.lane][self.index - 1]
        if leading.turned:
            return float('inf')
        
        # Calculate actual dimensions based on vehicle orientation
        if abs(self.rotateAngle) >= 45:
            self_width = self.currentImage.get_rect().height
            self_height = self.currentImage.get_rect().width
            leading_width = leading.currentImage.get_rect().height
            leading_height = leading.currentImage.get_rect().width
        else:
            self_width = self.currentImage.get_rect().width
            self_height = self.currentImage.get_rect().height
            leading_width = leading.currentImage.get_rect().width
            leading_height = leading.currentImage.get_rect().height
        
        # Calculate distance based on direction
        if self.direction in ['right', 'left']:
            distance = leading.x - (self.x + self_width) if self.direction == 'right' else self.x - (leading.x + leading_width)
        else:
            distance = leading.y - (self.y + self_height) if self.direction == 'down' else self.y - (leading.y + leading_height)
        
        return max(distance, 0)

    def move(self):
        # Constants for movement behavior
        stopping_distance = 50
        min_speed = 0.1
        safe_distance = 35
        current_speed = self.speed
        
        def get_distance_to_stopline(self):
            if abs(self.rotateAngle) >= 45:
                width = self.currentImage.get_rect().height
                height = self.currentImage.get_rect().width
            else:
                width = self.currentImage.get_rect().width
                height = self.currentImage.get_rect().height
                
            if self.direction == 'right':
                return stopLines[self.direction] - (self.x + width)
            elif self.direction == 'left':
                return self.x - stopLines[self.direction]
            elif self.direction == 'down':
                return stopLines[self.direction] - (self.y + height)
            else:  # up
                return self.y - stopLines[self.direction]
        
        def should_stop_at_signal(self):
            if self.crossed == 1:
                return False
            
            return {
                'right': currentGreen != 0 or currentYellow == 1,
                'down': currentGreen != 1 or currentYellow == 1,
                'left': currentGreen != 2 or currentYellow == 1,
                'up': currentGreen != 3 or currentYellow == 1
            }[self.direction]

        # Calculate distances and conditions
        distance_to_stopline = get_distance_to_stopline(self)
        leading_distance = self.get_leading_vehicle_distance()
        should_stop = should_stop_at_signal(self)
        
        # Determine speed based on conditions
        if not self.crossed:
            if should_stop and 0 < distance_to_stopline < stopping_distance:
                current_speed = self.speed * max(distance_to_stopline / stopping_distance, min_speed)
                if distance_to_stopline < 5:
                    current_speed = 0
            
            if leading_distance < safe_distance:
                speed_factor = max(leading_distance / safe_distance, min_speed)
                current_speed = min(self.speed * speed_factor, current_speed)
                if leading_distance < 5:
                    current_speed = 0

        # Handle movement based on direction
        movement_handlers = {
            'right': self.__handle_right_movement,
            'down': self.__handle_down_movement,
            'left': self.__handle_left_movement,
            'up': self.__handle_up_movement
        }
        movement_handlers[self.direction](current_speed)

    def __handle_right_movement(self, current_speed):
        if self.crossed == 0 and self.x + self.currentImage.get_rect().width > stopLines[self.direction]:
            self.crossed = 1
            vehicles[self.direction]['crossed'] += 1
            
        if self.willTurn:
            if self.crossed == 0 or self.x + self.currentImage.get_rect().width < mid[self.direction]['x']:
                if current_speed > 0:
                    self.x += current_speed
            elif self.turned == 0:
                self.rotateAngle += rotationAngle
                self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                self.x += 2
                self.y += 1.8
                if self.rotateAngle >= 90:
                    self.turned = 1
            else:
                self.y += self.speed
        else:
            if current_speed > 0:
                self.x += current_speed

    def __handle_down_movement(self, current_speed):
        if self.crossed == 0 and self.y + self.currentImage.get_rect().height > stopLines[self.direction]:
            self.crossed = 1
            vehicles[self.direction]['crossed'] += 1
            
        if self.willTurn:
            if self.crossed == 0 or self.y + self.currentImage.get_rect().height < mid[self.direction]['y']:
                if current_speed > 0:
                    self.y += current_speed
            elif self.turned == 0:
                self.rotateAngle += rotationAngle
                self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                self.x -= 2.5
                self.y += 2
                if self.rotateAngle >= 90:
                    self.turned = 1
            else:
                self.x -= self.speed
        else:
            if current_speed > 0:
                self.y += current_speed

    def __handle_left_movement(self, current_speed):
        if self.crossed == 0 and self.x < stopLines[self.direction]:
            self.crossed = 1
            vehicles[self.direction]['crossed'] += 1
            
        if self.willTurn:
            if self.crossed == 0 or self.x > mid[self.direction]['x']:
                if current_speed > 0:
                    self.x -= current_speed
            elif self.turned == 0:
                self.rotateAngle += rotationAngle
                self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                self.x -= 1.8
                self.y -= 2.5
                if self.rotateAngle >= 90:
                    self.turned = 1
            else:
                self.y -= self.speed
        else:
            if current_speed > 0:
                self.x -= current_speed

    def __handle_up_movement(self, current_speed):
        if self.crossed == 0 and self.y < stopLines[self.direction]:
            self.crossed = 1
            vehicles[self.direction]['crossed'] += 1
            
        if self.willTurn:
            if self.crossed == 0 or self.y > mid[self.direction]['y']:
                if current_speed > 0:
                    self.y -= current_speed
            elif self.turned == 0:
                self.rotateAngle += rotationAngle
                self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                self.x += 1
                self.y -= 1
                if self.rotateAngle >= 90:
                    self.turned = 1
            else:
                self.x += self.speed
        else:
            if current_speed > 0:
                self.y -= current_speed
                    
class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0
        self.totalRedTime = 0  # Add counter for total red time
        self.priority = 0  # Add priority field

    def updatePriorities():
        """Calculate and return current priorities for all intersections"""
        priorities = {}
        counter = VehicleCounter()
        
        # Priority values
        priority_values = {
            'bike': 1,
            'car': 2,
            'rickshaw': 2,
            'bus': 3,
            'truck': 3
        }
        
        for direction in directionNumbers.values():
            # Calculate priority for each direction
            priority = (
                counter.getBikeCount(direction) * priority_values['bike'] +
                counter.getCarCount(direction) * priority_values['car'] +
                counter.getRickshawCount(direction) * priority_values['rickshaw'] +
                counter.getBusCount(direction) * priority_values['bus'] +
                counter.getTruckCount(direction) * priority_values['truck']
            )
            priorities[direction] = priority
            
        return priorities
    

def setTime():
    global currentGreen, currentYellow, nextGreen
    min_green_time = 10
    forced_green_time = 15
    max_red_time = 50
    
    while True:
        # Get current priorities
        priorities = TrafficSignal.updatePriorities()
        current_direction = directionNumbers[currentGreen]
        
        # Check for signals that have been red too long
        for i in range(noOfSignals):
            if i != currentGreen and signals[i].totalRedTime >= max_red_time:
                print(f"\nForcing green for signal {i} due to excessive red time")
                
                # Transition current green to red
                currentYellow = 1
                signals[currentGreen].yellow = defaultYellow
                signals[currentGreen].green = 0
                time.sleep(defaultYellow)
                
                # Reset signal states
                signals[currentGreen].yellow = 0
                currentYellow = 0
                signals[currentGreen].red = defaultRed
                signals[currentGreen].totalGreenTime = 0
                
                # Set forced green signal
                currentGreen = i
                nextGreen = (currentGreen + 1) % noOfSignals
                signals[currentGreen].green = forced_green_time
                signals[currentGreen].totalRedTime = 0
                
                updateValues()
                time.sleep(forced_green_time)
                continue
        
        # Normal priority-based signal management
        max_priority_direction = max(priorities, key=priorities.get)
        max_priority = priorities[max_priority_direction]
        current_priority = priorities[current_direction]
        
        # Store priorities in signal objects for display
        for i in range(noOfSignals):
            signals[i].priority = priorities[directionNumbers[i]]
        
        if (signals[currentGreen].totalGreenTime >= min_green_time and 
            max_priority_direction != current_direction and 
            max_priority > current_priority):
            
            currentYellow = 1
            signals[currentGreen].yellow = defaultYellow
            signals[currentGreen].green = 0
            time.sleep(defaultYellow)
            
            signals[currentGreen].yellow = 0
            currentYellow = 0
            signals[currentGreen].red = defaultRed
            signals[currentGreen].totalGreenTime = 0
            
            currentGreen = list(directionNumbers.values()).index(max_priority_direction)
            nextGreen = (currentGreen + 1) % noOfSignals
            signals[currentGreen].green = defaultGreen
            signals[currentGreen].totalRedTime = 0
        
        updateValues()
        time.sleep(1)

def updateValues():
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):
                signals[i].green-=1
                signals[i].totalGreenTime+=1
                signals[i].totalRedTime = 0  # Reset red time when green
            else:
                signals[i].yellow-=1
        else:
            signals[i].red-=1
            signals[i].totalRedTime+=1  

# Initialization of signals with default values
def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    thread = threading.Thread(name="timing", target=setTime, args=())
    thread.daemon = True
    thread.start()



# Generating vehicles in the simulation
def generateVehicles():
    while(True):
        vehicle_type = random.randint(0,4)
        if(vehicle_type==4):  #trucks always go in the non bike lane
            lane_number = 0
        else:
            lane_number = random.randint(0,1) + 1  #any 
        will_turn = 0
        if(lane_number==2):     #this function is basically uses rng to see if vehicle turns or not, serves no real purpose except irl traffic management
            temp = random.randint(0,4)
            if(temp<=2):
                will_turn = 1
            elif(temp>2):
                will_turn = 0
        direction_number = random.randint(0, 3)  # Ensures an equal chance for each direction (0, 1, 2, 3)
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number], will_turn)
        time.sleep(1.2)

def simulationTime():
    global timeElapsed, simTime
    while(True):
        timeElapsed += 1
        time.sleep(1)
        if(timeElapsed==simTime):
            totalVehicles = 0
            for i in range(noOfSignals):
                totalVehicles += vehicles[directionNumbers[i]]['crossed']
            os._exit(1)

class Main:
    def __init__(self):
        self.vehicle_counter = VehicleCounter()

        # Initialize threads
        self.thread4 = threading.Thread(name="simulationTime", target=simulationTime, args=())
        self.thread4.daemon = True
        self.thread4.start()

        self.thread2 = threading.Thread(name="initialization", target=initialize, args=())
        self.thread2.daemon = True
        self.thread2.start()

        # Colours 
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)

        # Screensize 
        self.screenWidth = 1400
        self.screenHeight = 800
        self.screenSize = (self.screenWidth, self.screenHeight)

        # Setting background image i.e. image of intersection
        self.background = pygame.image.load(os.path.join(IMAGES_DIR, 'mod_int.png'))

        self.screen = pygame.display.set_mode(self.screenSize)
        pygame.display.set_caption("SIMULATION")

        # Loading signal images and font
        self.redSignal = pygame.image.load(os.path.join(IMAGES_DIR, 'signals', 'red.png'))
        self.yellowSignal = pygame.image.load(os.path.join(IMAGES_DIR, 'signals', 'yellow.png'))
        self.greenSignal = pygame.image.load(os.path.join(IMAGES_DIR, 'signals', 'green.png'))
        self.font = pygame.font.Font(None, 30)

        self.thread3 = threading.Thread(name="generateVehicles", target=generateVehicles, args=())
        self.thread3.daemon = True
        self.thread3.start()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            self.screen.blit(self.background, (0, 0))
            for i in range(0, noOfSignals):
                if (i == currentGreen):
                    if (currentYellow == 1):
                        if (signals[i].yellow == 0):
                            signals[i].signalText = "STOP"
                        else:
                            signals[i].signalText = signals[i].yellow
                        self.screen.blit(self.yellowSignal, signalCoods[i])
                    else:
                        if (signals[i].green == 0):
                            signals[i].signalText = "SLOW"
                        else:
                            signals[i].signalText = signals[i].green
                        self.screen.blit(self.greenSignal, signalCoods[i])
                else:
                    if (signals[i].red <= 10):
                        if (signals[i].red == 0):
                            signals[i].signalText = "GO"
                        else:
                            signals[i].signalText = signals[i].red
                    else:
                        signals[i].signalText = "---"
                    self.screen.blit(self.redSignal, signalCoods[i])
            signalTexts = ["", "", "", ""]

            # display signal timer and vehicle count
            for i in range(0, noOfSignals):
                priority_text = self.font.render(f"P: {int(signals[i].priority)}", True, self.white, self.black)
                self.screen.blit(priority_text, signalTimerCoods[i])

                # Get vehicle counts using VehicleCounter with proper direction
                direction = directionNumbers[i]  # Get the direction based on signal number
                displayText = 0
                
                # Count all vehicle types for the current direction
                if i == 0:
                    displayText = (
                        self.vehicle_counter.getBikeCount(direction) +
                        self.vehicle_counter.getCarCount(direction) +
                        self.vehicle_counter.getRickshawCount(direction) +
                        self.vehicle_counter.getBusCount(direction) +
                        self.vehicle_counter.getTruckCount(direction)
                    )
                elif i == 1:
                    displayText = (
                        self.vehicle_counter.getBikeCount(direction) +
                        self.vehicle_counter.getCarCount(direction) +
                        self.vehicle_counter.getRickshawCount(direction) +
                        self.vehicle_counter.getBusCount(direction) +
                        self.vehicle_counter.getTruckCount(direction)
                    )
                elif i == 2:
                    displayText = (
                        self.vehicle_counter.getBikeCount(direction) +
                        self.vehicle_counter.getCarCount(direction) +
                        self.vehicle_counter.getRickshawCount(direction) +
                        self.vehicle_counter.getBusCount(direction) +
                        self.vehicle_counter.getTruckCount(direction)
                    )
                elif i == 3:
                    displayText = (
                        self.vehicle_counter.getBikeCount(direction) +
                        self.vehicle_counter.getCarCount(direction) +
                        self.vehicle_counter.getRickshawCount(direction) +
                        self.vehicle_counter.getBusCount(direction) +
                        self.vehicle_counter.getTruckCount(direction)
                    )
                
                vehicleCountTexts[i] = self.font.render(str(displayText), True, self.black, self.white)
                self.screen.blit(vehicleCountTexts[i], vehicleCountCoods[i])

            timeElapsedText = self.font.render(("Time Elapsed: " + str(timeElapsed)), True, self.black, self.white)
            self.screen.blit(timeElapsedText, (1100, 50))

            # display the vehicles
            for vehicle in simulation:
                self.screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
                vehicle.move()
            pygame.display.update()
Main()