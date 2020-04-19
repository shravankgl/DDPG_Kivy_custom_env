# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture


# Importing the Dqn object from our AI in ai.py
#from ai import Dqn

import td3 

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
#brain = Dqn(5,3,0.9)
#action2rotation = [0,5,-5]
state_dim = (32, 32)
action_dim =  (2,1)
max_action = 1.


brain = td3.TD3(state_dim, action_dim, max_action)
max_rotation = 5
max_movement = 3
episode_max_steps = 10_000
last_reward = 0
scores = []
im = CoreImage("./images/MASK1_.png")

# textureMask = CoreImage(source="./kivytest/simplemask1.png")


# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    #global 

    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255

    #sand[0:5,0:-1] = 1
    #sand[1424:,0:-1] = 1
    #sand[0:-1,0:5]=1
    #sand[0:-1,655:] = 1
    goal_x = 1420
    goal_y = 622
    first_update = False
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
   

    def move(self, action):
        rotation = action[0]
        forward = action[1]
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = int(rotation*max_rotation)
        self.angle = self.angle + self.rotation
        self.velocity = Vector( int(forward * max_movement), 0).rotate(self.angle)       

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    max_timesteps = 5e5
    total_timesteps = 0
    batch_size = 100
    start_timesteps = 1e4
    replay_buffer = td3.ReplayBuffer()
    expl_noise = 0.1
    discount = 0.99
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2

    def serve_car(self):
        #self.car.center = self.center
        self.car.center = (450,325)
        self.car.velocity = Vector(6, 0)
        self.done = 0
        self.episode_steps = 0 
        self.episode_rewards = 0

    def update1(self, dt):
        pass    

    def update(self, dt):

        
        #global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        #orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        #last_signal = [0, 0, 0, orientation, -orientation]
        #action = brain.update(last_reward, last_signal)
        #scores.append(brain.score())
        #rotation = action2rotation[action]


        #-------------Train code
        curr_state = self.get_state("curr")

        if self.total_timesteps < 2000:#self.start_timesteps:
            action = self.get_action_sample()
        else:
            action = brain.select_action(curr_state)
            
            
            if expl_noise != 0:
                action[0] = (action[0] + np.random.normal(0, expl_noise, size=1)).clip(-1, 1)
                action[1] = (action[1] + np.random.normal(0, expl_noise, size=1)).clip(0, 1)

        

        next_state, last_distance = self.step(action,last_distance)

        if self.total_timesteps != 0 and self.done != 1:    
            self.replay_buffer.add((curr_state, next_state, action, last_reward, float(self.done)))
            print("while adding")
            print(curr_state.shape)
            print(next_state.shape)    
        
        #if self.total_timesteps == 100:
        #     print(self.replay_buffer.storage[0])
        #    App.get_running_app().stop()   

        # if distance < 25:
        #     if swap == 1:
        #         goal_x = 1420
        #         goal_y = 622
        #         swap = 0
        #     else:
        #         goal_x = 9
        #         goal_y = 85
        #         swap = 1
        #last_distance = distance
        self.episode_steps += 1
        
        self.episode_rewards += last_reward

        #if done train and reset 
        if self.done == 1:
            if self.total_timesteps !=0:    
                brain.train(self.replay_buffer, self.episode_steps, self.batch_size, self.discount, self.tau, self.policy_noise, self.noise_clip, self.policy_freq)
            self.serve_car()
        self.total_timesteps += 1    

        if self.total_timesteps == self.max_timesteps:
            App.get_running_app().stop()

    def step(self, action, last_distance):
        self.car.move(action)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        next_state = self.get_state("next")
       

        if sand[int(self.car.x),int(self.car.y)] > 0:
            last_reward = -1
            self.done = 1
        else: # otherwise
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        if self.car.x < 5:
            self.car.x = 5
            last_reward = -100
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            last_reward = -100
        if self.car.y < 5:
            self.car.y = 5
            last_reward = -100
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            last_reward = -100

        if distance == 0:
            done = 1
            last_reward = 1000    

        if  self.episode_steps == episode_max_steps:
            done = 1
            last_reward = -500 

        return next_state , distance    

        

    def get_state(self,name = ""):
        full_screen = np.copy(sand)
        gap = 40
        full_screen = np.pad(full_screen, pad_width=gap, mode='constant', constant_values=1)

        #obs = full_screen[int(self.car.x)-gap:int(self.car.x)+gap, int(self.car.y)-gap:int(self.car.y)+gap]
        obs = full_screen[int(self.car.x):int(self.car.x)+(2*gap), int(self.car.y):int(self.car.y)+(2*gap)]
        obs_img = PILImage.fromarray(obs.astype("uint8")*255)
        bg_w, bg_h = obs_img.size

        car_img = PILImage.open("./images/yellow-car.png") 
        car_img = car_img.rotate(self.car.angle, PILImage.NEAREST, expand = 1)
        img_w, img_h = car_img.size 
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
        obs_img.paste(car_img, offset, car_img)
        obs_img = obs_img.resize((32,32))
        #if self.total_timesteps % 200 == 0:
            #obs_img.save("./images/obs"+ str(self.total_timesteps) +"_"+name+".jpg")
        #print(np.asarray(obs_img).shape)    
        return np.asarray(obs_img)    

    def get_action_sample(self):
        angle = np.random.uniform(-1,1,1)[0]
        forward = np.random.uniform(0,1,1)[0]    
        return angle, forward




# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")
            
    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
