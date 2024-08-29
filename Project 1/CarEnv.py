import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random

# Enable interactive mode
plt.ion()

class CarEnv:
    def __init__(self, x_limit, y_limit, start_position, start_orientation, target_position, target_orientation):
        
        self.start_position = start_position
        self.start_orientation = start_orientation
        
        self.x, self.y = start_position[0], start_position[1]
        self.orientation = start_orientation
        self.target_position = target_position
        self.target_orientation = target_orientation
        
        self.actions = [ 'straight', 'right', 'left' ]
        self.velocities = [1, 2, 3]
        self.directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        self.x_bounds = (-x_limit, x_limit)
        self.y_bounds = (-y_limit, y_limit)
        
        self.movements = []  # [(x, y), orientation, velocity]
        self.steps = []
        self.current_index = 0
        self.fig, self.ax = plt.subplots()

        # Buttons
        axprev = plt.axes([0.7, 0.02, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.02, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next)
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev)

        # Change the window title (for TkAgg backend)
        if plt.get_backend() == 'TkAgg':
            manager = plt.get_current_fig_manager()
            manager.window.title('Car Movement Visuals')  # Set the window title to 'CAR'

    def step(self, action, velocity, log = False):
            
        if(log):
            print("Move made: ", action, velocity, " @ ", (self.x, self.y, self.orientation), end = " ==> ")
        
        steering, velocity = action, velocity
        
        cur_step = [(self.x, self.y), self.orientation, steering, velocity]
        self.update_orientation(steering)
        
        if(log):
            self.movements.append(((self.x, self.y), self.orientation, velocity))
        
        self.update_position(velocity)
        
        cur_step.append((self.x, self.y))
        cur_step.append(self.orientation)
        
        if(log):
            self.steps.append(cur_step)
        
        if(log):
            print((self.x, self.y, self.orientation))
        
        if(log):
            self.update_plot()
        
        done = self.check_done()
        reward = self.get_reward(done, steering, velocity)
        return (self.x, self.y, self.orientation), reward, done
    
    def update_orientation(self, steering):
        for i, direction in enumerate(self.directions):
            if direction == self.orientation:
                current_idx = i
                break
        
        # print("Steering: ", steering, " @ ", self.orientation, end = " -> ")
            
        if steering == 'left':
            self.orientation = self.directions[(8 + current_idx - 1) % 8]
        elif steering == 'right':
            self.orientation = self.directions[(8 + current_idx + 1) % 8]
            
        # print(self.orientation)
    
    def update_position(self, velocity):
        direction_map = {
            'N': (0, 1), 'NE': (1, 1), 'E': (1, 0), 'SE': (1, -1),
            'S': (0, -1), 'SW': (-1, -1), 'W': (-1, 0), 'NW': (-1, 1)
        }
        x_sign, y_sign = direction_map[self.orientation]
        
        self.x = self.x + velocity * x_sign
        self.y = self.y + velocity * y_sign
        
        # self.x = np.clip(self.x + velocity * x_sign, *self.x_bounds)
        # self.y = np.clip(self.y + velocity * y_sign, *self.y_bounds)
    
    def check_done(self):
        return (self.x, self.y) == self.target_position and self.orientation == self.target_orientation
    
    def get_reward(self, done, steering, velocity):
        if done:
            return 10000  # Large reward for reaching target
        else:
            time = 1  # Small penalty for each time step
            return -time
    
    def arrow_buffer(self, velocity, dx, dy):
        if(velocity <= 10):
            return 0.8
        if(velocity <= 20):
            return 0.9
        if(velocity <= 60):
            return 0.95
        return 0.975
    
    def update_plot(self):
        # Check if the figure still exists, if not, create a new one
        if not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots()
            
            # Re-add buttons in the new figure
            axprev = plt.axes([0.7, 0.02, 0.1, 0.075])
            axnext = plt.axes([0.81, 0.02, 0.1, 0.075])
            self.bnext = Button(axnext, 'Next')
            self.bnext.on_clicked(self.next)
            self.bprev = Button(axprev, 'Previous')
            self.bprev.on_clicked(self.prev)

            # Set window title for TkAgg backend
            if plt.get_backend() == 'TkAgg':
                manager = plt.get_current_fig_manager()
                manager.window.title('Car Movement Visuals')  # Set the window title to 'CAR'
                    
        self.ax.clear()
        self.ax.set_xlim(self.x_bounds)
        self.ax.set_ylim(self.y_bounds)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('Move #{}'.format(self.current_index + 1))
        # self.ax.grid(True)
        
        start_pos, direction, velocity = self.movements[self.current_index]
        direction_vectors = {
            'N': (0, 1), 'NE': (1, 1), 'E': (1, 0), 'SE': (1, -1),
            'S': (0, -1), 'SW': (-1, -1), 'W': (-1, 0), 'NW': (-1, 1)
        }
        dx, dy = direction_vectors[direction]
        end_pos = (start_pos[0] + velocity * dx, start_pos[1] + velocity * dy)
        
        buff = self.arrow_buffer(velocity, dx, dy)
        arr_buff = 1
        if(velocity == 1):
            arr_buff = 0.5
        if(velocity == 2):
            arr_buff = 0.75
        if(velocity == 3):
            arr_buff = 1
        
        self.ax.arrow(start_pos[0], start_pos[1], dx * velocity * buff, dy * velocity * buff,
                      head_width=0.5*arr_buff, head_length=0.5*arr_buff, fc='black', ec='black')
        
        self.ax.scatter(start_pos[0], start_pos[1], color='blue', label='Previous Position')
        self.ax.scatter(end_pos[0], end_pos[1], color='red', label='Current Position')
        self.ax.scatter(self.target_position[0], self.target_position[1], color='green', label='Target')
        self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        s = self.steps[self.current_index]
        
        console_text = """
        Previous State:
        Position: {} 
        Orientation: {}
        
        Action:
        Direction: {}
        Velocity: {}
        
        Current State:
        Position: {}
        Orientation: {}
        """.format(s[0], s[1], s[2], s[3], s[4], s[5])
        
        x_pos = 1.08  # X position to the right of the plot
        y_pos = 0.75   # Y position within the plot area (adjust as needed)

        # Add the text to the plot
        self.ax.text(x_pos, y_pos, console_text, fontsize=10, verticalalignment='top',
         bbox=dict(facecolor='lightgrey', alpha=0.5, edgecolor='black'), transform=self.ax.transAxes)

        plt.tight_layout()
        self.fig.canvas.draw()

    def next(self, event):
        if self.current_index < len(self.movements) - 1:
            self.current_index += 1
            self.update_plot()

    def prev(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_plot()
            
    def render(self):
        # Disable interactive mode
        plt.ioff()
        plt.show()
        plt.ion()
        
    def reset(self):
        self.movements = []
        self.steps = []
        self.current_index = 0
        
        self.x = self.start_position[0]
        self.y = self.start_position[1]
        self.orientation = self.start_orientation
        
    def off_interactive(self):
        plt.ioff()
        
    def on_interactive(self):
        plt.ion()
        
## Testing the CarEnv class
# env = CarEnv(5, 5, (0, 0), 'N', target_position=(10, 10), target_orientation='E')
# env.step('straight', 2, 1)
# env.step('left', 2, 1)
# env.step('left', 1, 1)
# env.step('left', 1, 1)
# env.step('left', 3, 1)

# env.render()

# env.reset()
# env.step('straight', 2, 1)
# env.step('left', 1, 1)
# env.step('left', 1, 1)
# env.step('left', 1, 1)

# env.render()