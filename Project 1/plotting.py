import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_value_function(V, title="Value Function", save = 0):
    """
    Plots the value function as a series of surface plots, one for each orientation.
    """
    # Determine the range of x and y from the keys in V
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())
    
    # List of possible orientations
    orientations = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    # Create the range for x and y
    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)
    
    folder_path = './results_plots'
    file_name = 'value_function.png'

    def plot_surface(X, Y, Z, title):
        """
        Helper function to plot a 3D surface.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(25, -135)  # Set a good viewing angle for 3D plot
        fig.colorbar(surf)
        
        if(save):
            plt.savefig(os.path.join(folder_path, file_name))
        else:
            plt.show()

    # Plot the value function for each orientation
    for orien in orientations:
        Z = np.zeros_like(X, dtype=np.float64)

        # Fill the Z array with the corresponding values from V
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x_val = X[i, j]
                y_val = Y[i, j]
                Z[i, j] = V.get((x_val, y_val, orien), 0)  # Use a default of 0 if key is not found

        file_name = '{}_{}.png'.format(orien, title)
        plot_surface(X, Y, Z, "{} (Orientation: {})".format(title, orien))

# # Example usage
# # Assuming you have a value function V that maps (x, y, orientation) to a float value
# V = {
#     (0, 0, 'N'): 0.1, (1, 0, 'N'): 0.2, (2, 0, 'N'): 0.3,
#     (0, 1, 'N'): 0.2, (1, 1, 'N'): 0.4, (2, 1, 'N'): 0.5,
#     # Add more values for other orientations and coordinates as needed
# }

# plot_value_function(V, title="Value Function")
