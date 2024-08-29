import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


def generate_place_fields(x,
                          y,
                          env_corners,
                          num_place_cells=9,
                          influence_radius=0.1):
    """
    Generates place fields for multiple place cells based on the (x, y) coordinates 
    and environment geometry defined by corners.

    Args:
        x (float): X coordinate of the current position.
        y (float): Y coordinate of the current position.
        env_corners (list of tuples): List of (x, y) points defining the environment's corners.
        num_place_cells (int): Number of place cells.
        influence_radius (float): Influence radius of place fields.

    Returns:
        place_fields (array): Activation levels of the place fields for each place cell.
    """
    # Define the environment boundary using the given corners
    env_boundary = Path(env_corners)

    # Create a grid to visualize place fields
    grid_size = 100
    xx, yy = np.meshgrid(np.linspace(0, 1, grid_size),
                         np.linspace(0, 1, grid_size))
    grid_points = np.vstack((xx.flatten(), yy.flatten())).T

    # Determine which points are inside the environment
    inside = env_boundary.contains_points(grid_points).reshape(
        grid_size, grid_size)

    # Initialize place fields for each place cell
    place_fields = []

    for i in range(num_place_cells):
        # Randomly initialize place cell centers within the environment
        while True:
            cx, cy = np.random.rand(2)  # Random center positions in [0, 1]
            if env_boundary.contains_point((cx, cy)):
                break

        # Compute place field activation as a Gaussian function
        place_field = np.exp(-((xx - cx)**2 + (yy - cy)**2) /
                             (2 * influence_radius**2))
        place_field *= inside  # Zero out values outside the environment
        place_fields.append(place_field)

    # Calculate activation rates for each place cell at the current (x, y)
    current_position = np.array([x, y])
    rates = [
        np.exp(-np.linalg.norm(current_position - np.array([cx, cy]))**2 /
               (2 * influence_radius**2)) for cx, cy in env_corners
    ]

    return rates, place_fields


# Define environment corners for a more complex shape (e.g., L-shape)
env_corners = [(0.1, 0.1), (0.8, 0.1), (0.8, 0.4), (0.5, 0.4), (0.5, 0.8),
               (0.1, 0.8)]

# Generate place fields for a position within the environment
x, y = 0.3, 0.3  # Example coordinates inside the environment
rates, place_fields = generate_place_fields(x, y, env_corners)

# Visualize the place fields
plt.figure(figsize=(15, 10))
for i, place_field in enumerate(place_fields):
    plt.subplot(3, 3, i + 1)
    plt.imshow(place_field,
               extent=[0, 1, 0, 1],
               origin='lower',
               cmap='viridis')
    plt.colorbar(label='Firing Rate')
    plt.title(f'Place Cell {i+1}')
    plt.plot(x, y, 'rx')  # Mark the current position
    plt.plot(*zip(*env_corners), 'wo-',
             linewidth=2)  # Show environment boundary
    plt.axis('equal')

plt.tight_layout()
plt.suptitle('Generated Place Fields for Complex Environment', fontsize=16)
plt.subplots_adjust(top=0.92)
plt.show()
