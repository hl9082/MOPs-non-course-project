import random
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.spatial.distance import cdist
#from sklearn.neighbors import NearestNeighbors 

def plot_2_graphs(lower_x,upper_x,lower_y,upper_y,width,height,limit,angle):
    
    if lower_x > upper_x or lower_y>upper_y:
        print("The range is invalid!")
        
    # Generate random data for the first scatter plot
    x1 = np.random.rand(limit)
    y1 = np.random.rand(limit)
    
    original_points=np.array([x1,y1])
    
    plt.figure(figsize=(width, height))

    plt.subplot(1, 2, 1)
    plt.scatter(x1, y1)
    plt.plot(x1,y1,'--', 'o',color='blue')
    plt.title('Original Scatter Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    # Define the rotation angle in degrees
    
    theta = np.radians(angle)

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]])

    # Apply the rotation matrix to the data
    rotated_data = np.dot(rotation_matrix, original_points)

    #x2 = rotated_data[lower_x, :upper_x]
    #y2 = rotated_data[lower_y, :upper_y]
    
    x2 = rotated_data[0, :]
    y2 = rotated_data[0, :]

    plt.subplot(1, 2, 2)
    plt.scatter(x2, y2)
    plt.plot(x2,y2, '--', 'o',color='red')
    plt.title('Rotated Scatter Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    ''' 
    # Draw lines connecting each original point to its corresponding rotated point
    for i in range(len(original_points)):
        plt.plot([original_points[i, 0], rotated_data[i, 0]],
                 [original_points[i, 1], rotated_data[i, 1]], 'k-')
        # 'k-' specifies a solid black line
    #plt.legend()
    plt.show()

# Print the matches
    for i, (orig, rot) in enumerate(zip(original_points, rotated_data)):
        print(f"Original point {i}: {orig} -> Corresponding rotated point: {rot}")
   '''

    
def main():
    lower_x=int(input("Enter the lower x limit: "))
    upper_x=int(input("Enter the upper x limit: "))
    lower_y=int(input("Enter the lower y limit: "))
    upper_y=int(input("Enter the upper y limit: "))
    width=int(input("Enter the width: "))
    height=int(input("Enter the height : "))
    limit=int(input("Enter the data limit: "))
    angle=float(input("Enter the angle to rotate: "))
    plot_2_graphs(lower_x,upper_x,lower_y,upper_y,width,height,limit,angle)
if __name__=='__main__':
    main()
