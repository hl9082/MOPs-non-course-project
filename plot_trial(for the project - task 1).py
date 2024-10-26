'''
@file plot_trial(for the project - task 1).py
@description This is where we plot 2 scatter plots, with one being the rotated version of the other,
and we also apply ICP to find the distance between each point in the 1st graph with its correspondance.
@author Huy Le (hl9082)
@author Jorge (please enter your full name and RIT username)
@author Eric (please enter your full name and RIT username)
'''
import random #for randomization
import matplotlib.pyplot as plt #for plotting
import numpy as np #for matrix operations
import math #for abs function
from scipy.spatial import cKDTree #to optimize ICP using KDTree to find nearest neighbor

'''
@brief Plot 2 scatter plots, with the 2nd being the rotated version of the 1st. As well as,
printing out necessary matrices, alongside the pairs of original-rotated points along with distances in-between.
@param lower_x lower x limit
@param lower_y lower y limit
@param width width of the screen.
@param height height of the screen.
@param limit the number of points.
@param angle the angle to rotate.
@pre x must be 0, y must be 1.
'''

def plot_2_graphs(lower_x,lower_y,width,height,limit,angle):
    
    if lower_x != 0 or lower_y!=1:
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

    
    
    x2 = rotated_data[lower_x, :] #it goes all columns in the row.
    y2 = rotated_data[lower_y, :]

    plt.subplot(1, 2, 2)
    plt.scatter(x2, y2)
    plt.plot(x2,y2, '--', 'o',color='red')
    plt.title('Rotated Scatter Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    aligned_points, rotation_matrix, translation_vector,indices,distances = icp(original_points, rotated_data)
    print("Original Points:\n",original_points)
    print("Rotated data:\n",rotated_data)
    print("Rotation matrix:\n",rotation_matrix)
    print("Translation vector:\n",translation_vector)
    
    for i in range(len(original_points)):
        print(f"Original Point {original_points[i]} -> Aligned Point {rotated_data[indices[i]]}, Distance: {distances[i]:.4f}")

def icp(source, target, max_iterations=100, tolerance=1e-6):
    """
    Apply the Iterative Closest Point algorithm to align the source points to the target points.

    Parameters:
        source (numpy.ndarray): Original scatterplot points (NxD array).
        target (numpy.ndarray): Rotated scatterplot points (NxD array).
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Tolerance for convergence.

    Returns:
        numpy.ndarray: Aligned source points.
        numpy.ndarray: Rotation matrix.
        numpy.ndarray: Translation vector.
    """
    prev_error = float('inf')
    for i in range(max_iterations):
         # Step 1: Find the closest points in the target for each point in the source
         tree=cKDTree(target)
         distances,indices=tree.query(source)
    
        #Step 2: Calculate the centroid of source and target points
         src_centroid=np.mean(source,axis=0)
         tgt_centroid=np.mean(target[indices],axis=0)
    
        #Step 3: Center the points.
         src_centered = source - src_centroid
         tgt_centered = target[indices] - tgt_centroid
    
        # Step 4: Calculate the covariance matrix
         H=np.dot(src_centered.T,tgt_centered)
    
        #Step 5: Singular Value Decomposition
         U, S, Vt=np.linalg.svd(H)
         R=np.dot(Vt.T,U.T)
         
         # Ensure a proper rotation (det(R) = 1)
         if np.linalg.det(R) < 0:
             Vt[1, :] *= -1
             R = np.dot(Vt.T, U.T)
    
        #Step 6: Calculate translation
         t=tgt_centroid-np.dot(R,src_centroid)
    
        #Step 7: Update the source points
         source=np.dot(source,R)+t
         
         # Compute mean squared error
         mean_error = np.mean(np.linalg.norm(target[indices] - source, axis=1))
    
        #Check for convergence
         if abs(prev_error - mean_error)<tolerance: #comparing the vector length with the convergence
             break
         prev_error=mean_error
    
    return source,R,t,indices,distances
     

'''
Where we implement everything, with the inputted necessary data.
'''
def main():
    lower_x=int(input("Enter the lower x limit: "))
    #upper_x=int(input("Enter the upper x limit: "))
    lower_y=int(input("Enter the lower y limit: "))
    #upper_y=int(input("Enter the upper y limit: "))
    width=int(input("Enter the width: "))
    height=int(input("Enter the height : "))
    limit=int(input("Enter the data limit: "))
    angle=float(input("Enter the angle to rotate: "))
    plot_2_graphs(lower_x,lower_y,width,height,limit,angle)
if __name__=='__main__':
    main()
