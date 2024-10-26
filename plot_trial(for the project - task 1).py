import random
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.spatial import cKDTree
#from sklearn.neighbors import NearestNeighbors 

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

    #x2 = rotated_data[lower_x, :upper_x]
    #y2 = rotated_data[lower_y, :upper_y]
    
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
