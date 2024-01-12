import numpy as np
from sklearn.cluster import KMeans

# Given data
X = np.array([[5.9,3.2],
[4.6,2.9],
[6.2,2.8],
[4.7,3.2],
[5.5,4.2],
[5.0,3.0],
[4.9,3.1],
[6.7,3.1],
[5.1,3.8],
[6.0,3.0]
])

# Initialize cluster centers
initial_centers = np.array([[6.2, 3.2],  # μ1 (red)
                            [6.6, 3.7],  # μ2 (green)
                            [6.5, 3.0]   # μ3 (blue)
                           ])

# Run k-means clustering with k=3
kmeans = KMeans(n_clusters=3, init=initial_centers, n_init=1, algorithm='full', random_state=42)
kmeans.fit(X)

# Get cluster centers after one iteration
centers_iteration_1 = kmeans.cluster_centers_

# Get cluster centers after two iterations
kmeans.fit(X)  # Fit again to perform the second iteration
centers_iteration_2 = kmeans.cluster_centers_

# Get cluster centers when clustering converges
converged_centers = kmeans.cluster_centers_

# Get the number of iterations required for convergence
num_iterations_to_converge = kmeans.n_iter_

# Round the results to three decimal places
centers_iteration_1 = np.round(centers_iteration_1, 3)
centers_iteration_2 = np.round(centers_iteration_2, 3)
converged_centers = np.round(converged_centers, 3)

print(f"Center of the first cluster (red) after one iteration: {centers_iteration_1[0]}")
print(f"Center of the second cluster (green) after two iterations: {centers_iteration_2[1]}")
print(f"Center of the third cluster (blue) when clustering converges: {converged_centers[2]}")
print(f"Number of iterations required for convergence: {num_iterations_to_converge}")
