import numpy as np
from scipy import stats

# Given array
data = [115.3, 195.5, 120.5, 110.2, 90.4, 105.6, 110.9, 116.3, 122.3, 125.4]

# Mean
mean_value = np.mean(data)
print(f"Mean: {mean_value}")

# Median
median_value = np.median(data)
print(f"Median: {median_value}")

# Mode
mode_value = stats.mode(data)[0][0]
print(f"Mode: {mode_value}")

# Standard Deviation
std_dev = np.std(data)
print(f"Standard Deviation: {std_dev}")

# Variance
variance = np.var(data)
print(f"Variance: {variance}")

# Min-Max Normalization
min_max_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
print(f"Min-Max Normalized: {min_max_normalized}")

# Standardization (Z-score)
z_score_normalized = (data - np.mean(data)) / np.std(data)
print(f"Standardized (Z-score): {z_score_normalized}")
 
------------------------------------------------------------------------------------------------

#import statistics as st
arr = [115.3, 195.5, 120.5,120.5,120.5, 110.2, 90.4, 105.6, 110.9, 116.3, 122.3, 125.4]
sum = 0
for num in arr:
    sum += num
mean = sum/(len(arr))
print("mean = ", mean)

arr1 = sorted(arr)
median = (arr1[len(arr1) // 2 - 1] + arr1[len(arr1) // 2]) / 2     
print("median = ", median)

mode = {}
for ele in arr:
    if ele not in mode:
        mode[ele] = 0
    else:
        mode[ele] += 1
    count = [g for g,l in mode.items() if l==max(mode.values())]
print("mode = ",count[0])
print("sd = ",st.stdev(arr))
print("varience = ",st.variance(arr))
print("max = ", max(arr))
print("min = ", min(arr))

