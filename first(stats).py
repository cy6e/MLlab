import statistics as st

x = [115.3, 195.5, 120.5, 110.2, 90.4, 105.6, 110.9, 116.3, 122.3, 125.4, 90.4]

sum_x = sum(x)
mean = sum_x / len(x)

x.sort()

if len(x) % 2 == 0:
    a = x[len(x)//2]
    b = x[(len(x)//2) - 1]
    median = (a + b) / 2
else:
    median = x[len(x)//2]

frequency_dict = {}
for i in x:
    if i in frequency_dict:
        frequency_dict[i] += 1
    else:
        frequency_dict[i] = 1

mx = 0
mode = 0
for key, value in frequency_dict.items():
    if value >= mx:
        mx = value
        mode = key

var = sum((i - mean)**2 for i in x) / (len(x) - 1)

std = var**0.5

xm = min(x)
xmax = max(x)

MMS = [(i - xm) / (xmax - xm) for i in x]

SS = [(i - mean) / std for i in x]

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Standard Deviation:", std)
print("Variance:", var)
print("Min-Max Scaling:", MMS)
print("Standardization (Z-score):", SS)
