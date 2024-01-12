import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('iris.csv')
df.head()

X = df.drop(['species'],axis=1)
y=df['species']

X = StandardScaler().fit_transform(X)

# features = X.T
cov_matrix = np.cov(X.T)
values, vectors = np.linalg.eig(cov_matrix)

var = []
for i in range(len(values)):
    var.append((values[i] / np.sum(values))*100)

print("variances of each feature",var)

plt.bar(range(4),var)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Dimensions')

projected_1 = X.dot(vectors.T[0])
projected_2 = X.dot(vectors.T[1])
res = pd.DataFrame(projected_1, columns=['PC1'])
res['PC2'] = projected_2
res['Y'] = y
print(res.head())

sns.FacetGrid(res, hue="Y").map(plt.scatter, 'PC1', 'PC2').add_legend()
plt.show()
