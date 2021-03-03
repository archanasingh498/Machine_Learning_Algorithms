from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import numpy as np
from scipy.stats import mode
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

print("Creating datasets\n")
#Generate blob dataset
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

print("Displaying ScatterPlot")
#Looking at graph
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()

print("Running KElbowVisualizer")
#determine the best k for k-means
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12))
visualizer.fit(X)        
visualizer.show()

print("Running Kmeans")
#Kmeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

print("Displaying Colored ScatterPlot")
#Plotting graph after Kmeans
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis');
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()

#Creating labels
labels = np.zeros_like(y_kmeans)
for i in range(4):
    mask = (y_kmeans == i)
    labels[mask] = mode(y_true[mask])[0]

print("Displaying Confusion Matrix")
#draw a confusion matrix
mat = confusion_matrix(y_true,labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False )
plt.show()


#calculate accuracy for best K
print("The accuracy score is: ",accuracy_score(y_true,labels)*100, "%")

