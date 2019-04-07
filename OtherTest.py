import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

test = np.loadtxt('./Aggregation.txt', delimiter=',')
clf = KMeans(n_clusters=7)
s = clf.fit(test)
print(clf.cluster_centers_)
print(clf.labels_)
print(clf.inertia_)

ax = plt.subplot()
plt.title("n = 4")
plt.scatter([i[0] for i in test], [i[1] for i in test], c=clf.labels_)
for point in clf.cluster_centers_:
    ax.annotate(u"Cluster Center",xy=point,
                xytext=(26,15),size=10,
                va="center",ha="center",
                bbox=dict(boxstyle='sawtooth',fc="w"),
                arrowprops=dict(arrowstyle="-|>",  #"-|>"代表箭头头上是实心的
                                connectionstyle="angle,rad=0.4",fc='r')  #rad代表箭头是否是弯的，+-定义弯的方向
                )
plt.show()