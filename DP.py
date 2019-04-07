import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics


class DensityPeak:

    def __init__(self, points, TAO, cluster_num, kernel, halo):
        self.TAO = TAO
        self.points = points
        self.kernel = kernel
        self.distance_compress = pdist(points, 'mahalanobis', VI=None)
        self.distance = squareform(self.distance_compress, force='no', checks=True)
        self.distance_compress.sort()
        print(self.distance_compress)
        self.cutoff_distance = self.distance_compress[round(self.TAO * len(self.distance_compress))]
        self.cluster_num = cluster_num
        self.densityList = []
        self.distanceList = []
        self.labelList = []  # 用于记录聚类信息
        self.clusterCenter = []
        self.boarderRegionDensity = {}
        self.halo = halo


    def findCutoffDistance(self):
        cutoffDistanceList = np.linspace(0.05, 10, 200)
        entropyList = []
        for cutoff_distance in cutoffDistanceList:
            densityList = []
            for i in range(0, len(self.points)):
                pointdensity = 0
                for j in range(0, len(self.points)):
                    distance = self.calDistanceBetweenTwoPoints(i, j)
                    pointdensity += np.exp(-(distance / cutoff_distance) ** 2)
                densityList.append(pointdensity)
            Z = sum(densityList)
            H = sum((np.array(densityList) / Z) * np.log(np.array(densityList) / Z) * (-1))
            entropyList.append(H)
            print(cutoff_distance)

        minCur = -1
        min = float("inf")
        for i in range(0, len(cutoffDistanceList)):
            if entropyList[i] < min:
                min = entropyList[i]
                minCur = cutoffDistanceList[i]
        ax = plt.subplot()
        plt.plot(cutoffDistanceList, entropyList)
        ax.annotate(u"{}, {:.2f}".format(minCur, min),xy=[minCur, min],
                    xytext=(5, 5),size=10,
                    va="center",ha="center",
                    bbox=dict(boxstyle='sawtooth',fc="w"),
                    arrowprops=dict(arrowstyle="-|>",  #"-|>"代表箭头头上是实心的
                                    connectionstyle="angle,rad=0.4",fc='r')  #rad代表箭头是否是弯的，+-定义弯的方向
                    )
        plt.xlabel('sigma')
        plt.ylabel('H')
        plt.show()


    def findClusterNumber(self):
        self.calDensity()
        self.calDistance()
        gamma = np.array(self.densityList) * np.array(self.distanceList)
        gamma = sorted(gamma, reverse=True)
        ax = plt.subplot()
        plt.scatter([i for i in range(1, len(gamma)+1)], gamma)
        plt.xlabel('n')
        plt.ylabel('gamma')
        for i in range(0, 9):
            ax.annotate(u"{}, {:.2f}".format(i+1, gamma[i]),xy=[i, gamma[i]],
                        xytext=((i+1)*50, gamma[i]+i*8),size=10,
                        va="center",ha="center",
                        bbox=dict(boxstyle='sawtooth',fc="w"),
                        arrowprops=dict(arrowstyle="-|>",  #"-|>"代表箭头头上是实心的
                                        connectionstyle="angle,rad=0.4",fc='r')  #rad代表箭头是否是弯的，+-定义弯的方向
                        )
        plt.show()


    def beginCluster(self):
        self.calDensity()
        self.calDistance()
        self.calClusterCenter()
        self.labelAllPoint()
        if self.halo is True:
            self.calHalo()
        ss = metrics.silhouette_score(self.points, self.labelList)
        print('ss = {}'.format(ss))
        self.draw()

    
    def calDistanceBetweenTwoPoints(self, point1, point2):
        return self.distance[point1, point2]


    def calDensity(self):
        self.densityList.clear()
        for i in range(0, len(self.points)):
            pointdensity = 0
            for j in range(0, len(self.points)):
                distance = self.calDistanceBetweenTwoPoints(i, j)
                # if distance < self.cutoff_distance:
                if distance < self.cutoff_distance and self.kernel == 'cutoff':
                    pointdensity += 1
                elif self.kernel == 'gaussian':
                    pointdensity += np.exp(-(distance / self.cutoff_distance) ** 2)
            self.densityList.append(pointdensity)

    
    def calDistance(self):
        self.distanceList.clear()
        self.labelList.clear()
        for i in range(0, len(self.densityList)):
            distance = float("inf")  # 初始距离为无穷大
            label = -1  # 初始聚类结果：-1
            for j in range(0, len(self.densityList)):
                if self.densityList[i] < self.densityList[j]:
                    new_dist = self.calDistanceBetweenTwoPoints(i, j)
                    if new_dist < distance:  # 找到这个点到所有比它本身密度大的点的距离的最小值
                        distance = new_dist
                        label = j  # 将来聚类时，i点将被归入j点所在的类
            if distance == float("inf"):  # 这个点本身密度最大
                distance = -1
                label = i  # i点本身聚类中心
                self.clusterCenter.append(self.points[i])
                for j in range(0, len(self.densityList)):
                    new_dist = self.calDistanceBetweenTwoPoints(i, j)
                    if new_dist > distance:  # 找到密度最大点到其他点的最远距离
                        distance = new_dist
            self.distanceList.append(distance)
            self.labelList.append(label)


    def calClusterCenter(self):
        self.clusterCenter.clear()
        kth = -self.cluster_num  # 找到第kth大的值
        threshold = np.partition(self.distanceList, kth, axis=None)[kth]
        print(threshold)
        for i in range(0, len(self.distanceList)):
            if self.distanceList[i] >= threshold:
                self.labelList[i] = i  # 选为聚类中心
                self.clusterCenter.append(self.points[i])


    def labelAllPoint(self):
        for i in range(0, len(self.labelList)):
            j = i
            while j != self.labelList[j]:  # 未追溯到聚类中心
                j = self.labelList[j]
            self.labelList[i] = j


    def calHalo(self):
        for label in set(self.labelList):
            self.boarderRegionDensity[label] = 0
        for i in range(0, len(self.labelList)):
            label = self.labelList[i]
            for j in range(0, len(self.labelList)):
                # 找到属于boarder region的点
                if label != self.labelList[j] and self.calDistanceBetweenTwoPoints(i, j) < self.cutoff_distance:
                    self.boarderRegionDensity[label] = (self.densityList[i] if (self.densityList[i] > self.boarderRegionDensity[label])
                                                                            else self.boarderRegionDensity[label])
        for i in range(0, len(self.densityList)):
            if self.densityList[i] < self.boarderRegionDensity[self.labelList[i]]:
                self.labelList[i] += 0.1  # 标记为cluster halo
            

    def draw(self):
        labelList = self.labelList
        labelSet = set(labelList)
        num = 0
        for i in labelSet:
            num += 1
            for j in range(0, len(labelList)):
                if i == labelList[j]:
                    labelList[j] = num
        ax = plt.subplot(2, 1, 1)
        plt.title("d = {}".format(self.cutoff_distance))
        plt.scatter([i[0] for i in self.points], [i[1] for i in self.points], c=labelList)
        for point in self.clusterCenter:
            ax.annotate(u"Cluster Center",xy=point,
                        xytext=(26,15),size=10,
                        va="center",ha="center",
                        bbox=dict(boxstyle='sawtooth',fc="w"),
                        arrowprops=dict(arrowstyle="-|>",  #"-|>"代表箭头头上是实心的
                                        connectionstyle="angle,rad=0.4",fc='r')  #rad代表箭头是否是弯的，+-定义弯的方向
                        )
        plt.subplot(2, 1, 2)
        plt.scatter(self.densityList, self.distanceList, c=labelList)
        plt.xlabel('Rou')
        plt.ylabel('Phi')
        plt.show()
        # plt.savefig("{}.jpg".format(self.cutoff_distance))


if __name__ == "__main__":
    test = np.loadtxt('./Aggregation.txt', delimiter=',')
    DP = DensityPeak(test, 0.02, 7, 'gaussian', halo=False)
    DP.findClusterNumber()
    # DP.findCutoffDistance()
    DP.beginCluster()
    print(DP.labelList)
    print(len(set(DP.labelList)))