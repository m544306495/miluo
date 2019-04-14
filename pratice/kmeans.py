from pylab import  *
from numpy import  *
from math  import  *
import  codecs
import  matplotlib.pyplot as plt


# 读取文件
def loadDataSet (filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

#返回欧几里得距离
def disEclud(vecA, vecB):
    return math.sqrt(sum(power((vecA-vecB), 2)))


# 构建聚簇中心，去k个
def ranCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        maxJ = max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, disMeans = disEclud, createCent = ranCent):

    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChange = True

    while clusterChange:
        clusterChange = False;
        for i in range(m):
            minDist = inf; minIndesx = -1
            for j in range(k):
                distJI = disMeans(centroids[j, :],dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if (clusterAssment[i, 0]) != minIndex:
                clusterChange = True
            clusterAssment[i, :] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):   # 重新计算中心点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
            centroids[cent,:] = mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
    return centroids, clusterAssment
# print(np.random.rand(2,1))
# a= np.mat(loadDataSet("testSet.txt"))
# print(ranCent(a,3))
dataMat = mat(loadDataSet("testSet.txt"))

myCentroids, clustAssing = kMeans(dataMat,4)

# for i in range(len(dataMat)):
#     plt.scatter(dataMat[i,0],dataMat[i,1],c="blue")
# plt.show()
color = 'rgbycmykw'
plt.subplot(2, 2, 1)
print(dataMat.shape)
for i in range(len(dataMat)):
    plt.plot(dataMat[i,0],dataMat[i,1],"ro",c= "blue")


plt.subplot(2,2,2)
for i in range(len(myCentroids)):
    plt.plot(myCentroids[i,0],myCentroids[i,1],"ro","red")


plt.subplot(2,2,3)
for i in range(len(clustAssing)):
    plt.plot(clustAssing[i,0],clustAssing[i,1],"ro",c=color[int(clustAssing[i,0])])


plt.subplot(2,2,4)
for i in range(len(dataMat)):
    plt.plot(dataMat[i,0],dataMat[i,1],"ro",c=color[int(clustAssing[i,0])])


plt.show()