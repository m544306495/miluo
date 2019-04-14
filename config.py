import matplotlib.pyplot as plt
import numpy as np
import  os

#读取文件
def getfile(filename):

    data = np.array(np.loadtxt(filename, dtype=int))
    data1 = data[200:220]  #取其200行
    return data

#初始化矩阵
def data_deal(data,k):

    labelmat = np.zeros((data.shape[0],1))  #标签
    N = data.shape[0]   #样本数量
    M = data.shape[1]   #属性数量
    P = np.random.rand(N, int(k))
    Q = np.random.rand(M, int(k))
    w = np.zeros((M,1)) #权重向量
    return labelmat,M,P,Q,w

#数据归一化
def nomal(data):

    max, min = np.amax(data), np.amin(data)
    data = (data - min) / (max - min)
    return data

#控制数字大小，使其不产生inf，nan
def  control(P,Q,w):
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if np.isinf(P[i][j]) or np.isnan(P[i][j]):
                P[i][j] = 0.0000001
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if np.isinf(Q[i][j]) or np.isnan(Q[i][j]):
                Q[i][j] = 0.0000001
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if np.isinf(w[i][j]) or np.isnan(w[i][j]):
                w[i][j] = 0.0000001
    return P,Q,w

 #矩阵迭代
def matrix_factorization(data,labelmat,M,P, Q, k,step,alpha,beta):
    Q = Q.T
    w = np.ones((M, 1))
    # .T操作表示矩阵的转置 Q的维度2*6
    # print(data.shape)
    # print(labelmat.shape)
    # print(M)
    # print(P.shape)
    # print(Q.shape)
    # print(w.shape)
    result = []

    for step in range(step):
        e = np.linalg.norm((data - np.dot(P, Q)), ord=2) ** 2 + beta / 2 * (np.linalg.norm(P, ord=2) ** 2) + np.linalg.norm(Q, ord=2) ** 2 + np.linalg.norm(np.dot(np.dot(P, Q), w) - labelmat, ord=2) ** 2
        e1 = data - np.dot(P, Q)
        P = P + alpha * (
        2 * np.dot(e1, Q.T) - beta * P - 2 * np.dot((np.dot(np.dot(P, Q), w) - labelmat), (np.dot(Q, w).T)))
        Q = Q + alpha * (2 * np.dot(e1.T, P).T - beta * Q - 2 * (np.dot(np.dot(P.T, (np.dot(np.dot(P, Q), w) - labelmat)), w.T)))  # 凑得
        w = w - 2 * alpha * (np.dot((np.dot(np.dot(P, Q), w) - labelmat).T, np.dot(P, Q))).T
        P,Q,w = control(P,Q,w)

        if np.isinf(e):
            e = 0.0000001

        result.append(e)

    return P, Q.T, w, result

#画图，横坐标为迭代次数，纵坐标为损失
def draw(result):
    n = len(result)
    x = range(n)
    plt.plot(x, result, color='r', linewidth=3)
    plt.title("Convergence curve")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()

def main(path):
    if os.path.isdir(path):  #输入的参数是文件夹
        step = int(input("请输入迭代次数:"))
        alpha = float(input("请输入学习率:"))
        beta = float(input("请输入正则化参数:"))
        k = int(input("请输入矩阵规模k:"))

        fileName = os.listdir(path)
        for file  in fileName:
            file = path+'\\'+file
            print (file)
            data = nomal(getfile(file))
            labelmat, M, P, Q, w = data_deal(data, k)
            nP, nQ, nw, result = matrix_factorization(data, labelmat, M, P, Q, k, step, alpha, beta)
            draw(result)
    elif os.path.isfile(r"data\{}".format(path)):
        # filename = input("请输入文件名：")
        filename = r"data\{}".format(path)
        step = int(input("请输入迭代次数:"))
        alpha = float(input("请输入学习率:"))
        beta = float(input("请输入正则化参数:"))
        k = int(input("请输入矩阵规模k:"))

        data = nomal(getfile(filename))
        labelmat, M, P, Q, w = data_deal(data, k)
        nP, nQ, nw, result = matrix_factorization(data, labelmat, M, P, Q, k, step, alpha, beta)
        draw(result)
    else:
        print(path)
        print(type(path))
if __name__ == "__main__":
    # filename = input("请输入文件名：")
    # filename = r"data\{}".format(filename)
    # step = int(input("请输入迭代次数:"))
    # alpha = float(input("请输入学习率:"))
    # beta = float(input("请输入正则化参数:"))
    # k = int(input("请输入矩阵规模k:"))
    # data = nomal(file(filename))
    # labelmat,M,P,Q,w= data_deal(data,k)
    # nP, nQ, nw, result = matrix_factorization(data,labelmat,M,P, Q, k,step,alpha,beta)
    # draw(result)
    path = input("请输入想修复的数据：")
    main(path)