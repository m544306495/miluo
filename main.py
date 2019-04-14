import matplotlib.pyplot as plt
from math import pow
import  numpy as np
import  config

def matrix_factorization(data,P,Q,K,steps=10,alpha=0.00002,beta=0.002):
    Q=Q.T  # .T操作表示矩阵的转置 Q的维度2*6
    result=[]
    w = np.ones((M,1))

    for step in range(steps):
        e = np.linalg.norm((data - np.dot(P, Q)),ord=2)**2+beta/2*(np.linalg.norm(P,ord=2)**2)+np.linalg.norm(Q,ord=2)**2+np.linalg.norm(np.dot(np.dot(P,Q),w)-labelmat,ord=2)**2
        e1= data - np.dot(P,Q)
        P = P + alpha*(2*np.dot(e1,Q.T) - beta*P - 2*np.dot((np.dot(np.dot(P,Q),w)-labelmat),(np.dot(Q,w).T)))
        Q = Q + alpha*(2*np.dot(e1.T,P).T - beta*Q - 2*(np.dot(np.dot(P.T,(np.dot(np.dot(P,Q),w)-labelmat)),w.T))) #凑得
        w = w - 2*alpha*(np.dot((np.dot(np.dot(P,Q),w)-labelmat).T,np.dot(P,Q))).T
        # P = np.nan_to_num(P)
        # Q = np.nan_to_num(Q)
        # w = np.nan_to_num(w)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if np.isinf(P[i][j])or np.isnan(P[i][j]):
                    P[i][j] = 0.0000001
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                if np.isinf(Q[i][j])or np.isnan(Q[i][j]):
                   Q[i][j] =0.0000001
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                if np.isinf(w[i][j])or np.isnan(w[i][j]):
                    w[i][j] = 0.0000001
        if np.isinf(e):
            e = 0.0000001


        result.append(e)
        print(e)


    return P,Q.T,w,result
if __name__ == "__main__":
    R = np.loadtxt("mfeat-fac.csv",dtype=int)
    R = np.array(R)
    max,min = np.amax(R),np.amin(R)
    R = (R-min)/(max-min)

    K = 4
    data = R[200:400, :]
    labelmat = np.zeros((200,1))
    N = len(data)
    M = len(data[0])
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    nP, nQ ,nw,result= matrix_factorization(data, P, Q, K)
    T = np.dot(nP,nQ.T)
    n = len(result)
    x = range(n)


    plt.plot(x,result,color = 'r',linewidth=3)
    plt.title("Convergence curve")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()

