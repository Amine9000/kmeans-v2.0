import numpy as np
import random


class KMeans:
    def __init__(self,n_clusters,init="random",n_init=10,max_iters=300,tol=1e-4,loss='mse',random_state=None) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.loss = loss
        self.error = {
            'sse':self.SSE,
            'mse':self.MSE,
            'rmse':self.RMSE
        }
        self.inits = {
            'random':self._init_random,
            'kfarthest':self._init_kfarthest
        }

    def _init_kfarthest(self,X,y):
        if self.random_state:
                np.random.seed(self.random_state)
        centers = np.array([X[y==0][np.random.choice(X[y==0].shape[0],1,replace=False)]])
        for i in range(1,self.n_clusters):
            # check if i > n_cluster
            if(np.isin(i,y)):
                n = i%self.n_clusters
            data = X[y==n]
            result = data[:,np.newaxis,:] - centers
            result = (result**2).sum(axis=(2,3))
            min_dist = np.min(result,axis=1)
            min_dist_idx = min_dist.argmax()
            new_center = np.array([[data[min_dist_idx]]])
            centers = np.concatenate((centers,new_center),axis=1)
        
        new_shape = centers.shape[1:]
        centers = centers.reshape(new_shape)
        self.centers = centers


    def _init_random(self,X,y):
        firstD = (self.n_clusters,)
        restD = X.shape[1:]
        shape = firstD+restD
        centers = np.empty(shape)
        for i in range(self.n_clusters):
            if self.random_state:
                np.random.seed(self.random_state)
            i_cluster = X[y==i]
            random_index = np.random.randint(0,i_cluster.shape[0])
            centers[i] = i_cluster[random_index]
        self.centers = centers
    
    def assign_labels(self,X):
        self.labels = np.empty(len(X))
        for i_record,record in enumerate(X):
            dists = np.empty((self.n_clusters,2))
            for c_index,center in enumerate(self.centers):
                sub = record - center
                squared = sub**2
                summ = np.sum(squared)
                d = np.sqrt(summ)
                dists[c_index] = np.array([d,c_index])
            min_index = np.argmin(dists,axis=0)
            self.labels[i_record] = dists[min_index[0]][1]
        
    def mean(self,X,y):
        num_rows = (self.n_clusters,)
        ele_shape = X.shape[1:]
        centroids = np.empty(num_rows+ele_shape)
        for i in range(self.n_clusters):
            i_cluster = X[y==i]
            sum_cluster = np.sum(i_cluster,axis=0)
            mean_cluster = sum_cluster/len(i_cluster)
            centroids[i] = mean_cluster
        return centroids

    def distance(self,point,center,dist_alg="euclidean"):
        dist = 0
        if(dist_alg == "euclidean"):
            sub = point - center
            squared = sub**2
            summ = squared.sum()
            dist = np.sqrt(summ)
        if(dist_alg == "manhaten"):
            sub = point - center
            dist = sub.sum()
        return dist

    def SSE(self,X):
        sse = np.empty(self.n_clusters)
        for i,center in enumerate(self.centers):
            for recored in X[X==i]:
                sse[i] += self.distance(recored,center)**2
        return sse.sum()

    def MSE(self,X):
        mse = np.empty(self.n_clusters)
        for i,center in enumerate(self.centers):
            for recored in X[X==i]:
                mse[i] += self.distance(recored,center)**2
            mse[i] /= len(X[X==i])
        return mse.sum()
    
    def RMSE(self,X):
        rmse = np.empty(self.n_clusters)
        for i,center in enumerate(self.centers):
            for recored in X[X==i]:
                rmse[i] += self.distance(recored,center)**2
            rmse[i] /= len(X[X==i])
        return rmse.sum()**0.5



    def calc_accuracy(self,y):
        dif = y - self.labels
        correct = dif[dif==0]
        acc = len(correct)/len(dif)
        return acc

    def fit(self,X,y):
        self.inits[self.init](X,y)
        self.assign_labels(X)
        self.accuracy = self.calc_accuracy(y)
        last_err = self.error[self.loss](X)
        tol_state = False
        for i in range(self.n_init):
            for j in range(self.max_iters):
                centeroids = self.mean(X,y)
                self.assign_labels(X)
                acc = self.calc_accuracy(y)
                if self.accuracy > acc:
                    self.accuracy = acc
                    self.centers = centeroids
                err = self.error[self.loss](X)

                max_len = 100
                x = (j*max_len)/self.max_iters
                done = "#"*int(x/2)
                rest = " "*int((max_len - x - 1)/2)
                print("\r[{0}{1}] {2}%".format(done,rest,x),end="")
                
                if abs(err - last_err) <= self.tol:
                    tol_state = True
                    break
                else:
                    last_err = err
            print()
            if(tol_state):
                break

    def predict(self,X):
        self.assign_labels(X)
        return self.labels
