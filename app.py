from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from KMeans import KMeans


digits = load_digits()
images = digits.images
targets = digits.target


N = int(len(images)*0.8)


X_train = images[:N]
X_test = images[N:]

y_train = targets[:N]
y_test = targets[N:]


kmeans = KMeans(10,'kfarthest',n_init=2,max_iters=100,tol=1e-6,loss='rmse')
kmeans.fit(X_train,y_train)
y_pred = kmeans.predict(X_test)


print(y_test[:10])
print(y_pred[:10].astype(int))
acc = kmeans.calc_accuracy(y_test)

print(acc)

fig,axes = plt.subplots(nrows=5,ncols=2,figsize=(8,8))
for ax,image,label in zip(axes.flat,digits.images[N:N+10],digits.target[N:N+10]):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r)
    ax.set_title("label "+str(label))
plt.show()