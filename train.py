import numpy as np
import ot
import tensorflow as tf
import matplotlib.pyplot as plt

def h(X,W):
    """
    :param X: train feature X=[x1,x2,x3...xm].T dimension: m*d
    :param W: the weight of h(x)  W=[w1,w2,w3..wL].T dimension: L*d
    :return: h(x) L*m
    """

    temp = np.exp(np.dot(W, X.T))
    col_sum = temp.sum(axis=0)
    result = temp/col_sum
    return result



def mapping_learning_gradient(M, W, X, Y, reg):
    """
    get the gradient of loss function
    loss function: F(W) = epsilon l(h(x),y)
    :param M: cost matrix
    :param W: the weight of h(x)  W=[w1,w2,w3..wL].T dimension: L*d
    :param X: train feature X=[x1,x2,x3...xm].T dimension: m*d
    :param Y: train label  Y = [y1,y2,y3...ym].T dimension: m*L
    :param lam: the hyper parameter of entropy regularization
    :return: the gradient of loss function
    """
    hh_x = h(X, W).T
    gradient = 0
    loss = 0
    coupling = 0
    L = W.shape[0]
    for i in range(Y.shape[0]):
        h_x = hh_x[i]
        x = X[i]
        y = Y[i]
        ot_distance, couple, u, v = ot.sinkhorn(h_x, y, M, reg)
        loss += ot_distance
        coupling += couple

        #compute the gra_h, dimension L*1
        gra_h = np.log(u)/reg - np.log(np.sum(u))/(reg*L)

        gra = np.zeros(W.shape)

        #compute the gra_w, dimension L*d
        for j in range(len(h_x)):
            temp_yy = h_x * -h_x[j]
            temp_y = temp_yy.copy()
            temp_y[j] = temp_yy[j] + h_x[j]
            temp_y = temp_y.reshape(-1, 1)
            temp_x = x.reshape(1, -1)
            temp_x = temp_x * temp_y
            temp_gra = np.sum(gra_h.reshape(-1, 1) * temp_x, axis=0)
            gra[j] = temp_gra

        gradient = gradient + gra

    return loss, coupling, gradient

"""
        #compute the gra_w, dimension L*d
        x = x.reshape(1, x.shape[0])
        temp_1 = np.dot(W, x.T)
        temp_2 = temp_1*x*(np.sum(temp_1)-temp_1)
        gra_w = temp_2/(np.sum(temp_1)*np.sum(temp_1))

        gradient = gradient + gra_h.reshape(2, 1)*gra_w
"""


def mapping_train(X, Y, M, learning_rate = 1e-4, num_iters = 300,batch_size = 100, reg = 0.05, verbose = False):
    """
    :param X: train feature X=[x1,x2,x3...xm].T dimension: m*d
    :param Y: train label  Y = [y1,y2,y3...ym].T dimension: m*L
    :param M: cost matrix
    :param learning_rate:
    :param num_iters:
    :param batch_size:
    :param verbose:
    :return:
    """

    num_train, dim = X.shape
    L = Y.shape[1]
    W = 0.001 * np.random.randn(L,dim)
    coupling = 0
    loss_history = []
    for it in range(num_iters):
        X_batch = None
        Y_batch = None

        batch_idx = np.random.choice(num_train, batch_size, replace=False)
        X_batch = X[batch_idx]
        Y_batch = Y[batch_idx]

        loss, couple, grad = mapping_learning_gradient(M,W,X_batch,Y_batch,reg)
        loss_history.append(loss)
        coupling = couple     #get the newest couple
        W += -learning_rate * grad

#        if it % 100 == 0:
#            print('iteration %d: loss %f' % (it, loss))
    return loss_history, coupling

def ground_train(P, Y, C):
    """
    :param C: hyper parameter between OT and K
    :param P: coupling matrix dimension: L*L
    :param Y: train label  Y = [y1,y2,y3...ym].T dimension: m*L
    :return:
    """
    K_0 = np.dot(Y.T, Y)
    K_temp = K_0 + 2*P
    for i in range(Y.shape[1]):
        K_temp[i][i] = K_0[i][i]-np.sum(P[i])-np.sum(P[:, i])+2*P[i][i]
    eigvals, eigvectors = np.linalg.eig(K_temp)
    eigvals = np.maximum(eigvals, 0)
    K = np.dot(eigvectors*eigvals.T, eigvectors.T)
    K_diag = np.diag(K)
    K_diag = K_diag.reshape(-1, 1)
    M = K_diag - 2*K + K_diag.T
    return M/C

def compression_train(Y, M, l):
    """
    learn the mapping from high dimension to low dimension, Y:m*L -> Y':m*l
    :param Y:the label matrix
    :param M:the compressive label correlation matrix
    :param l:hyper parameter, the low dimension
    :return:low dimension label m*l
    """
    def weight_variable(shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    L = Y.shape[1]
    x = tf.placeholder("float", [None, L])

    W_1 = weight_variable([L, 150])
    bias_1 = bias_variable([150])
    layer_1 = tf.nn.relu(tf.matmul(x, W_1) + bias_1)

    W_2 = weight_variable([150, l])
    bias_2 = bias_variable([l])
    layer_2 = tf.nn.relu(tf.matmul(layer_1, W_2) + bias_2)

    sess = tf.InteractiveSession()

    loss = tf.norm(tf.matmul(tf.transpose(layer_2), layer_2)-M)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        train_step.run(feed_dict={x: Y})
        if i%100 == 0:
            loss_value = loss.eval(feed_dict={x: Y})
            print("step %d, loss %d", (i,loss_value))
    layer = layer_2.eval(feed_dict={x: Y})
    temp = np.dot(layer.T,layer)
    print(layer)
if __name__ == '__main__':
    # for testing
    X = np.random.rand(100, 50)
    Y = np.random.rand(100, 10)
    Y_sum = np.sum(Y, axis=1).reshape(-1, 1)
    Y = Y/Y_sum
    C = 1
    M = np.eye(5)*0.05
    compression_train(Y,M,5)
"""    for iter in range(100):
        loss_history, coupling = mapping_train(X, Y, M)
        print("the iter last loss: %f", loss_history[-1])
#        x_cor = range(len(loss_history))
#        plt.plot(x_cor, loss_history, 'ro-')
#        plt.show()
        M = ground_train(coupling, Y, C)
"""