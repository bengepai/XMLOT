import numpy as np
import ot


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
    L = W.shape[0]
    for i in range(Y.shape[0]):
        h_x = hh_x[i]
        x = X[i]
        y = Y[i]
        ot_distance, u, v = ot.sinkhorn(h_x, y, M, reg)
        loss += ot_distance

        #compute the gra_h, dimension L*1
        gra_h = np.log(u)/reg - np.log(np.sum(u))/(reg*L)

        #compute the gra_w, dimension 1*d
        temp_1 = np.dot(W, x.T)
        temp_2 = temp_1[i]*(np.sum(temp_1)-temp_1[i])*x
        gra_w = temp_2/(np.sum(temp_1)*np.sum(temp_1))

        gradient = gradient + np.dot(gra_h,gra_w)

    return loss, gradient


def mapping_train(X, Y, M, learning_rate = 1e-3, num_iters = 100,batch_size = 200, verbose = False):
    """
    :param X:
    :param Y:
    :param M:
    :param learning_rate:
    :param num_iters:
    :param batch_size:
    :param verbose:
    :return:
    """

    num_train, dim = X.shape
    L = Y.shape[1]
    W = 0.001 * np.random.randn(dim,L)

    loss_history = []
    for it in range(num_iters):
        X_batch = None
        Y_batch = None

        batch_idx = np.random.choice(num_train, batch_size, replace=True)
        X_batch = X[batch_idx]
        Y_batch = Y[batch_idx]

        loss, grad = mapping_learning_gradient(M,W,X_batch,Y_batch,reg)
        loss_history.append(loss)

        W += -learning_rate * grad

        if verbose and it % 100 == 0:
            print('iteration %d: loss %f' % (it, num_iters, loss))
    return loss_history

