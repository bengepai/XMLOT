import numpy as np
import ot
import tensorflow as tf
import scipy.io
from scipy import sparse
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

        temp_yy = h_x * -h_x.reshape(-1, 1)
        temp_y = temp_yy.copy()
        temp_y = temp_y + np.diag(h_x)
        temp_y_sum = np.sum(temp_y*gra_h.reshape(-1, 1), axis=1)
        temp_x = x.reshape(1, -1)
        gra = temp_x * temp_y_sum.reshape(-1, 1)

#        for j in range(len(h_x)):
#            temp_yy = h_x * -h_x[j]
#            temp_y = temp_yy.copy()
#            temp_y[j] = temp_yy[j] + h_x[j]
#            temp_y = temp_y.reshape(-1, 1)
#            temp_x = x.reshape(1, -1)
#            temp_x = temp_x * temp_y
#            temp_gra = np.sum(gra_h.reshape(-1, 1) * temp_x, axis=0)
#            gra[j] = temp_gra

        gradient = gradient + gra

    return loss, coupling, gradient

def mapping_train(X, Y, M, learning_rate = 1e-6, num_iters = 100,batch_size = 4000, reg = 0.05, verbose = False):
    """
    learn the h(x)
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
    W = 0.001 * np.random.randn(L, dim)
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

        if it % 1 == 0:
            print('mapping train iteration %d: loss %.20f' % (it, loss))
    return loss_history, coupling, W

def ground_train(P, Y, C, lam, pre_M):
    """
    learn the cost matrix
    :param C: hyper parameter between OT and K
    :param P: coupling matrix dimension: L*L
    :param Y: train label  Y = [y1,y2,y3...ym].T dimension: m*l
    :param lam: the regularization hyper parameter about phi(y)
    :return:
    """
    K_0 = np.dot(Y.T, Y)
    cof = lam*np.linalg.norm(pre_M - K_0)
    K_temp = K_0 + 2*P + 2*cof
    for i in range(Y.shape[1]):
        K_temp[i][i] = K_0[i][i]-np.sum(P[i])-np.sum(P[:, i])+2*P[i][i]+cof*(2-2*Y.shape[1])
    eigvals, eigvectors = np.linalg.eig(K_temp)
    eigvals = np.maximum(eigvals, 0)
    K = np.dot(eigvectors*eigvals.T, eigvectors.T)
    K_diag = np.diag(K)
    K_diag = K_diag.reshape(-1, 1)
    M = K_diag - 2*K + K_diag.T
    return M/C

def compression_train(Y, M, l,compress_number = 2000):
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

    W_1 = weight_variable([L, L])
    bias_1 = bias_variable([L])
    layer_1 = tf.nn.relu(tf.matmul(x, W_1) + bias_1)

    W_2 = weight_variable([L, l])
    bias_2 = bias_variable([l])
    layer_2 = tf.nn.relu(tf.matmul(layer_1, W_2) + bias_2)

    sess = tf.InteractiveSession()

    loss = tf.norm(tf.matmul(tf.transpose(layer_2), layer_2)-M)
    layer = tf.nn.softmax(layer_2)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    sess.run(tf.initialize_all_variables())
    for i in range(compress_number):
        train_step.run(feed_dict={x: Y})
        if i%100 == 0:
            loss_value = loss.eval(feed_dict={x: Y})
#            print("step %d, loss %d", (i, loss_value))
    layer_r = layer.eval(feed_dict={x: Y})
    return layer_r

def alternative_train(X, N, Y, M, l, C, lam, compress_number, sinkhorn_number, reg):
    """
    train the model h(x), phi(y), cost matrix M
    :param N: the total number of train
    :param Y: the label matrix
    :param M: the cost matrix
    :param l: the low dimension
    :param C: the regularization hyper parameter about M
    :param lam: the regularization hyper parameter about phi(y)
    :return:
    """

    for iter in range(N):
        Y_ = compression_train(Y, M, l, compress_number)
        loss_history, coupling, W = mapping_train(X, Y_, M, 1e-7, sinkhorn_number, 500, reg)
        print("the iter last loss: %f",loss_history[-1])
        M = ground_train(coupling, Y_, C, lam, M)
    return Y_, M, W

#TODO using bicluster
def predict(X_test, W, X, Y, Y_, knn_number=10):
    """
    calculate the h(X), then bi-cluster it. justify which cluster is closet to X_test, then choose knn from this cluster
    to average as the final label
    :param X_test: the test feature dimension: n*d
    :param W: the weight of h(x) dimension:L*d
    :param X: the train feature: m*d
    :param Y: the train label:m*L
    :return: Y_test
    """
#    temp_HX = np.exp(np.dot(W, X.T))
#    temp_HX_sum = np.sum(temp_HX, axis=0).reshape(1, -1)
#    H_X = temp_HX/temp_HX_sum  #dimension: L*m
#    H_X = H_X.T #dimension: m*L

    test_HX = np.exp(np.dot(W, X_test.T))
    test_HX_sum = np.sum(test_HX, axis=0).reshape(1,-1)
    pre_H_X = test_HX/test_HX_sum
    pre_H_X = pre_H_X.T #dimension: n*L
    pre_y = []
    for i in range(pre_H_X.shape[0]):
        temp = pre_H_X[i]
        temp_re = (temp-Y_)**2
        candidate = np.sum(temp_re, axis=1)
        dis_index = np.argsort(candidate)
        pre_label = 0
        for i in range(knn_number):
            pre_label = pre_label + Y[dis_index[i]]
        pre_label /= knn_number
        pre_y.append(pre_label)
    pre_y = np.array(pre_y)
    return pre_y

def read_data():
    ft = scipy.io.loadmat('ft.mat')
    lbl = scipy.io.loadmat('lbl.mat')
    feature_matrix = ft['ft_mat']
    label_matrix = lbl['lbl_mat']
    L = label_matrix.shape[0]
    number = label_matrix.shape[1]
    dimension = feature_matrix.shape[0]

    dataset_x = feature_matrix.toarray()
    dataset_y = label_matrix.toarray()
    return dataset_x.T, dataset_y.T, L, number, dimension


"""
hyper parameter:
N : the number of alternative train
C : the regularization hyper parameter about M
lam: the regularization hyper parameter about phi(y)
l : the compress dimension of label
knn_number : the number of knn for prediction
compress_number: the number of deep learning for compression
reg: the regularization of entropy Sinkhorn
sinkhorn_number: the number of sinkhorn for iteration
scale_m : the scale of cost matrix
"""

"""
if __name__ == '__main__':
    N = 3
    C = 4
    lam = 1
    l = 70
    reg = 0.02
    knn_number = 100
    compress_number = 10000
    sinkhorn_number = 300

    scale_m = 1.5
    M = np.eye(l)*scale_m

    feature_matrix, label_matrix, L, number, dimension = read_data()

    train_X = feature_matrix[0:10000]
    train_Y = label_matrix[0:10000]
    test_X = feature_matrix[10001:11000]
    test_Y = label_matrix[10001:11000]

    Y_, M, W = alternative_train(train_X, N, train_Y, M, l, C, lam, compress_number, sinkhorn_number, reg)
    print("begin predict")
    pre_Y = predict(test_X, W, train_X, train_Y, Y_, knn_number)

    # save the mat file to matlab

    train_Y = sparse.csc_matrix(train_Y.T)
    test_Y = sparse.csc_matrix(test_Y.T)
    pre_Y = sparse.csc_matrix(pre_Y.T)

    scipy.io.savemat('train_Y.mat', {'train_Y': train_Y})
    scipy.io.savemat('pre_Y.mat', {'pre_Y': pre_Y})
    scipy.io.savemat('test_Y.mat', {'test_Y': test_Y})


"""


if __name__ == '__main__':
    l_set = [10, 30, 50, 70, 90]
    C_set = [0.01, 0.1, 0.5, 1, 5, 10]
    lam_set = [0.01, 0.1, 0.5, 1, 5, 10]
    scale_m_set = [0.01, 0.1, 0.5, 1, 5, 10]

    feature_matrix, label_matrix, L, number, dimension = read_data()
    error_list = []

    num_training = 5000
    mask = list(range(num_training))
    train_X = feature_matrix[mask]
    train_Y = label_matrix[mask]

    num_folds = 5

    x_train_folds = np.array_split(train_X, num_folds)
    y_train_folds = np.array_split(train_Y, num_folds)

    reg = 0.05
    scale_m = 1.5
    N = 2
    l = 70
    M = np.eye(l) * scale_m
    knn_number = 100
    compress_number = 5000
    sinkhorn_number = 50
    file1 = open('result.txt', 'w')
    for C in C_set:
        for lam in lam_set:
            error = 0
            for fold in range(num_folds):
                print('C = %f, lam = %f, fold = %f' % (C, lam, fold))
                temp_X = x_train_folds[:]
                temp_y = y_train_folds[:]
                x_validate_fold = temp_X.pop(fold)
                y_validate_fold = temp_y.pop(fold)
                temp_X = np.array([y for x in temp_X for y in x])
                temp_y = np.array([y for x in temp_y for y in x])
                Y_, M, W = alternative_train(temp_X, N, temp_y, M, l, C, lam, compress_number, sinkhorn_number, reg)
                print('predict')
                pre_Y = predict(x_validate_fold, W, temp_X, temp_y, Y_, knn_number)
                error += np.linalg.norm(pre_Y - y_validate_fold)
            error = error/num_folds
            print('C = %f, lam = %f, error = %f' % (C, lam, error))
            file1.write('C = %f, lam = %f, error = %f' % (C, lam, error))



# test
"""
    N = 3
    X = np.random.rand(100, 10)
    X_test = np.random.rand(3, 10)
    Y = np.random.rand(100, 10)
    Y_sum = np.sum(Y, axis=1).reshape(-1, 1)
    Y = Y/Y_sum
    C = 1
    lam = 1
    l = 5
    M = np.eye(l)
    Y_, M, W = alternative_train(X, N, Y, M, l, C, lam)
    pre_label = predict(X_test, W, X, Y)
    print(pre_label)
"""


"""
N = 3
C = 1
lam = 1
l = 50

precision at 1--5
    0.5829
    0.5704
    0.5092
    0.4397
    0.3739

nDCG at 1--5
    0.5829
    0.5795
    0.5455
    0.5096
    0.4758

propensity weighted precision at 1--5
    0.3455
    0.3980
    0.4059
    0.3916
    0.3693

propensity weighted nDCG at 1--5
    0.3455
    0.3846
    0.3919
    0.3847
    0.3722
***********************************************************

test_Data:
when         N = 3
        C = 1
        lam = 1
    l in range(10,90)
    the 70 is the best
    
    when N = 3
    l = 70
    lam = 1
    C in range(0.5, 4,5)
    the 4 is the best
    
    when N = 3
    l = 70 
    C = 4
    lam in range(0.5, 4.5)
    the 1 is the best
    
    when N = 3
    l = 70
    C = 4
    lam = 1
    scale_m in range(0.1, 2)
    the 1.5 is best


N = 3
C = 4
lam = 1
l = 70
precision at 1--5
    0.6533
    0.6206
    0.5377
    0.4611
    0.4090

nDCG at 1--5
    0.6533
    0.6331
    0.5920
    0.5525
    0.5342

propensity weighted precision at 1--5
    0.3850
    0.4294
    0.4271
    0.4101
    0.4041

propensity weighted nDCG at 1--5
    0.3850
    0.4181
    0.4184
    0.4089
    0.4055
************************************************************

N = 3
C = 4
lam = 1.5
l = 70
knn_number : 100
compress_number: 5000
reg: 0.05
sinkhorn_number: 1000
scale_m : 1.5

precision at 1--5
    0.7538
    0.7060
    0.5176
    0.4347
    0.3970

nDCG at 1--5
    0.7538
    0.7207
    0.5976
    0.5503
    0.5366

propensity weighted precision at 1--5
    0.4402
    0.4860
    0.4089
    0.3847
    0.3913

propensity weighted nDCG at 1--5
    0.4402
    0.4743
    0.4256
    0.4092
    0.4113
    
    
    
C = 0.010000, lam = 0.010000, error = 53.551489
C = 0.010000, lam = 0.100000, error = 55.296172
C = 0.010000, lam = 0.500000, error = 54.456501
"""
