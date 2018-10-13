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
#        gra_h = np.log(u)/reg - np.log(np.sum(u))/(reg*L)
        gra_h = np.log(u)/reg - np.sum(np.log(u))/(reg*L)

        #compute the gra_w, dimension L*d
        temp_yy = h_x * -h_x.reshape(-1, 1)
        temp_y = temp_yy.copy()
        temp_y = temp_y + np.diag(h_x)
        temp_m = np.dot(temp_y, gra_h.T)
        gra = x.reshape(1, -1) * temp_m.reshape(-1, 1)

        gradient = gradient + gra

    return loss, coupling, gradient

def mapping_train(X, Y, M, W, learning_rate = 1e-6, num_iters = 50,batch_size = 4000, reg = 0.01, verbose = False):
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
    coupling = 0
    loss_history = []
    for it in range(num_iters):
        X_batch = None
        Y_batch = None

#        batch_idx = np.random.choice(num_train, batch_size, replace=False)
#        X_batch = X[batch_idx]
#        Y_batch = Y[batch_idx]
        X_batch = X
        Y_batch = Y

        loss, couple, grad = mapping_learning_gradient(M,W,X_batch,Y_batch,reg)
        loss_history.append(loss)
        coupling = couple     #get the newest couple
        W -= learning_rate * grad

        if it % 1 == 0:
            print('mapping train iteration %d: loss %.30f' % (it, loss))
    return loss_history, coupling, W

def ground_train(P, Y, C, lam_1, pre_M):
    """
    learn the cost matrix
    :param C: hyper parameter between OT and K
    :param P: coupling matrix dimension: L*L
    :param Y: train label  Y = [y1,y2,y3...ym].T dimension: m*l
    :param lam: the regularization hyper parameter about phi(y)
    :return:
    """
    K_0 = np.dot(Y.T, Y)
    #grad_f_K = -2*P - diag(diag(-2*P)) + diag(sum(P,2) + sum(P,1)' - 2*diag(P));
    #K = K_0 - 1/C*grad_f_K;

    # the gradient of K- K_0
    grad_f_k = -2*P - np.diag(np.diag(-2*P)) + np.diag(np.sum(P, axis=0) + np.sum(P, axis=1) - 2*np.diag(P))

    # the gradient of dm(phi(Y)) - M
    L_num = pre_M.shape[0]
    grad_f_phi_temp = np.ones((L_num, L_num))*(-2) + 2*L*np.eye(L_num)
    grad_f_phi = (pre_M - compute_squared(Y)) * grad_f_phi_temp

    K_temp = K_0 - (1/C)*grad_f_k - (1/lam_1)*grad_f_phi

    eigvals, eigvectors = np.linalg.eig((K_temp+K_temp.T)/2)
    eigvals = np.maximum(eigvals, 0)
    K = np.dot(eigvectors*eigvals.T, eigvectors.T)
    K_diag = np.diag(K)
    K_diag = K_diag.reshape(-1, 1)
    M = np.abs(K_diag - 2*K + K_diag.T)
    M = M/np.max(M)
    return M

# the network structure is so bad
def compression_train(Y, M, DC, l, lam_2, compress_number = 2000, seed = 1):
    """
    learn the mapping from high dimension to low dimension, Y:m*L -> Y':m*l
    :param Y:the label matrix
    :param M:the compressive label correlation matrix
    :param l:hyper parameter, the low dimension
    :return:low dimension label m*l
    """

    def compute_distance_matrix(X, n):
#        G = tf.matmul(tf.transpose(X), X)
#        diag_value = tf.reshape(tf.diag_part(tf.matmul(tf.transpose(X), X)), [1, -1])
#        H = tf.tile(diag_value, [n, 1])
#        return H + tf.transpose(H) - 2 * G
        return tf.tile(tf.reshape(tf.diag_part(tf.matmul(tf.transpose(X), X)), [1, -1]), [n, 1]) + tf.transpose(tf.tile(tf.reshape(tf.diag_part(tf.matmul(tf.transpose(X), X)), [1, -1]), [n, 1])) - 2 * tf.matmul(tf.transpose(X), X)

    def weight_variable(shape, lambda1):
        initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
        var = tf.Variable(initial)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
        return var

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    L = Y.shape[1]
    num = Y.shape[0]
    x = tf.placeholder("float", [None, L])
    dc = tf.placeholder("float", [num, num])

    W_1 = weight_variable([L, int((L+l)/2)], 0.003)
    bias_1 = bias_variable([int((L+l)/2)])
    layer_1 = tf.nn.relu(tf.matmul(x, W_1) + bias_1)

    W_2 = weight_variable([int((L+l)/2), l], 0.003)
    bias_2 = bias_variable([l])
    layer_2 = tf.matmul(layer_1, W_2) + bias_2

#    layer_2 = tf.add(layer_2, 1e-15)

    layer = tf.nn.softmax(layer_2)
#    layer = layer_2 / tf.reshape(tf.reduce_sum(layer_2, 1), [-1, 1])

    loss_label = tf.norm(compute_distance_matrix(layer, l) - M)
    loss_data = tf.norm(compute_distance_matrix(tf.transpose(layer), num) - dc)

    loss = loss_label * loss_label + lam_2 * loss_data * loss_data
    tf.add_to_collection("losses", loss)
#    losses = tf.add_n(tf.get_collection('losses'))

    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(compress_number):
            sess.run(train_step, feed_dict={x: Y, dc: DC})
            if i % 500 == 0:
                loss_value = sess.run(loss, feed_dict={x: Y, dc: DC})
                print("step %d, loss %.10f" % (i, loss_value))
        layer_r = sess.run(layer, feed_dict={x: Y, dc: DC})
    return layer_r

def alternative_train(X, N, Y, M, DC, l, C, lam_1, lam_2, compress_number, sinkhorn_number, batch_size, learning_rate, reg,seed):
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
    W = np.ones((l, X.shape[1]))*0.5
    for iter in range(N):
        print("the %d th iteration" % (iter))
        Y_ = compression_train(Y, M, DC, l, lam_2, compress_number,seed)
        if iter == 0:
            K = np.dot(Y_.T, Y_)
            K_diag = np.diag(K)
            K_diag = K_diag.reshape(-1, 1)
            M = np.abs(K_diag - 2 * K + K_diag.T)
            M = M/np.max(M)
        loss_history, coupling, W = mapping_train(X, Y_, M, W, learning_rate, sinkhorn_number, batch_size, reg)
#        print("the iter last loss: %f",loss_history[-1])
        M = ground_train(coupling, Y_, C, lam_1, M)
        pre_Y = predict(X, W, X, Y, 80)
        print("the %d th iteraton error: %f" %(iter, np.linalg.norm(pre_Y-Y)))
    return Y_, M, W

#TODO using bicluster
def predict(X_test, W, X, Y, knn_number=10):
    """
    calculate the h(X), then bi-cluster it. justify which cluster is closet to X_test, then choose knn from this cluster
    to average as the final label
    :param X_test: the test feature dimension: n*d
    :param W: the weight of h(x) dimension:L*d
    :param X: the train feature: m*d
    :param Y: the train label:m*L
    :return: Y_test
    """
    temp_HX = np.exp(np.dot(W, X.T))
    temp_HX_sum = np.sum(temp_HX, axis=0).reshape(1, -1)
    H_X = temp_HX/temp_HX_sum  #dimension: L*m
    H_X = H_X.T #dimension: m*L

    test_HX = np.exp(np.dot(W, X_test.T))
    test_HX_sum = np.sum(test_HX, axis=0).reshape(1, -1)
    pre_H_X = test_HX/test_HX_sum
    pre_H_X = pre_H_X.T #dimension: n*L
    pre_y = []
    weight = list(range(2*(knn_number-1), -1, -2))
    weight = [t/(knn_number*(knn_number-1)) for t in weight]
    for i in range(pre_H_X.shape[0]):
        temp = pre_H_X[i]
        temp_re = (temp-H_X)**2
        candidate = np.sum(temp_re, axis=1)
        dis_index = np.argsort(candidate)
        pre_label = 0
        #add weight?

        for j in range(knn_number):
            pre_label = pre_label + weight[j]*Y[dis_index[j]]
#        pre_label /= knn_number
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
def compute_squared(X):
  m,n = X.shape
  G = np.dot(X.T, X)
  H = np.tile(np.diag(G), (n, 1))
  temp_r = H + H.T - 2*G
  return temp_r/np.max(temp_r)

def read_split_file(file_name, N):
    file_data = open(file_name, 'r')
    contents = file_data.readlines()
    result = []
    for ii in range(len(contents)):
        content = contents[ii]
        temp_cont = content.split()
        temp_con = [int(p)-1 for p in temp_cont]
        result.append(temp_con)
    temp_result = np.array(result)
    temp_r = temp_result[:, N]
    file_data.close()
    return temp_r.tolist()

if __name__ == '__main__':

    C_set = [0.05, 0.1, 0.5]
    lam_1_set = [50, 100, 150, 200]
    lam_2_set = [50, 100, 150, 200]
    rate_set = [5e-9, 1e-9, 5e-8, 1e-8, 1e-7, 1e-6, 1e-5]
    knn_set = [5, 10, 20, 50, 70, 80]
    N_set = [3, 5, 10, 20, 50, 100]
    l_set = [5, 10, 20, 30, 50, 70, 90]
    seed_set = [11, 12, 13, 14, 15]
    seed = 1
    N = 3
    C = 0.1
    lam_1 = 100  #In ground learning, the hyper parameter of phi(y)
    lam_2 = 100   #In compression, balance the loss_label and loss_data
    l = 50
    reg = 0.05
    knn_number = 80
    compress_number = 10000
    sinkhorn_number = 1
    num_training = 1000
    num_testing = 500
    batch_size = int(num_training)
    learning_rate = 1e-9

    M = np.ones((l, l)) - np.eye(l)
    M = M / np.max(M)

    feature_matrix, label_matrix, L, number, dimension = read_data()

    mask = read_split_file("mediamill_trSplit.txt", 0)
    mask = mask[0:num_training]
    train_X = feature_matrix[mask]

#    num_testing = 500
    mask_test = read_split_file("mediamill_tstSplit.txt", 0)
    mask_test = mask[0:num_testing]
    train_Y = label_matrix[mask]
# train_Y has term all zero
    train_Y_sum = np.sum(train_Y, axis=1).reshape(-1, 1)
    train_Y_sum = np.maximum(train_Y_sum, 1e-20)
    origin_train_Y = train_Y
    train_Y = train_Y / train_Y_sum

    test_X = feature_matrix[mask_test]
    test_Y = label_matrix[mask_test]

    test_Y_sum = np.sum(test_Y, axis=1).reshape(-1, 1)
    test_Y_sum = np.maximum(test_Y_sum, 1e-20)
    origin_test_Y = test_Y
    test_Y = test_Y / test_Y_sum

    # DC represent the data correlation
    DC = compute_squared(train_Y.T)

    seq = 0
    file = open('result.txt', 'w')
    for knn_number in knn_set:
        Y_, M, W = alternative_train(train_X, N, train_Y, M, DC, l, C, lam_1, lam_2, compress_number, sinkhorn_number, batch_size, learning_rate, reg, seed)
        print("begin predict")
        pre_Y = predict(test_X, W, train_X, origin_train_Y, knn_number)
        print("knn_number = %f, error = %f" % (knn_number, np.linalg.norm(pre_Y-test_Y)))
        file.write("knn_number = %f, error = %f\n" % (knn_number, np.linalg.norm(pre_Y-test_Y)))

        # save the mat file to matlab
        train_Y_temp = sparse.csc_matrix(origin_train_Y.T)
        test_Y_temp = sparse.csc_matrix(origin_test_Y.T)
        pre_Y_temp = sparse.csc_matrix(pre_Y.T)

        train_name = 'train_Y'+str(seq)+'.mat'
        pre_name = 'pre_Y'+str(seq)+'.mat'
        test_name = 'test_Y'+str(seq)+'.mat'

        scipy.io.savemat(train_name, {'train_Y': train_Y_temp})
        scipy.io.savemat(pre_name, {'pre_Y': pre_Y_temp})
        scipy.io.savemat(test_name, {'test_Y': test_Y_temp})
        seq += 1



"""
>> load('pre_Y.mat')
>> load('train_Y.mat')
>> load('test_Y.mat')
>> weights = inv_propensity(train_Y,0.55,1.5);
>> [metrics]=get_all_metrics(pre_Y, test_Y,[weights]);

"""
"""
# the current result
precision at 1--5
    0.7480
    0.6380
    0.5693
    0.4850
    0.4300

nDCG at 1--5
    0.7480
    0.6677
    0.6342
    0.5995
    0.5896

propensity weighted precision at 1--5
    0.3918
    0.4006
    0.4096
    0.4026
    0.4072

propensity weighted nDCG at 1--5
    0.3918
    0.3984
    0.4046
    0.4008
    0.4035
"""