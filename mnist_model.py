"""

"""

import random
import pickle
import numpy as np
from activation import Softmax, ReLU, LeakyReLU
from layer import FullyConnected
from loss import CrossEntropyLoss
from optimizer import SGD
import parameter
import network
from utils import one_hot, testResult2labels
from data.mnist import fetch_testingset, fetch_traingset


def mnist_model():

    model = network.Network()
    model.add(FullyConnected(in_feature=784, out_feature=512), name='fc1')
    model.add(LeakyReLU(), name='leaky_relu1')
    model.add(FullyConnected(in_feature=512, out_feature=256), name='fc2')
    model.add(LeakyReLU(), name='leaky_relu2')
    model.add(FullyConnected(in_feature=256, out_feature=256), name='fc3')
    model.add(LeakyReLU(), name='leaky_relu3')
    model.add(FullyConnected(in_feature=256, out_feature=128), name='fc4')
    model.add(LeakyReLU(), name='leaky_relu4')
    model.add(FullyConnected(in_feature=128, out_feature=10), name='fc5')
    model.add(Softmax(), name='softmax')

    model.add_loss(CrossEntropyLoss())

    optimizer = SGD(lr=1e-4)

    print(model)
    traingset = fetch_traingset()
    train_images, train_labels = traingset['images'], traingset['labels']
    batch_size = 256
    training_size = len(train_images)
    loss_list = np.zeros((50, int(training_size/batch_size)))
    for epoch in range(50):
        for i in range(int(training_size/batch_size)):
            batch_images = np.array(train_images[i*batch_size:(i+1)*batch_size])
            batch_labels = np.array(train_labels[i*batch_size:(i+1)*batch_size])
            batch_labels = one_hot(batch_labels, 10)
            _, loss = model.forward(batch_images, batch_labels)
            if i % 50 == 0:
                loss_list[epoch][i] = loss
                print("e:{}, i:{} loss: {}".format(epoch, i, loss))
            model.backward()
            model.optimize(optimizer)

    filename = 'model.data'
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()

    loss_fname = 'loss.data'
    f = open(loss_fname, 'wb')
    pickle.dump(loss_list, f)
    f.close()

    testset = fetch_testingset()
    test_images, test_labels = testset['images'], testset['labels']
    test_images = np.array(test_images[:])
    test_labels_one_hot = one_hot(test_labels, 10)

    y_, test_loss = model.forward(test_images, test_labels_one_hot)
    test_labels_pred = testResult2labels(y_)
    test_labels = np.array(test_labels)
    right_num = np.sum(test_labels==test_labels_pred)
    accuracy = 1.0 * right_num/test_labels.shape[0]
    print('test accuracy is: ', accuracy)

    a = 0


if __name__ == '__main__':

    mnist_model()







