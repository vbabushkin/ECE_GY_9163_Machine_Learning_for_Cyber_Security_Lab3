import keras
import sys
import h5py
import numpy as np

data_filename = str(sys.argv[1])
b_model_filename = str(sys.argv[2])
bprime_model_filename = str(sys.argv[3])


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))

    return x_data, y_data


def main():
    x, y = data_loader(data_filename)
    B = keras.models.load_model(b_model_filename)
    B_prime = keras.models.load_model(bprime_model_filename)

    yhat = np.argmax(B(x), axis=1)
    yhat_prime = np.argmax(B_prime(x), axis=1)

    badnet_accuracy = np.mean(np.equal(yhat, y)) * 100
    rep_accuracy = np.mean(np.equal(yhat_prime, y)) * 100
    print(
        "Badnet classification accuracy: {0:>7.6f}\nGoodnet classification accuracy: {1:>5.6f}\n".format(badnet_accuracy,
                                                                                                       rep_accuracy))
    res = np.array([yhat[i] if yhat[i] == yhat_prime[i] else 1283 for i in range(y.shape[0])])
    for i in range(res.shape[0]):
        print(
            "Badnet predicted label: {0:>15d}\nRepaired Network predicted label: {1:>5d}\nGoodnet G predicted label: {2:>12d}".format(
                yhat[i], yhat_prime[i], res[i]))
    return res


if __name__ == '__main__':
    main()
