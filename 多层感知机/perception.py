from ctypes import resize
import torch
from torch import nn
from d2l import torch as d2l


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)


num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))


def main():
    # batch_size = 256
    # train_iter, test_iter = d2l.load_data_fashion_mnist(
    #     batch_size=batch_size, resize=None)

    # params = [W1, b1, W2, b2]

    # loss = nn.CrossEntropyLoss()

    # num_epochs, lr = 10, 0.1
    # updater = torch.optim.SGD(params, lr=lr)
    # d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_iter, test_iter = d2l.load_data_fashion_mnist(
        batch_size=batch_size, resize=None)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()


if __name__ == '__main__':
    main()
