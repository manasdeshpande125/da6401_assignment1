from tensorflow.keras.datasets import fashion_mnist,mnist
import numpy as np
from  matplotlib import pyplot as plt
import time
import math
from sklearn.model_selection import train_test_split
import wandb
import argparse

def load(args):
    my=args.dataset
    if my=="mnist":
        dataset= mnist.load_data()
    else:
        dataset= fashion_mnist.load_data()
    (X_train_and_validation, y_train_and_validation), (X_test, y_test) = dataset
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_and_validation, y_train_and_validation, test_size=0.1, random_state=42)
    X_train = (X_train/255.0).astype(np.float32)
    X_validation = (X_validation/255.0).astype(np.float32)
    X_test = (X_test/255.0).astype(np.float32)
    X_train = np.array(X_train.reshape(X_train.shape[0], 784,1))
    X_test = np.array(X_test.reshape(X_test.shape[0], 784,1))
    X_validation = np.array(X_validation.reshape(X_validation.shape[0], 784,1))
    train(X_train, y_train, X_validation, y_validation,num_inputs_nodes=784,num_hidden_layers=args.num_layers,
                          hidden_layer_size = args.hidden_size,out_num = 10,
                          batch_size = args.batch_size, lr_rate = args.learning_rate,
                          init_type = args.weight_init, op_name = args.optimizer,
                          act_type=args.activation,weight_decay = args.weight_decay,
                          l_type = args.loss, epochs = args.epochs)


def layer_init(arr,n1,n2,init_type):
    np.random.seed(10)
    if init_type=="random":
        arr.append(np.random.randn(n1,n2)*0.1)
    elif init_type=="xavier":
        arr.append(np.random.randn(n1,n2)*np.sqrt(2/(n1+n2)))
    return arr

def param(num_input_nodes, num_hidden_layers, hidden_layer_size, out_num, init_type):
    W = []
    B = []

    layers = [num_input_nodes]  # Input layer
    layers.extend([hidden_layer_size] * num_hidden_layers)  # Dynamic hidden layers
    layers.append(out_num)  # Output layer

    for i in range(len(layers) - 1):
        W = layer_init(W, layers[i + 1], layers[i], init_type)
        B = layer_init(B, layers[i + 1], 1, init_type)

    return W, B

#Activation function
def activation(activation_function):
    if activation_function == 'sigmoid':
        return sigmoid
    if activation_function == 'tanh':
        return tanh
    if activation_function == 'ReLU':
        return relu
    if activation_function == 'identity':
        return identity


def sigmoid(x, derivative = False):
    if derivative:
        return sigmoid(x)*(1-sigmoid(x))
    return 1/(1 + np.exp(-x))

def tanh(x, derivative = False):
    if derivative:
        return 1 - tanh(x)**2
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu(x, derivative = False):
    if derivative:
        return (x>0)*1
    return x*(x>0)

def softmax(x,derivative = False):
    if derivative:
        return softmax(x)*(1- softmax(x))
    return np.exp(x)/np.sum(np.exp(x), axis = 0)

def identity(x,derivative=False):
  if derivative:
        return np.ones_like(x)
  return x

def softmax1(x,derivative = False):
    if derivative:
        return softmax1(x)*(1- softmax1(x))
    x = np.array(x)
    x -= np.max(x, axis=0, keepdims=True)  # Normalize values to avoid large exponentials
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x, axis=0, keepdims=True) + 1e-10)  # Prevent divide by zero

def one_hot(y, num_output_nodes):
    v = np.zeros((num_output_nodes, len(y)))
    for i,j in enumerate(y):
        v[j,i] = 1
    return v


def forward(x, W, B, activation_type):
    h = []
    a = []
    sigma = activation(activation_type)  #activation
    h.append(x)   #h0 = x
    a.append(np.dot(W[0], h[0]) + B[0])
    for i in range(len(W)-1):
        h.append(sigma(a[-1]))
        a.append(np.dot(W[i+1], h[-1]) + B[i+1])
    y_hat = softmax1(a[-1])

    return y_hat, h, a

def loss(y,y_hat,l_type,W,reg,n_class):
    if l_type=='cross_entropy':
        #err=-1*np.sum(np.multiply(one_hot(y,n_class),np.log(y_hat)))/one_hot(y,n_class).shape[1]
        err = -1 * np.sum(np.multiply(one_hot(y, n_class), np.log(np.clip(y_hat, 1e-10, 1.0)))) / one_hot(y, n_class).shape[1]
    elif l_type=='squared_error':
        err=np.sum((one_hot(y,n_class)-y_hat)**2)/(2*one_hot(y,n_class)).shape[1]

    if W:
        r=0
        for i in range(len(W)):
            r+=np.sum((np.array(W,dtype=object)**2)[i])
        err=err+reg*r
    return err

def eval_acc(y_hat, y_true):
    return np.mean(np.argmax(y_hat, axis = 0) ==y_true )*100


def back_prop(x, y, y_hat, a, h , W, B, batch_size,l_type,act_type):
    grad_h,grad_a,grad_W,grad_B = [0]*len(h),[0]*len(a),[0]*len(W),[0]*len(B)
    sigma = activation(act_type)

    if l_type == "cross_entropy":
        grad_h[-1] = -1 * (y / (y_hat + 1e-10))
        grad_a[-1] = -1*(y-y_hat)
    if l_type == "squared_error":
        grad_h[-1] = -2 * (y - y_hat)  # Correct derivative of squared error loss
        grad_a[-1] = grad_h[-1] * (y_hat * (1 - y_hat))  # Chain rule for activation function

    for i in range(len(W)-1, -1, -1):
        grad_W[i] = np.dot(grad_a[i], h[i].T)
        #grad_B[i] = np.dot(grad_a[i], np.ones((batch_size,1)))
        grad_B[i] = np.sum(grad_a[i], axis=1, keepdims=True)
        if i > 0:
            grad_h[i-1] = np.dot(W[i].T, grad_a[i])
            grad_a[i-1]  = np.multiply(grad_h[i-1],sigma(a[i-1], derivative = True))

    return grad_W, grad_B, grad_h, grad_a

def sgd_step(W, B, grad_W, grad_B, lr, reg):
    # Convert lists to numpy arrays
    W = [w - lr * reg * w - lr * gw for w, gw in zip(W, grad_W)]
    B = [b - lr * reg * b - lr * gb for b, gb in zip(B, grad_B)]

    return W, B


def momentum_step(w, b, gW, gB, lr, moment, reg):
    params = {'w': w, 'b': b}

    # Initialize momentum buffers for weights and biases
    Wmoments = [np.zeros_like(wi) for wi in params['w']]
    Bmoments = [np.zeros_like(bi) for bi in params['b']]

    # Update momentum buffers
    Wmoments = [moment * wm + lr * gw for wm, gw in zip(Wmoments, gW)]
    Bmoments = [moment * bm + lr * gb for bm, gb in zip(Bmoments, gB)]

    # Update weights and biases
    W = [(1 - lr * reg) * wi - wm for wi, wm in zip(params['w'], Wmoments)]
    B = [(1 - lr * reg) * bi - bm for bi, bm in zip(params['b'], Bmoments)]

    return W, B


def RMSprop_step(w, b, gW, gB, lr, beta, epsilon=1e-7,reg=0):
    params = {'w': w, 'b': b}

    # Initialize moving average of squared gradients
    vW = [np.zeros_like(wi) for wi in params['w']]
    vB = [np.zeros_like(bi) for bi in params['b']]

    # Update moving averages of squared gradients
    vW = [beta * vw + (1 - beta) * (gw ** 2) for vw, gw in zip(vW, gW)]
    vB = [beta * vb + (1 - beta) * (gb ** 2) for vb, gb in zip(vB, gB)]

    # Update parameters
    W = [wi - (lr / np.sqrt(vw + epsilon)) * gw for wi, vw, gw in zip(params['w'], vW, gW)]
    B = [bi - (lr / np.sqrt(vb + epsilon)) * gb for bi, vb, gb in zip(params['b'], vB, gB)]

    return W, B

def nesterov_sgd_step(w, b, gW, gB, lr, moment, reg=0):
    params = {'w': w, 'b': b}

    # Initialize momentum terms
    Wmoments = [np.zeros_like(wi) for wi in params['w']]
    Bmoments = [np.zeros_like(bi) for bi in params['b']]

    # Lookahead step: Apply momentum before computing gradients
    lookahead_W = [wi - moment * wm for wi, wm in zip(params['w'], Wmoments)]
    lookahead_B = [bi - moment * bm for bi, bm in zip(params['b'], Bmoments)]

    # Compute updated momentum
    Wmoments = [moment * wm + lr * gw for wm, gw in zip(Wmoments, gW)]
    Bmoments = [moment * bm + lr * gb for bm, gb in zip(Bmoments, gB)]

    # Apply weight decay regularization and update parameters
    W = [(1 - lr * reg) * wi - wm for wi, wm in zip(lookahead_W, Wmoments)]
    B = [(1 - lr * reg) * bi - bm for bi, bm in zip(lookahead_B, Bmoments)]

    return W, B


def adam_sgd_step(w, b, gW, gB, lr, beta1, beta2, epsilon, reg=0, t=1):
    params = {'w': w, 'b': b}

    # Initialize moment estimates as lists of zero arrays for each layer
    Wm = [np.zeros_like(wi) for wi in params['w']]
    Wv = [np.zeros_like(wi) for wi in params['w']]
    Bm = [np.zeros_like(bi) for bi in params['b']]
    Bv = [np.zeros_like(bi) for bi in params['b']]

    # Convert gradients to NumPy arrays
    gW = [np.array(gi) for gi in gW]
    gB = [np.array(gi) for gi in gB]

    # Update biased first moment estimate
    Wm = [beta1 * wm + (1 - beta1) * gw for wm, gw in zip(Wm, gW)]
    Bm = [beta1 * bm + (1 - beta1) * gb for bm, gb in zip(Bm, gB)]

    # Update biased second raw moment estimate
    Wv = [beta2 * wv + (1 - beta2) * (gw ** 2) for wv, gw in zip(Wv, gW)]
    Bv = [beta2 * bv + (1 - beta2) * (gb ** 2) for bv, gb in zip(Bv, gB)]

    # Compute bias-corrected moment estimates
    Wm_hat = [wm / (1 - beta1 ** t) for wm in Wm]
    Wv_hat = [wv / (1 - beta2 ** t) for wv in Wv]
    Bm_hat = [bm / (1 - beta1 ** t) for bm in Bm]
    Bv_hat = [bv / (1 - beta2 ** t) for bv in Bv]

    # Update parameters
    W = [(1 - lr * reg) * wi - lr * (wm_h / (np.sqrt(wv_h) + epsilon)) for wi, wm_h, wv_h in zip(params['w'], Wm_hat, Wv_hat)]
    B = [(1 - lr * reg) * bi - lr * (bm_h / (np.sqrt(bv_h) + epsilon)) for bi, bm_h, bv_h in zip(params['b'], Bm_hat, Bv_hat)]

    return W, B


def nadam_sgd_step(w, b, gW, gB, lr, beta1, beta2, epsilon, reg=0, t=1):
    params = {'w': w, 'b': b}

    # Initialize moment estimates properly
    Wm = [np.zeros_like(wi) for wi in params['w']]
    Wv = [np.zeros_like(wi) for wi in params['w']]
    Bm = [np.zeros_like(bi) for bi in params['b']]
    Bv = [np.zeros_like(bi) for bi in params['b']]

    # Convert gradients to NumPy arrays
    gW = [np.array(gi) for gi in gW]
    gB = [np.array(gi) for gi in gB]

    # Compute lookahead momentum term for Nesterov-like update
    Wm = [beta1 * wm + (1 - beta1) * gw for wm, gw in zip(Wm, gW)]
    Bm = [beta1 * bm + (1 - beta1) * gb for bm, gb in zip(Bm, gB)]

    Wm_nesterov = [beta1 * wm + (1 - beta1) * gw for wm, gw in zip(Wm, gW)]
    Bm_nesterov = [beta1 * bm + (1 - beta1) * gb for bm, gb in zip(Bm, gB)]

    # Update biased second raw moment estimate
    Wv = [beta2 * wv + (1 - beta2) * (gw ** 2) for wv, gw in zip(Wv, gW)]
    Bv = [beta2 * bv + (1 - beta2) * (gb ** 2) for bv, gb in zip(Bv, gB)]

    # Compute bias-corrected moment estimates
    Wm_hat = [wm_n / (1 - beta1 ** t) for wm_n in Wm_nesterov]
    Wv_hat = [wv / (1 - beta2 ** t) for wv in Wv]
    Bm_hat = [bm_n / (1 - beta1 ** t) for bm_n in Bm_nesterov]
    Bv_hat = [bv / (1 - beta2 ** t) for bv in Bv]

    # Update parameters
    W = [(1 - lr * reg) * wi - lr * (wm_h / (np.sqrt(wv_h) + epsilon)) for wi, wm_h, wv_h in zip(params['w'], Wm_hat, Wv_hat)]
    B = [(1 - lr * reg) * bi - lr * (bm_h / (np.sqrt(bv_h) + epsilon)) for bi, bm_h, bv_h in zip(params['b'], Bm_hat, Bv_hat)]

    return W, B


def train(X_train, y_train, x_val, y_val, num_inputs_nodes, num_hidden_layers, hidden_layer_size, out_num, init_type, epochs,
          batch_size, l_type, act_type, op_name, lr_rate, m=0.5,weight_decay=0, beta=0.5, beta1=0.5, beta2=0.5, epsilon=0.000001):

    # Initialize weights and biases dynamically
    W, B = param(num_inputs_nodes, num_hidden_layers, hidden_layer_size, out_num, init_type)

    N = X_train.shape[0]
    n_batches = int(np.floor(N / batch_size))

    for epoch in range(epochs):
        train_loss = []
        train_accuracy = []
        val_loss = []
        val_accuracy = []
        l, acc, temp, ds, steps = 0, 0, 0, 0, 1

        while ds < N:
            mini_batch_size = min((N - ds), batch_size)
            x = np.squeeze(X_train[ds:ds + mini_batch_size]).T
            y = one_hot(y_train[ds:ds + mini_batch_size], out_num)
            y_hat, h, a = forward(x, W, B, act_type)
            grad_W, grad_B, grad_h, grad_a = back_prop(x, y, y_hat, a, h, W, B, batch_size, l_type, act_type)

            # Choose optimizer dynamically
            if op_name == 'sgd':
                W, B = sgd_step(W, B, grad_W, grad_B, lr_rate, weight_decay)
            elif op_name == 'momentum':
                W, B = momentum_step(W, B, grad_W, grad_B, lr_rate, m, weight_decay)
            elif op_name == 'rmsprop':
                W, B = RMSprop_step(W, B, grad_W, grad_B, lr_rate, beta=beta, reg=weight_decay)
            elif op_name == "nesterov":
                W, B = nesterov_sgd_step(W, B, grad_W, grad_B, lr_rate, m, weight_decay)
            elif op_name == "adam":
                W, B = adam_sgd_step(W, B, grad_W, grad_B, lr=lr_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, reg=weight_decay,t=steps)
            elif op_name == "nadam":
                W, B = nadam_sgd_step(W, B, grad_W, grad_B, lr=lr_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, reg=weight_decay,t=steps)

            l += loss(y_train[ds:ds + mini_batch_size], y_hat, l_type, W, weight_decay, out_num)
            acc += eval_acc(y_hat, y_train[ds:ds + mini_batch_size])
            steps += 1
            ds += batch_size

        l /= (n_batches + mini_batch_size)
        acc /= steps

        train_loss.append(l)
        train_accuracy.append(acc)

        y_val_hat, _, _ = forward(np.squeeze(x_val).T, W, B, act_type)
        val_acc = eval_acc(y_val_hat, y_val)
        val_l = loss(y_val, y_val_hat, l_type, W=None, reg=weight_decay, n_class=out_num)
        val_accuracy.append(val_acc)
        val_loss.append(val_l)

        wandb.log({"epoch": epoch, "Train_loss": l, "Train_acc": acc, "val_loss": val_l, "val_Accuracy": val_acc})

    return W, B, train_loss, train_accuracy, val_loss, val_accuracy






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", "-we",help = "Wandb Entity used to track experiments in the Weights & Biases dashboard.", default="cs24m024")
    parser.add_argument("--wandb_project", "-wp",help="Project name used to track experiments in Weights & Biases dashboard", default="Trial")
    parser.add_argument("--dataset", "-d", help = "dataset", choices=["mnist","fashion_mnist"])
    parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network", type= int, default=10)
    parser.add_argument("--batch_size","-b",help="Batch size used to train neural network", type =int, default=32)
    parser.add_argument("--optimizer","-o",help="batch size is used to train neural network", default= "adam", choices=["sgd","momentum","nag","rmsprop","adam","nadam"])
    parser.add_argument("--loss","-l", default= "cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("--learning_rate","-lr", default=0.1, type=float)
    parser.add_argument("--momentum","-m", default=0.5,type=float)
    parser.add_argument("--beta","-beta", default=0.5, type=float)
    parser.add_argument("--beta1","-beta1", default=0.5,type=float)
    parser.add_argument("--beta2","-beta2", default=0.5,type=float)
    parser.add_argument("--epsilon","-eps",type=float, default = 0.000001)
    parser.add_argument("--weight_decay","-w_d", default=0.005,type=float)
    parser.add_argument("-w","--weight_init", default="xavier",choices=["random","xavier"])
    parser.add_argument("--num_layers","-nhl",type=int, default=4)
    parser.add_argument("--hidden_size","-sz",type=int, default=64)
    parser.add_argument("-a","--activation",choices=["identity","sigmoid","tanh","relu"], default="ReLU")

    args = parser.parse_args()
    print(args.epochs)
    wandb.login()
    wandb.init(project=args.wandb_project,entity=args.wandb_entity)
    load(args)
    wandb.finish()



