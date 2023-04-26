import numpy as np
from load_mnist import load_mnist

class NeuralNetworkClassifier:

    def __init__(self, layer_dimensions, activations):
        np.random.seed(1)
        self.layers = len(layer_dimensions) - 1 # should be 3
        self.W = []
        self.b = []
        self.activations = activations
        self.model_state = {"linear_outputs": [], "activation_outputs": []}

        sigma = 0.01
        for i in range(1, self.layers + 1):
            weight = np.random.normal(scale=sigma**2, size=(layer_dimensions[i], layer_dimensions[i-1]))
            #bias = np.zeros((layer_dimensions[i], 1))
            bias = np.random.randn(layer_dimensions[i], 1)
            self.W.append(weight)
            self.b.append(bias)
        
        if self.activations[-1] != "softmax":
            print("Last activation layer should be softmax.")
    
    # Activation Functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0.0, x)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0)
    
    # Forward Propagation
    def linear_forward(self, A_prev, W, b):
        """ param A_prev: the activation values from the previous layer or input data
            param W: the weight matrix (size of current layer x previous layer)
            param b: the bias vector: (size of current layer x 1)
            return: the input of the activation function
        """
        return np.dot(W, A_prev) + b

    def activation_forward(self, x, activation):
        """ param x: linear outputs (output of the linear_forward)
            return activation outputs
        """
        if activation == "sigmoid":
            return self.sigmoid(x)
        elif activation == "relu":
            return self.relu(x)
        elif activation == "softmax":
            return self.softmax(x)
    
    def model_forward(self, x):
        """ param x: the input data
        """
        self.model_state["activation_outputs"] = [np.copy(x.T)]
        self.model_state["linear_outputs"] = []

        for l in range(self.layers):
            linear_output = self.linear_forward(self.model_state["activation_outputs"][-1], self.W[l], self.b[l])
            activation_output = self.activation_forward(linear_output, self.activations[l])
            self.model_state["linear_outputs"].append(linear_output)
            self.model_state["activation_outputs"].append(activation_output)
        
        return activation_output
    
    # Loss Function
    def compute_loss(self, y):
        """ param y: expected output
        """
        z = self.model_state["linear_outputs"][-1].T
        if np.max(z) > 0:
            z -= np.max(z)
        loss = np.sum(np.log(np.sum(np.exp(z), axis=0)) - np.sum(z[y==1])) / y.shape[1]
        #loss2 = np.linalg.norm(y - self.model_state["activation_outputs"][-1].T)
        return loss
    
    # Activation Functions Derivatives
    def sigmoid_backward(self, x):
        #return np.exp(-x) / ((1 + np.exp(-x)) ** 2)
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def relu_backward(self, x):
        return [[0 if i < 0 else 1 for i in y] for y in x]
    
    # Backwards Propogation
    def linear_backward(self, W, delta):
        return np.dot(W.T, delta)
    
    def activation_backward(self, x, activation): # f'(Z)
        if activation == "sigmoid":
            return self.sigmoid_backward(x)
        elif activation == "relu":
            return self.relu_backward(x)
        
    def model_backward(self, Y):
        data_points = Y.shape[0]
        deltas = [None] * self.layers
        self.gradients = {"dW": [], "db": []}
        
        activation_output = self.model_state["activation_outputs"][-1]
        deltas[-1] = ((Y.T - activation_output))

        for i in reversed(range(self.layers - 1)):
            deltas[i] = self.linear_backward(self.W[i+1], deltas[i+1]) * self.activation_backward(self.model_state["linear_outputs"][i], self.activations[i])
        
        for i, d in enumerate(deltas):
            self.gradients["dW"].append(np.dot(d, self.model_state["activation_outputs"][i].T) / float(data_points))
            self.gradients["db"].append(np.dot(d, np.ones((data_points,1))) / float(data_points))

    def update_parameters(self, alpha):
        for i in range(self.layers):        
            self.W[i] += alpha * self.gradients["dW"][i]
            self.b[i] += alpha * self.gradients["db"][i]

    def predict(self, x):
        return self.model_forward(x)

    def random_mini_batches(self, x, y, batch_size = 64):
        if batch_size == -1:
            return [x], [y]
        
        x, y = self.shuffle_data(x, y) # to make sure that the batches are different every time
    
        x_mini = []
        y_mini = []
        i = 0
        while(i < y.shape[0]):
            x_mini.append( x[i:i+batch_size, :] )
            y_mini.append( y[i:i+batch_size, :] )
            i += batch_size
            
        return x_mini, y_mini
        
    def shuffle_data(self, a, b):
        shuffled_indices = np.random.permutation(a.shape[0])
        return a[shuffled_indices, :], b[shuffled_indices, :]

    def train_model(self, x_train, y_train, iterations = 1, alpha = 0.01, batch_size = 64):
        losses = []
        
        for i in range(iterations):
            self.model_forward(x_train)
            self.model_backward(y_train)
            self.update_parameters(alpha)
            losses.append(self.compute_loss(y_train))
            print("Loss \t{}: {}".format(i, losses[-1]))
        return losses
    
    def predict_accuracy(self, x, y):
        y_pred = self.predict(x)
        y_pred_argmax = np.argmax(y_pred, axis=0)
        success_rate = y_pred_argmax == np.argmax(y, axis=1)
        success_rate = success_rate.sum() / len(success_rate)
        return success_rate
    


def main():
    X_train, Y_train, X_test, Y_test = load_mnist()
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    #layer_dimensions = [X_train.shape[1], 512, 128, Y_train.shape[1]]
    #activations = ["sigmoid", "relu", "softmax"]
    layer_dimensions = [X_train.shape[1], 300, Y_train.shape[1]]
    activations = ["sigmoid", "softmax"]

    model = NeuralNetworkClassifier(layer_dimensions, activations)

    model.train_model(X_train, Y_train, 50, 0.5)

    print(model.predict_accuracy(X_test, Y_test))


if __name__ == "__main__":
    main()
