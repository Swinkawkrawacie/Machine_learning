import numpy as np
from functools import cached_property
from keras.datasets import mnist
from keras.utils import to_categorical

class DenseLayer:
    def __init__(self,
                 n_in: int,
                 n_neu: int,
                 activation_function: str):
        self.activ_name = activation_function
        self.n_in = n_in
        # self.activ_func = self.get_activ_func(activation_function)
        # self.d_activ_func = self.get_d_activ_func(activation_function)
        # if not activation_function=='softmax':
        #     self.activ_func = np.vectorize(self.activ_func)
        #     self.d_activ_func = np.vectorize(self.d_activ_func)
        self.weights = 2*np.random.random((n_neu, n_in+1))-1

    @cached_property
    def activ_func(self):
        activ_func_dict = {'sigmoid': lambda x: np.exp(x) / (1 + np.exp(x)),
                           'softplus': lambda x: np.log(np.exp(x) + 1),
                           'exponential': lambda x: np.exp(x),
                           'relu': lambda x: max((x,0)),
                           'softmax': lambda x: np.asarray([np.exp(i) / sum(np.exp(x)) for i in x])}
        func = activ_func_dict.get(self.activ_name)
        if not self.activ_name=='softmax':
            func = np.vectorize(func)
        return func
    
    @cached_property
    def d_activ_func(self):
        activ_func_dict = {'sigmoid': lambda x: np.exp(x) / (1 + np.exp(x))**2,
                           'softplus': lambda x: np.exp(x)/(np.exp(x) + 1),
                           'exponential': lambda x: np.exp(x),
                           'relu': lambda x: (x>0)*1}
        func = activ_func_dict.get(self.activ_name)
        if not self.activ_name=='softmax':
            func = np.vectorize(func)
        return func
    
    def forward_prop(self, input):
        self.input = np.vstack((1, input))
        self.out =  self.weights @ self.input
        return self.activ_func(self.out)
    
    def back_prop(self, output, dlda,  learning_rate: float):
        self.delta = dlda if self.activ_name=='softmax' else dlda * self.d_activ_func(output)
        self.result =  self.delta * self.input.reshape(1,-1)
        res = self.weights.T @ self.delta
        self.weights -= learning_rate * self.result
        return res

class NeuralNetwork:
    def __init__(self, loss_function='mse'): 
        self.layers = [] #list of layers
        self.loss_name = loss_function
    
    def add_layer(self,
                  n_in,
                  n_neu: int,
                  activation_function: str):
        self.layers.append(DenseLayer(n_in, n_neu, activation_function))

    @cached_property
    def loss_func(self):
        loss_func_dict = {'mse': lambda x: np.mean((x[0]-x[1])**2), #x[0]-true label
                          'categorical_crossentropy': lambda x: -sum(x[0]*np.log(x[1]))}
        return loss_func_dict.get(self.loss_name)
    
    @cached_property
    def d_loss_func(self):
        loss_func_dict = {'mse': lambda x: 2*(x[1]-x[0])/x[0].size, #x[0]-true label
                          'categorical_crossentropy': lambda x: sum(x[0])*x[1]-x[0]}
        return loss_func_dict.get(self.loss_name)

    def fit(self, train_features: np.array, train_targets: np.array, epochs: int, learning_rate, val=None):
        self.loss = []
        self.acc = []
        train_features = train_features.copy()
        train_targets = train_targets.copy()
        if val:
            self.val_loss = []
            self.val_acc = []
        for i in range(epochs):
            loss=0
            acc = []
            order = np.arange(train_features.shape[0])
            np.random.shuffle(order)
            train_features = train_features[order,:]
            train_targets = train_targets[order,:]
            for j in range(train_features.shape[0]):
                train = train_features[j,:].reshape(train_features[j,:].size,1)
                layer_out = train
                for layer in self.layers:
                    layer_out = layer.forward_prop(layer_out)
                targets = train_targets[j,:].reshape(train_targets[j,:].size,1)
                acc.append(np.argmax(targets)==np.argmax(layer_out))
                loss += self.loss_func((targets,layer_out))

                dlda = self.d_loss_func((targets,layer_out))
                for layer in self.layers[::-1]:
                    dlda = layer.back_prop(layer.out, dlda, learning_rate)[1:]
            self.loss.append(loss/train_features.shape[0])
            self.acc.append(np.mean(acc))
            if val:
                self.predict(val[0], val[1])
                self.val_loss.append(self.test_loss)
                self.val_acc.append(self.test_acc)
            print(f'epoch: {i+1}, loss: {self.loss[-1]}, acc: {self.acc[-1]}')
            if val:
                print(f'val_loss: {self.val_loss[-1]}, val_acc: {self.val_acc[-1]}')
            

    def predict(self, test_features: np.array, test_targets: np.array):
        test_acc = []
        test = test_features.copy()
        self.test_loss = 0
        for j in range(test.shape[0]):
            layer_out = test[j,:].reshape(test[j,:].size, 1)
            for layer in self.layers:
                layer_out = layer.forward_prop(layer_out)
            target = test_targets[j,:].reshape(test_targets[j,:].size,1)
            test_acc.append(np.argmax(target)==np.argmax(layer_out))
            self.test_loss += self.loss_func((target,layer_out))
        self.test_acc = np.mean(test_acc)
        self.test_loss /= test.shape[0]
        return layer_out

if __name__ == "__main__":

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    first_network = NeuralNetwork()
    first_network.add_layer(28*28, 128,'relu')
    # first_network.add_layer(128, 64,'relu')
    # first_network.add_layer(64, 32,'sigmoid')
    first_network.add_layer(128, 10,'softmax')

    #lista liczb neuronów i żeby samo się utworzyło
    #zbiór obserwacji podzielić na wsady (2**... czy po sto), po każdej liczymy gradient ale wagi updatujemy na koniec, rozmiar batcha żeby się dało podać (ostatni batch to to zostało jak zbiór się nie dZieli)
    #funkcja aktywacji czy funkcja kosztu do wyboru
    first_network.fit(train_images, train_labels, epochs=10, learning_rate=.01, val=(test_images, test_labels))
    predicted = first_network.predict(test_images, test_labels)