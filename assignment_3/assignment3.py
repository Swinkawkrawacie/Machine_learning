import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

class DenseLayer:
    def __init__(self,
                 n_in: int,
                 n_neu: int,
                 activation_function: str):
        self.activ_name = activation_function
        self.n_in = n_in
        self.activ_func = self.get_activ_func(activation_function)
        self.d_activ_func = self.get_d_activ_func(activation_function)
        if not activation_function=='softmax':
            self.activ_func = np.vectorize(self.activ_func)
            self.d_activ_func = np.vectorize(self.d_activ_func)
        self.weights = 2*np.random.random((n_in, n_neu))-1

    def get_activ_func(self, activation_function: str):
        activ_func_dict = {'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
                           'softplus': lambda x: np.log(np.exp(x) + 1),
                           'exponential': lambda x: np.exp(x),
                           'relu': lambda x: max((x,0)),
                           'softmax': lambda x: np.asarray([np.exp(i) / sum(np.exp(x)) for i in x])}
        return activ_func_dict.get(activation_function)
    
    def get_d_activ_func(self, activation_function: str):
        activ_func_dict = {'sigmoid': lambda x: np.exp(-x) / (1 + np.exp(-x))**2,
                           'softplus': lambda x: np.exp(x)/(np.exp(x) + 1),
                           'exponential': lambda x: np.exp(x),
                           'relu': lambda x: 1 if x>=1 else 0}
        return activ_func_dict.get(activation_function)
    
    def forward_prop(self, input):
        # print(self.n_in)
        self.input = input
        self.out = input @ self.weights
        return self.activ_func(self.out)
    
    def back_prop(self, output, dlda,  learning_rate: float):
        self.delta = dlda if self.activ_name=='softmax' else dlda * self.d_activ_func(output)
        self.result = self.input.reshape(-1,1) * self.delta.reshape(1,-1)
        res = self.weights @ self.delta
        self.weights -= learning_rate * self.result
        return res

class NeuralNetwork:
    def __init__(self): 
        self.layers = [] #list of layers
    
    def add_layer(self,
                  n_in,
                  n_neu: int,
                  activation_function: str):
        self.layers.append(DenseLayer(n_in, n_neu, activation_function))

    def fit(self, train_features: np.array, train_targets: np.array, epochs: int, learning_rate):
        self.loss = []
        self.acc = []
        train_features = train_features.copy()
        train_targets = train_targets.copy()
        for i in range(epochs): #shuffle będzie potrzebny
            print(i)
            loss=0
            acc = []
            order = np.arange(train_features.shape[1])
            np.random.shuffle(order)
            train_features = train_features[order,:]
            train_targets = train_targets[order,:]
            for j in range(train_features.shape[1]):
                layer_out = train_features[j,:]
                for layer in self.layers:
                    layer_out = layer.forward_prop(layer_out)

                acc.append(np.array_equal(train_targets.copy()[j,:],layer_out.round()))
                loss -= sum(train_targets.copy()[j,:]*np.log(layer_out))

                dlda = train_targets[j,:].sum()*layer_out - train_targets.copy()[j,:]
                for layer in self.layers[::-1]:
                    dlda = layer.back_prop(layer.out, dlda, learning_rate)
            self.loss.append(loss/train_features.shape[1])
            self.acc.append(np.mean(acc))

    def predict(self, test_features: np.array, test_targets: np.array):
        errors = []
        layer_out = test_features.copy()
        self.test_loss = 0
        for j in range(test_features.shape[1]):
            layer_out = test_features.copy()[j,:]
            for layer in self.layers:
                layer_out = layer.forward_prop(layer_out)
            errors.append(np.array_equal(test_targets[j,:],layer_out.round()))
            self.test_loss -= sum(test_targets.copy()[j,:]*np.log(layer_out))
        self.errors = np.mean(errors)
        self.test_loss /= test_features.shape[1]
        return layer_out

# if __name__ == "__main__":

#     (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#     train_images = train_images.reshape((60000, 28 * 28))
#     train_images = train_images.astype('float32') / 255
#     test_images = test_images.reshape((10000, 28 * 28))
#     test_images = test_images.astype('float32') / 255

#     train_labels = to_categorical(train_labels)
#     test_labels = to_categorical(test_labels)

#     first_network = NeuralNetwork()
#     first_network.add_layer(28*28, 128,'relu')
#     # first_network.add_layer(128, 64,'relu')
#     # first_network.add_layer(64, 32,'sigmoid')
#     first_network.add_layer(128, 10,'softmax')

#     #lista liczb neuronów i żeby samo się utworzyło
#     #zbiór obserwacji podzielić na wsady (2**... czy po sto), po każdej liczymy gradient ale wagi updatujemy na koniec, rozmiar batcha żeby się dało podać (ostatni batch to to zostało jak zbiór się nie dZieli)
#     #funkcja aktywacji czy funkcja kosztu do wyboru
#     first_network.fit(train_images, train_labels, epochs=3, learning_rate=.1)
#     predicted = first_network.predict(test_images, test_labels)