#Fundamental unit of a Neural network
class Neuron():
    def __init__(self, w: list, b, learning_rate = 0.01):
        self.w = w
        self.b = b
        self.learning_rate = learning_rate

    #Makes prediction
    def forward(self, x: list): 
        dot = 0

        #Checks for dimensions of the input and weights
        if len(self.w) != len(x):
            raise ValueError("Dimensions of weights and inputs are not equal")

        #No need of else and we evaluate the dot product of weights and inputs
        for i in range(len(x)):
            dot += self.w[i] * x[i]
        
        self.input = x
        self.output =  dot + self.b

        return self.output

    #Returns a Gradient Vector
    def backward(self, gradient):
        #dL/dW Weight Gradient
        dLw = []
        for data in self.input:
            dLw.append(gradient * data)
        
        #dL/db Bias Gradient
        dLb = gradient

        temp_w = []
        for i in range(len(self.w)):
            temp_w.append(self.w[i] - (self.learning_rate * dLw[i]))

        dLdx = []
        for weigh in self.w:
            dLdx.append(weigh * gradient)
        
        self.w = temp_w        
        self.b = self.b - (self.learning_rate * dLb)

        return dLdx
        

#Layer consists neurons
class Layer():
    def __init__(self, neurons: list):
        self.neurons = neurons

    #Sends the predictions to the next neuron
    def forward(self, x: list):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(x))
        
        return outputs

    #Returns a list of sum of all the Gradient Vectors
    def backward(self, gradients):
        #Stores the gradients of the layer given by the neurons
        dLdx = []

        #Taking al the gradients from the neuron
        for neuron, gradient in zip(self.neurons, gradients):
            dLdx.append(neuron.backward(gradient))

        #Returns the vector sum of the current layer's neurons
        gradient = [0] * len(dLdx[0])

        for j in range(len(dLdx[0])):
            for i in range(len(dLdx)): #gardient[0] = dLdx[0][0] + dLdx[1][0] + ...
                gradient[j] += dLdx[i][j]

        return gradient

  
#NeuralNetwork consists layers
class NeuralNetwork():
    def __init__(self, layers: list):
        self.layers = layers

    #Sends the output from first layer to the next layer
    def forward(self, x: list):
        output = x
        for i in range(len(self.layers)):
            output = self.layers[i].forward(output)
        
        return output

    #Sends the Gradient from last layer to the previous layer
    def backward(self, gradients):
        gradient = gradients
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    #Calculates the loss of the whole Neural Network
    def calculate_loss(self, actual, prediction):
        loss = 0

        for actual_value, predicted_value in zip(actual, prediction):
            loss += (actual_value - predicted_value) ** 2

        return loss
    

neuron1 = Neuron([0.5, -0.2], 0)
neuron2 = Neuron([0.1, 0.8], 0)
layer = Layer([neuron1, neuron2])
network = NeuralNetwork([layer])

#Training the network
def train(dataset):
    for epoch in range(10000):
        epoch_loss = 0
        for stuff in dataset:
            x, y = stuff

            prediction = network.forward(x)
            loss = network.calculate_loss(y, prediction)
            epoch_loss += loss

            gradients = []
            for actual_value, predicted_value in zip(y, prediction):
                gradients.append(2 * (predicted_value - actual_value))

            network.backward(gradients)
    print(f"Epoch Loss: {epoch_loss}")


dataset = [
    ([0, 0], [1, 2]),
    ([1, 0], [3, 1]),
    ([0, 1], [2, 5]),
    ([1, 1], [4, 4]),
    ([2, 1], [6, 3]),
    ([1, 2], [5, 7]),
    ([2, 2], [7, 6]),
    ([3, 1], [8, 2]),
]

train(dataset)

for x, y in dataset:
    print(x, network.forward(x), y)