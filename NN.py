import random, sys
from activation_functions import ReLu, Sigmoid


class Neuron():
    
    def __init__(self):
        self.weights = []
        self.biases = []
        self.connected_to = []
        self.value = 0
        self.pending_changes_weights = []
        self.pending_changes_biases = []
        self.function = ReLu
    
    def __get_random_weight__(self):
        random_num = random.randrange(9,12) * 0.2 -1
        result = (2/100)**0.5
        return random_num * result

    def __get_random_bias__(self):
        random_num = random.randrange(4,7) * 0.2 -1
        return random_num 

    def serialize(self, precision=8):
        result = ','.join([str(round(x,precision)) for x in self.weights]) + ":" + ','.join([str(round(x,precision)) for x in self.biases])
        return result
    
    def load(self, configuration):
        pass

    def connect(self, n):
        self.connected_to.append(n)
        self.weights.append(self.__get_random_weight__())
        self.biases.append(self.__get_random_bias__())
        self.pending_changes_biases.append([])
        self.pending_changes_weights.append([])

    def FeedForward(self):
        if len(self.connected_to) <= 0: return self.value
        return self.function.calc(self)


    def Derivative(self,  number_layer, number_neuron, connection_number, Bias=False):
        if len(self.connected_to) <= 0: return self.value
        return self.function.calc_derivative(self,  number_layer, number_neuron, connection_number, Bias=False)


    def get_derivative(self, number_layer, number_neuron, connection_number, Bias=False, exact=0):
        return (self.value - exact) * self.Derivative(number_layer, number_neuron, connection_number, Bias) 

class Neural_Network():

    def __init__(self):
        self.layers = []
    
    def FeedForward(self):
        if len(self.layers) == 0:
            print("No layers found!")
            return

        if len(self.layers) == 1:
            print("Needs more layers!")
            return
        
        result = []

        for output_neuron in self.layers[-1]:
            result.append(output_neuron.FeedForward())
        
        return result

    def add_layer(self, amount_of_neurons=2, activation_function=ReLu):
        self.layers.append([])
        for _ in range(amount_of_neurons):
            new_neuron = Neuron()
            new_neuron.function = activation_function
            if len(self.layers) > 1:
                for previous_neuron in self.layers[-2]:
                    new_neuron.connect(previous_neuron)
            self.layers[-1].append(new_neuron)

    def set_inputs(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].value = inputs[i]

    def save(self, filepath = 'model.txt', precision=4):
        serialized_data = []

        for layer_number in range(1,len(self.layers)):
            layer_data = []
            for neuron in self.layers[-layer_number]:
                layer_data.append(neuron.serialize(precision))
            serialized_data.append(';'.join(layer_data))

        with open(filepath,'w') as file_:
            file_.write("&".join(serialized_data))

    def load(self, filepath = "model.txt"):
        with open(filepath, 'r') as file_:

            layers = "".join(file_.readlines()).replace("\n","").split("&")
            for line in range(len(layers)):
                neurons = layers[line].split(";")
                for neuron in range(len(neurons)):
                    weights = neurons[neuron].split(":")[0].split(",")
                    biases = neurons[neuron].split(":")[1].split(",")
                    for parameter in range(len(weights)):
                        self.layers[-(line + 1)][neuron].weights[parameter] = float(weights[parameter])
                        self.layers[-(line + 1)][neuron].biases[parameter] = float(biases[parameter])
        print("\nLoaded Model!")

    def train(self, x_list, y_list, iterations = 1000, learning_rate=0.0001, show_progress=False):
        for i in range(iterations):
            
            error = []

            if show_progress: 
                sys.stdout.write(f"\r{i/iterations * 100:.2f}%")
                sys.stdout.flush()

            for x,y in zip(x_list, y_list):
                
                for features_ in range(len(x)):
                    self.layers[0][features_].value = x[features_]

                outputs = self.FeedForward()

                error.append(y_list[0][0] - outputs[0])                

                for outputs in range(len(self.layers[-1])):
                    for connections_ in range(len(self.layers[-1][outputs].connected_to)):
                        self.layers[-1][outputs].pending_changes_weights[connections_].append(learning_rate * self.layers[-1][outputs].get_derivative(1, outputs, connections_, Bias=False, exact=y[outputs]))
                        self.layers[-1][outputs].pending_changes_biases[connections_].append(learning_rate * self.layers[-1][outputs].get_derivative(1, outputs, connections_, Bias=True ,exact=y[outputs]))

                    for layer in range(2,len(self.layers)):
                        for neuron in range(len(self.layers[-1 * layer])):
                            for connections in range(len(self.layers[-1 * layer][neuron].connected_to)):
                                self.layers[-layer][neuron].pending_changes_weights[connections].append(learning_rate * self.layers[-1][outputs].get_derivative(layer, neuron, connections, Bias=False, exact=y[outputs]))
                                self.layers[-layer][neuron].pending_changes_biases[connections].append(learning_rate * self.layers[-1][outputs].get_derivative(layer, neuron, connections, Bias=True, exact=y[outputs]))

            self.evaluate_changes()

        print("\nTraining Done!")

    def evaluate_changes(self):
        for layer in self.layers:
            for neuron in layer:
                for weight_number in range(len(neuron.pending_changes_weights)):
                    avg = sum(neuron.pending_changes_weights[weight_number])
                    neuron.weights[weight_number] -= avg
                    neuron.pending_changes_weights[weight_number].clear()

                    avg = sum(neuron.pending_changes_biases[weight_number])
                    neuron.biases[weight_number] -= avg
                    neuron.pending_changes_biases[weight_number].clear()
                    




def example():

    network = Neural_Network()

    network.add_layer(2)
    network.add_layer(2)
    network.add_layer(1, activation_function=Sigmoid)

    network.train( 
                    [[1,2]],
                    [[0.5]],
                    iterations=10000,
                    learning_rate=0.01,
                    show_progress=True
                 )

    #network.save('test.txt', precision=8)

    network.set_inputs([1,2])
    print(network.FeedForward())

