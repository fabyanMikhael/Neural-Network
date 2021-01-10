class ReLu():
    @staticmethod
    def calc(neuron_):
        
        neuron_.value = 0

        for i in range(len(neuron_.connected_to)):
            neuron_.value += neuron_.connected_to[i].FeedForward() * neuron_.weights[i] + neuron_.biases[i]
            
        if neuron_.value <= 0: neuron_.value = 0

        return neuron_.value

    @staticmethod
    def calc_derivative(neuron_,  number_layer, number_neuron, connection_number, Bias=False):

        if neuron_.value <= 0:
            return 0

        if len(neuron_.connected_to) <= 0: return neuron_.value

        if number_layer == 2:
            return neuron_.weights[number_neuron] * neuron_.connected_to[number_neuron].Derivative(number_layer - 1, number_neuron, connection_number, Bias)   

        if number_layer == 1:
            if not Bias: return neuron_.connected_to[connection_number].value
            return 1


        sum_of_derivatives = 0

        for i in range(len(neuron_.connected_to)):
            sum_of_derivatives += neuron_.weights[i] * neuron_.connected_to[i].Derivative(number_layer - 1, number_neuron, connection_number, Bias) 

        return sum_of_derivatives


eulers_number = 2.7182818284
class Sigmoid():

    @staticmethod
    def calc(neuron_):

        neuron_.value = 0

        for i in range(len(neuron_.connected_to)):
            neuron_.value += neuron_.connected_to[i].FeedForward() * neuron_.weights[i] + neuron_.biases[i]
            
        neuron_.value = Sigmoid.sigmoid_function(neuron_.value)

        return neuron_.value

    @staticmethod
    def sigmoid_function(x):
        return 1/(1 + (eulers_number ** (-x)))

    @staticmethod
    def calc_derivative(neuron_,  number_layer, number_neuron, connection_number, Bias=False):

        if number_layer == 2:
            return (neuron_.value * (1-neuron_.value)) * neuron_.weights[number_neuron] * neuron_.connected_to[number_neuron].Derivative(number_layer - 1, number_neuron, connection_number, Bias)   

        if number_layer == 1:
            if not Bias: return (neuron_.value * (1-neuron_.value)) * neuron_.connected_to[connection_number].value
            return 0.196611933241

        sum_of_derivatives = 0

        for i in range(len(neuron_.connected_to)):
            sum_of_derivatives += neuron_.weights[i] * neuron_.connected_to[i].Derivative(number_layer - 1, number_neuron, connection_number, Bias) 

        return neuron_.value * (1 - neuron_.value) * sum_of_derivatives