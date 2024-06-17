import math
import random
import matplotlib.pyplot as plt
import numpy as np

class BaseNetwork:
    def __init__(self, data, network_structure, best_weight_array=None):
        """
        Initializes a neural network with given data, structure, and optional initial weights.

        Parameters:
        - data: Input data for the network.
        - network_structure: List defining the structure of the network (input, hidden layers, output).
        - best_weight_array: Optional list of initial weights for the network.

        Attributes:
        - network: Represents the structure of the neural network.
        - weight_array: Stores the weights of the network.
        """
        self.data = data
        self.network_structure = network_structure
        self.network = []
        self.weight_array = best_weight_array if best_weight_array else []

    @staticmethod
    def summation(array, node_pre_layer):
        """
        Computes the weighted sum of inputs for a node in the network.

        Parameters:
        - array: Array containing inputs and weights.
        - node_pre_layer: Number of nodes in the previous layer.

        Returns:
        - Total sum of inputs weighted by their respective weights.
        """
        total_sum = 0
        for i in range(node_pre_layer):
            total_sum += array[i][0] * array[i][1]
        return total_sum

    @staticmethod
    def reLU(x):
        """
        Rectified Linear Unit (ReLU) activation function.

        Parameters:
        - x: Input value.

        Returns:
        - Output value after applying ReLU activation.
        """
        return max(0, x)

    def init_weights(self):
        """
        Initializes weights for the neural network if not already initialized.
        """
        if not self.weight_array:
            total_weights = sum(self.network_structure[i] * self.network_structure[i + 1]
                                for i in range(len(self.network_structure) - 1))
            self.weight_array = [random.uniform(0, 1) for _ in range(total_weights)]

    def randomize_weights(self):
        """
        Randomly adjusts weights within a small range [-0.5, 0.5].
        """
        self.weight_array = [w + random.uniform(-0.5, 0.5) if 0 <= w + random.uniform(-0.5, 0.5) <= 1 else w
                             for w in self.weight_array]

    def modify_network(self):
        """
        Modifies the network structure by adding layers or nodes based on random probabilities.
        """
        layer_prob = random.uniform(0, 1)
        node_prob = random.uniform(0, 1)

        if layer_prob > 0.9:
            self.network_structure.insert(-1, 1)

        if node_prob > 0.5 and self.weight_array and self.network:
            self.network_structure[-2] += 1
            self.network[-2].append([])
            for _ in range(self.network_structure[-3]):
                self.network[-2][-1].append([0, random.uniform(0, 1)])
            self.network[-2][-1].append(0)

    def init_layers(self):
        """
        Initializes the layers of the neural network with proper connections and weights.
        """
        self.network.append(self.data)
        self.network.append([])

        for i in range(len(self.network_structure) - 1):
            self.network.append([])

        for i in range(1, len(self.network_structure)):
            nodes_layer = self.network_structure[i]
            nodes_pre_layer = self.network_structure[i - 1]

            for _ in range(nodes_layer):
                self.network[i + 1].append([])
                for _ in range(nodes_pre_layer):
                    self.network[i + 1][-1].append([0, self.weight_array.pop(0)])
                self.network[i + 1][-1].append(0)

        for layer in range(2, len(self.network_structure) + 1):
            for node in range(self.network_structure[layer - 1]):
                nodes_pre_layer = self.network_structure[layer - 2]
                for edge in range(nodes_pre_layer):
                    if layer == 2:
                        self.network[2][node][edge][0] = self.network[2][edge]
                    if layer != 2:
                        self.network[layer][node][edge][0] = self.network[layer - 1][edge][-1]
                self.network[layer][node][-1] = self.reLU(self.summation(self.network[layer][node], nodes_pre_layer))

    def return_weights(self):
        """
        Returns the current weights of the neural network.

        Returns:
        - List of current weights.
        """
        return self.weight_array

    def return_difference(self):
        """
        Computes the difference between the network's output and expected output.

        Returns:
        - Difference value.
        """
        x = self.network[-1][0][-1]
        return math.pow(x, 3) - math.pow(self.data[0], 3)

def main(population, generation):
    """
    Main function to evolve and evaluate neural networks over multiple generations.

    Parameters:
    - population: Number of networks to evaluate per generation.
    - generation: Number of generations to evolve the networks.

    Displays:
    - Scatter plot of generation vs. square difference between network output and expected output.
    """
    network_structure = [1, 7, 6, 4, 1]
    best_difference = 1
    generations = []
    differences = []
    best_weight_array = []

    for i in range(generation):
        generations.append(i)
        differences.append(abs(best_difference))

        for _ in range(population):
            random_input = [random.uniform(-10, 10)]
            new_network = BaseNetwork(random_input, network_structure, best_weight_array)

            new_network.init_weights()
            new_network.randomize_weights()
            new_network.modify_network()
            new_network.init_layers()

            difference = new_network.return_difference()
            if abs(difference) < best_difference:
                best_difference = difference
                best_weight_array = new_network.return_weights()

    plt.style.use('seaborn-darkgrid')
    plt.scatter(generations, differences)
    plt.xlabel("Generation")
    plt.ylabel("Square Difference")
    plt.axhline(y=0, color='r', linestyle='-')
    plt.show()

if __name__ == "__main__":
    main(1, 10)