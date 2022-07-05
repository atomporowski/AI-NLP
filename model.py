import torch.nn as nn


# design the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        ------------------------------------------------------------
        This function helps to initialize the neural network model
        as soon as call the constructor
        ------------------------------------------------------------
        :param input_size:
        :param hidden_size:
        :param num_classes:
        """
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        ---------------------------------------------------------------------------------------------------------
        Function controls the data flow through the network which makes
        it responsible for feedforward.
        ---------------------------------------------------------------------------------------------------------
        Feedforward neural networks are also known as Multi-layered Network of Neurons (MLN).
        These network of models are called feedforward because the information only travels forward
        in the neural network, through the input nodes then through the hidden layers (single or many layers)
        and finally through the output nodes.
        ---------------------------------------------------------------------------------------------------------
        :param x:
        :return:
        """
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
