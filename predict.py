from math import exp

# Calculate neuron activation for an input
import pandas


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


data = pandas.read_csv("data/penguins.csv")
dataset = []

for (i, row) in data.iterrows():
    m = row['body_mass_g'] / 10000
    l = row['flipper_length_mm'] / 1000
    if m == m and l == l and (row['sex'] in ['MALE', 'FEMALE']):
        s = row['sex'] == 'MALE'
        dataset.append([m, l, s])

network = [[{'weights': [-40.94629894065542, -11.669891530438365, 13.820550901114899], 'output': 2.089172898285934e-05,
             'delta': 1.63803720563232e-06},
            {'weights': [72.6476760733058, 98.49999526427582, -58.03906287317645], 'output': 0.8971596132869817,
             'delta': -0.007667422301416595},
            {'weights': [27.723604307836176, 197.57687090915215, -52.326622144124045], 'output': 0.9912381788266127,
             'delta': 0.0008278153840785506},
            {'weights': [-111.10027259841168, -24.77230977508195, 48.60813206469706], 'output': 5.801773039158376e-08,
             'delta': 2.7621143850075556e-09}],
           [{'weights': [10.39028137315633, -11.014164769847701, 12.628903408162524, 6.3088442766640105,
                         -5.3338237216026725], 'output': 0.06347321123437019, 'delta': 0.003773124589712803}, {
                'weights': [-10.390281371888577, 11.014164770046897, -12.62890340840483, -6.308844276838723,
                            5.333823721719132], 'output': 0.9365267887689316, 'delta': -0.0037731245893335616}]]

for row in dataset:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))
