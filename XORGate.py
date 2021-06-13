    #Author: Ian Edwards EDWIAN004
    #CSC3022F ML assignment 2
    #Value Iteration driver file

from Perceptron import Perceptron
import random
from random import seed
from random import randint
import sys
import argparse
#text for argument parser:
text = 'The XOR gate program implements a XOR gate with a feed forward Perceptron ANN' 

def main():
    NOT = Perceptron(1)
    AND = Perceptron(2, bias = -1)
    OR = Perceptron(2, bias = -0.25, seeded_weights=[1,1])

    x_train, AND_train, x_val, AND_val = generateData(300, 100, "AND")
    print("Training GATE_0 - AND Gate")
    AND, success = train(AND, x_train, AND_train, x_val, AND_val, 0.001)

    while not success:
        x_train, AND_train, x_val, AND_val = generateData(300, 100, "AND")
        AND, success = train(AND, x_train, AND_train, x_val, AND_val, 0.001)

    x_train, NOT_train, x_val, NOT_val = generateData(300, 100, "NOT")
    print("Training GATE_1 - NOT Gate")
    NOT, success = train(NOT, x_train, NOT_train, x_val, NOT_val, 0.001)

    while not success:
        x_train, NOT_train, x_val, NOT_val = generateData(300, 100, "NOT")
        NOT, success = train(NOT, x_train, NOT_train, x_val, NOT_val, 0.001)

    #x_train, OR_train, x_val, OR_val = generateData(100, 100, "OR")
    # x_train = [[0,0], [0, 1], [1,0], [1,1]]
    # OR_train = [0,1,1,1]

    # print("OR")
    #OR = train(OR, x_train, OR_train, x_val, OR_val, 0.001)
    # print("OR answer:", OR.activate([x1,x2]))
    print("Constructing Network...")
    print("Done!")
    data = input("Please enter two inputs:\n")
    while(data != "exit"):
        x1, x2 = data.split()
        output = classify(AND, NOT, (float)(x1), (float)(x2))
        print("XOR Gate:", output)
        data = input("Please enter two inputs:\n")
    print("Exiting...")
    
    

def train(perceptron, training_data, training_labels, val_data, val_labels, lr):
    valid_percentage = perceptron.validate(training_data, training_labels, verbose=True)
    #print("before %:", valid_percentage)

    i = 0
    while valid_percentage < 0.98: # We want our Perceptron to have an accuracy of at least 80%
        i += 1
        perceptron.train(training_data, training_labels, lr)  # Train our Perceptron
        valid_percentage = perceptron.validate(val_data, val_labels, verbose=True) # Validate it
        if i > 2000:
            return perceptron, False
    return perceptron, True

def generateData(num_train, num_val, gateType="AND"):
    training_examples = []
    training_labels = []

    for i in range(num_train):
        if gateType == "AND":
            training_examples.append([round(random.uniform(-0.25,1.25), 2), round(random.uniform(-0.25,1.25), 2)])
            training_labels.append(1.0 if training_examples[i][0] > 0.75 and training_examples[i][1] > 0.75 else 0.0)
        elif gateType == "OR":
            training_examples.append([round(random.uniform(-0.25,1.25), 2), round(random.uniform(-0.25,1.25), 2)])
            training_labels.append(1.0 if training_examples[i][0] > 0.75 or training_examples[i][1] > 0.75 else 0.0)
        elif gateType == "NOT":
            training_examples.append([round(random.uniform(-0.25,1.25), 2)])
            training_labels.append(0.0 if training_examples[i][0] > 0.75 else 1.0)

    validate_examples = []
    validate_labels = []

    for i in range(num_val):
        if gateType == "AND":
            validate_examples.append([round(random.uniform(-0.25,1.25), 2), round(random.uniform(-0.25,1.25), 2)])
            validate_labels.append(1.0 if validate_examples[i][0] > 0.75 and validate_examples[i][1] > 0.75 else 0.0)
        elif gateType == "OR":
            validate_examples.append([round(random.uniform(-0.25,1.25), 2), round(random.uniform(-0.25,1.25), 2)])
            validate_labels.append(1.0 if validate_examples[i][0] > 0.75 or validate_examples[i][1] > 0.75 else 0.0)
        elif gateType == "NOT":
            validate_examples.append([round(random.uniform(-0.25,1.25), 2)])
            validate_labels.append(0.0 if validate_examples[i][0] > 0.75 else 1.0)

    return training_examples, training_labels, validate_examples, validate_labels

def classify(AND, NOT, x1, x2):
    intermediary = NOT.activate([AND.activate([NOT.activate([x1]),NOT.activate([x2])])])
    return AND.activate([NOT.activate([AND.activate([x1,x2])]),intermediary])
    #return AND.activate([NOT.activate([AND.activate([x1,x2])]), OR.activate([x1,x2])])

if __name__ == "__main__":
    main()
