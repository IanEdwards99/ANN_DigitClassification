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

def main(): #get arguments passed in.
    parser = argparse.ArgumentParser(description=text) #setup argument parser.
    parser.add_argument('x1', type = float)
    parser.add_argument('x2', type = float)
    args = parser.parse_args()

    x1 = args.x1
    x2 = args.x2

    NOT = Perceptron(1)
    AND = Perceptron(2)
    OR = Perceptron(2)

    x_train, AND_train, x_val, AND_val = generateData(100, 100, "AND")
    print("AND")
    AND = train(AND, x_train, AND_train, x_val, AND_val, 0.01)
    ANDanswer = AND.activate([x1,x2])
    print("AND answer:", ANDanswer)

    x_train, NOT_train, x_val, NOT_val = generateData(100, 100, "NOT")

    print("NOT")
    NOT = train(NOT, x_train, NOT_train, x_val, NOT_val, 0.001)
    print("NOT answer:", NOT.activate([ANDanswer]))

    x_train, OR_train, x_val, OR_val = generateData(100, 100, "OR")
    # x_train = [[0,0], [0, 1], [1,0], [1,1]]
    # OR_train = [0,1,1,1]

    # print("OR")
    # OR = train(OR, x_train, OR_train, x_val, OR_val, 0.001)
    # print("OR answer:", OR.activate([x1,x2]))

    output = classify(AND, NOT, OR, x1, x2)
    print("Answer:", output)



def train(perceptron, training_data, training_labels, val_data, val_labels, lr):
    valid_percentage = perceptron.validate(training_data, training_labels, verbose=True)
    print("before %:", valid_percentage)

    i = 0
    while valid_percentage < 0.95: # We want our Perceptron to have an accuracy of at least 80%
        i += 1
        perceptron.train(training_data, training_labels, lr)  # Train our Perceptron
        valid_percentage = round(perceptron.validate(val_data, val_labels, verbose=True),2) # Validate it
        print(valid_percentage)
    print("after %:", perceptron.validate(val_data, val_labels, verbose=True))
    return perceptron

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

def classify(AND, NOT, OR, x1, x2):
    intermediary = NOT.activate([AND.activate([NOT.activate([x1]),NOT.activate([x2])])])
    return AND.activate([NOT.activate([AND.activate([x1,x2])]),intermediary])
    #return AND.activate([NOT.activate([AND.activate([x1,x2])]), OR.activate([x1,x2])])

if __name__ == "__main__":
    main()
