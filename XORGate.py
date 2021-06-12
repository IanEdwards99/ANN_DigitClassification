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

    NAND = Perceptron(2, bias = 1.0)
    AND = Perceptron(2, bias = -1.0)
    OR = Perceptron(2, bias = -1)

    x_train, AND_train, NAND_train, OR_train, x_val, AND_val, NAND_val, OR_val = generateData(100)
    print("AND")
    AND = train(AND, x_train, AND_train, x_val, AND_val)
    print("NAND")
    NAND = train(NAND, x_train, NAND_train, x_val, NAND_val)
    print("NAND answer:", NAND.activate([x1,x2]))
    print("OR")
    OR = train(OR, x_train, OR_train, x_val, OR_val)
    print("OR answer:", OR.activate([x1,x2]))
    output = classify(AND, NAND, OR, x1, x2)
    print("Answer:", output)



def train(perceptron, training_data, training_labels, val_data, val_labels):
    valid_percentage = perceptron.validate(training_data, training_labels, verbose=True)
    print("before %:", valid_percentage)

    i = 0
    while valid_percentage < 0.96: # We want our Perceptron to have an accuracy of at least 80%
        i += 1
        perceptron.train(training_data, training_labels, 0.1)  # Train our Perceptron
        # print('------ Iteration ' + str(i) + ' ------')
        # print(perceptron.weights)
        valid_percentage = perceptron.validate(val_data, val_labels, verbose=True) # Validate it
        # print(valid_percentage)
        if i == 5000: 
            break
    print("after %:", perceptron.validate(val_data, val_labels, verbose=True))
    return perceptron

def generateData(num_train):
    training_examples = []
    training_labels_NAND = []
    training_labels_AND = []
    training_labels_OR = []

    for i in range(num_train):
        training_examples.append([random.random(), random.random()])
        training_labels_AND.append(1.0 if training_examples[i][0] >= 0.75 and training_examples[i][1] >= 0.75 else 0.0)
        training_labels_OR.append(1.0 if training_examples[i][0] >= 0.75 or training_examples[i][1] >= 0.75 else 0.0)
        training_labels_NAND.append(0.0 if training_examples[i][0] >= 0.75 and training_examples[i][1] >= 0.75 else 1.0)

    validate_examples = []
    validate_labels_NAND = []
    validate_labels_AND = []
    validate_labels_OR = []

    for i in range(num_train):
        validate_examples.append([random.random(), random.random()])
        validate_labels_AND.append(1.0 if validate_examples[i][0] >= 0.75 and validate_examples[i][1] >= 0.75 else 0.0)
        validate_labels_OR.append(1.0 if validate_examples[i][0] >= 0.75 or validate_examples[i][1] >= 0.75 else 0.0)
        validate_labels_NAND.append(0.0 if validate_examples[i][0] >= 0.75 and validate_examples[i][1] >= 0.75 else 1.0)

    return training_examples, training_labels_AND, training_labels_NAND, training_labels_OR, validate_examples, validate_labels_AND, validate_labels_NAND, validate_labels_OR

def classify(AND, NAND, OR, x1, x2):
    return AND.activate([NAND.activate([x1,x2]), OR.activate([x1,x2])])

if __name__ == "__main__":
    main()
