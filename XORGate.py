    #Author: Ian Edwards EDWIAN004
    #CSC3022F ML assignment 2
    #Value Iteration driver file

from Perceptron import Perceptron
from random import seed
from random import randint
import sys
import argparse
#text for argument parser:
text = 'The XOR gate program implements a XOR gate with a feed forward Perceptron ANN' 
x_train, AND_train, OR_train, NAND_train, x_val, AND_val, OR_val, NAND_val = generateData()

def main(): #get arguments passed in.
    parser = argparse.ArgumentParser(description=text) #setup argument parser.
    parser.add_argument('x1', type = int)
    parser.add_argument('x2', type = int)
    args = parser.parse_args()

    x1 = args.x1
    x2 = args.x2

    NAND = Perceptron(2)
    AND = Perceptron(2)
    OR = Perceptron(2)

    #x_train, AND_train, OR_train, NAND_train, x_val, AND_val, OR_val, NAND_val = generateData()
    AND, NAND, OR = train(AND, NAND, OR)
    output = classify(AND, NAND, OR, x1, x2)
    print(output)



def train(AND, NAND, OR):
    valid_percentage_AND = AND.validate(x_val, AND_val, verbose=True)
    valid_percentage_NAND = NAND.validate(x_val, NAND_val, verbose=True)
    valid_percentage_OR = OR.validate(x_val, OR_val, verbose=True)
    print(valid_percentage_AND)
    print(valid_percentage_NAND)
    print(valid_percentage_OR)

    i = 0
    while valid_percentage_AND < 0.98: # We want our Perceptron to have an accuracy of at least 80%
        i += 1
        AND.train(x_train, AND_train, 0.2)  # Train our Perceptron
        print('------ Iteration ' + str(i) + ' ------')
        print(AND.weights)
        valid_percentage_AND = AND.validate(x_val, AND_val, verbose=True) # Validate it
        print(valid_percentage_AND)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train. 
        if i == 50: 
            break

    i = 0
    while valid_percentage_NAND < 0.98: # We want our Perceptron to have an accuracy of at least 80%
        i += 1
        NAND.train(x_train, NAND_train, 0.2)  # Train our Perceptron
        print('------ Iteration ' + str(i) + ' ------')
        print(NAND.weights)
        valid_percentage_NAND = NAND.validate(x_val, NAND_val, verbose=True) # Validate it
        print(valid_percentage_NAND)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train. 
        if i == 50: 
            break

        i = 0
    while valid_percentage_OR < 0.98: # We want our Perceptron to have an accuracy of at least 80%
        i += 1
        OR.train(x_train, OR_train, 0.2)  # Train our Perceptron
        print('------ Iteration ' + str(i) + ' ------')
        print(OR.weights)
        valid_percentage_OR = OR.validate(x_val, OR_val, verbose=True) # Validate it
        print(valid_percentage_OR)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train. 
        if i == 50: 
            break

    return AND, NAND, OR

def generateData():
    training_examples = []
    training_labels_NAND = []
    training_labels_AND = []
    training_labels_OR = []

    for i in range(num_train):
        training_examples.append([random.random(), random.random()])
        training_labels_AND.append(1.0 if training_examples[i][0] > 0.8 and training_examples[i][1] > 0.8 else 0.0)
        training_labels_OR.append(1.0 if training_examples[i][0] > 0.8 or training_examples[i][1] > 0.8 else 0.0)
        training_labels_NAND.append(0.0 if training_examples[i][0] > 0.8 and training_examples[i][1] > 0.8 else 1.0)

    validate_examples = []
    validate_labels_NAND = []
    validate_labels_AND = []
    validate_labels_OR = []

    for i in range(num_train):
        validate_examples.append([random.random(), random.random()])
        validate_labels_AND.append(1.0 if validate_examples[i][0] > 0.8 and validate_examples[i][1] > 0.8 else 0.0)
        validate_labels_OR.append(1.0 if validate_examples[i][0] > 0.8 or validate_examples[i][1] > 0.8 else 0.0)
        validate_labels_NAND.append(0.0 if validate_examples[i][0] > 0.8 and validate_examples[i][1] > 0.8 else 1.0)

    return training_examples, training_labels_AND, training_labels_NAND, training_labels_OR, validate_examples, validate_labels_AND, validate_labels_NAND, validate_labels_OR

def classify(AND, NAND, OR, x1, x2):
    return AND.activate(NAND.activate([x1,x2]), OR.activate([x1,x2]))

if __name__ == "__main__":
    main()
