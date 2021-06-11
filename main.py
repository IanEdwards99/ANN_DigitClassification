#Author: Ian Edwards EDWIAN004
#CSC3022F ML assignment 2
#Value Iteration driver file

from ValueIteration import ValueIteration
from random import seed
from random import randint
import sys
import argparse
#text for argument parser:
text = 'The XOR gate program implements a XOR gate with a feed forward Perceptron ANN' 

def main(): #get arguments passed in.
    parser = argparse.ArgumentParser(description=text) #setup argument parser.
    parser.add_argument('x1', type = int)
    parser.add_argument('x2', type = int)
    args = parser.parse_args()

    x1 = args.x1
    x2 = args.x2


if __name__ == "__main__":
    main()
