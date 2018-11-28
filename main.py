# SCENE COMPLETION WITH GAN #
"""
MAIN SCRIPT 
- Load Data
- Build, Train And Evaluate the Model
"""
# import libraries
import os
import argparse
import numpy as np
import tensorflow as tf
import random as rnd

# import scripts
import model as m
import helper as h

# load data
def load_data(args):

    #load data X
    path = args.X_path

    # iterate through data - save strings
    X = []
    for model in os.listdir(path):

        # if not folder continue
        objPath = os.path.join(path, model)
        if(os.path.isfile(objPath) == False or not objPath.endswith(".npz")):
            continue

        # append      
        X.append(objPath)

    #load data Y
    path = args.Y_path

    # iterate through data - save strings
    Y = []
    for model in os.listdir(path):

        # if not folder continue
        objPath = os.path.join(path, model)
        if(os.path.isfile(objPath) == False or not objPath.endswith(".npz")):
            continue

        # append      
        Y.append(objPath)

    # label map path
    label_names = {}
    label_colors = {}
    if args.label_map_path:
        with open(args.label_map_path, "r") as fid:
            for line in fid:
                line = line.strip()
                if not line:
                    continue
                label = int(line.split(":")[0].split()[0])
                name = line.split(":")[0].split()[1]
                color = tuple(map(int, line.split(":")[1].split()))
                label_names[label] = name
                label_colors[label] = color

    X = np.sort(X)
    Y = np.sort(Y)
    return X, Y, label_names, label_colors

# parse arguments
def parse_args():

    # define the parser
    parser = argparse.ArgumentParser()

    # define command line arguments
    parser.add_argument("--X_path", default='./Scene/gt/')
    parser.add_argument("--Y_path", default='./Scene/dc/')
    parser.add_argument("--label_map_path", default='./Scene/labels.txt')
    parser.add_argument("--sample_output_path", default='./samples')
    parser.add_argument("--nclasses", default=38)
    parser.add_argument("--load_model", default=-1)

    # return
    return parser.parse_args()

# main
def main():

    # get the arguments when launching the script
    args = parse_args() 

    # initialize seed with 0 - same random number every initialization
    np.random.seed(0)
    tf.set_random_seed(0)

    # set level of logging
    tf.logging.set_verbosity(tf.logging.INFO)

    # load data
    X_Data, Y_Data, label_names, label_colors = load_data(args)

    # parameters
    epochs=100
    batchSize=32

    # construct
    print("INITIALIZE AND BUILD MODEL! . . . ")
    vaeGAN = m.Model(epochs=epochs, batchSize=batchSize, loadModel=args.load_model)
    print(vaeGAN.currentArgs)

    # run model
    with tf.Session() as sess:

        # initialize variables
        if(args.load_model == -1):
            print("INITIALIZE VARIABLES! . . . ")
            sess.run(tf.global_variables_initializer())
            print("DONE!")

        # run training
        print("START TRAINING! . . . ")
        losses = vaeGAN.train(sess, X_Data, Y_Data, label_names, label_colors, args.label_map_path, args.sample_output_path)
        print("DONE!")
    
if __name__ == "__main__":
    main()
