# SCENE COMPLETION WITH 3DVAAEGAN #
"""
TEST SCRIPT 
- Load Test Data
- Evaluate Model
- Collect result data
"""
# import libraries
import os, itertools
import argparse
import numpy as np
import tensorflow as tf
import random as rnd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import *
import math

# import scripts
import model as m
import helper as h

# parse arguments
def parse_args():

    # define the parser
    parser = argparse.ArgumentParser()

    # define command line arguments
    parser.add_argument("--test_folder", default='./Scene_Test')
    parser.add_argument("--result_folder", default='./TEST')
    parser.add_argument("--checkpoints_path", default="./checkpoints")
    parser.add_argument("--load_model", default=78)

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

    # Load Data
    xPath = args.test_folder + '/gt/'
    yPath = args.test_folder + '/dc/'
    labelMapPath = args.test_folder + '/labels.txt'
    recFolder = args.result_folder + '/REC/'
    genFolder = args.result_folder + '/GEN/'
    X, Y, label_names, label_colors = h.load_test_data(xPath, yPath, labelMapPath)

    # values
    nTestScenes = len(Y)
    subVolumeShape = [1, 32, 32, 32, 38]
    zDim=[1, 2, 2, 2, 16]
    batchSize=32

    # booleans
    doRec = True
    doGen = True

    # start evaluating
    with tf.Session() as sess:

        # load model
        print("LOAD THE MODEL . . .")
        new_saver = tf.train.import_meta_graph('./checkpoints/VAE_GAN-' + str(args.load_model) + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))
        gan = tf.get_default_graph()
        print("DONE!")

        print("GET TENSORS . . .")  
        # get noise placeholder
        z = gan.get_tensor_by_name("noiseZ:0")
        # get datacost placeholder
        y = gan.get_tensor_by_name("datacostData:0")
        # get encoding
        encZ = gan.get_tensor_by_name("encScope/EConvMean/BiasAdd:0")
        # get generated sample
        genSample = gan.get_tensor_by_name("genScope/GSOFTMAXOUT:0")
        print("DONE!")

        if(doRec):

            # test reconstruction
            print("TESTING RECONSTRUCTION . . .")
            totalAcc = np.empty(shape=(nTestScenes,4))
            classAcc = np.zeros(shape=(nTestScenes, subVolumeShape[-1]))
            classCount = np.zeros(shape=(nTestScenes, subVolumeShape[-1]))
            roomEncoding = np.empty(shape=(nTestScenes, 2))
            encMatrix = np.empty(shape=(0, zDim[1]*zDim[2]*zDim[3]*zDim[4]))
            encIdx = np.zeros(shape=(nTestScenes,2))
            roomNames = []
            for test in range(nTestScenes):

                # get ground truth
                groundtruth = np.load(X[test])['probs'][None]
                gtShape = groundtruth.shape

                # get datacost
                datacost = np.load(Y[test])['volume'][None]
                dcShape = datacost.shape

                # get room name and append it to list
                roomName = Y[test].split('/')[-1]
                roomName = roomName.split('.')[0]
                roomNames.append(roomName)

                # tesselate datacost in list of 32 block
                Y_, Yindexes, numBlocks, numBlockAxis, padVal = h.processDatacost(datacost, subVolumeShape)

                # encode datacost blocks and reconstruct them in gt form
                resInt = int(subVolumeShape[1])
                resFloat = float(subVolumeShape[1])
                recBlocks = int(math.ceil(numBlocks/resFloat) * resInt)
                encSample = np.zeros(shape=(int(recBlocks),zDim[1], zDim[2], zDim[3], zDim[-1]))
                recSample = np.zeros(shape=(int(recBlocks),subVolumeShape[1], subVolumeShape[2], subVolumeShape[3], subVolumeShape[-1]))
                for iter in range(0,int(numBlocks),batchSize):

                    # get index of datacost and get block
                    startIndex = iter
                    endIndex = startIndex + subVolumeShape[1]
                    thisY = Y_[startIndex:endIndex]

                    # encode and reconstruct
                    encSample[startIndex:endIndex] = encZ.eval({y : thisY})
                    recSample[startIndex:endIndex] = genSample.eval({z : encSample[startIndex:endIndex]})
       
                # rebuild datacost reconstructed in original shape
                recDatacost = np.empty(shape=(1,int(padVal[0]), int(padVal[1]), int(padVal[2]), 38))
                for block in range(numBlocks):
                    Idx = Yindexes[block,:]
                    Idx = Idx.astype(int)
                    oldBlock = recDatacost[0, Idx[0]:Idx[1], Idx[2]:Idx[3], Idx[4]:Idx[5], :]
                    recDatacost[0, Idx[0]:Idx[1], Idx[2]:Idx[3], Idx[4]:Idx[5], :] = np.mean([oldBlock, recSample[block]],0)
                recDatacostDC = recDatacost[:,:dcShape[1],:dcShape[2],:dcShape[3],:]

                # build encoding matrix
                encIdx[test, 0] = encMatrix.shape[0]
                encIdx[test, 1] = encIdx[test, 0] + numBlocks
                encMatrix = np.concatenate((encMatrix, np.reshape(encSample, newshape=(-1,zDim[1]*zDim[2]*zDim[3]*zDim[4]))))
                
                # discretize datacost to one hot representation
                recDatacostOH = np.copy(groundtruth)
                recDatacostOH[:,:dcShape[1],:dcShape[2],:dcShape[3],:] = 0
                for row in range(dcShape[1]):
                    for col in range(dcShape[2]):
                        for slices in range(dcShape[3]):

                            # fill reconstructed datacost one hot encoded
                            idx = np.argmax(recDatacostDC[0,row,col,slices,:])
                            recDatacostOH[0,row,col,slices,idx] = 1 

                # compute accuracy with a loop
                groundtruth = groundtruth.astype(int)
                recDatacostOH = recDatacostOH.astype(int)
                thisFullAccuracy = 0
                thisFsAccuracy = 0
                thisOAccuracy = 0
                thisSAccuracy = 0
                thisFsCount = 0
                thisOCount = 0
                nVoxels = 0
                for row in range(dcShape[1]):
                    for col in range(dcShape[2]):
                        for slices in range(dcShape[3]):

                            # get groundtruth voxel array
                            gtVal = groundtruth[0, row, col, slices, :]
                            dcVal = recDatacostOH[0, row, col, slices, :]

                            # get class index
                            gtIdx = np.argmax(gtVal)
                            dcIdx = np.argmax(dcVal)

                            # increment class count
                            classCount[test, gtIdx] += 1

                            # full accuracy
                            nVoxels += 1
                            if(gtIdx == dcIdx):

                                # increment full and per-class accuracy
                                thisFullAccuracy += 1
                                classAcc[test, gtIdx] += 1

                            # free space accuracy
                            if(gtIdx == (subVolumeShape[-1]-1)):

                                # increment free space counter
                                thisFsCount += 1

                                # increment free space accuracy
                                if(gtIdx == dcIdx):
                                    thisFsAccuracy += 1

                            # occupied accuracy
                            elif(gtIdx < (subVolumeShape[-1]-1)):

                                # increment occupied counter
                                thisOCount += 1

                                # increment occupied accuracy
                                if(dcIdx < (subVolumeShape[-1]-1)):
                                    thisOAccuracy += 1

                                # increment semantic accuracy accuracy
                                if(gtIdx == dcIdx):
                                    thisSAccuracy += 1
                
                # compute final value per room
                thisFullAccuracy = thisFullAccuracy/float(nVoxels)
                thisFsAccuracy = thisFsAccuracy/float(thisFsCount)
                thisOAccuracy = thisOAccuracy/float(thisOCount)
                thisSAccuracy = thisSAccuracy/float(thisOCount)

                # save accuracies
                totalAcc[test] = [thisFullAccuracy, thisFsAccuracy, thisOAccuracy, thisSAccuracy]

                # compute per class accuracy
                for label in range(subVolumeShape[-1]):

                    # check if label count > 0
                    if(classCount[test, label] > 0):
                        classAcc[test, label] = classAcc[test, label]/float(classCount[test, label])
                    # else acc = 1
                    else:
                        classAcc[test, label] = 1.0

                print("DEBUG : ")
                print("n test : " + str(test))
                print("nVoxels : " + str(nVoxels))
                print("thisFullAccuracy : " + str(thisFullAccuracy))
                print("thisFsAccuracy : " + str(thisFsAccuracy))
                print("thisOAccuracy : " + str(thisOAccuracy))
                print("thisSAccuracy : " + str(thisSAccuracy))
                print("classAcc[test,:] : ")
                print(classAcc[test,:])

                # save reconstruction
                # h.save_data(recDatacost, 'rec_' + roomName, recFolder, label_names, label_colors, labelMapPath, False)

            print("DONE!")

            # compute final general accuracy values and save file
            finalAcc = np.mean(totalAcc,0)
            np.save(args.result_folder + '/RoomAccuracy_' + str(args.load_model), totalAcc)
            np.save(args.result_folder + '/FinalAccuracy_' + str(args.load_model), finalAcc)

            # compute final class accuracies and count and save file
            finalClassAcc = np.mean(classAcc, 0)
            np.save(args.result_folder + '/RoomClassAccuracy_' + str(args.load_model), classAcc)
            np.save(args.result_folder + '/FinalClassAccuracy_' + str(args.load_model), finalClassAcc)
            np.save(args.result_folder + '/classCount_' + str(args.load_model), classCount)

            # visualization analysis
            # perform svd on transpose of encoding matrix feature[]#blocks
            encMatrix = np.transpose(encMatrix)
            u, s, vh =  np.linalg.svd(encMatrix)

            # project every room encoding on 2principal axis
            for room in range(nTestScenes):
                # recall the encoding of room
                thisEncMatrix = encMatrix[ :, int(encIdx[room, 0]):int(encIdx[room, 1])]
                # compute projection on the 2-principal axis of room
                thisProjEnc = np.dot(np.transpose(u[:,:2]), thisEncMatrix)
                # compute mean block of encoding of room
                ThisProjEncMean = np.mean(thisProjEnc,1)
                # store it
                roomEncoding[room] = ThisProjEncMean

            # save encoding array and names
            np.save(args.result_folder + '/roomEncoding_' + str(args.load_model), roomEncoding)
            np.save(args.result_folder + '/roomNames_' + str(args.load_model), roomNames)

        if(doGen):

            # play with it
            print("TESTING GENERATOR . . .")
            nTest = 1
            theta = np.linspace(0,1,10)
            sampleZA = np.random.normal(0,1,(batchSize,zDim[1],zDim[2],zDim[3],zDim[4]))
            sampleZB = np.random.normal(0,1,(batchSize,zDim[1],zDim[2],zDim[3],zDim[4]))
            for test in range(nTest):
            
                # random sample
                sampleZ = np.add(np.multiply(theta[test],sampleZA),np.multiply(1 - theta[test],sampleZB))
                
                # generate
                thisSample = genSample.eval({z : sampleZ})
            
                # visualize and save it
                h.visualize_data(thisSample[0], 'gen_' + str(test), genFolder, label_names, label_colors, labelMapPath, True)
            print("DONE!")

if __name__ == "__main__":
    main()

