# HELPER CLASS # 
"""
UTIL SCRIPT
- Wrapper for helpful functions
"""
# libraries
import os, itertools
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import glob
import shutil
import argparse
import tempfile
import numpy as np
import tensorflow as tf
import plyfile
import utils
from skimage.measure import marching_cubes_lewiner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

# create folder if path does not exist
def mkdir_if_not_exists(path):
    assert os.path.exists(os.path.dirname(path.rstrip("/")))
    if not os.path.exists(path):
        os.makedirs(path)

# write2ply to visualize data
def write_ply(path, points, color):
    with open(path, "w") as fid:
        fid.write("ply\n")
        fid.write("format ascii 1.0\n")
        fid.write("element vertex {}\n".format(points.shape[0]))
        fid.write("property float x\n")
        fid.write("property float y\n")
        fid.write("property float z\n")
        fid.write("property uchar diffuse_red\n")
        fid.write("property uchar diffuse_green\n")
        fid.write("property uchar diffuse_blue\n")
        fid.write("end_header\n")
    for i in range(points.shape[0]):
        fid.write("{} {} {} {} {} {}\n".format(points[i, 0], points[i, 1],
                                                points[i, 2], *color))

# extract mesh from tsdf
def extract_mesh_marching_cubes(path, volume, color=None, level=0.5, step_size=1.0, gradient_direction="ascent"):

    if level > volume.max() or level < volume.min():
        return

    verts, faces, normals, values = marching_cubes_lewiner(
        volume, level=level, step_size=step_size,
        gradient_direction=gradient_direction)

    ply_verts = np.empty(len(verts),
                        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    ply_verts["x"] = verts[:, 0]
    ply_verts["y"] = verts[:, 1]
    ply_verts["z"] = verts[:, 2]
    ply_verts = plyfile.PlyElement.describe(ply_verts, "vertex")

    if color is None:
        ply_faces = np.empty(
            len(faces), dtype=[("vertex_indices", "i4", (3,))])
    else:
        ply_faces = np.empty(
            len(faces), dtype=[("vertex_indices", "i4", (3,)),
                            ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        ply_faces["red"] = color[0]
        ply_faces["green"] = color[1]
        ply_faces["blue"] = color[2]
    ply_faces["vertex_indices"] = faces
    ply_faces = plyfile.PlyElement.describe(ply_faces, "face")

    with tempfile.NamedTemporaryFile(dir=".", delete=False) as tmpfile:
        plyfile.PlyData([ply_verts, ply_faces]).write(tmpfile.name)
        shutil.move(tmpfile.name, path)

# visualize data
def visualize_data(volume, name, outputPath, labelNames, labelColors, labelMapPath, one_hot):

    # create folder
    outputVolume = outputPath + '/' + str(name)
    print(outputVolume)
    mkdir_if_not_exists(outputPath)
    mkdir_if_not_exists(outputVolume)

    # reshape it if it's 5-dim
    if(len(volume.shape) == 5):
        volume = volume[0]

    # save npz
    name = name + '.npz'
    np.savez_compressed(os.path.join(outputPath, name),
                        probs=volume)

    # make one hot
    if (one_hot == True):
        volume = makeOH(volume)

    # create ply
    for label in range(volume.shape[-1]):
        if labelMapPath:
            path = os.path.join(outputVolume,
                                "{}-{}.ply".format(label, labelNames[label]))
            color = labelColors[label]
        else:
            path = os.path.join(outputVolume, "{}.ply".format(label))
            color = None
    
        extract_mesh_marching_cubes(path, volume[..., label], color=color)

# visualize data
def save_data(volume, name, outputPath, labelNames, labelColors, labelMapPath, one_hot):

    # create folder
    outputVolume = outputPath + '/' + str(name)
    print(outputVolume)
    mkdir_if_not_exists(outputPath)
    mkdir_if_not_exists(outputVolume)

    # reshape it if it's 5-dim
    if(len(volume.shape) == 5):
        volume = volume[0]

    # make one hot
    if (one_hot == True):
        volume = makeOH(volume)

    # save npz
    name = name + '.npz'
    np.savez_compressed(os.path.join(outputPath, name),
                        probs=volume)

# convert generator output to correct format
def makeOH(thisSample):

    # get props
    nRow = thisSample.shape[0]
    nCol = thisSample.shape[1]
    nSlices = thisSample.shape[2]

    # convert it to right format
    for row in range(nRow):
        for col in range(nCol):
            for slices in range(nSlices):

                # get vector of classes
                classes = thisSample[row,col,slices]

                # index of min element -> most likely
                maxIndex = np.argmax(classes)

                # replace vector with 0/1
                classes = np.zeros(38)
                classes[maxIndex] = 1
                thisSample[row,col,slices] = classes
    
    # return
    return thisSample

# create partial image
def createPartial(test_image, batchSize):

    # create mask
    mask = np.random.randint(2,size=test_image.shape)

    # get random indexes
    indexes = np.random.randint(0, batchSize, 2)
    startInd = np.min(indexes)
    endInd = np.max(indexes)
    mask[startInd:endInd,startInd:endInd,:] = 0

    # multiply
    partialY = np.multiply(mask, test_image) 

    return partialY

# load data
def load_data(dataPath, labelMapPath):

    #load data
    print(dataPath)

    # iterate through data - save strings
    X = []
    for folder in os.listdir(dataPath):

        # if not folder continue
        modelPath = os.path.join(dataPath, folder)
        if(os.path.isdir(modelPath) == False): # or not folder.startswith('0')):
            continue

        for model in os.listdir(modelPath):

            # if not folder continue
            objPath = os.path.join(modelPath, model)
            if(os.path.isfile(objPath) == False or not objPath.endswith(".npy")):
                continue

            # append      
            X.append(objPath)

    # label map path
    label_names = {}
    label_colors = {}
    if labelMapPath:
        with open(labelMapPath, "r") as fid:
            for line in fid:
                line = line.strip()
                if not line:
                    continue
                label = int(line.split(":")[0].split()[0])
                name = line.split(":")[0].split()[1]
                color = tuple(map(int, line.split(":")[1].split()))
                label_names[label] = name
                label_colors[label] = color

    return X, label_names, label_colors

# load test data
def load_test_data(X_path, Y_path, label_map_path):

    #load data X
    path = X_path

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
    path = Y_path

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
    if label_map_path:
        with open(label_map_path, "r") as fid:
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

# process Scene
def processScene(gtScenePath, dcScenePath, subVolumeShape):

    # get groundtruth and datacost
    batchSize = subVolumeShape[0]
    gt = np.load(gtScenePath)['probs'][None]
    dc = np.load(dcScenePath)['volume'][None]

    # parameters
    gtShape = gt.shape
    dcShape = dc.shape
    xMax = np.amin([gt.shape[1], dc.shape[1]]) - subVolumeShape[1]
    yMax = np.amin([gt.shape[2], dc.shape[2]]) - subVolumeShape[2]
    zMax = np.amin([gt.shape[3], dc.shape[3]]) - subVolumeShape[3]

    # fill batch Vector
    batchGT = np.empty(shape=subVolumeShape)
    batchDC = np.empty(shape=subVolumeShape)
    # get every batch
    for batch in range(batchSize):

        # while sub sample is not valid
        valid = False
        while not valid:

            # sample subvolume
            xStart = np.random.randint(0,xMax)
            xEnd = xStart + subVolumeShape[1]
            yStart = np.random.randint(0,yMax)
            yEnd = yStart + subVolumeShape[2]
            zStart = np.random.randint(0,zMax)
            zEnd = zStart + subVolumeShape[3]
            subVolumeGT = gt[0, xStart:xEnd, yStart:yEnd, zStart:zEnd, :]
            subVolumeDC = dc[0, xStart:xEnd, yStart:yEnd, zStart:zEnd, :]

            # check if valid
            check = np.sum(np.sum(subVolumeGT[:,:,:,0:35],(0,1,2)))
            if(check > 0):
                batchGT[batch] = subVolumeGT
                batchDC[batch] = subVolumeDC
                valid = True

    return batchGT, batchDC

# process datacost when evaluating
def processDatacost(datacost, subVolumeShape):

    # load datacost
    resInt = int(subVolumeShape[1])
    resFloat = float(resInt)
    datacostShape = datacost.shape
    print(datacostShape)

    # pad datacost with free space to fit 32
    XpadVal = resInt * int(math.ceil(datacostShape[1]/resFloat))
    YpadVal = resInt * int(math.ceil(datacostShape[2]/resFloat))
    ZpadVal = resInt * int(math.ceil(datacostShape[3]/resFloat))
    padVal = [XpadVal,YpadVal,ZpadVal]
    datacostPad = np.pad(datacost, ((0,0), (0,XpadVal - datacostShape[1]), (0,YpadVal - datacostShape[2]), (0,ZpadVal - datacostShape[3]), (0,0)), 'constant', constant_values=(0))
    datacostPad[0,datacostShape[1]:,datacostShape[2]:,datacostShape[3]:,-1] = -2
    datacostPadShape = datacostPad.shape

    # tesselate datacost
    numBlockAxis = [int(XpadVal/resInt), int(YpadVal/resInt), int(ZpadVal/resInt)]    
    thisDatacost = np.zeros(shape=(0, 32, 32, 32, 38))
    thisDatacostIndexes = np.empty(shape=(0,6))
    block = 0
    for row in range(numBlockAxis[0]):
        startRow = resInt * row
        endRow = startRow + resInt
        startRow2 = startRow + 8
        endRow2 = endRow + 8
        startRow3 = startRow + 16
        endRow3 = endRow + 16
        startRow4 = startRow + 24
        endRow4 = endRow + 24
        for col in range(numBlockAxis[1]):
            startCol = resInt * col
            endCol = startCol + resInt
            startCol2 = startCol + 8
            endCol2 = endCol + 8
            startCol3 = startCol + 16
            endCol3 = endCol + 16
            startCol4 = startCol + 24
            endCol4 = endCol + 24
            for slices in range(numBlockAxis[2]):
                startSlices = resInt * slices
                endSlices = startSlices + resInt
                startSlices2 = startSlices + 8
                endSlices2 = endSlices + 8
                startSlices3 = startSlices + 16
                endSlices3 = endSlices + 16
                startSlices4 = startSlices + 24
                endSlices4 = endSlices + 24
                # fill datacost
                thisDatacost = np.concatenate((thisDatacost, datacostPad[:, startRow:endRow, startCol:endCol, startSlices:endSlices, :]))
                # thisDatacost[block] = datacostPad[0, startRow:endRow, startCol:endCol, startSlices:endSlices, :]
                thisDatacostIndexes = np.concatenate((thisDatacostIndexes, [[startRow, endRow, startCol, endCol, startSlices, endSlices]]))
                # thisDatacostIndexes[block] = [startRow, endRow, startCol, endCol, startSlices, endSlices]
                # dont go out of datacost
                if(endRow4 <= XpadVal and endCol4 <= YpadVal and endSlices4 <= ZpadVal):
                    thisDatacost = np.concatenate((thisDatacost, datacostPad[:, startRow2:endRow2, startCol2:endCol2, startSlices2:endSlices2, :]))
                    thisDatacostIndexes = np.concatenate((thisDatacostIndexes, [[startRow2, endRow2, startCol2, endCol2, startSlices2, endSlices2]]))
                    thisDatacost = np.concatenate((thisDatacost, datacostPad[:, startRow3:endRow3, startCol3:endCol3, startSlices3:endSlices3, :]))
                    thisDatacostIndexes = np.concatenate((thisDatacostIndexes, [[startRow3, endRow3, startCol3, endCol3, startSlices3, endSlices3]]))
                    thisDatacost = np.concatenate((thisDatacost, datacostPad[:, startRow4:endRow4, startCol4:endCol4, startSlices4:endSlices4, :]))
                    thisDatacostIndexes = np.concatenate((thisDatacostIndexes, [[startRow4, endRow4, startCol4, endCol4, startSlices4, endSlices4]]))
                    '''
                    thisDatacost[block+1] = datacostPad[0, startRow2:endRow2, startCol2:endCol2, startSlices2:endSlices2, :]
                    thisDatacostIndexes[block+1] = [startRow2, endRow2, startCol2, endCol2, startSlices2,endSlices2]
                    thisDatacost[block+2] = datacostPad[0, startRow3:endRow3, startCol3:endCol3, startSlices3:endSlices3, :]
                    thisDatacostIndexes[block+2] = [startRow3, endRow3, startCol3, endCol3, startSlices3,endSlices3]
                    thisDatacost[block+3] = datacostPad[0, startRow4:endRow4, startCol4:endCol4, startSlices4:endSlices4, :]
                    thisDatacostIndexes[block+3] = [startRow4, endRow4, startCol4, endCol4, startSlices4,endSlices4]
                    '''
                    block = block + 4
                else:
                    block = block + 1
    numBlocks=block 
    # pad in case numBlocks is not divisible per res
    padRecBlocks = int(math.ceil(numBlocks/resFloat) * resInt) - int(numBlocks)
    thisDatacost = np.concatenate((thisDatacost, np.zeros(shape=(padRecBlocks, 32, 32, 32, 38))))
    # return
    return thisDatacost, thisDatacostIndexes, numBlocks, numBlockAxis, padVal
    
