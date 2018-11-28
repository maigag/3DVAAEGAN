# DCGAN MODEL CLASS #
"""
MODEL SCRIPT 
- Define the VAE-DCGAN network architecture along with its variables and methods
"""
# libraries
import numpy as np
import os, itertools
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
from utils import *

# import scripts
import helper as h

# define class
class Model:

	# initialization
	def __init__(self, batchSize=1, learningRate=1.e-4, epochs=1,
		zDim=[1, 2, 2, 2, 16], subVolumeShape=[1, 32, 32, 32, 38], 
		kernelShape=[5, 5, 5], poolShape=[2, 2, 2], 
		paddingType='same', stride=2, adamBeta1=0.5, leakySlope=0.2,
		alphaDivergence=1, alphaRec=1,
		isTraining=True, loadModel=-1):

		# clears the default graph stack and resets the global default graph.
		tf.reset_default_graph()

		# print parameters
		self.currentArgs = locals()

		# train parameters
		self.batchSize = batchSize
		self.learningRate = learningRate
		self.epochs = epochs
		self.adamBeta1 = adamBeta1
		self.leakySlope = leakySlope
		self.alphaDivergence = alphaDivergence
		self.alphaRec = alphaRec
		
		# input parameters
		zDim[0] = batchSize
		self.zDim = zDim
		subVolumeShape[0] = batchSize
		self.subVolumeShape = subVolumeShape
		self.nClasses = subVolumeShape[-1]

		# disc parameters
		self.kernelShape = kernelShape
		self.poolShape = poolShape

		# network parameter
		self.paddingType = paddingType
		self.stride = stride

		# is training mode
		self.isTraining = isTraining
		
		# build model automatically
		self.loadModel = int(loadModel)
		self.buildModel()

	# define encoder
	def encoder(self, thisInput, reuse=False, isTraining=True):
		
		# define encoder variable scope
		with tf.variable_scope("encScope", reuse=reuse) as scope:

			# convolutions
			conv1 = self.myConv3D(thisInput, 64, self.kernelShape, withBN=False, isTraining=self.isTraining, thisName='EConv1')
			conv2 = self.myConv3D(conv1, 128, self.kernelShape, isTraining=self.isTraining, thisName='EConv2')
			conv3 = self.myConv3D(conv2, 256, self.kernelShape, isTraining=self.isTraining, thisName='EConv3')
			conv4 = self.myConv3D(conv3, 512, self.kernelShape, isTraining=self.isTraining, thisName='EConv4')

			# compute mean and sigma
			mean = tf.layers.conv3d(inputs=conv4, filters=self.zDim[-1], kernel_size=self.kernelShape, padding=self.paddingType, activation=None, name='EConvMean')
			sigma = tf.layers.conv3d(inputs=conv4, filters=self.zDim[-1], kernel_size=self.kernelShape, padding=self.paddingType, activation=None, name='EConvSigma')
			eps = tf.random_normal(self.zDim)
			encZ = tf.add(mean, tf.multiply(sigma, eps))

			return encZ, mean, sigma

	# define encoder discriminator
	def codeDiscriminator(self, thisInput, reuse=False):

		# define encoder discriminator variable scope
		with tf.variable_scope("encDiscScope", reuse=reuse) as scope:

			# reuse variables in case
			if reuse:
				scope.reuse_variables()

			# 3 layers MLP
			cd1 = self.MLP(thisInput, self.zDim[1]*self.zDim[2]*self.zDim[3]*self.zDim[4], thisName='EncD1')
			cd2 = self.MLP(cd1, 750, thisName='EncD2')
			cd3 = self.MLP(cd2, 750, thisName='EncD3')

			# flattening step
			flattening = tf.contrib.layers.flatten(cd3)
			# compute sigmoid output #  
			logit = tf.layers.dense(inputs=flattening, units=1, name='ENCDISCOUT')
			# compute output
			encDiscOut = tf.sigmoid(x=logit, name='ENCDISCSIGOUT')

			return encDiscOut, logit

	# define generator
	def generator(self, thisInput, reuse=False, isTraining=True):	

		# define discriminator variable scope
		with tf.variable_scope("genScope", reuse=reuse) as scope:
		
			# project and reshape z
			z1 = tf.layers.dense(inputs=thisInput, units=512, name='GZReshape')
			reshapedZ1 = tf.reshape(z1, [-1, 2, 2, 2, 512])
			# apply batch normalization
			batchNormZ1 = tf.layers.batch_normalization(inputs=reshapedZ1, training=isTraining, name='GBatchN1')
			# apply activation function
			activZ1 = tf.nn.leaky_relu(features=batchNormZ1, name='GRelu1')

			# start upconvolution pipeline - maybe loop until correct shape is reached
			z2 = self.myConv3DT(activZ1, 256, self.kernelShape, isTraining=isTraining, thisName='GConvT1') 
			z3 = self.myConv3DT(z2, 128, self.kernelShape, isTraining=isTraining, thisName='GConvT2')
			z4 = self.myConv3DT(z3, 64, self.kernelShape, isTraining=isTraining, thisName='GConvT3') 

			# output layer
			logit = tf.layers.conv3d_transpose(inputs=z4, filters=self.nClasses, kernel_size=self.kernelShape, padding=self.paddingType, strides=self.stride, activation=None, name='GConvT3')
			gOut = tf.nn.softmax(logit, name='GSOFTMAXOUT')

			return gOut, logit

	# define discriminator
	def discriminator (self, thisInput, reuse=False):

		# define discriminator variable scope
		with tf.variable_scope("discScope", reuse=reuse) as scope:

			# reuse variables in case
			if reuse:
				scope.reuse_variables()

			# convolution 
			conv1 = self.myConv3D(thisInput, 64, self.kernelShape, withBN=False, isTraining=self.isTraining, thisName='DConv1')
			conv2 = self.myConv3D(conv1, 128, self.kernelShape, isTraining=self.isTraining, thisName='DConv2') 
			conv3 = self.myConv3D(conv2, 256, self.kernelShape, isTraining=self.isTraining, thisName='DConv3') 
			conv4 = self.myConv3D(conv3, 512, self.kernelShape, isTraining=self.isTraining, thisName='DConv4') 

			# flattening step
			flattening = tf.contrib.layers.flatten(conv4)
			# compute sigmoid output #  
			logit = tf.layers.dense(inputs=flattening, units=1, name='DFCLOUT')
			# compute output
			discOut = tf.sigmoid(x=logit, name='DSIGOUT')

			return discOut, logit
	
	# define single layer MLP
	def MLP(self, thisInput, nOfUnits, withBN=True, isTraining=True, thisName='myMLP'):

		# apply fcl
		mlpName = thisName + '/flc'
		h1 = tf.layers.dense(thisInput, nOfUnits, activation=None, name=mlpName)
		h2 = h1

		# apply batch normalization
		batchName = thisName + '/batch'
		if(withBN):
			batchNorm2 = tf.layers.batch_normalization(inputs=h1, training=isTraining, name=batchName)
			h2 = batchNorm2

		# apply leaky relu
		activName = thisName + '/activ'
		activ3 = tf.nn.leaky_relu(h2,name=activName)

		# return flc - batchNorm - activeF
		return activ3

	# define my convolution
	def myConv3D(self, thisInput, thisFilter, thisKernel, thisPad='SAME', thisStride=(2, 2, 2), withBN=True, isTraining=True,  thisName='myConv3D'):

		# apply convolution
		convName = thisName + '/conv'
		conv1 = tf.layers.conv3d(inputs=thisInput, filters=thisFilter, kernel_size=thisKernel, padding=thisPad, strides=thisStride, activation=None, name=convName)
		activ2 = conv1

		# apply batch normalization
		batchName = thisName + '/batch'
		if(withBN):
			batchNorm2 = tf.layers.batch_normalization(inputs=conv1, training=isTraining, name=batchName)
			activ2 = batchNorm2

		# apply leaky relu
		activName = thisName + '/activ'
		activ3 = tf.nn.leaky_relu(activ2,name=activName)

		# return conv - batchNorm - activeF
		return activ3

	# define transpoe convolution
	def myConv3DT(self, thisInput, thisFilter, thisKernel, thisPad='SAME', thisStride=(2, 2, 2), withBN=True, isTraining=True,  thisName='myConv3DT'):

		# apply transpose convolution
		convName = thisName + '/convT'
		conv1 = tf.layers.conv3d_transpose(inputs=thisInput, filters=thisFilter, kernel_size=thisKernel, padding=thisPad, strides=thisStride, activation=None, name=convName)
		activ2 = conv1

		#apply batch normalization
		batchName = thisName + '/batchT'
		if (withBN):
			batchNorm2 = tf.layers.batch_normalization(inputs=conv1, training=isTraining, name=batchName)
			activ2 = batchNorm2

		# apply leaky relu
		activName = thisName + '/activT'
		activ3 = tf.nn.leaky_relu(activ2,name=activName)

		# return convT - batchNorm - activeF
		return activ3

	# build DCGAN
	def buildModel(self):

		# define data placeholders
		self.z = tf.placeholder(shape=self.zDim, dtype=tf.float32, name='noiseZ')
		self.x = tf.placeholder(shape=self.subVolumeShape, dtype=tf.float32, name='groundTruthData')
		self.y = tf.placeholder(shape=self.subVolumeShape, dtype=tf.float32, name='datacostData')

		# encode partial data with encoder
		self.encZ, self.eMean, self.eSigma = self.encoder(self.y)

		# classify prior
		self.encDiscReal, self.encdLogitReal = self.codeDiscriminator(self.z)
		# classify encoding
		self.encDiscFake, self.encdLogitFake = self.codeDiscriminator(self.encZ, reuse=True)

		# generate new sample with generator
		self.genSample, self.gLogit = self.generator(self.z)		
		# generate encoded sample with generator
		self.genSampleEnc, self.gLogitEnc = self.generator(self.encZ, reuse=True)

		# classify real data
		self.discReal, self.dLogitReal = self.discriminator(self.x)
		# classify generated sample
		self.discGen, self.dLogitGen = self.discriminator(self.genSample,reuse=True)
		# classify decoded sample
		self.discDec, self.dLogitDec = self.discriminator(self.genSampleEnc,reuse=True)

		# compute encoder discriminator accuracy
		encDiscRealPercArr = tf.cast(tf.greater(tf.reshape(self.encDiscReal, shape=[self.batchSize]),0.5), tf.int32)
		encDiscFakePercArr = tf.cast(tf.less_equal(tf.reshape(self.encDiscFake, shape=[self.batchSize]),0.5), tf.int32)
		encDiscRealPerc = tf.reduce_sum(encDiscRealPercArr)
		encDiscFakePerc = tf.reduce_sum(encDiscFakePercArr)
		self.encDiscAccuracy = tf.divide(tf.add(encDiscRealPerc, encDiscFakePerc), 2*self.batchSize)

		# compute discriminator accuracy
		discRealPercArr = tf.cast(tf.greater(tf.reshape(self.discReal, shape=[self.batchSize]),0.5), tf.int32)
		discGenPercArr = tf.cast(tf.less_equal(tf.reshape(self.discGen, shape=[self.batchSize]),0.5), tf.int32)
		discRealPerc = tf.reduce_sum(discRealPercArr)
		discGenPerc = tf.reduce_sum(discGenPercArr)
		self.discAccuracy = tf.divide(tf.add(discRealPerc, discGenPerc), 2*self.batchSize)

		# tensorboard summary 
		tf.summary.histogram('Z', self.z)
		tf.summary.histogram('encZ', self.encZ)
		tf.summary.histogram('genSample', self.genSample)
		tf.summary.histogram('genSampleEnc', self.genSampleEnc)

		# define encoder Discriminator loss
		self.encDLossReal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.encdLogitReal, labels=tf.ones_like(self.encDiscReal)))
		self.encDLossGen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.encdLogitFake, labels=tf.zeros_like(self.encDiscFake)))
		self.encDLoss = self.encDLossReal + self.encDLossGen
		
		# define reconstruction loss
		meanX = tf.reduce_mean(self.x, [0,1,2,3])
		ones = tf.ones_like(meanX)
		inverse = tf.div(ones, tf.add(meanX, ones))
		weight = inverse * tf.div(1., tf.reduce_sum(inverse))
		self.recLoss = -tf.reduce_sum(0.97*self.x* tf.log(1e-6 + self.genSampleEnc) + (0.03) * (1-self.x) * tf.log(1e-6 + 1-self.genSampleEnc), [1,2,3]) 
		self.recLoss = tf.reduce_mean(tf.reduce_sum(self.recLoss * weight, 1))

		# define GAN loss + reconstruction loss
		self.dLossReal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dLogitReal, labels=tf.ones_like(self.discReal)))
		self.dLossGen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dLogitGen, labels=tf.zeros_like(self.discGen)))
		self.dLossDec = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dLogitDec, labels=tf.zeros_like(self.discDec)))
		self.dLoss = self.dLossReal + self.dLossGen + self.dLossDec
		self.gLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dLogitGen, labels=tf.ones_like(self.discGen)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dLogitDec, labels=tf.ones_like(self.discDec)) + self.alphaRec*self.recLoss)

		# define encoder and reconstruction loss
		self.encLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.encdLogitFake, labels=tf.ones_like(self.encDiscFake)) + self.alphaRec*self.recLoss)

		# tensorboard summary 
		tf.summary.scalar('encDLoss', self.encDLoss)
		tf.summary.scalar('dLoss', self.dLoss)
		tf.summary.scalar('encLoss', self.encLoss)
		tf.summary.scalar('gLoss', self.gLoss)

		# get variables
		allVars = tf.trainable_variables()
		self.encDiscVar = [var for var in allVars if var.name.startswith('encDiscScope')]
		self.discVar = [var for var in allVars if var.name.startswith('discScope')]
		self.encVar = [var for var in allVars if var.name.startswith('encScope')]
		self.genVar = [var for var in allVars if var.name.startswith('genScope')]

		# define optimizer -  ensures that we execute the update_ops before performing the train_step
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.encDiscTrainOpt = tf.train.AdamOptimizer(self.learningRate, beta1=self.adamBeta1).minimize(self.encDLoss, var_list=self.encDiscVar)
			self.discTrainOpt = tf.train.AdamOptimizer(self.learningRate, beta1=self.adamBeta1).minimize(self.dLoss, var_list=self.discVar)
			self.encTrainOpt = tf.train.AdamOptimizer(self.learningRate, beta1=self.adamBeta1).minimize(self.encLoss, var_list=self.encVar)
			self.genTrainOpt = tf.train.AdamOptimizer(self.learningRate, beta1=self.adamBeta1).minimize(self.gLoss, var_list=self.genVar)

		# initialize saver
		self.saver = tf.train.Saver(max_to_keep=1)

	# train the model
	def train(self, sess, X, Y, label_names, label_colors, label_map_path, sampleOutputPath):

		# load model in case
		if(self.loadModel > -1):
			print("LOAD THE MODEL . . .")
			self.saver.restore(sess, './checkpoints/VAE_GAN-' + str(self.loadModel))
			print("DONE!")

		# if new model set tensorboard
		summaryOp = tf.summary.merge_all()
		summaryWriter = tf.summary.FileWriter("train/gan_{}".format(datetime.datetime.now().strftime("%s")), sess.graph)

		# get length of dataset
		nData = len(X)
		# sample noise Z to generate image - normal distributed
		sampleZ = np.random.normal(0,1, size=self.zDim)

		# start training
		self.step = 0
		self.losses = []

		# train bool
		trainEncD = True
		trainD = True

		# epoch loop
		startEpoch = int(self.loadModel) + 1
		for epoch in range(startEpoch, self.epochs):

			# print
			print("Starting epoch n {}/{} : ".format(epoch+1,self.epochs))

			# compute number of iteraton for an epoch
			nIter = nData
			# randomize indexes for batches
			batchAllIndexes = np.random.random_integers(0,nData-1,size=nData)

			# iter loop
			for iter in range(nIter):

				# increment step counter
				self.step += 1
				print("Starting iteration n {}/{} : ".format(iter+1,nIter))

				# get batch indexes
				batchIndexes = batchAllIndexes[iter]

				# get data and convert it to correct format
				thisX, thisY = h.processScene(X[batchIndexes], Y[batchIndexes], self.subVolumeShape)

				# apply some noise to the data
				noiseX = np.random.normal(loc=0, scale=0.001, size=self.subVolumeShape)
				noiseY = np.random.normal(loc=0, scale=0.001, size=self.subVolumeShape)
				thisX = np.add(thisX, noiseX)
				thisY = np.add(thisY, noiseY)
					
				# sample latent sample
				thisZ = np.random.normal(0,1, size=self.zDim)

				# UPDATE E
				_ = sess.run(self.encTrainOpt, feed_dict={self.x : thisX, self.y : thisY, self.z : thisZ})

				# UPDATE G
				_, thisSummary = sess.run([self.genTrainOpt, summaryOp], feed_dict={self.x : thisX, self.y : thisY, self.z : thisZ})

				# UPDATE D
				if(trainD):
					# sample random noise for D
					_ = sess.run(self.discTrainOpt, feed_dict={self.x : thisX, self.y : thisY, self.z : thisZ})

				# UPDATE ENC D
				if(trainEncD):
					_ = sess.run(self.encDiscTrainOpt, feed_dict={self.x : thisX, self.y : thisY, self.z : thisZ})

				# write summary
				summaryWriter.add_summary(thisSummary, self.step)

				# eval losses and at the end of each iter, get the losses and print them out
				encDiscTrainLoss = self.encDLoss.eval({self.x : thisX, self.y : thisY, self.z : thisZ})
				discTrainLoss = self.dLoss.eval({self.x : thisX, self.y : thisY, self.z : thisZ})
				encTrainLoss = self.encLoss.eval({self.x : thisX, self.y : thisY, self.z : thisZ})
				genTrainLoss = self.gLoss.eval({self.x : thisX, self.y : thisY, self.z : thisZ})
				encDAccuracy = self.encDiscAccuracy.eval({self.x : thisX, self.y : thisY, self.z : thisZ})
				dAccuracy = self.discAccuracy.eval({self.x : thisX, self.y : thisY, self.z : thisZ})
				print("Encoder Discriminator Accuracy : {:.2f}...".format(encDAccuracy))
				print("Discriminator Accuracy : {:.2f}...".format(dAccuracy))

				# check training next iteration
				trainEncD = True
				trainD = True
				if (encDAccuracy >= 0.8):
					trainEncD = False
				if (dAccuracy >= 0.8):
					trainD = False

				# print losses
				print("Encoder Discriminator Loss: {:.4f}...".format(encDiscTrainLoss),
					"Discriminator Loss: {:.4f}...".format(discTrainLoss),
					"Encoder Loss: {:.4f}".format(encTrainLoss),
					"Generator Loss: {:.4f}".format(genTrainLoss))

			# get sample
			thisSample, thisSampleLogit = sess.run( self.generator(self.z, reuse=True, isTraining=False), feed_dict={self.z : sampleZ})

			# visualize and save it
			sampleName = 'gen_' + str(epoch)
			h.save_data(thisSample[0], sampleName, './samples', label_names, label_colors, label_map_path, False)

			# save session
			print("SAVING SESSION! . . . ")
			self.saver.save(sess, './checkpoints/VAE_GAN', global_step=epoch)

		# return the losses
		return self.losses
