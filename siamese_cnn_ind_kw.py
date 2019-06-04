""" This program defines a siamese CNN model and trains it on the KPWhale_sp database in order to distinguish between sounds produced by individual killer whales.

    This is part of the following pipeline:
			
			download_audio_files.py        
				|_create_db.py 
					|_train_ResNet.py
			 	|_create_sp_db.py
			-->		|_train_siamese_net.py
                   
    
    Authors: Fabio Frazao
    contact: fsfrazao@gmail.com
     
    License: GPL3

	This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
"""

import tensorflow as tf
import numpy as np
from batch_gen import SiameseBatchGenerator
import random
import tables
import os

   

class SiameseCNNBranch(tf.keras.Model):
	""" Create one branch of the siamese network.

		There are 3 convolutional layers followed by  1 fully connected layers.
		Each branch works as an encoder

		Returns:
			branch: Model
				A model that will be used as one of the branches in a siamese network.
			
	"""
	
    def __init__(self):
        super(SiameseCNNBranch, self).__init__()
        self.seq = tf.keras.models.Sequential(name="siamese_branch")
        self.conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(10,10), use_bias=False,
                                            kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(2e-4))
        self.max_pool_1 = tf.keras.layers.MaxPooling2D()


        self.conv_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(7,7),
                                            kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                            bias_initializer=tf.random_normal_initializer())
        self.max_pool_2 = tf.keras.layers.MaxPooling2D()

        self.conv_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4),
                                            kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                            bias_initializer=tf.random_normal_initializer())
        self.max_pool_3 = tf.keras.layers.MaxPooling2D()

        self.conv_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4),
                                            kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                            bias_initializer=tf.random_normal_initializer())
        self.max_pool_3 = tf.keras.layers.MaxPooling2D()

        self.flatten = tf.keras.layers.Flatten()
        self.fully_connected = tf.keras.layers.Dense(512,kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                            bias_initializer=tf.random_normal_initializer())

    def call(self,inputs):
        branch = self.seq(inputs)
        branch = self.conv_1(branch)
        branch = tf.nn.relu(branch)
        branch = self.max_pool_1(branch)
        branch = self.conv_2(branch)
        branch = tf.nn.relu(branch)
        branch = self.max_pool_2(branch)
        branch = self.conv_3(branch)
        branch = tf.nn.relu(branch)
        branch = self.max_pool_3(branch)
        branch = self.flatten(branch)
        branch = self.fully_connected(branch)
        branch = tf.keras.activations.sigmoid(branch)

        return branch



class SiameseCNN(tf.keras.Model):
	""" Create a Siamese Network with 2 branches (instances of SiameseCNNBranch)

		Each of the branches works as an encoder. The encoding are than compared and the network is trained to tell whether or not two inputs (spectral representations of killer whale calls) were produced by the same individual.
		
		Return
			out: Model
				The Siamese Convolutional Network model
	"""
	

    def __init__(self):
        super(SiameseCNN, self).__init__()
                
        self.branch1 = SiameseCNNBranch()
        self.branch2 = SiameseCNNBranch()
        
        
        self.L1_layer = tf.keras.layers.Lambda(lambda tensors:tf.keras.backend.abs(tensors[0] - tensors[1]))
        self.final_fully_connected = tf.keras.layers.Dense(1,activation='sigmoid',bias_initializer=tf.random_normal_initializer())


    def call(self, input1, input2):
       
        branch1_in = self.branch1(input1)
        branch2_in = self.branch2(input2)
        
        out = self.L1_layer([branch1_in, branch2_in])
        out = self.final_fully_connected(out)
        
        return out

h5 = tables.open_file("sp_database.h5",'r')
train_table = h5.get_node("/train/kw")
val_table = h5.get_node("/val/kw")
test_table = h5.get_node("/test/kw")

#These are the identification codes for individual killer whales.
classes =[ 5, 6 ,7, 8, 9, 10, 11, 12, 13, 14, 22, 23, 24, 25, 26]
train_generator = SiameseBatchGenerator(train_table, batch_size=32, n_batches=100, classes=classes, y_field="wid", shuffle=True)
val_generator = SiameseBatchGenerator(test_table, batch_size=32, n_batches=100,  classes=classes, y_field="wid", shuffle=True)
test_generator = SiameseBatchGenerator(train_table, batch_size=32, n_batches=100, classes=classes, y_field="wid", shuffle=True)



model = SiameseCNN()
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(lr = 0.00006)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

checkpoint_path = "saved_models/Siamese_ind_kw/checkpoints"
os.makedirs(checkpoint_path, exist_ok=True)
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
model.load_weights(latest_checkpoint)
# model.compile(optimizer="Adam",loss="binary_crossentropy") 

@tf.function
def train_step(input1, input2, label):
  with tf.GradientTape() as tape:
    predictions = model(input1, input2)
    loss = loss_object(label, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(label, predictions)

@tf.function
def val_step(input1, input2, label):
  predictions = model(input1, input2)
  t_loss = loss_object(label, predictions)

  val_loss(t_loss)
  val_accuracy(label, predictions)

EPOCHS = 250
N_BATCHES = 100
BATCH_SIZE = 32

template_batch = 'Epoch {}, Batch {}, Loss: {:.3f}, Accuracy: {:.3f}'
template_epoch = 'Epoch {}, Loss: {:.3f}, Accuracy: {:.3f}, Val Loss: {:.3f}, Val Accuracy: {:.3f}'
for epoch in range(EPOCHS):
  
  for batch in range(N_BATCHES):  
    input_batch1, input_batch2, labels = next(train_generator)
    train_step(input_batch1, input_batch2, labels)

    print(template_batch.format(epoch+1,
                         batch,
                         train_loss.result(),
                         train_accuracy.result()*100))

  val_input_batch1, val_input_batch2, val_labels = next(val_generator)
  val_step(val_input_batch1, val_input_batch2, val_labels)
  
  print("\n==================================================================")
  print(template_epoch.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         val_loss.result(),
                         val_accuracy.result()*100))
  print("==================================================================\n")


checkpoint_name = "cp-{:04d}.ckpt".format(epoch)
model.save_weights(os.path.join(checkpoint_path, checkpoint_name))
