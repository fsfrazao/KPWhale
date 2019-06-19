""" This program defines a ResNet model and trains it on the KPWhale database in order to distinguish between sounds produced by pilot whales and killer whales.


    This is part of the following pipeline:
			
			download_audio_files.py        
				|_create_db.py 
			-->		|_train_ResNet.py
			 	|_create_sp_db.py
					|_train_siamese_net.py
                   
    
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
from batch_gen import BatchGenerator
import numpy as np
import tables
import datetime
import os

def to1hot(sp):
    """ 1hot encode the sp(species) label  

	Args:
		sp: integer
			The species code (1 for killer whale, 2 for pilot whale)
	Returns:
		one_hot: numpy array
			The corresponding 1x2 1hot encoded vector	
	"""
    one_hot = np.zeros(2)
    one_hot[sp-1]=1.0
    return one_hot

def transform_batch(x,y):
	""" Function to be applied to each batch of data before it is fed to the network

		Reshape the input batch and 1hot encode the labels.

	Args:
		x: numpy array
			Array containing the batch of inputs passed by the a BatchGenerator instance.
		y: numpy array
			Array containing the batch of labels passed by the a BatchGenerator instance.

	Return:
		(X,Y): tuple containing the transformed x and y batches
	"""
    X = x.reshape(x.shape[0],x.shape[1], x.shape[2],1)
    Y = np.array([to1hot(sp) for sp in y])
    return (X,Y)


class ResNetBlock(tf.keras.Model):
	"""Create a ResNet block

		A full path has the following layers:
			Batch Normalization + ReLU activation			
			2D Convolution
			Batch Normalization + ReLU activation			
			2D Convolution
			
	   Args:
			channels: int
				The number of channels used by each Convolutional Layer
			strides: int
				The number of cells/pixels used for stride by the filters in the first convolutional layer
			residual_block: bool
				If true, build a block with residual path (with only one 1x1 convolutional layer.
				If false, build a full block (i.e.: without skip connections).

	Return: model
		A residual block. Full path + inputs if 'residual_block' is false. Full path + residual path if true.
				
	"""

    def __init__(self, channels, strides=1, residual_path=False):
        super(ResNetBlock, self).__init__()

        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=(3,3), strides=self.strides,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=(3,3), strides=1,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

        if residual_path == True:
            self.conv_down = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=(1,1), strides=self.strides,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
            self.batch_norm_down = tf.keras.layers.BatchNormalization()

    def call(self,inputs, training=None):
        residual = inputs

        x = self.batch_norm_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1(x)
        x = self.batch_norm_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_2(x)

        if self.residual_path:
            residual = self.batch_norm_down(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.conv_down(residual)

        x = x + residual
        return x


class ResNet(tf.keras.Model):
	""" Create a ResNet model

	Args:

		block_list: list of integers
			A list describing how many blocks each residual layer will have. Ex [2,3] A set of 2 residual blocks followed by a set of 3 residual blocks.

		n_classes: integer
			The number of classes the model will learn to classify

		initial_filters: integer
			How many filters the first Convolutional layer will have.
	
	Return: Model
		A ResNet model with the specified residual block structure.

		
		
	"""

    def __init__(self, block_list, n_classes, initial_filters=16, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.n_blocks = len(block_list)
        self.n_classes = n_classes
        self.block_list = block_list
        self.input_channels = initial_filters
        self.output_channels = initial_filters
        self.conv_initial = tf.keras.layers.Conv2D(filters=self.output_channels, kernel_size=(3,3), strides=1,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())

        self.blocks = tf.keras.models.Sequential(name="dynamic_blocks")

        for block_id in range(self.n_blocks):
            for layer_id in range(self.block_list[block_id]):
                #Frst layer of every block except the first
                if block_id != 0 and layer_id == 0:
                    block = ResNetBlock(self.output_channels, strides=2, residual_path=True)
                
                else:
                    if self.input_channels != self.output_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = ResNetBlock(self.output_channels, residual_path=residual_path)

                self.input_channels = self.output_channels

                self.blocks.add(block)
            
            self.output_channels *= 2

        self.batch_norm_final = tf.keras.layers.BatchNormalization()
        self.average_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fully_connected = tf.keras.layers.Dense(self.n_classes)

    def call(self, inputs, training=None):

        output = self.conv_initial(inputs)

        output = self.blocks(output, training=training)
        output = self.batch_norm_final(output, training=training)
        output = tf.nn.relu(output)
        output = self.average_pool(output)
        output = self.fully_connected(output)

        return output



if __name__ == "__main__":

    model = ResNet(block_list = [2, 2, 2], n_classes=2)


    opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=opt,
                loss=loss_function,
                metrics=['accuracy'])

    model.build(input_shape=(None, 200, 200, 1))

    log_dir="logs/fit/mine_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    tensorboard_callback.set_model(model)

    def name_logs(model, logs, prefix="train_"):
        named_logs = {}
        for l in zip(model.metrics_names, logs):
            named_logs[prefix+l[0]] = l[1]
        return named_logs

    checkpoint_path = "saved_models/ResNet_sp/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    model.load_weights(latest_checkpoint)

    h5 = tables.open_file("database.h5",'r')
    train_table = h5.get_node("/train/specs")
    val_table = h5.get_node("/val/specs")
    test_table = h5.get_node("/test/specs")


    train_generator = BatchGenerator(train_table, 128, transform_batch, y_field="sp", shuffle=True, refresh_on_epoch_end=True )
    val_generator = BatchGenerator(val_table, 128, transform_batch, y_field="sp",  shuffle=True, refresh_on_epoch_end=True )
    test_generator = BatchGenerator(train_table, 128, transform_batch, y_field="sp", shuffle=True, refresh_on_epoch_end=True )


    metrics_names = model.metrics_names

    for epoch in range(20):
        #Reset the metric accumulators
        model.reset_metrics()
            
        for train_batch_id in range(train_generator.n_batches):
            train_X, train_Y = next(train_generator)  
            train_result = model.train_on_batch(train_X, train_Y)
            

            print("train: ",
                "Epoch:{} - batch:{} | {}: {:.3f}".format(epoch, train_batch_id, model.metrics_names[0], train_result[0]),
                "{}: {:.3f}".format(model.metrics_names[1], train_result[1]))
        for val_batch_id in range(val_generator.n_batches):
            val_X, val_Y = next(val_generator)
            val_result = model.test_on_batch(val_X, val_Y, 
                                        # return accumulated metrics
                                        reset_metrics=False)

        tensorboard_callback.on_epoch_end(epoch, name_logs(model, train_result, "train_"))    
        tensorboard_callback.on_epoch_end(epoch, name_logs(model, val_result, "val_"))    
        if epoch % 5:
            checkpoint_name = "cp-{:04d}.ckpt".format(epoch)
            model.save_weights(os.path.join(checkpoint_path, checkpoint_name))

        print("\neval: ",
                "{}: {:.3f}".format(model.metrics_names[0], val_result[0]),
                "{}: {:.3f}".format(model.metrics_names[1], val_result[1]))


    tensorboard_callback.on_train_end(None)
    
    
