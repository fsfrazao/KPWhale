import numpy as np

class BatchGenerator():
    """ Creates batches to be fed to a model

        Instances of this class are python generators. They will load one batch at a time from a HDF5 database, which is particularly useful when working with larger than memory datasets.
        Yield (X,Y) or (ids,X,Y) if 'return_batch_ids' is True. X is a batch of data as a np.array of shape (batch_size,mx,nx) where mx,nx are the shape of on instance of X in the database. Similarly, Y is an np.array of shape[0]=batch_size with the corresponding labels.

        Args:
            hdf5_table: pytables table (instance of table.Table()) 
                The HDF5 table containing the data
            batch_size: int
                The number of instances in each batch. The last batch of an epoch might have fewer examples, depending on the number of instances in the hdf5_table.
            instance_function: function
                A function to be applied to the batch, transforming the instances. Must accept 'X' and 'Y' and, after processing, also return  'X' and 'Y' in a tuple.
            x_field: str
                The name of the column containing the X data in the hdf5_table
            y_field: str
                The name of the column containing the Y labels in the hdf5_table
            shuffle: bool
                If True, instances are selected randomly (without replacement). If False, instances are selected in the order the appear in the database
            refresh_on_epoch: bool
                If True, and shuffle is also True, resampling is performed at the end of each epoch resulting in different batches for every epoch. If False, the same batches are used in all epochs.
                Has no effect if shuffle is False.
            return_batch_ids: bool
                If False, each batch will consist of X and Y. If True, the instance indeces (as they are in the hdf5_table) will be included ((ids, X, Y)).

            Attr:
                n_instances: int
                    The number of intances (rows) in the hdf5_table
                n_batches: int
                    The number of batches of size 'batch_size' for each epoch
                entry_indices:list of ints
                    A list of all intance indices, in the order used to generate batches for this epoch
                batch_indices: list of tuples (int,int)
                    A list of (start,end) indices for each batch. These indices refer to the 'entry_indices' attribute.
                batch_count: int
                    The current batch within the epoch
        
            
    """
    def __init__(self, hdf5_table, batch_size, instance_function=None, x_field='data', y_field='boxes', shuffle=False, refresh_on_epoch_end=False, return_batch_ids=False):
        self.data = hdf5_table
        self.batch_size = batch_size
        self.x_field = x_field
        self.y_field = y_field
        self.n_instances = self.data.nrows
        self.n_batches = int(np.ceil(self.n_instances / self.batch_size))
        self.shuffle = shuffle
        self.instance_function = instance_function
        self.entry_indices = self.__update_indices__()
        self.batch_indices = self.__get_batch_indices__()
        self.batch_count = 0
        self.refresh_on_epoch_end = refresh_on_epoch_end
        self.return_batch_ids = return_batch_ids

    
    def __update_indices__(self):
        """Updates the indices used to divide the instances into batches.

            A list of indices is kept in the self.entry_indices attribute.
            The order of the indices determines which instances will be placed in each batch.
            If the self.shuffle is True, the indices are randomly reorganized, resulting in batches with randomly selected instances.

            Returns
                indices: list of ints
                    The list of instance indices
        """
        indices = np.arange(self.n_instances)
        if self.shuffle:
            np.random.shuffle(indices)
        return indices

    def __get_batch_indices__(self):
        """Selects the indices for each batch

            Divides the instances into batchs of self.batch_size, based on the list generated by __update_indices__()

            Returns:
                list_of_indices: list of tuples
                    A list of tuple, each containing two integer values: the start and end of the batch. These positions refer to the list stored in self.entry_indices.                
        
        """
        ids = self.entry_indices
        n_complete_batches = int( self.n_instances // self.batch_size) # number of batches that can accomodate self.batch_size intances
        last_batch_size = self.n_instances % n_complete_batches
    
        list_of_indices = [list(ids[(i*self.batch_size):(i*self.batch_size)+self.batch_size]) for i in range(self.n_batches)]
        if last_batch_size > 0:
            last_batch_ids = list(ids[-last_batch_size:])
            list_of_indices.append(last_batch_ids)

        return list_of_indices

    def __iter__(self):
        return self

    def __next__(self):
        """         
            Return: tuple
            A batch of instances (X,Y) or, if 'returns_batch_ids" is True, a batch of instances accompanied by their indeces (ids, X, Y) 
        """

        batch_ids = self.batch_indices[self.batch_count]
        X = self.data[batch_ids][self.x_field]
        Y = self.data[batch_ids][self.y_field]

        self.batch_count += 1
        if self.batch_count > (self.n_batches - 1):
            self.batch_count = 0
            if self.refresh_on_epoch_end:
                self.entry_indices = self.__update_indices__()
                self.batch_indices = self.__get_batch_indices__()

        if self.instance_function is not None:
            X,Y = self.instance_function(X,Y)

        if self.return_batch_ids:
            return (batch_ids,X,Y)
        else:
            return (X, Y)



class SiameseBatchGenerator():
""" Creates batches to be fed to a model

        Instances of this class are python generators. They will load one batch at a time from a HDF5 database, which is particularly useful when working with larger than memory datasets.
        Yield (input_batch_1,input_batch_2, labels_batch). input_batch_1 and input_batch_2 are batches of data as a np.array. Similarly, labels is an np.array of shape[0]=batch_size with the corresponding labels telling whether or not the input pairs were produced by the same individual whale.

        Args:
            hdf5_table: pytables table (instance of table.Table()) 
                The HDF5 table containing the data
            batch_size: int
                The number of instances in each batch. The last batch of an epoch might have fewer examples, depending on the number of instances in the hdf5_table.
            instance_function: function
                A function to be applied to the batch, transforming the instances. Must accept 'X' and 'Y' and, after processing, also return  'X' and 'Y' in a tuple.
            x_field: str
                The name of the column containing the X data in the hdf5_table
            y_field: str
                The name of the column containing the Y labels in the hdf5_table
            shuffle: bool
                If True, instances are selected randomly (without replacement). If False, instances are selected in the order the appear in the database
            refresh_on_epoch: bool
                If True, and shuffle is also True, resampling is performed at the end of each epoch resulting in different batches for every epoch. If False, the same batches are used in all epochs.
                Has no effect if shuffle is False.
            

            Attr:
                n_instances: int
                    The number of intances (rows) in the hdf5_table
                n_batches: int
                    The number of batches of size 'batch_size' for each epoch
                entry_indices:list of ints
                    A list of all intance indices, in the order used to generate batches for this epoch
                batch_indices: list of tuples (int,int)
                    A list of (start,end) indices for each batch. These indices refer to the 'entry_indices' attribute.
                batch_count: int
                    The current batch within the epoch
        
            
    """
    
    def __init__(self, hdf5_table, batch_size, n_batches, instance_function=None, x_field='data', y_field='sp', classes=[1,2], shuffle=False, refresh_on_epoch_end=False, return_batch_ids=False):
        self.data = hdf5_table
        self.batch_size = batch_size
        self.x_field = x_field
        self.y_field = y_field
        self.classes = classes
        self.class_coord = self.__get_class_coordinates__()        
        self.n_instances = self.data.nrows
        self.n_batches = n_batches
        self.n_same = int(self.batch_size/2)
        self.n_diff = int(self.batch_size/2)
        self.shuffle = shuffle
        self.instance_function = instance_function
        self.batch_count = 0
        self.refresh_on_epoch_end = refresh_on_epoch_end
        self.return_batch_ids = return_batch_ids


    def __get_class_coordinates__(self):
        class_coord = {}
        for input_class in self.classes:
            condition = "{} == {}".format(self.y_field, input_class)
            class_coord[input_class] = self.data.get_where_list(condition)
            
        return class_coord



    def __get_same_pair__(self, chosen_class):
        first_input = np.random.choice(self.class_coord[chosen_class])
        second_input = np.random.choice(self.class_coord[chosen_class])
        target = 1

        return (first_input, second_input, target)


    def __get_diff_pair__(self, chosen_class):    
        first_input = np.random.choice(self.class_coord[chosen_class])
        other_classes = [c for c in self.classes if c != chosen_class]
        second_class = np.random.choice(other_classes)
        second_input = np.random.choice(self.class_coord[second_class])
        target = 0

        return (first_input, second_input, target)
    
    def __get_batch_indices__(self):
        """Selects the indices for one batch


            Returns:
                list_of_indices: list of tuples
                    A list of tuples, each containing
                     three integer values: the coodinates (row number) for the first input,
                     the coordinates for the second input and the target value (1 if the inputs
                     belong to the same class, 0 if not).
        
        """
        
        list_of_indices=[]
        for same in range(self.n_same):
            same_chosen_class = np.random.choice(self.classes)
            list_of_indices.append(self.__get_same_pair__(chosen_class=same_chosen_class))
        for diff in range(self.n_diff):
            diff_chosen_class = np.random.choice(self.classes)
            list_of_indices.append(self.__get_diff_pair__(chosen_class=diff_chosen_class))
        return list_of_indices

    def __iter__(self):
        return self

    def __next__(self):
        """         
            Return: tuple of numpy arrays
            A batch of instances (input_batch_1,input_batch_2, labels_batch) 
        """

        batch_ids = self.__get_batch_indices__()
        input1_ids = [ids[0] for ids in batch_ids]
        input2_ids = [ids[1] for ids in batch_ids]
        input1_arrays = self.data.read_coordinates(input1_ids)[:][self.x_field]
        input2_arrays = self.data.read_coordinates(input2_ids)[:][self.x_field]
        labels = np.array([ids[2] for ids in batch_ids])


        n_inputs, width, height = input1_arrays.shape
        input_batch_1 = input1_arrays.reshape((n_inputs, width, height, 1))
        input_batch_2 = input2_arrays.reshape((n_inputs, width, height, 1))
        labels_batch = labels.reshape(labels.shape[0], 1)

        if self.shuffle:
            shuffle_ids = np.random.choice(list(range(n_inputs)), size=n_inputs, replace=False)
            input_batch_1 = input_batch_1[shuffle_ids]
            input_batch_2 = input_batch_2[shuffle_ids]
            labels_batch = labels_batch[shuffle_ids]

        return input_batch_1, input_batch_2, labels_batch        
        
      

       