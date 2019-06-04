""" This program creates spectrograms from the KPWhale audio files and stores them in and HDF5 database.

    This database will be used to train a neural network to distinguish between two species:
	Pilot whales and Killer whales.


    This is part of the following pipeline:
			
			download_audio_files.py        
		--> 	|_create_db.py 
					|_train_ResNet.py
					|_create_sp_db.py
						|_train_siamese_net.py
            

    The HDF5 database has 3 groups(training, validation and test):
    The file is named KPWhale_db.h5 and has the following structure
        KPWhale_db.h5
         |---train
            |---specs
         |---validation
		    |---specs
         |---test
		    |---specs
    
    
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



from ketos.audio_processing.audio import AudioSignal
from ketos.audio_processing.spectrogram import MagSpectrogram
from skimage.transform import resize 
import multiprocessing as mp
import random
import tables
import librosa
import os
from tqdm import tqdm


class TableDescription(tables.IsDescription):
	"""Describe the fields in an HDF5 table.
	
	Attributes:
		cid: IntCol
			An in teger field for the unique call id numbers
		wid: IntCol
			An integer field for the whale id numbers. Each whale has an unique identifier
		sp: IntCol
			An integer column for the species code. 1=killer whale, 2=Pilot whale
		data: Float32Col
			A float32 2d array for the spectral representation of each whale call
	"""
			
            cid = tables.IntCol()
            wid = tables.IntCol()
            sp = tables.IntCol()
            data = tables.Float32Col((200,200))

def parse_seg_name(seg_name):
	"""Parse the file name containing the call.

	   From the file name, obtain the call id, the species and the whale id.

	Args:
		seg_name: string
			The file name including the '.mp3' extension.
			Example: 'cid_96_sp_pw_wid_15.mp3'

	Returns:
		dict:
		A dictionary containing
	    'cid'(call id): integer identifying the call
		'sp'(species code): integer identifying the species (1=Killer Whale, 2=Pilot Whale)
		'wid' (whale id): integer identifying the whale that produced that call
 
	"""
	
    sp_key = {"kw":1, "pw":2}
    
    split_str = seg_name.split("_")
    cid = split_str[1]
    sp = split_str[3]
    wid = split_str[5].split(".mp3")[0]

    return {"cid":cid, "sp":sp_key[sp], "wid":wid}


def create_spec(seg_name):
	""" Create a magnitude spectrogram from an audio file.
	

	Obs:The spectrogram is created using a window length of 0.085 seconds and window step of 0.002.
		Spectrograms are cropped at 8 KHz and resized to 200 x 200 pixels.
		All audio signals are resampled to 44100 Hz and spectrograms.
		
	Args:

		seg_name: string
			The audio file name including extension
			
	Returns
		rimg: numpy array
		A 2d array of shape (200,200) containing the magnitude spectrogram for the given audio file.
		
	"""
    sig, rate = librosa.load(os.path.join(path_to_segs, seg_name), sr=44100)
    audio = AudioSignal(rate=rate, data=sig) 
    spec = MagSpectrogram(audio,  winlen=0.085, winstep=0.002)
    spec.crop(fhigh=8000)
    img = spec.image
    rimg = resize(img, (200,200))
	rimg = rimg.T

    return rimg

def write_spec(table, spec, cid, wid, sp):
	""" Write a spetrogram to a HDF5 database

	Args:
		table: pytables table
			The table where the spectrograms will be saved
		spec: numpy array
			A 2D array containing the spectrogram
		cid: integer
			The call id associated with the spectrogram
		wid: integer
			The whale id associated with the spectrogram
		sp: integer
			The species code (1 for killer whales, 2 for pilot whales)

	"""
    row = table.row
    row['data'] = spec
    row['cid'] = cid
    row['sp'] = sp
    row['wid'] = wid
    row.append()


def populate_table(table, list_of_segs, pool):
	""" Populate HDF5 database with spectrograms

	Obs: Uses multiprocessing to speed up the database creation.

	Args:
		table: pytables table
			The HDF5 table where the spectrograms will be saved
		list_of_segs: list of strings
			The list of audio file names from which the spectrograms will be created.
		pool: multiprocessing pool
			A pool object to enable multiprocessing

		
	"""
	
   
    n = len(list_of_segs)
    n_sets = 100
    specs_per_set = n // n_sets
    last_set =  specs_per_set + (n % n_sets)


    for s in tqdm(range(n_sets)):
        n_specs = specs_per_set
        if s == (n_sets - 1):
            n_specs = last_set

        sublist_of_segs = list_of_segs[s*n_specs:(s*n_specs+n_specs)]
        spec_set = pool.map(create_spec, [sublist_of_segs[s_id] for s_id in range(n_specs)])

        for i, spec in enumerate(spec_set):
            info = parse_seg_name(sublist_of_segs[i])
            write_spec(table=table, spec=spec, cid=info['cid'], sp=info['sp'], wid=info['wid'])
    table.flush()



if __name__ == '__main__':

	"""Split all the audio segments available into 3 groups:
		Training (10000 samples)
		Validation (2000 samples)
		Test (~3000 samples

		Create a HDF5 database with these groups, each containing one dataset with the following fields:
	
		cid: IntCol
			An in teger field for the unique call id numbers
		wid: IntCol
			An integer field for the whale id numbers. Each whale has an unique identifier
		sp: IntCol
			An integer column for the species code. 1=killer whale, 2=Pilot whale
		data: Float32Col
			A float32 2d array for the spectral representation of each whale call
			
	"""

    path_to_segs = "/media/fsfrazao/data/KPWhale/audio"

    seg_names = os.listdir(path_to_segs)
    random.shuffle(seg_names)

    pool = mp.Pool(mp.cpu_count())
    h5 = tables.open_file("KPWhale_db.h5", 'a')
        


    train_list = seg_names [:10000]
    train_group = h5.create_group("/", "train", createparents=True)
    train_table = h5.create_table(train_group, "specs", TableDescription)

    populate_table(train_table, train_list, pool)

    val_list = seg_names [10000:12000]
    val_group = h5.create_group("/", "val", createparents=True)
    val_table = h5.create_table(val_group, "specs", TableDescription)

    populate_table(val_table, val_list, pool)

    test_list = seg_names [12000:]
    test_group = h5.create_group("/", "test", createparents=True)
    test_table = h5.create_table(test_group, "specs", TableDescription)

    populate_table(test_table, test_list, pool)

    
