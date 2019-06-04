""" This program organizes the KPWhale database in a convenient structure to train a siamese network.

    It simply separates the spectrogram by species, so that it is easier to train a siamese network
	to distinguish indentify individual whales within each species by their sounds.


    This is part of the following pipeline:
			
			download_audio_files.py        
				|_create_db.py 
					|_train_ResNet.py
			--> 	|_create_sp_db.py
						|_train_siamese_net.py
            

    The HDF5 database has 3 groups(training, validation and test)
	Each group has two datasets:
		 kw:containing killer whale signals
		 pw: containing pilot whale signals

    The file is named KPWhale_sp_db.h5 and has the following structure
        KPWhale_sp_db.h5
         |---train
            |---kw
			|---pw
         |---validation
            |---kw
			|---pw
         |---test
            |---kw
			|---pw
    
    
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


import tables
from create_db import TableDescription

#The database containing both species
h5_orig = tables.open_file("KPWhale_db.h5", 'r')
h5_sp = tables.open_file("KPWhale_db.h5", 'w')



train_group = h5_sp.create_group("/", "train", createparents=True)
kw_train_table = h5_sp.create_table(train_group, "kw", TableDescription)
pw_train_table = h5_sp.create_table(train_group, "pw", TableDescription)
orig_train_table = h5_orig.get_node("/train/specs")

kw_orig_train = orig_train_table.where("sp == 1")
for spec in kw_orig_train:
    kw_train_row = kw_train_table.row
    kw_train_row['cid'] = spec['cid']
    kw_train_row['wid'] = spec['wid']
    kw_train_row['sp'] = spec['sp']
    kw_train_row['data'] = spec['data']
    kw_train_row.append()
kw_train_table.flush()

pw_orig_train = orig_train_table.where("sp == 2")
for spec in pw_orig_train:
    pw_train_row = pw_train_table.row
    pw_train_row['cid'] = spec['cid']
    pw_train_row['wid'] = spec['wid']
    pw_train_row['sp'] = spec['sp']
    pw_train_row['data'] = spec['data']
    pw_train_row.append()
pw_train_table.flush()



val_group = h5_sp.create_group("/", "val", createparents=True)
kw_val_table = h5_sp.create_table(val_group, "kw", TableDescription)
pw_val_table = h5_sp.create_table(val_group, "pw", TableDescription)
orig_val_table = h5_orig.get_node("/val/specs")

kw_orig_val = orig_val_table.where("sp == 1")
for spec in kw_orig_val:
    kw_val_row = kw_val_table.row
    kw_val_row['cid'] = spec['cid']
    kw_val_row['wid'] = spec['wid']
    kw_val_row['sp'] = spec['sp']
    kw_val_row['data'] = spec['data']
    kw_val_row.append()
kw_val_table.flush()

pw_orig_val = orig_val_table.where("sp == 2")
for spec in pw_orig_val:
    pw_val_row = pw_val_table.row
    pw_val_row['cid'] = spec['cid']
    pw_val_row['wid'] = spec['wid']
    pw_val_row['sp'] = spec['sp']
    pw_val_row['data'] = spec['data']
    pw_val_row.append()
pw_val_table.flush()


test_group = h5_sp.create_group("/", "test", createparents=True)
kw_test_table = h5_sp.create_table(test_group, "kw", TableDescription)
pw_test_table = h5_sp.create_table(test_group, "pw", TableDescription)
orig_test_table = h5_orig.get_node("/test/specs")

kw_orig_test = orig_test_table.where("sp == 1")
for spec in kw_orig_test:
    kw_test_row = kw_test_table.row
    kw_test_row['cid'] = spec['cid']
    kw_test_row['wid'] = spec['wid']
    kw_test_row['sp'] = spec['sp']
    kw_test_row['data'] = spec['data']
    kw_test_row.append()
kw_test_table.flush()

pw_orig_test = orig_test_table.where("sp == 2")
for spec in pw_orig_test:
    pw_test_row = pw_test_table.row
    pw_test_row['cid'] = spec['cid']
    pw_test_row['wid'] = spec['wid']
    pw_test_row['sp'] = spec['sp']
    pw_test_row['data'] = spec['data']
    pw_test_row.append()
pw_test_table.flush()


h5_orig.close()
h5_sp.close()

