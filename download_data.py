""" This program doenloads the KPWhale audio files. 
	
	Each of the 15,000+ files contains one whale call produced by either a killer or Pilot whale.
	The individuals producing the calls were also identified.

	The audio data will be processed into spectrograms and stored in HDF5 databases that will feed Neural networks to perform 2 tasks:
	1) Distinguish between Killer and Pilot whales (ResNet)
	2) Given 2 calls, determine if they were produced by the same individual (Siamese CNN)

    This is part of the following pipeline:
			
			--> download_audio_files.py        
				|_create_db.py 
					|_train_ResNet.py
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

import urllib
import os
import pandas as pd
from time import sleep


audio_dir = "data/audio"
#spec_dir = "data/spec"
#spec_12_dir = "data/spec_12"

links = pd.read_csv("whale_fm_anon_04-03-2015_assets.csv")

def extract_call_data(row):
	""" Extract the relevant data from a row in the spreadsheet.
	
	Args:
		row: pandas series
			One row from the links dataframe
	Return:
		call_data: dict
		A dictionary with the following keys:
			"call_id":integer identifying the call,
            "audio_link":string containing the link to the audio file in the server,
            "species":string identifying the whale species ('killerwhale' or 'pilotwhale'),
            "whale_id":integer identifying the individual whale associated with that call

	"""
    call_data={"call_id":row['id'],
            "audio_link":row['location'],
            "spec_link":row['spectrogram'],
            "spec_link_12":row['spectrogram_12'],
            "species":row['whale_type'],
            "whale_id":row['whale_id']}
    return call_data

def create_name(call_data):
	""" Create a file name based on the data for that call

			
	Return:
		name: string
			A new file name including the call id, the species and the individual id.
			Ex:"cid_105_sp_kw_wid_12"
	"""
    sp_codes={'pilotwhale':'pw','killerwhale':'kw'}
    species = sp_codes[call_data['species'].replace(' ','')] 
    name = "cid_{0}_sp_{1}_wid_{2}".format(call_data['call_id'],
                                            species,
                                            call_data['whale_id'])
    return name

def download_audio(call_data):
	""" Download the audio file for one call.

	Args:
		call_data: dict
		A dictionary containing call data (as return by the 'extract_call_data' function). 

	"""
    file_name = create_name(call_data) + ".mp3"
    file_path = os.path.join(audio_dir, file_name)
    urllib.request.urlretrieve(call_data['audio_link'],        
                               file_path)


if __name__ == "__main__":

    for i,call in links[2000:].iterrows(): # the first 2000 lines of the spreadsheet do not contain links, but notes on data collection.
        call_data = extract_call_data(call)
        try:        
            download_audio(call_data)
            print("Downloading call {0}".format(i))

            sleep(0.3) # Wait a little between requests to avoid causing problems to the server
        except:
            print("Something went wrong with call {0}".format(i))            
