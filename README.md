# KPWhale


This project was implemented as a proof of concept to demonstrate the utility of deep neural network model to a bioacoustic lab.

They had a collection of 15,000 audio clips in the their server. Each clip was approximately 3 seconds long and contained the sound produced by either a Killer Whale or Pilot Whale (a "call"). Each call was associated with one individual whale, which had and unique id.

![killer whale](_static/killer_whale.png "Killer Whale")  ![pilot whale](_static/pilot_whale.png "Pilot Whale")

I chose two tasks:

1) To distinguish between Pilot and Killer whales (using a ResNet) and
2) To match determine wether 2 killer whale calls were produced by the same individual (using a siamese CNN)


The pipeline included:
* Downloading the audio files from the server

  `download_data.py`
* Pre-processing each file to create an spectrogram and storing the processed samples in a database

  `create_db.py` and `create_sp_db.py`
* Implementing and training the Neural Networks

  `ResNet_sp.py` and `siamese_cnn_ind_kw.py`
  



The ResNet achieved an accuracy of *98.44%* on the test set and the Siamese CNN achieved *94.6%*.
