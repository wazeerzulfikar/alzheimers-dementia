#!/bin/bash

for entry in /path/to/audio/files/*; #change this to the path of the directory that contains the audio files then put this /* to loop on all the files
do
  fn=${entry%.*}

  # change the first part of this path to the path of the opensmile directory you downloaded during installation
  SMILExtract -C /path to/opensmile-2.3.0/config/IS13_ComParE.conf -I "$entry" -O "compare_${fn##*/}.csv"
done
