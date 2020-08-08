# Compare Feature Installation

Download the [opensmile](https://www.audeering.com/opensmile/) toolkit. You will find comprehensive guidance on how to build and use openSMILE inside of the downloaded directory in doc/openSMILE_book.pdf. For quick installation on various operating systems, you can refer to INSTALL inside of the openSMILE downloaded directory.

## Quick Installation on MacOS without portAudio

### Use the following commands in the terminal to do the installation:
1. `tar -zxvf openSMILE-2.x.x.tar.gz`
2. `cd openSMILE-2.x.x`
3. `sed -i -e 's/-lrt//' buildStandalone.sh`, this fixes the issue of missing librt on macOS.
4. `sh buildStandalone.sh`, This will configure, build, and install the openSMILE binary SMILExtract to the inst/bin subdirectory. Add this directory to your path. You can also use this instead: sh buildStandalone.sh -p /path/to/install/to
  
### Another way to do it:
1. `tar -zxvf openSMILE-2.x.x.tar.gz`
2. `cd openSMILE-2.x.x`
3. Run this twice `sh autogen.sh` (You must run autogen.sh a second time in order to have all necessary files created!)
4. `make -j4`
5. `make`
6. `make install`
  
## How to use it
1. Place [generate_compare_feats.sh](https://github.com/wazeerzulfikar/ad-mmse/blob/master/generate_compare_feats.sh) script in the directory you want to create the compare features inside.
2. Change the paths as described in the comments of the [generate_compare_feats.sh](https://github.com/wazeerzulfikar/ad-mmse/blob/master/generate_compare_feats.sh) script.
3. Run the script using the terminal using `sh generate_compare_feats.sh`, and you should automatically get compare features (inside the directory you placed the script in) for all audio files you provided the path to in the script (for each audio file a csv file is created with its compare features).    

***Voila! You have all your compare features!***



