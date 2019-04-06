# **Welcome to the optic-nerve training example**


## Description:
This repo is meant to provide an overview of basics for practical implementation of deep learning techniques in biological image analysis challenges.


# ** Usage notes**


## Installation:
### Windows installation (~15 - 30 mins)
  #### 1.	Install Anaconda
  * Download here: [https://www.anaconda.com/download/](https://www.anaconda.com/download/)
  * Select "Python 3.7 version" and the corresponding "64-bit" or "32-bit" processor version
  * Follow the instructions
     
  #### 2. Install Python
  * In Windows Start Menu, search for "Anaconda Prompt" and select it
  * In the black command window that pops up, type:
  
        conda install python=3.6
  
  * Since Tensorflow is only compatible with Python 3.6, please ensure that you enter the command above to install Python 3.6 alongside the default 3.7 in anaconda.
  * enter "y" if prompted
       
  #### 3.	Install packages
  In Anaconda command-terminal, type:
  
        pip install natsort opencv-python
        
  Then
  
        conda install -c https://conda.anaconda.org/conda-forge mahotas 
   
   For tensorflow installation:

        pip install tensorflow
       
  #### 4.	Download files
  * Navigate to home-page of this repository again
  * On the right-hand side, click the green button to "Clone or download ZIP file" of repo
  * Download ZIP and extract all files
  * Save anywhere on your computer
         
   
### **Mac installation:**

  #### 1. Check to update python version
  * ensure version 3.6
  * if not, find a downloadable version here: [https://www.python.org/downloads/release/python-368/](https://www.python.org/downloads/release/python-368/)

  #### 2.	Install packages
  Open a command-terminal and type:
  
      pip3 install numpy pillow scipy matplotlib natsort scikit-image opencv-python tensorflow
      
  To install the final package, mahotas, you will need to first install xcode:
  
      xcode-select --install
  
  A pop-up will jump out after the command above. Follow the instructions to install. Then type:
  
      pip3 install mahotas
      

  #### 3.	Download files
  * Navigate to home-page of this repository again
  * On the right-hand side, click the green button to "Clone or download ZIP file" of repo
  * Download ZIP and extract all files
  * Save anywhere on your computer
  

## Usage:
  ### 1.	Data format
   *  Please ensure all images are “.tiff” format
   *	Channels are NOT separated
   *  The stained sheaths (either MBP or O4) are in the **RED** channel.
   *  Cell nuclei are in the **BLUE** channel
   *	All files to be analyzed are located in a SINGLE folder (see "Demo-data" folder for example)

  ### 2.	Run main file
  1. (a) For Anaconda (Windows):
  
      * Search for "Spyder" in Windows search bar
      * Open the file "main_UNet.py" using Spyder
      * run by pressing the green run button
      
  1. (b) For Mac (command console):
  
      * In command console type:
           
           python3 main_UNet.py
  

  ### 3. Understanding the output/results:
  Under the directory you selected to save all files, you should find:
  
  For examples of these files, check under "Results/Demo-data-output/"
    
## Demo run:
  ### Run the "main_UNet.py" file by following the directions in "Usage" above
  * when prompted with GUI, select the following folders:
      *  create your own folder, then select it for output folder
      * "Demo-data" for input folder      


## Troubleshooting:
1.  Recommended computational specifications:
    * > 8GB RAM

2.	If program does not run or computer freezes:
    * Check the size of your images. If they are larger than 5000 x 5000 pixels, you may need to:
        * move to a computer with RAM > 8 GB
        * crop your image and analyze half of it at a time
        
3.  If you would like to use your own checkpoint from training:
    * navigate to the folder “Checkpoints” and replace the files with your own checkpoint files.
    
    
4. If the analysis is not picking up any sheaths, but they are evident to the human eye, consider enhancing the contrast in your images, either manually using Fiji, or an alternative algorithm like CLAHE.


