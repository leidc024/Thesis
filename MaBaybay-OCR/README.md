# MaBaybay: A Matlab-based Baybayin Optical Character Recognition Package

<img src="https://github.com/rbp0803/MaBaybay/blob/main/MaBaybayIconV3.png" align="left" width="130px"/>

### This work aims to recognize Baybayin scripts from individual characters to a whole block of texts. The software offers a user-friendly graphical user interface (GUI) console with a simple design, allowing for easy use and interaction with the application.


![Alt text](https://github.com/rbp0803/MaBaybay/blob/main/MaBaybayOCR_console.jpg)


## Operating Environments and Dependencies
System:
* Windows 8 and later

Software:
* MATLAB versions R2023b
* Statistics and Machine Learning Toolbox
* Image Processing Toolbox
* Computer Vision Toolbox
* OCR Language Data Files (Tagalog)
* Parallel Computing Toolbox
  
_**Note:** The posted source codes here are only compatible with MATLAB versions R2023a and later since the [ocr](https://www.mathworks.com/help/vision/ref/ocr.html) built-in function from Mathworks was recently updated in version R2023a. Visit the [Release Section](https://github.com/rbp0803/MaBaybay/releases/tag/v1.0) if you may opt to access the MaBaybay-OCR codes using older MATLAB versions (R2019a-R2022b)._

## Usage
* /Algorithms folder contains essential code functions to implement the recognition program. Due to the upload maximum size (25 MB) restriction per file, the Baybayin characters SVM classier and the Baybayin GUI executable file are deposited in the Release section: https://github.com/rbp0803/MaBaybay/releases/tag/v1.0.   
* Upon acquisition of the complete package (please ensure that the classifier is also in the same folder), simply run the `MaBaybayOCR_App.mlapp` file to access and explore the Baybayin GUI.
* /Filipino Word Corpus folder contains a csv file that holds a list of 74,400+ Filipino words. Words recorded from this file will be considered legitimate by the system.
* /Sample images folder contains 20 sample images that the user can opt to try. 

## References

For each system's detailed discussion, we refer the readers to the following:
 1. Pino R, Mendoza R, Sambayan R. 2021. Optical character recognition system for Baybayin scripts using support vector machine. PeerJ Comput. Sci. 7:e360 http://doi.org/10.7717/peerj-cs.360
 2. Pino R, Mendoza R, Sambayan R. 2021. A Baybayin word recognition system. PeerJ Comput. Sci. 7:e596 http://doi.org/10.7717/peerj-cs.596 
 3. Pino R, Mendoza R, Sambayan R. 2022. Block-level optical character recognition system for automatic transliterations of baybayin texts using support vector machine. Philipp. J. Sci. 151 (1), 303-315, https://doi.org/10.56899/151.01.23.

## Acknowledgement
We would like to acknowledge the implementation of an external MATLAB function (<i> insertInImage <i>) being used in our program for embedding the bounding boxes and the transliterated words into the image.
  1. Brett Shoelson (2019, October 23). insertInImage, MATLAB Central File Exchange. Retrieved (Jan 2024) from : https://www.mathworks.com/matlabcentral/fileexchange/38721-embed-text-and-graphics-in-an-image
     

## Queries

Should you have any concerns, feel free to contact me: rbpino@up.edu.ph
