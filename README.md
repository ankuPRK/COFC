# COFC
An Online Algorithm for Constrained Face Clustering in Videos

FACENET features represent faces very effectively using 108 dimensional features. We use the FACENET features, 
along with the must-link and cannot-link constraints from videos, for clustering the faces in an Online fashion.

Link to paper: https://github.com/ankuPRK/COFC/blob/master/ICIP/kulshreshtha_ICIP.pdf

The code used in ICIP publication was a bit complicated and scattered, with shot-detection using Johan Mathe's ShotDetect, face detection using the faces provided in the datasets, or using a C++ code, and the Online Clustering was in Python. This made the whole thing messy. So I have re-implemented the algorithm entirely in Python, and it can be run from commandline on a given video file.

Usage: 
$ git clone https://github.com/ankuPRK/COFC
$ cd COFC
$ python run_COFC_on_video.py --help

usage: run_COFC_on_video.py [-h] [-vp VID_PATH] [-sd SAVE_DIR]
                            [-ft FEAT_THRESH] [-ot OVERLAP_THRESH]
                            [-st SIM_THRESH]

optional arguments:
  -h, --help          show this help message and exit
  -vp VID_PATH        Path to the video file
  -sd SAVE_DIR        Directory path for saving the output
  -ft FEAT_THRESH     Threshold of distance bw features to belong to different
                      persons
  -ot OVERLAP_THRESH  Threshold of overlap above which two faces in
                      consecutive frames will belong to same track
  -st SIM_THRESH      Threshold of Similarity for facetracks to belong to a
                      cluster

One simple usage can be:

$ python run_COFC_on_video.py -vp <path_to_video> -sd <path_to_save_dir> -ft 1.0 -ot 0.85 -st 3.0
