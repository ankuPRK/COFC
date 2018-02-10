# OnlineFaceClustering
An Online Algorithm for Constrained Face Clustering in Videos

# Abstract
We address the problem of face clustering in long, real world videos. This is a challenging task because faces in such videos exhibit wide variability in scale, pose, illumination, expressions, and may also be partially occluded. The majority of the existing face clustering algorithms are offline, i.e., they assume the availability of the entire data at once. However, in many practical scenarios, complete data may not be available at the same time or may be too large to process or may exhibit significant variation in the data distribution over time. We propose an online clustering algorithm that processes data sequentially in short segments (of variable length), and groups the detected faces in each segment either to an existing cluster or by creating a new one. Our algorithm uses several temporal constraints, and a convolutional neural network (CNN) to obtain a robust representation of the faces in order to achieve high clustering accuracy on two benchmark databases (82.1% and 93.8%). Despite being an online method (usually known to have lower accuracy), our algorithm achieves comparable or better results than several existing offline methods.

# Resources:
Download the resources (Python 2.7 compatible) for BF0502 Dataset from: https://drive.google.com/file/d/1Ve-yFGqfpj2TfIxfn2PUxJTIyDY7-CLu/view?usp=sharing
Extract the .7z file in the same location as the original code. The resource consists of following files:
1) buffy_tl.npy: Track-Ids and labels for all 17737 faces as np.array format ()
2) file_Names_buffy.txt: File names are changed to contain the information of shot-id, frame-id, and then face no in that frame.
3) labels_buffy.txt: Ground truth label for each of the faces
4) loc_info_buffy.txt: Rectangular co-ordinates for each of the faces
5) ls_feats_list_buffy: An 17337x128 dimensional np array containing 128-dimensional FACENET features for each of the faces
6) shot_info_buffy.txt: The file contains the shot-begin frame, followed by the length of shot (in frames count)


# Code
The code consists of three files:
1) GMM_Update_Uni.py: The model for updating the Centroids. It can also be easily modified to use Gaussians instead of just normal mean. However, in our case Gaussian was performing poorer. 
2) 
