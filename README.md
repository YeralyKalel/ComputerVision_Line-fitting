# ComputerVision_Line-fitting

Finds lines in the image

Procedural steps:
1) Loads the image
2) Finds edge points by running Canny edge detector and turning resultant matrix into binary map by some specified threshold value
3) Runs iteratively RANSAC algorithm to estimate lines using the edge points (Number of iteration = number of lines)
