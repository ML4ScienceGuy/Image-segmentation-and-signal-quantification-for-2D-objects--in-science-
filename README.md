# Image segmentation and signal quantification for 2D objects (in science)
 Using OpenCV, Scipy and Scikit Image, you will be able to quickly segment thousands of cells and extract intensity data for statistical analysis.
 
 
This is a two part tutorial on how to segment images obtained from a life science experiment, although for real-life images of objects with a clearly defined background and not too much noise, it can function also. The images were generated in Photoshop instead of lab microscopes, in as close to the real thing as possible. Only a control and a test set were used for simplicity.

This script is written in Python, with a mix of imaging tools including OpenCV, Scipy and Scikit Image. The script assumes you have, in one master folder, two or more folders where you store control and test image sets. Here, in our example, there are 4 images in each folder, each image with more than 10 cells, providing a decent statistical power. The images should have at least two channels, in this case green and red, representing fluorescent markers. The imagined scenario is that the red channel is relatively constant across all cells and conditions, and the green signal varies as a consequence of some chemical treatment. The experimenter would like to see if the green signal is ampliffied or decreased.

The script contains two phases:
A. Cell segmentation - cells are extracted in individual images and stored in their experimental folder together.
B. Image analysis - each cell is scanned for the mean green intensity normalized to the red channel, giving the experimenter the statistical results.

The inputs will be images with black backgorund and yellow to orange hues:

![yellow 1](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/4a42ac81-d4a7-4e0c-b303-053fcb532194)
![orange 1](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/43f5a73e-2cb6-43a5-915f-1e24ad1a16cf)


The outputs will be, for the two phases:
A. Cell segmentation:
1. Masks - simplified images of the original, for each image. They are produced using Otsu thresholding, and are used to separate individual cells from the original images.
<br >

![orange 2_mask](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/764d8a24-80ca-4cab-8865-1c7ace663a5b)

<br >
<br >
3. Split images - for each cell, an image with the original size but only one cell visible.
<br >

![orange 3 tif_object-1](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/7e11ec3f-9cd8-49e6-b019-a67ac265e5a7)

<br >
<br >
5. Cropped images - from each split image, a single cell is cropped by drawing a box around its extremities, and all black signal is subtracted, leaving only the color ranges for the markers inside that cell.
<br >

![orange 3 tif_object-1_cropped](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/0506f50e-6777-4e0c-a2e2-443da236d5ed)

<br >
<br >
B. Image analysis:
1. A single csv - statistics on data.csv - it contains data on minimum, mode and maximum values of pixel intensity for each channel, as well as the green mean intensity normalized to the red channel mean. The latter is used for the graphic statistical output.
2. A single image (outputs.png) with the violin plot of green mean intensities for the two experimental conditions. 


