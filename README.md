# Image segmentation and signal quantification for 2D objects (in science)
 Using OpenCV, Scipy and Scikit Image, you will be able to quickly segment thousands of cells and extract intensity data for statistical analysis.
 
 
This is a two part tutorial on how to segment images obtained from a life science experiment, although for real-life images of objects with a clearly defined background and not too much noise, it can function also. The images were generated in Photoshop instead of lab microscopes, in as close to the real thing as possible. Only a control and a test set were used for simplicity.

This script is written in Python, with a mix of imaging tools including OpenCV, Scipy and Scikit Image. The script assumes you have, in one master folder, two or more folders where you store control and test image sets. Here, in our example, there are 4 images in each folder, each image with more than 10 cells, providing a decent statistical power. The images should have at least two channels, in this case green and red, representing fluorescent markers. The imagined scenario is that the red channel is relatively constant across all cells and conditions, and the green signal varies as a consequence of some chemical treatment. The experimenter would like to see if the green signal is ampliffied or decreased.

The script contains two phases:
A. Cell segmentation - cells are extracted in individual images and stored in their experimental folder together.
B. Image analysis - each cell is scanned for the mean green intensity normalized to the red channel, giving the experimenter the statistical results.

The inputs will be images with black backgorund and yellow to orange hues:

![yellow 1](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/de1059cc-c2ab-474c-b066-7ad8c5eb5c6e)
![orange 1](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/cdeb43c2-085a-4025-b3c4-a0a732f46d5f)


The images should have as little noise as possible for this script to work well, for reasons that will be explained bellow, and of course not be over-saturated, to propperly compare the signal intensities between image sets.


The outputs will be, for the two phases: <br >
A. Cell segmentation:
1. Masks - simplified images of the original, for each image. They are produced using Otsu thresholding, and are used to separate individual cells from the original images.
<br >

![orange 2_mask](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/0594529f-4f9b-427f-81b1-c0021d2ab7ed)


<br >
<br >
3. Split images - for each cell, an image with the original size but only one cell visible.
<br >

![orange 3 tif_object-1](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/e9ed2710-27fc-4484-8a46-616fade4263e)


<br >
<br >
5. Cropped images - from each split image, a single cell is cropped by drawing a box around its extremities, and all black signal is subtracted, leaving only the color ranges for the markers inside that cell.
<br >

![orange 3 tif_object-1_cropped](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/a7544373-26c1-4b36-8527-2a1ba429f0af)


<br >
<br >
An important note on input images: The background should be kept to a minimum by selecting out the noise in the microscope settings. That is because the program will count the small pixels as blobs to count, which slows down the scanning process. In contrast to the cleaner image used for the mask above, this next input an mask pair visibly create confusion in the thresholding step:
![yellow 4](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/9d7f7ad5-d41b-4e7e-86db-d522375ca8c4)
![yellow 4_mask](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/434dc3d5-1adf-4124-a42d-cfd42cfceb0d)

<br >
<br >
<br >
B. Image analysis:<br >
1. A single csv - statistics on data.csv - it contains data on minimum, mode and maximum values of pixel intensity for each channel, as well as the green mean intensity normalized to the red channel mean. The latter is used for the graphic statistical output.<br >
2. A single image (outputs.png) with the violin plot of green mean intensities for the two experimental conditions. <br>

![outputs](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/544ac713-243d-480a-aa3a-fefbc6603fa3)

<br >
<br >
Image processing takes place in the following way:
1. The RGB image channels are split and stored as numpy arrays.
2. Histograms are calculated for each channel. Here are some possible scenarios (visualization is possible inside the script):

![overlapped](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/8e6320f0-62cd-4f1f-a7d1-86d3a25cb320)

(ratio of green/red ~1)<br >

![shifted](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/ec7531ca-8a97-4a3f-b309-8d03efc1ead8)

(ratio slightly lower than 1)<br >

![large shift](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/fe702e7e-a5d4-4d5f-aa18-2a955a1ead6f)

(ratio of much lower than 1)<br ><br >

5. Means are calculated for the histograms
6. The green histogram mean intensity is normalized to the red histogram mean intensity.
7. Mean green intensities are compared between datasets.

The principle behind normalization is that experimental setup (cell position on slide or inside the focal field of the image) can alter the green color channel (of interest here), and the red channel can serve as a standard for correction. The channel corrected mean now can be used to accurately determine if the chemical added to the experimental condition 1 or 2 had an effect. The statistics are not included here but can be easily added or created using the statistics csv generated.

If you plan to use any part of this script, please cite using the cff file details.
