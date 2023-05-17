# Image segmentation and signal quantification for 2D objects (in science)
 Using OpenCV, Scipy and Scikit Image, you will be able to quickly segment thousands of cells and extract intensity data for statistical analysis.
 
 
This is a two part tutorial on how to segment images obtained from a life science experiment, although for real-life images of objects with a clearly defined background and not too much noise, it can function also. The images were generated in Photoshop instead of lab microscopes, in as close to the real thing as possible. Only a control and a test set were used for simplicity.

This script is written in Python, with a mix of imaging tools including OpenCV, Scipy and Scikit Image. The script assumes you have, in one master folder, two or more folders where you store control and test image sets. Here, in our example, there are 4 images in each folder, each image with more than 10 cells, providing a decent statistical power. The images should have at least two channels, in this case green and red, representing fluorescent markers. The imagined scenario is that the red channel is relatively constant across all cells and conditions, and the green signal varies as a consequence of some chemical treatment. The experimenter would like to see if the green signal is ampliffied or decreased.

The script contains two phases:
A. Cell segmentation - cells are extracted in individual images and stored in their experimental folder together.
B. Image analysis - each cell is scanned for the mean green intensity normalized to the red channel, giving the experimenter the statistical results.

The inputs will be images with black backgorund and yellow to orange hues:

![yellow 1](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/de1059cc-c2ab-474c-b066-7ad8c5eb5c6e)
![orange 1](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/43f5a73e-2cb6-43a5-915f-1e24ad1a16cf)

The images should have as little noise as possible for this script to work well, for reasons that will be explained bellow, and of course not be over-saturated, to propperly compare the signal intensities between image sets.


The outputs will be, for the two phases: <br >
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
An important note on input images: The background should be kept to a minimum by selecting out the noise in the microscope settings. That is because the program will count the small pixels as blobs to count, which slows down the scanning process. In contrast to the cleaner image used for the mask above, this next input an mask pair visibly create confusion in the thresholding step:
![yellow 4](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/0afa7094-608b-43e4-9b5a-584b28c31f35)
<br >
<br >
<br >
B. Image analysis:<br >
1. A single csv - statistics on data.csv - it contains data on minimum, mode and maximum values of pixel intensity for each channel, as well as the green mean intensity normalized to the red channel mean. The latter is used for the graphic statistical output.<br >
2. A single image (outputs.png) with the violin plot of green mean intensities for the two experimental conditions. <br>

![outputs](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/653f5ae4-4c74-4092-8670-5749f0420ff1)

 <br >
 
![yellow 4_mask](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/887683f6-9a31-45b8-b6e6-e00094be5b91)

<br >
<br >
Image processing takes place in the following way:
1. The RGB image channels are split and stored as numpy arrays.
2. Histograms are calculated for each channel. Here are some possible scenarios (visualization is possible inside the script):

![overlapped](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/a498aad0-7bcd-484f-989f-edfe94da9818)
(ratio of green/red ~1)<br >

![shifted](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/c1e97a73-0cbe-45c2-a9a1-2ee043069be2)
(ratio slightly lower than 1)<br >

![large shift](https://github.com/ML4ScienceGuy/Image-segmentation-and-signal-quantification-for-2D-objects--in-science-/assets/47111504/54ff4ddb-74ca-44ed-9bee-8fa587cb2a26)
(ratio of much lower than 1)<br ><br >

5. Means are calculated for the histograms
6. The green histogram mean intensity is normalized to the red histogram mean intensity.
7. Mean green intensities are compared between datasets.

The principle behind normalization is that experimental setup (cell position on slide or inside the focal field of the image) can alter the green color channel (of interest here), and the red channel can serve as a standard for correction. The channel corrected mean now can be used to accurately determine if the chemical added to the experimental condition 1 or 2 had an effect. The statistics are not included here but can be easily added or created using the statistics csv generated.

If you plan to use any part of this script, please cite using the cff file details.
