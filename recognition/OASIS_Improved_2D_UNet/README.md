# OASIS 2D Brain Segmentation using Improved 2D UNet

## Report

### Problem & Objective:
This project (Project 1) performs **2D brain tissue segmentation** on the OASIS PNG slices using an **Improved U-Net** algorithm.  
The target for this project was to achieve **Dice similarity coefficient greater than 0.90 for each label** on the test set. The mean Dice similarity coefficient was **0.97** with the lowest being Cerberal Spinal Fluid at **0.92**.

Improved UNet works by first compressing the input with an encoder to learn broad context of the image, this is done via convolution. In 4-5 stages, the image is downsampled allowing each kernel to "see" a larger portion of the original image since each pixel now depicts more than just 1 pixel on the original image - thus learning context (classification based on a pixel and what surrounds it). At each stage, the learned information is refined and stored.  Then once compressed, it reconstructs the information to full resolution. The decoder upsamples the feature map and reinjects the information stored during each step in the encoding stage. At the end, the final map features are converted to per pixel classes. The process of taking an image, downsampling and upsampling is where the name UNet comes from as the resolution of the map resembles a U shape.

![Diagram depicting UNet Algorithm.](UNet_Visual.jpg?raw=true)

The left prong of the "U" depicts the downsampling / encoding stage and the right prong depicts the upsampling / decoding stage. The arrows indicate the way stored information is injected from each stage while decoding.

### Dataset & Layout:
Expected OASIS PNG layout:
```
<data_dir>/
keras_png_slices_train/ # images (train)
keras_png_slices_validate/ # images (val)
keras_png_slices_test/ # images (test)
keras_png_slices_seg_train/ # masks (train)
keras_png_slices_seg_validate/ # masks (val)
keras_png_slices_seg_test/ # masks (test)
```

Training is done through `train.py` and taking `config` as an argument. The specific config file is under `json\train_config.JSON`
Segmentation is done through `predit.py` with `config` as an argument. The specific config file is under `json\predict_config.JSON`

These config files were used to avoid having to type all arguments into terminal at execution.
The default values set for `predict.py` are:
```
{
  "ids": "splits/val.txt",
  "weights": "runs/oasis_unet_improved/best.pt",
  "out_dir": "runs/oasis_unet_improved/val_report",
  "data_dir": "data",
  "num_classes": 4,
  "remap_json": "json/remap_labels_oasis_png.json",
  "base_ch": 32,
  "use_se": true,
  "deep_supervision": true,
  "dropout": 0.1,
  "batch": 4,
  "workers": 0,
  "device": "dml",
  "class_names": "bg,csf,gm,wm"
}
```

Yielding the following results:

![Histogram of Dice Similarity Coefficient of the 4 classes Background (bg), Cerebrospinal Fluid (csf), Grey Matter (gm), and White Matter (wm).](runs/oasis_unet_improved/val_report/plots/dice_bar.png?raw=true)

![Histogram of Intersection over Union (IoU) of the 4 classes Background (bg), Cerebrospinal Fluid (csf), Grey Matter (gm), and White Matter (wm).](runs/oasis_unet_improved/val_report/plots/iou_bar.png?raw=true)

![Image of the all test cases. Left image - Original Slice, Middle Image - True Answer, Right Image - Model Answer](runs/oasis_unet_improved/val_report/plots/overlay_grid.png?raw=true)


### Table of results:

| Class               | Dice   | IoU    | Precision (% of Correct Predictions) | Recall (% of found classes) |
| ------------------- | ------ | ------ | ------------------------------------ | --------------------------- |
| Background          | 0.9994 | 0.9989 | 99.94                                | 99.94                       |
| Cerebrospinal Fluid | 0.9630 | 0.9286 | 96.96                                | 95.65                       |
| Grey Matter         | 0.9693 | 0.9402 | 96.46                                | 97.39                       |
| White Matter        | 0.9819 | 0.9645 | 98.38                                | 98.00                       |
| Mean                | 0.9784 | 0.9581 |                                      |                             |


### Pre-Processing:

* **Normalisation**: After each slice is converted to `float32`, they are normalised to `[0, 1]`. This stabalised optimisation and makes brightness intensities more comparable between slices.

* **Lable Remapping**: Colours are remapped to class IDs for so all operations work on class numbers rather than colours.

* **Tensor Shapes**: Slices are configured to shape `(1, H, W)` to be used with convolve. Masks configured to `(H, W)` with integer class IDs for easier comparison in the model

* **Augmentation**: In training, flips and rotations are used to reduce overfitting and to make the model more robust.

### Splits:
The data was split distinctly into training, validating, and testing. No images were shared between the 3 folders to reduce any bias due to having seen the image before. Typically, a ratio of `80/10/10` is used to split the data between training, validating and testing.
In this model, a split of approximately `85/10/5` was used. This split places more emphasis on training and gives the model better accuracy overall. The 10 percent validation was used to retain the accuracy and stability while the 5 percent testing allowed for faster predictions though risking slightly less precise testing.


### Requirements and Reproducabiliy:
Library requirements are outlined in `requirements.txt`.
Beyond the in-built libraries, the model uses:
* **numpy**
* **pillow**
* **matplotlib**
* **tqdm**
* **torch**

*Note: This model was written and tested using an AMD PC. The code is written to be compatibile with NVIDA, AMD and CPU computations, but results may vary between the 3.*

## Work Log:

### Day 1:

First started by asking ChatGPT to produce framework and code to solve the problem. To do so, I provided ChatGPT the project description file. Once it provided a solution, I studied the given solution to learn how it works and to see if it was suitable.
I asked ChatGPT to provide step by step instructions on implementing the given code including proper file directories and any additional required scripts.
Then I created the text files as well as the script requried to generate the list of ID names for the algorithm to use to pull images.


### Day 2:

I have written the code for dataset.py and I am now studying and confirming the code is correct.
After spending an hour reviewing dataset.py, I have discovered ChatGPT does a lot of edge testing and confirmation. I have removed all the ones I have deemed unneccessary such as checking if the file directories are in "OASIS PNG" or "flat" layout, and alterations to the code which allows for both layouts. 
After studying the code further, I've created a remapping json file for the mask images and confirmed the remapping is correct with the help of ChatGPT.
I have now fully implemented the dataset.py file and have confirmed the functionality with ChatGPTs help.


I just finished implementing modules.py and I am now verifying functionality while also studying what each line of code does and its purpose in the algorithm.


### Day 3:

Implemented, studied and validated the utils.py file. I'm discovering pretty quickly that although ChatGPT gives a lot of information, its pretty bad at explaining concepts from the very bottom and often includes complex language.
Implemented train.py. Same issue as before, ChatGPT includes so many safety nets like making a list of possible class names when importing from another class.
Came across a new issue - CUDA is for nvidia GPUS only. Now I'm wrangling with ChatGPT to help me rewrite the code to support AMD. This entire process is causing so many errors which I have to deal with one by one.
I've rewritten the concerned code to support DML.

Just finished writing train.py and testing it. Same issue as earlier, lots of code to be rewritten to support DML while also maintainging CUDA and CPU compatibility. But now that is all done, so I will leave the model to train.

### Day 4:

Training is done and now I am implementing predict.py.
Just ran predict.py and the results are fantastic.
AI is kinda crazy.