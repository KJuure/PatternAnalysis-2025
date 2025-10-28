# Recognition Tasks
Various recognition tasks solved in deep learning frameworks.

Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion


OASIS Improved 2D UNet - Project 1

Day 1:

First started by asking ChatGPT to produce framework and code to solve the problem. To do so, I provided ChatGPT the project description file. Once it provided a solution, I studied the given solution to learn how it works and to see if it was suitable.

I asked ChatGPT to provide step by step instructions on implementing the given code including proper file directories and any additional required scripts.

Then I created the text files as well as the script requried to generate the list of ID names for the algorithm to use to pull images.


Day 2:

I have written the code for dataset.py and I am now studying and confirming the code is correct.

After spending an hour reviewing dataset.py, I have discovered ChatGPT does a lot of edge testing and confirmation. I have removed all the ones I have deemed unneccessary such as checking if the file directories are in "OASIS PNG" or "flat" layout, and alterations to the code which allows for both layouts. 

After studying the code further, I've created a remapping json file for the mask images and confirmed the remapping is correct with the help of ChatGPT.

I have now fully implemented the dataset.py file and have confirmed the functionality with ChatGPTs help.


I just finished implementing modules.py and I am now verifying functionality while also studying what each line of code does and its purpose in the algorithm.


Day 3:

Implemented, studied and validated the utils.py file. I'm discovering pretty quickly that although ChatGPT gives a lot of information, its pretty bad at explaining concepts from the very bottom and often includes complex language.

Implemented train.py. Same issue as before, ChatGPT includes so many safety nets like making a list of possible class names when importing from another class.

Came across a new issue - CUDA is for nvidia GPUS only. Now I'm wrangling with ChatGPT to help me rewrite the code to support AMD. This entire process is causing so many errors which I have to deal with one by one.

I've rewritten the concerned code to support DML.