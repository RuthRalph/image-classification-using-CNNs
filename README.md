# image-classification-using-CNNs.
The other image classification used MLP, among its limitations, it flatten images into long 1D vectors before it can be fed to the network. Thus, the model loses all sense of locality and spatial hierarchy.  Why CNNs? Convolutional Neural Networks (CNNs) are designed for structured grid-like data like images.

What I did here: I explored two classical CNNs that laid the foundation for modern deep vision (LeNet & AlexNet):
Implement LeNet — one of the earliest CNNs, by Yann LeCun.
Build a simplified AlexNet — a deeper network that won ImageNet 2012.

Component	  Purpose
Conv2d	    Detects local patterns in image patches
ReLU	      Enables non-linear transformations
MaxPool2d	  Reduces size, adds robustness
Linear	    Final decision-making (classification)

LeNet Architecture and Shape Transformations
Layer	   Input Shape	  Output Shape	   Notes
Input	   32×32×3	            —	         RGB image (CIFAR-10)
Conv1	   32×32×3	      28×28×6	         6 filters, 5×5 kernel, stride=1
Pool1	   28×28×6	      14×14×6	         2×2 max pooling
Conv2	   14×14×6	      10×10×16	       16 filters, 5×5 kernel, stride=1
Pool2	   10×10×16	      5×5×16	         2×2 max pooling
Flatten	 5×5×16	        400	             Flatten 3D tensor to 1D vector
FC1	     400	          120	             Fully connected, followed by ReLU
FC2	     120	          84	             Fully connected, followed by ReLU
Output	 84	            10	             Output logits for 10 CIFAR-10 classes
