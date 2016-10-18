# Deep Convolutional Variational Autoencoder w/ Adversarial Network

A combination of the DCGAN implementation and the variational autoencoder. Currently, the model is set up to upsample and downsample 128x128 color images, with three color channels. The code can, however, be easily modified to accept any number of different dimensions or color channel combinations. I may modify the script to allow for this to be automated, depending on the level of interest.  

## Prerequisites 
..1. Torch7
..2. CUDA
..3. CUDNN
..4. DPNN
..5. 

To run as a variational autoencoder, simply execute the script using 

``` th daliVAE.lua -i [input folder destination] -o [output folder destination] -s [size of dataset (number of image files)] -c [destination for saving model checkpoints]
```

where the input folder is expected to contain 128x128 color images. Output folder will be for generated samples from the model, although if you want to reconstruct test samples, this can be done by adding a test tensor and running it through a forward pass on the model. 
