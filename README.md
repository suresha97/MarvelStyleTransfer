# StyleTrasfer

As described in the original paper on neural style transfer (https://arxiv.org/pdf/1508.06576.pdf), the basic 
idea is to use neural networks (specifically CNNs) to generate content from one image
in the style of another, resulting in strange and fantasic works of art. I wanted to try this out for 
myself to understand it a bit better.

Using a pre-trained VGG-19 net, these were some of the experiments I ran:
- Varying image size.
- Varying learning rate.
- Relative importance of style and content losses.
- Higher weight to style loss componenets from latter layers vs. higher weight to earlier layers.
- Use content loss from shallow vs. deep layers.
- Adam vs. LBFGS optimiser.
- Initialise generated image to content image or style image rather than a white noise image.
- Standardise images.
- Replace max pools with average pooling.


Here is an example. I wanted to generate the image of Tony Stark bravely sacrificing himself, in the style of the abstract
space man below:

![alt-text-1](images/iron_man.jpg "title-1") ![alt-text-2](images/style5.jpeg "title-2")

The result: 

![Test Image !](images/iron_man_generated.png)

