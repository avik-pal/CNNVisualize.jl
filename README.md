# CNNVisualize.jl

This Package implements popular CNN Visualization techniques and is built on top of `Flux.jl`. Most of the models from `Metalhead.jl` will work out of the box. To visualize custom models look at the documentation.

## Implemented Algorithms

1. Gradient Visualizations using Vanilla BackPropagation
2. Gradient Visualizations using DeconvNet
3. Gradient Visualizations using Guided BackPropagation
4. Gradient Weight Class Activation Maps
5. Guided Gradient Weight Class Activation Maps
6. DeepDream

## TODO

1. Class Specific Image Generation
2. CNN Filter Visualization
3. Inverted Image Representations
4. Smooth Grad
5. Adversarial Techniques
    * Fast Gradient Sign, Untargeted
    * Fast Gradient Sign, Targeted
    * Gradient Ascent, Adversarial Images
    * Gradient Ascent, Fooling Images (Unrecognizable images predicted as classes with high confidence)

## Some Notes

1. DeepDream Code in this repo is a direct copy of the code in my other repository [DeepDream.jl](https://github.com/avik-pal/DeepDream.jl). The code here demonstrates only a small fraction of what is actually implemented in the original repository. For more advanced operations like `guided dreams` and `video generation` have a look at the other repository

## References

[1] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. Striving for Simplicity: The All Convolutional Net, https://arxiv.org/abs/1412.6806

[2] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, A. Torralba. Learning Deep Features for Discriminative Localization, https://arxiv.org/abs/1512.04150

[3] R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, https://arxiv.org/abs/1610.02391

[4] K. Simonyan, A. Vedaldi, A. Zisserman. Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034

[5] A. Mahendran, A. Vedaldi. Understanding Deep Image Representations by Inverting Them, https://arxiv.org/abs/1412.0035

[6] H. Noh, S. Hong, B. Han, Learning Deconvolution Network for Semantic Segmentation https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Noh_Learning_Deconvolution_Network_ICCV_2015_paper.pdf

[7] A. Nguyen, J. Yosinski, J. Clune. Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images https://arxiv.org/abs/1412.1897

[8] D. Smilkov, N. Thorat, N. Kim, F. Viégas, M. Wattenberg. SmoothGrad: removing noise by adding noise https://arxiv.org/abs/1706.03825

[9] D. Erhan, Y. Bengio, A. Courville, P. Vincent. Visualizing Higher-Layer Features of a Deep Network https://www.researchgate.net/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network

[10] A. Mordvintsev, C. Olah, M. Tyka. Inceptionism: Going Deeper into Neural Networks https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html

[11] I. J. Goodfellow, J. Shlens, C. Szegedy. Explaining and Harnessing Adversarial Examples https://arxiv.org/abs/1412.6572

[12] I. J. Goodfellow, J. Shlens, C. Szegedy. Explaining and Harnessing Adversarial Examples https://arxiv.org/abs/1412.6572

[13] A. Nguyen, J. Yosinski, J. Clune. Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images https://arxiv.org/abs/1412.1897

This repo draws deep inspiration from https://github.com/utkuozbulak/pytorch-cnn-visualizations which implements similar algorithms in Pytorch
