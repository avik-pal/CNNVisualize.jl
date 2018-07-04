__precompile__()

module CNNVisualize

using Flux, Plots, Images, Augmentor
using GR
using Flux: Tracker
using Flux.Tracker: data

clibrary(:colorcet)

include("backpropagation.jl")
include("gradcam.jl")
include("deepdream.jl")
include("utils.jl")

export GradCAM, GuidedGradCAM, Backprop, VanillaBackprop, GuidedBackprop,
       Deconvolution, save_gradient_images, save_gradcam, save_grayscale_gradient,
       positive_negative_saliency, im2arr_rgb, image_to_arr, load_image, save_image,
       load_model, deepdream, make_step

# Some re exports

export save, load, cpu, gpu

end # module
