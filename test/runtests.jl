using CNNVisualize, Metalhead

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

img = load_image("../images/original/cat_dog.png")
model = VGG19()

@testset "BackPropagation" begin
  @test_nowarn begin
    vbprop = VanillaBackprop(model.layers)
    grads = vbprop(img)
    pgrads, ngrads = positive_negative_saliency(grads[1][1])
  end

  @test_nowarn begin
    gbprop = GuidedBackprop(model.layers)
    grads = gbprop(img)
  end

  @test_nowarn begin
    deconv = Deconvolution(model.layers)
    grads = deconv(img, 2) # Test in case multiple visualizations are to be generated
  end

  save_gradient_images(grads[1][1], "../images/results/save1.png")
  save_grayscale_gradient(grads[2][1], "../images/results/save1.png")
end

@testset "Grad CAM" begin
  @test_nowarn begin
    gcam = GradCAM(model.layers, 12)
    grads = gcam(img)
    save_gradcam(grads[1][1], "../images/original/cat_dog.png", "../images/results/save2.png", "../images/results/save3.png", "../images/results/save4.png")
  end

  @test_warn "target should be less than the model length" GradCAM(model.layers, length(model.layer.layers) + 5)

  @test_nowarn begin
    ggcam = GuidedGradCAM(model.layers, 12)
    grads = ggcam(img)
  end
end
