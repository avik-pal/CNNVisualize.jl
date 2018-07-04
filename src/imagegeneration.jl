struct ImageGenerator
  model
  img
  target_class::Int
end

function ImageGenerator(model, target_class::Int)
  target_class > 1000 && error("Invalid Target Class")
  img = param(rand(224, 224, 3, 1) * 255) |> gpu
  ImageGenerator(model |> gpu, img, target_class)
end

function (IG::ImageGenerator)(save_dir, img_name = "generated"; niters::Int = 150, lr = 6.0)
  isdir(save_dir) || error("Storage must be a directory")
  mask = zeros(1000, 1) |> gpu
  mask[IG.target_class, 1] = 255.0
  local mean_img = reshape([0.485, 0.456, 0.406], 1, 1, 3, 1) .* 255 |> gpu
  local std_img = reshape([0.229, 0.224, 0.225], 1, 1, 3, 1) .* 255 |> gpu
  for i in 1:niters
    IG.img.data .= (IG.img.data .- mean_img) ./ std_img
    outputs = IG.model(IG.img)
    loss = Flux.Tracker.data(outputs) .* mask
    println("Loss after Iteration $i is $(sum(loss))")
    Flux.back!(outputs, loss)
    IG.img.data .= IG.img.data .+ lr * IG.img.grad/sqrt(mean(abs2.(IG.img.grad)))
    save("$(save_dir)/$(img_name)_iteration_$i.png", generate_image(Flux.Tracker.data(IG.img)))
  end
end
