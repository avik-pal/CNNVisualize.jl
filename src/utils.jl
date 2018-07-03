function get_topk(probs, k = 5)
  T = eltype(probs)
  prob = Array{T, 1}()
  idx = Array{Int, 1}()
  while(k!=0)
    push!(idx, indmax(probs))
    push!(prob, probs[idx[end]])
    probs[idx[end]] = 0.0
    k -= 1
  end
  (prob, idx)
end

function one_hot_encode(preds, idx)
  one_hot = zeros(eltype(preds.data), size(preds)[1], 1)
  one_hot[idx ,1] = 255.0
  one_hot |> gpu
end

function save_gradient_images(gradient, file_name)
  gradient = gradient - minimum(gradient)
  gradient /= maximum(gradient)
  gradient = permutedims(squeeze(gradient, 4), (3, 2, 1))
  img = colorview(RGB{eltype(gradient)}, gradient)
  display(img)
  save(file_name, img)
end

function save_grayscale_gradient(gradient, file_name)
  gradient = gradient - minimum(gradient)
  gradient /= maximum(gradient)
  gradient = permutedims(squeeze(gradient, 4), (3, 2, 1))
  img = colorview(RGB{eltype(gradient)}, gradient)
  img = Gray.(img)
  display(img)
  save(file_name, img)
end

function positive_negative_saliency(gradient)
  pos_saliency = max.(zero(gradient), gradient) / maximum(gradient)
  neg_saliency = max.(zero(gradient), -gradient) / maximum(-gradient)
  (pos_saliency, neg_saliency)
end

im2arr_rgb(img) = permutedims(float.(channelview(imresize(img, (224, 224)))), (3, 2, 1))

im_mean = reshape([0.485, 0.456, 0.406], 1, 1, 3) |> gpu
im_std = reshape([0.229, 0.224, 0.225], 1, 1, 3) |> gpu

# The below code is a directly copied from https://github.com/avik-pal/DeepDream.jl
# There might be some code redundancy

function image_to_arr(img; preprocess = true)
  local x = img
  x = Float32.(channelview(img))
  x = permutedims(x, [3,2,1]) |> gpu
  if(preprocess)
    x = (x .- im_mean)./im_std
  end
  x = reshape(x, size(x,1), size(x,2), size(x,3), 1)
end

function load_image(path, resize = false; size_save = true)
  img = load(path)
  if(size_save)
    global original_size = size(img)
  end
  if(resize)
    image_to_arr(imresize(img, (224, 224)))
  else
    image_to_arr(img)
  end
end

function generate_image(x, resize_original = false)
  x = reshape(x, size(x)[1:3]...)
  x = x .* im_std .+ im_mean
  x = clamp.(permutedims(x, [3,2,1]), 0, 1) |> cpu
  if resize_original
    imresize(colorview(RGB, x), original_size)
  else
    colorview(RGB, x)
  end
end

save_image(path, x) = save(path, generate_image(x, true))
