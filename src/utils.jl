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
