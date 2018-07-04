#------------------------Grad CAM--------------------------

struct GradCAM
  model::Chain
end

function save_gradcam(gradient, original_image_path, grad_file_name, heatmap_file_name, combined_file_name)
  gradient = max.(gradient, zero(gradient))
  gradient = permutedims((gradient - minimum(gradient))/(maximum(gradient) - minimum(gradient)), (2, 1))
  img = Gray.(gradient)
  img = imresize(img, 224, 224)
  display(img)
  h1 = plot(heatmap(float.(augment(img, FlipY())), color = :rainbow))
  try
    display(h1)
  catch
    info("The heatmap could not be displayed. The file will be saved at $heatmap_file_name")
  end
  mapped = float.(channelview(load(original_image_path))) * 0.9 .+ reshape(float.(img), 1, 224, 224) * 3
  mapped -= minimum(mapped)
  mapped /= maximum(mapped)
  h2 = plot(heatmap(float.(Gray.(augment(colorview(RGB{eltype(mapped)}, mapped), FlipY()))), color = :rainbow))
  try
    display(h2)
  catch
    info("The heatmap could not be displayed. The file will be saved at $combined_file_name")
  end
  save(grad_file_name, img)
  try
    savefig(h1, heatmap_file_name)
    savefig(h2, combined_file_name)
  catch
    info("Encountered Errors while trying to save heatmaps. So the heatmaps are being returned")
    info("They need to be saved manually")
    (h1, h2)
  end
end

function save_gradient(x)
  global target_output = x |> cpu
  x
end

save_gradient(x::TrackedArray) = Tracker.track(save_gradient, x)

normalize_grad(grad) = grad / (sqrt(mean(abs2.(grad))) + eps())

function Tracker.back(::typeof(save_gradient), Δ, x)
  global guided_gradients = Δ |> cpu
  Tracker.@back(x, Δ)
end

function GradCAM(model::Chain, target::Int)
  length(model.layers) - 1 < target && error("target should be less than the model length")
  updated_model = []
  for (i, l) in enumerate(model.layers[1:end-1])
    push!(updated_model, l)
    i == target && push!(updated_model, save_gradient)
  end
  GradCAM(Chain(updated_model...) |> gpu)
end

function (m::GradCAM)(img, top = 1)
  if !Tracker.istracked(img)
    img = param(img) |> gpu
  end
  preds = m.model(img)
  probs = softmax(preds)
  prob, inds = get_topk(probs.data, top)
  grads = []
  for (i, idx) in enumerate(inds)
    Flux.back!(preds, one_hot_encode(preds, idx))
    weights = reshape(maximum(normalize_grad(guided_gradients), [1, 2, 4]), 1, 1, size(target_output)[3], 1)
    cam = ones(size(target_output)[1:2])
    cam += squeeze(sum(weights .* target_output, [3, 4]), dims = (3, 4))
    push!(grads, (cam, prob[i], idx))
    img.grad .= 0.0
  end
  grads
end

#--------------------Guided Grad CAM-----------------------

struct GuidedGradCAM
  gcam::GradCAM
  backprop::Backprop
end

GuidedGradCAM(model::Chain, target::Int) = GuidedGradCAM(GradCAM(model, target), GuidedBackprop(model))

function (m::GuidedGradCAM)(img, top = 1)
  # Ideally we should not be doing the forward pass twice
  gcam_grads = m.gcam(img, top)
  backprop_grads = m.backprop(img, top)
  grads = []
  for i in 1:top
    normalized_camgrads = gcam_grads[i][1] - minimum(gcam_grads[i][1])
    normalized_camgrads /= maximum(normalized_camgrads)
    normalized_camgrads = reshape(float.(imresize(Gray.(normalized_camgrads), 224, 224)), 224, 224, 1, 1)
    normalized_bpropgrads = backprop_grads[i][1] - minimum(backprop_grads[i][1])
    normalized_bpropgrads /= maximum(normalized_bpropgrads)
    push!(grads, (normalized_camgrads .* normalized_bpropgrads, gcam_grads[i][2], gcam_grads[i][3]))
  end
  grads
end
