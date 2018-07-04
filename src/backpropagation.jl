#--------------------General Utilities---------------------

# Just a hack to add the guided relu functions and replace the relu
# Resnet model is not supported
# Google Net is partially supported
function change_activation(model::Chain, activation)
  updated_model = []
  for (i, l) in enumerate(model.layers)
    T = typeof(l)
    if T <: Dense
      push!(updated_model, Dense(l.W, l.b, identity))
      push!(updated_model, activation)
    elseif T <: Conv
      push!(updated_model, Conv(identity, l.weight, l.bias, l.stride, l.pad, l.dilation))
      push!(updated_model, activation)
    elseif T <: BatchNorm
      push!(updated_model, BatchNorm(identity, l.β, l.γ, l.μ, l.σ, l.ϵ, l.momentum, l.active))
      push!(updated_model, activation)
    elseif T <: Chain
      push!(updated_model, change_activation(l, activation))
    # elseif T<: Bottleneck # To handle the DenseNet Model
    #   push!(updated_model, Bottleneck(change_activation(l.layer, activation)))
    else
      push!(updated_model, l)
    end
  end
  Chain(updated_model...)
end

#--------------------BackPropagation----------------------

struct Backprop
  model::Chain
end

function (m::Backprop)(img, top = 1)
  if !Tracker.istracked(img)
    img = param(img) |> gpu
  end
  preds = m.model(img)
  probs = softmax(preds)
  prob, inds = get_topk(probs.data, top)
  grads = []
  for (i, idx) in enumerate(inds)
    Flux.back!(preds, one_hot_encode(preds, idx))
    push!(grads, (img.grad |> cpu, prob[i], idx))
    img.grad .= 0.0
  end
  grads
end

#-----------------Vanilla BackPropagation-----------------

VanillaBackprop(model::Chain) = Backprop(model[1:end-1] |> gpu)

#------------------Guided BackPropagation-----------------

guided_relu1(x) = max.(zero(x), x)

guided_relu1(x::TrackedArray) = Tracker.track(guided_relu1, x)

Tracker.back(::typeof(guided_relu1), Δ, x) = Tracker.@back(x, Int.(x .> zero(x)) .* max.(zero(Δ), Δ))

function GuidedBackprop(model::Chain)
  model = change_activation(model, guided_relu1)
  Backprop(model[1:end-1] |> gpu)
end

#---------------------Deconvolution-----------------------

guided_relu2(x) = max.(zero(x), x)

guided_relu2(x::TrackedArray) = Tracker.track(guided_relu2, x)

Tracker.back(::typeof(guided_relu2), Δ, x) = Tracker.@back(x, max.(zero(Δ), Δ))

function Deconvolution(model::Chain)
  model = change_activation(model, guided_relu2)
  Backprop(model[1:end-1] |> gpu)
end
