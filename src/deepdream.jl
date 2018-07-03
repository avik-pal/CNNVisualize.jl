#--------------------General Utilities---------------------

function zoom_image(x, scale_x, scale_y)
  img = generate_image(x)
  inv_scalex = 1 / scale_x
  inv_scaley = 1 / scale_y
  local s = size(img)
  local r1 = clamp(Int(ceil(s[1] * (1 - inv_scalex) / 2)), 1, s[1]) : clamp(Int(ceil(s[1] * (1 + inv_scalex) / 2)), 1, s[1])
  local r2 = clamp(Int(ceil(s[2] * (1 - inv_scaley) / 2)), 1, s[2]) : clamp(Int(ceil(s[2] * (1 + inv_scaley) / 2)), 1, s[2])
  img = imresize(img[r1, r2], s)
  image_to_arr(img)
end

function load_model(layer, m = VGG19)
  model = m()
  global model = Chain(model.layers[1:layer]...) |> gpu
end

function argmax(A, dims)
  z = findmax(A, dims)[2] .% size(A, dims)
  z[z.==0] .= size(A,dims)
  z
end

#---------------------Core Functions-----------------------

function make_step(img, iterations, η, solo_call = false; path = "")
  if(solo_call && path == "")
    error("Image Save Path must be specified for solo calls")
  end
  input = param(img)
  for i in 1:iterations
    out = model(input)
    Flux.back!(out, out.data)
    input.data .= input.data + η * input.grad / mean(abs.(input.grad))
    info("$i iterations complete")
  end
  if(solo_call)
    save_image(path, input.data)
  end
  return input.data
end

function deepdream(base_img, iterations, η, octave_scale, num_octaves, path, guide = 1.0)
  octaves = [copy(base_img)]
  for i in 1:(num_octaves-1)
    push!(octaves, zoom_image(octaves[end], octave_scale, octave_scale))
  end
  detail = zeros(octaves[end])
  out = base_img
  for (i, oct) in enumerate(octaves[length(octaves):-1:1])
    info("OCTAVE NUMBER = $i")
    w, h = size(oct)[1:2]
    if(i > 1)
      w1, h1 = size(detail)[1:2]
      detail = zoom_image(detail, w1 / w, h1 / h)
    end
    input_oct = (oct + detail) |> gpu
    if(guide != 1.0)
      out = guided_step(input_oct, guide, iterations, η)
    else
      out = make_step(input_oct, iterations, η)
    end
    detail = out - oct
  end
  save_image(path, out)
end
