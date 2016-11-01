require 'image'
require 'xlua'
require 'nn'
require 'dpnn'
require 'optim'
require 'lfs'
local VAE = require 'VAE'
local discriminator = require 'discriminator'

hasCudnn, cudnn = pcall(require, 'cudnn')
assert(hasCudnn) --check to make sure you have CUDA and CUDNN-enabled GPU 

local argparse = require 'argparse'
local parser = argparse('dcgan_vae', 'a Torch implementation of the deep convolutional generative adversarial network, with variational autoencoder')
parser:option('-i --input', 'input directory for image dataset')
parser:option('-o --output', 'output directory for generated images')
parser:option('-s --size', 'size of dataset')
parser:option('-c --checkpoints', 'directory for saving checkpoints')
parser:option('-r --reconstruction', 'directory to put samples of reconstructions')

args = parser:parse()

input = args.input
output_folder = args.output
dataset_size = args.size
checkpoints = args.checkpoints
reconstruct_folder = args.reconstruction

--ensure tensors are of correct type
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

function getFilenames()
    queue = {}
    count = 1
    for file in lfs.dir(input) do
        if file ~= '.' and file ~= '..' then
            queue[count] = file
            count = count + 1
        end
    end
    return queue
end

function getNumber(num)
  length = #tostring(num)
  filename = ""
  for i=1, (6 - length) do
    filename = filename .. 0
  end
  filename = filename .. num
  return filename
end

train_size = 200
batch_size = 50
channels = 3
dim = 64

train = torch.Tensor(train_size, channels, dim, dim)
train = train:cuda()

function fillTensor(tensor)
  filenames = getFilenames()
  for i = 1, train_size do
    local image_x = image.load(input .. filenames[torch.random(1, dataset_size)])
    local flip_or_not = torch.random(1, 2)
    if flip_or_not == 1 then
        image_x = image.hflip(image_x)
    end
    local image_ok, image_crop = pcall(image.crop, image_x, 'c', dim, dim)
    if image_ok then 
        tensor[i] = image_crop
    else
        print('image  cannot be cropped to ' .. dim .. 'x' .. dim .. '. Skipping...')
    end
  end
  return tensor
end

train = fillTensor(train)

feature_size = channels * dim * dim

--initialize the weights to non-zero 
function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

z_dim = 100
ndf = 64
ngf = 64
naf = 64

encoder = VAE.get_encoder(channels, naf, z_dim)
sampler = VAE.get_sampler()
decoder = VAE.get_decoder(channels, ngf, z_dim)

netG = nn.Sequential()
netG:add(encoder)
netG:add(sampler)
netG:add(decoder)
netG:apply(weights_init)

netD = discriminator.get_discriminator(channels, ndf)
netD:apply(weights_init)

netG = netG:cuda()
netD = netD:cuda()
cudnn.convert(netG, cudnn)
cudnn.convert(netD, cudnn)

criterion = nn.BCECriterion()
criterion = criterion:cuda()

m_criterion = nn.MSECriterion()
m_criterion = m_criterion:cuda()

optimStateG = {
   learningRate =  0.0002,
   beta1 = 0.5
}

optimStateD = {
   learningRate = 0.0002,
   beta1 = 0.5
}

--noise to pass through decoder to generate random samples from Z
noise_x = torch.Tensor(batch_size, z_dim, 1, 1)
noise_x = noise_x:cuda()
noise_x:normal(0, 0.01)

--label, real or fake, for our GAN
label = torch.Tensor(batch_size)

label = label:cuda()

real_label = 1
fake_label = 0

dNoise = .1

epoch_tm = torch.Timer()
tm = torch.Timer()
data_tm = torch.Timer()

--to keep track of our reconstructions
reconstruct_count = 1

parametersD, gradParametersD = netD:getParameters()
parametersG, gradParametersG = netG:getParameters()

errD = 0
errG = 0
errA = 0


--training evaluation for discriminator.
fDx = function(x)
    if x ~= parametersD then
        parametersD:copy(x)
    end
    gradParametersD:zero()

    -- train with real
    label:fill(real_label)
    --slightly noise inputs to help stabilize GAN and allow for convergence
    input_x = nn.WhiteNoise(0, dNoise):cuda():forward(input_x)
    output = netD:forward(input_x)
    errD_real = criterion:forward(output, label)
    df_do = criterion:backward(output, label)
    if (errG < .7 or errD > 1.0) then netD:backward(input_x, df_do) end

    -- train with fake
    noise_x:normal(0, 0.01)
    fake = decoder:forward(noise_x)
    --input_x:copy(fake)
    label:fill(fake_label)
    output = netD:forward(fake)
    errD_fake = criterion:forward(output, label)
    df_do = criterion:backward(output, label)
    if (errG < .7 or errD > 1.0) then netD:backward(fake, df_do) end

    errD = errD_real + errD_fake
    gradParametersD:clamp(-5, 5)
    return errD, gradParametersD
end

--training evaluation for variational autoencoder
fAx = function(x)
    if x ~= parametersG then
        parametersG:copy(x)
    end
    --reconstruction loss
    gradParametersG:zero()
    output = netG:forward(input_x)
    --print(output:size(), input_x:size())
    errA = m_criterion:forward(output, input_x)
    df_do = m_criterion:backward(output, input_x)
    netG:backward(input_x, df_do)

    --KLLoss
    nElements = output:nElement()
    mean, log_var = table.unpack(encoder.output)
    var = torch.exp(log_var)
    KLLoss = -0.5 * torch.sum(1 + log_var - torch.pow(mean, 2) - var)
    KLLoss = KLLoss / nElements
    errA = errA + KLLoss
    gradKLLoss = {mean / nElements, 0.5*(var - 1) / nElements}
    encoder:backward(input_x, gradKLLoss)
    if reconstruct_count % 10 == 0 then
      if reconstruct_folder then
        image.save(reconstruct_folder .. 'reconstruction' .. getNumber(reconstruct_count) .. '.png', output[1])
      end
    end
    return errA, gradParametersG
end

--training evaluation for generator
fGx = function(x)
    if x ~= parametersG then
        parametersG:copy(x)
    end
    gradParametersG:zero()
    label:fill(real_label)
    output = netD.output
    errG = criterion:forward(output, label)
    df_do = criterion:backward(output, label)
    df_dg = netD:updateGradInput(input_x, df_do)
    if (errD < .7 or errG > 1.0) then decoder:backward(noise_x, df_dg) end
    gradParametersG:clamp(-5, 5)
    return errG, gradParametersG
end

--generate samples from Z
generate = function(epoch)
    noise_x:normal(0, 0.01)
    local generations = decoder:forward(noise_x)
    image.save(output_folder .. getNumber(epoch) .. '.png', generations[1])
end

require 'optim'
require 'cunn'

for epoch = 1, 50000 do
    epoch_tm:reset()
    --main training loop
    for i = 1, train_size, batch_size do
        --split training data into mini-batches
        local size = math.min(i + batch_size - 1, train_size) - i
        input_x = train:narrow(1, size, batch_size)
        tm:reset()
        --optim takes an evaluation function, the parameters of the model you wish to train, and the optimization options, such as learning rate and momentum
        optim.adam(fAx, parametersG, optimStateG)  --VAE
        optim.adam(fDx, parametersD, optimStateD) --decoder
        optim.adam(fGx, parametersG, optimStateG) --generator
        collectgarbage('collect')
    end
    reconstruct_count = reconstruct_count + 1
    if errG then
      print("Generator loss: " .. errG .. ", Autoencoder loss: " .. errA .. ", Discriminator loss: " .. errD)
      else print("Discriminator loss: " .. errD)
    end
    parametersD, gradParametersD = nil, nil
    parametersG, gradParametersG = nil, nil
    --save and/or clear model state for next training batch
    if epoch % 1000 == 0 then
        torch.save(checkpoints .. epoch .. '_net_G.t7', netG:clearState())
        torch.save(checkpoints .. epoch .. '_net_D.t7', netD:clearState())
    else
        netG:clearState()
        netD:clearState()
    end
    generate(epoch)
    train = fillTensor(train)
    parametersD, gradParametersD = netD:getParameters()
    parametersG, gradParametersG = netG:getParameters()
    --simulated annealing for the discriminator's noise parameter
    if epoch % 10 == 0 then dNoise = dNoise * 0.99 end
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
           epoch, 10000, epoch_tm:time().real))
end
