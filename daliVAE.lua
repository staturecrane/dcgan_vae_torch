require 'image'
require 'xlua'
require 'nn'
require 'dpnn'
require 'optim'
require 'lfs'

hasCudnn, cudnn = pcall(require, 'cudnn')
assert(hasCudnn)

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

function getDali()
    queue = {}
    count = 1
    for file in lfs.dir('dali') do
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

train_size = 100
dataset_size = 323
batch_size = 50
channels = 3
dim = 128

train = torch.Tensor(train_size, channels, dim, dim)
train = train:cuda()

function fillTensor(tensor)
  filenames = getDali()
  for i = 1, train_size do
    tensor[i] = image.load('dali/' .. filenames[torch.random(1, dataset_size)])
  end
  return tensor
end

train = fillTensor(train)

feature_size = channels * dim * dim

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

z_dim = 600
ndf = 64
ngf = 80
naf = 80

encoder = nn.Sequential()
encoder:add(nn.SpatialConvolution(channels, naf, 4, 4, 2, 2, 1, 1))
encoder:add(nn.ReLU())
encoder:add(nn.SpatialConvolution(naf, naf * 2, 4, 4, 2, 2, 1, 1))
encoder:add(nn.SpatialBatchNormalization(naf * 2)):add(nn.ReLU())
encoder:add(nn.SpatialConvolution(naf * 2, naf * 4, 4, 4, 2, 2, 1, 1))
encoder:add(nn.SpatialBatchNormalization(naf * 4)):add(nn.ReLU())
encoder:add(nn.SpatialConvolution(naf * 4, naf * 8, 4, 4, 2, 2, 1, 1))
encoder:add(nn.SpatialBatchNormalization(naf * 8)):add(nn.ReLU())
encoder:add(nn.SpatialConvolution(naf * 8, naf * 16, 4, 4, 2, 2, 1, 1))
encoder:add(nn.SpatialBatchNormalization(naf * 16)):add(nn.ReLU())

zLayer = nn.ConcatTable()
zLayer:add(nn.SpatialConvolution(naf * 16, z_dim, 4, 4))
zLayer:add(nn.SpatialConvolution(naf * 16, z_dim, 4, 4))
encoder:add(zLayer)

epsilonModule = nn.Sequential()
epsilonModule:add(nn.MulConstant(0))
epsilonModule:add(nn.WhiteNoise(0, 0.01))

noiseModule = nn.Sequential()
noiseModuleInternal = nn.ConcatTable()
stdModule = nn.Sequential()
stdModule:add(nn.MulConstant(0.5)) -- Compute 1/2 log σ^2 = log σ
stdModule:add(nn.Exp()) -- Compute σ
noiseModuleInternal:add(stdModule) -- Standard deviation σ
noiseModuleInternal:add(epsilonModule) -- Sample noise ε
noiseModule:add(noiseModuleInternal)
noiseModule:add(nn.CMulTable())

sampler = nn.Sequential()
samplerInternal = nn.ParallelTable()
samplerInternal:add(nn.Identity())
samplerInternal:add(noiseModule)
sampler:add(samplerInternal)
sampler:add(nn.CAddTable())


decoder = nn.Sequential()
decoder:add(nn.SpatialFullConvolution(z_dim, ngf * 16, 4, 4))
decoder:add(nn.SpatialBatchNormalization(ngf * 16)):add(nn.ReLU(true))
decoder:add(nn.SpatialFullConvolution(ngf * 16, ngf * 8, 4, 4, 2, 2, 1, 1))
decoder:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
decoder:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
decoder:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
decoder:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
decoder:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
decoder:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
decoder:add(nn.SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
decoder:add(nn.SpatialFullConvolution(ngf, channels, 4, 4, 2, 2, 1, 1))
decoder:add(nn.Sigmoid())

netG = nn.Sequential()
netG:add(encoder)
netG:add(sampler)
netG:add(decoder)

netG:apply(weights_init)

netD = nn.Sequential()

netD:add(nn.SpatialConvolution(channels, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
netD:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
netD:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
netD:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
netD:add(nn.SpatialConvolution(ndf * 8, ndf * 16, 4, 4, 2, 2, 1, 1))
netD:add(nn.SpatialBatchNormalization(ndf * 16)):add(nn.LeakyReLU(0.2, true))
netD:add(nn.SpatialConvolution(ndf * 16, 1, 4, 4))
netD:add(nn.Sigmoid())
netD:add(nn.View(1):setNumInputDims(3))

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
   learningRate = 0.0002,
   optimize = true
}

optimStateD = {
   learningRate = 0.00002,
   optimize = true
}

noise_x = torch.Tensor(batch_size, z_dim, 1, 1)
noise_x = noise_x:cuda()
noise_x:normal(0, 0.01)
label = torch.Tensor(batch_size)

label = label:cuda()

real_label = 1
fake_label = 0

epoch_tm = torch.Timer()
tm = torch.Timer()
data_tm = torch.Timer()

parametersD, gradParametersD = netD:getParameters()
parametersG, gradParametersG = netG:getParameters()

errD = 0
errG = 0
errA = 0

fDx = function(x)

    if x ~= parametersD then
        parametersD:copy(x)
    end
    gradParametersD:zero()
    -- train with real
    label:fill(real_label)
    output = netD:forward(input_x)
    errD_real = criterion:forward(output, label)
    df_do = criterion:backward(output, label)
    if (errG < 0.7 and errD > 0.7) then netD:backward(input_x, df_do) end

    -- train with fake
    noise_x:normal(0, 0.01)
    fake = decoder:forward(noise_x)
    --input_x:copy(fake)
    label:fill(fake_label)
    output = netD:forward(fake)
    errD_fake = criterion:forward(output, label)
    df_do = criterion:backward(output, label)
    if (errG < 0.7 and errD > 0.7) then netD:backward(fake, df_do) end

    errD = errD_real + errD_fake

    return errD, gradParametersD
end

fAx = function(x)
    if x ~= parametersG then
        parametersG:copy(x)
    end
    gradParametersG:zero()
    output = netG:forward(input_x)
    errA = m_criterion:forward(output, input_x)
    df_do = m_criterion:backward(output, input_x)
    netG:backward(input_x, df_do)

    nElements = output:nElement()
    mean, log_var = table.unpack(encoder.output)
    var = torch.exp(log_var)
    KLLoss = -0.5 * torch.sum(1 + log_var - torch.pow(mean, 2) - var)
    KLLoss = KLLoss / nElements
    errA = errA + KLLoss
    gradKLLoss = {mean / nElements, 0.5*(var - 1) / nElements}
    encoder:backward(input_x, gradKLLoss)
    return errA, gradParametersG
end

-- create closure to evaluate f(X) and df/dX of generator
fGx = function(x)
    if x ~= parametersG then
        parametersG:copy(x)
    end
    gradParametersG:zero()
    label:fill(real_label) -- fake labels are real for generator cost
    output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
    errG = criterion:forward(output, label)
    df_do = criterion:backward(output, label)
    df_dg = netD:updateGradInput(input_x, df_do)
    if (errD < 0.7 and errG > 0.7) then decoder:backward(noise_x, df_dg) end
    return errG, gradParametersG
end

generate = function(epoch)
    noise_x:normal(0, 0.01)
    local generations = decoder:forward(noise_x)
    image.save('dali_generated/' .. getNumber(epoch) .. '.png', generations[1])
end

require 'optim'
require 'cunn'

for epoch = 1, 50000 do
    epoch_tm:reset()
    for i = 1, train_size, batch_size do
        local size = math.min(i + batch_size - 1, train_size) - i
        input_x = train:narrow(1, size, batch_size)
        tm:reset()
        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))))
        optim.adam(fDx, parametersD, optimStateD)
        optim.adam(fAx, parametersG, optimStateG)
        optim.adam(fGx, parametersG, optimStateG)
        collectgarbage('collect')
    end
    if errG then
      print("Generator loss: " .. errG .. ", Autoencoder loss: " .. errA .. ", Discriminator loss: " .. errD)
      else print("Discriminator loss: " .. errD)
    end
    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil
    if epoch % 1000 == 0 then
        torch.save('dali_checkpoints/dali' .. epoch .. '_net_G.t7', netG:clearState())
        torch.save('dali_checkpoints/dali' .. epoch .. '_net_D.t7', netD:clearState())
        torch.save('dali_checkpoints/dali'  .. epoch .. 'encoder.t7', encoder:clearState())
        torch.save('dali_checkpoints/dali'.. epoch .. 'decoder.t7', decoder:clearState())
        torch.save('dali_checkpoints/dali' .. epoch .. 'sampler.t7', sampler:clearState())
    else
        netG:clearState()
        netD:clearState()
    end
    generate(epoch)
    train = fillTensor(train)
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
           epoch, 10000, epoch_tm:time().real))
end
