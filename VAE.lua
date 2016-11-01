require 'nn'
require 'dpnn'

local VAE = {}

function VAE.get_encoder(channels, naf, z_dim)
    encoder = nn.Sequential()
    encoder:add(nn.SpatialConvolution(channels, naf, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.ReLU())
    encoder:add(nn.SpatialConvolution(naf, naf * 2, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.SpatialBatchNormalization(naf * 2)):add(nn.ReLU())
    encoder:add(nn.SpatialConvolution(naf * 2, naf * 4, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.SpatialBatchNormalization(naf * 4)):add(nn.ReLU())
    encoder:add(nn.SpatialConvolution(naf * 4, naf * 8, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.SpatialBatchNormalization(naf * 8)):add(nn.ReLU())

    zLayer = nn.ConcatTable()
    zLayer:add(nn.SpatialConvolution(naf * 8, z_dim, 4, 4))
    zLayer:add(nn.SpatialConvolution(naf * 8, z_dim, 4, 4))
    encoder:add(zLayer)
    
    return encoder
end

function VAE.get_sampler()
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
    
    return sampler
end

function VAE.get_decoder(channels, ngf, z_dim)
    decoder = nn.Sequential()
    decoder:add(nn.SpatialFullConvolution(z_dim, ngf * 8, 4, 4))
    decoder:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
    decoder:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
    decoder:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
    decoder:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
    decoder:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
    decoder:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
    decoder:add(nn.SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
    decoder:add(nn.SpatialFullConvolution(ngf, channels, 4, 4, 2, 2, 1, 1))
    decoder:add(nn.Sigmoid())
    
    return decoder
end

return VAE
