require 'nn'

local discriminator = {}

function discriminator.get_discriminator(channels, ndf)
    netD = nn.Sequential()
    netD:add(nn.SpatialConvolution(channels, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    netD:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    netD:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    netD:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    netD:add(nn.SpatialConvolution(ndf * 8, 1, 4, 4))
    netD:add(nn.Sigmoid())
    netD:add(nn.View(1):setNumInputDims(3))
    
    return netD
end

return discriminator
