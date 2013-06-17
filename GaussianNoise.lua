-- The GaussianNoise module computes f(x) =  x + N(0, sigma^2)
-- where N the normal distribution. 


require 'torch'
require 'torch-env'
require 'nn'
require 'nnd'


local GaussianNoise, parent = torch.class('nnd.GaussianNoise', 'nn.Module')

function GaussianNoise:__init(sigma)
    sigma = sigma or 0.01
    parent.__init(self)
    self.sigma = sigma
end

function GaussianNoise:updateOutput(input)
    self.output:typeAs(input):resizeAs(input):copy(input):add(torch.randn(input:size()):mul(self.sigma))
    return self.output
end

function GaussianNoise:updateGradInput(input, gradOutput)
    self.gradInput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
    return self.gradInput
end

