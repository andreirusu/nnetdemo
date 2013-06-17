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
    self.noise = torch.Tensor()
    self.input = torch.Tensor()
end

function GaussianNoise:updateOutput(input)
    self.input:typeAs(input):resizeAs(input):copy(input)
    self.noise:typeAs(input):resizeAs(input):copy(torch.randn(input:size()):mul(self.sigma))
    self.output:typeAs(input):resizeAs(input):copy(input):add(self.noise)
    return self.output
end

function GaussianNoise:updateGradInput(input, gradOutput)
    self.gradInput:typeAs(input):resizeAs(input):copy(gradOutput)
    return self.gradInput
end

