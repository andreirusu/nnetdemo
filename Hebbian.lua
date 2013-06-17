-- The Hebbian module computes f(x) =  x + N(0, sigma^2)
-- where N the normal distribution. 


require 'torch'
require 'torch-env'
require 'nn'
require 'nnd'


local Hebbian, parent = torch.class('nnd.Hebbian', 'nn.Module')

function Hebbian:__init(sigma)
    parent.__init(self)
end

function Hebbian:updateOutput(input)
    self.output:typeAs(input):resizeAs(input):copy(input)
    return self.output
end

function Hebbian:updateGradInput(input, gradOutput)
    self.gradInput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput):mul(-1)
    return self.gradInput
end

