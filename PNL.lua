-- The PNL module computes f(x) =  a * NL(b * x + c) + d * x + e, 
-- where a, b, ..., e are fixed parameters, and x is the input.
--
-- The output is always the same dimension as the input.


require 'torch'
require 'torch-env'
require 'nn'
require 'nnd'


local PNL, parent = torch.class('nnd.PNL', 'nn.Module')

function PNL:__init(NL)
    parent.__init(self)
    self.NL = NL
end

function PNL:updateOutput(input)
    self.output:typeAs(input):resizeAs(input):copy(self.NL:forward(input))
    return self.output
end

function PNL:updateGradInput(input, gradOutput)
    self.gradInput:typeAs(input):resizeAs(input):copy(self.NL:backward(input, torch.ones(input:size())):cmul(gradOutput))
    return self.gradInput
end

