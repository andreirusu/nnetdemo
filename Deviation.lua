-- The Deviation module computes f(x) =  x + N(0, sigma^2)
-- where N the normal distribution. 


require 'torch'
require 'torch-env'
require 'nn'
require 'nnd'


local Deviation, parent = torch.class('nnd.Deviation', 'nn.Module')

function Deviation:__init(gamma, gradGamma)
    gamma = gamma or 0.95
    gradGamma = gradGamma or 0.99

    parent.__init(self)
    self.gamma = gamma
    self.gradGamma = gradGamma
    self.baseline = nil
    self.gradBaseline = nil
end

function Deviation:updateOutput(input)
    self.output:typeAs(input):resizeAs(input):copy(input)
    if not self.baseline then 
        self.baseline = self.output:clone()
    else 
        self.baseline:mul(self.gamma):add(self.output:clone():mul(1-self.gamma))
    end
    return self.output
end

function Deviation:updateGradInput(input, gradOutput)
    if not self.gradBaseline then 
        self.gradBaseline = gradOutput:clone()
    else 
        self.gradBaseline:mul(self.gradGamma):add(gradOutput:clone():mul(1-self.gradGamma))
    end
    self.gradInput:typeAs(input):resizeAs(input):copy(self.baseline):mul(-1):add(input):mul(-1):cmul(self.gradBaseline)
    return self.gradInput
end

