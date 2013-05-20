local VerySparseRandomProjections, parent = torch.class('nn.VerySparseRandomProjections', 'nn.Linear')

function VerySparseRandomProjections:__init(inputSize, outputSize)
   parent.__init(self, inputSize, outputSize)
   self:reset()
end

function VerySparseRandomProjections:reset()
    local prob = 0.5/torch.sqrt(self.weight:size(2))
    self.weight:apply(function()
        local val = torch.rand(1)[1]
        if val <= prob then
            return 1
        elseif val >= 1-prob then
            return -1
        else
            return 0
        end
    end)
    self.bias:fill(0)
end

