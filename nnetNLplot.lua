#!/usr/bin/env torch
require 'torch'
require 'torch-env'
require 'nn'
require 'nnet'
require 'sys'
require 'optim'



local function main()
    local options = nnet.parse_arg(arg, true)

    options = nnet.set_options(options)
    nnet.init_experiment(options)
    
    local samples = nnet.get_data(options)
    local mlp = nnet.get_model(options) 
   

    local funs = {  options.objectiveFunction, 
                    function(x)
                        return options.NL:forward(torch.Tensor(options.h1):fill(x))[1] 
                    end
                }

    nnet.plot(samples, funs, options)
    
end

main()
    
