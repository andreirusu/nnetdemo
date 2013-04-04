#!/usr/bin/env torch
require 'torch'
require 'torch-env'
require 'nn'
require 'nnet'
require 'sys'
require 'optim'



local function main()
    local options = nnet.parse_arg(arg)
    options = nnet.set_options(options)
    nnet.init_experiment(options)
    
    local samples = nnet.get_data(options)
    local mlp = nnet.get_model(options) 
   
    local function updatePNLParameters(mlp, options) 
        for i = 1,mlp:size(),1 do
            local layer = mlp:get(i)
            if torch.typename(layer) == 'nnd.PNL' then
                print('Updating parameters of: ', layer)
                layer:updateStaticParameters(options)
            end
        end
    end

    updatePNLParameters(mlp, options) 

    local funs = {  function(x)  
                        return torch.abs(x) * torch.sin(x) 
                    end, 
                    function(x)
                        return mlp:forward(torch.Tensor(1):fill(x)):squeeze() 
                    end
                }

    local mlp_loss = nnet.eval_net(samples, funs, mlp, options)
       
    nnet.plot(samples, funs, options)
    print(string.format('Current MLP: %.4f', mlp_loss))
    
end

main()
    
