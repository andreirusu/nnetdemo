#!/usr/bin/env torch
require 'torch'
require 'nn'
require 'nnet'
require 'util/arg' 
require 'sys'


local function main()
    local options = nnet.parse_arg(arg)
    
    options = nnet.set_options(options)
  
    nnet.init_experiment(options)

    local samples = nnet.get_data(options)


    local best_mlp 
    local best_mlp_loss = math.huge
    local mlp

    while true do
        local mlp = nnet.get_net(options)  
        --print(mlp)

        
        local funs = {  function(x) 
                            return torch.abs(x) * torch.sin(x) 
                        end, 
                        function(x) 
                            return mlp:forward(torch.Tensor(1):fill(x)):squeeze() 
                        end
                    }

        local mlp_loss = nnet.eval_net(samples, funs, mlp, options)
        if best_mlp_loss > mlp_loss then
            best_mlp = mlp
            best_mlp_loss = mlp_loss
        end
        print(string.format('Best MLP: %.4f\tCurrent MLP: %.4f', best_mlp_loss, mlp_loss))
       
        if best_mlp then
            table.insert(funs,
                            function(x) 
                                return best_mlp:forward(torch.Tensor(1):fill(x)):squeeze() 
                            end)
        end

        nnet.plot(samples, funs, options)
        --os.execute('sleep 2')
    end
end

main()
    
