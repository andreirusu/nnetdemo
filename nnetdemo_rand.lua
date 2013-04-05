#!/usr/bin/env torch
require 'torch'
require 'nn'
require 'nnet'
require 'util/arg' 
require 'sys'


local function main()
    local options = nnet.parse_arg(arg, true)
    
    options = nnet.set_options(options)
  
    nnet.init_experiment(options)

    local samples = nnet.get_data(options)
   

    local best_mlp 
    local best_mlp_loss = math.huge
    local mlp

    local epoch = 0 
    while true do
        options.seed = os.time()
        mlp = nnet.get_model(options, true) 
        
        local funs = {  options.objectiveFunction, 
                        function(x) 
                            return mlp:forward(torch.Tensor(1):fill(x)):squeeze() 
                        end
                    }

        if best_mlp then
            table.insert(funs,
                            function(x) 
                                return best_mlp:forward(torch.Tensor(1):fill(x)):squeeze() 
                            end)
        end

        nnet.plot(samples, funs, options)


        local mlp_loss = nnet.eval_net(samples, funs, mlp, options)
        if best_mlp_loss > mlp_loss then
            best_mlp = mlp:clone()
            best_mlp_loss = mlp_loss
        end
        if epoch % options.saveEvery == 0 then 
            nnet.save_network({network=best_mlp}, options)
        end
        print(string.format('Epoch %3d: Best MLP: %.4f\tCurrent MLP: %.4f', epoch, best_mlp_loss, mlp_loss))
        
        epoch = epoch + 1
    end
end

main()
    
