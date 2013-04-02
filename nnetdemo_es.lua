require 'torch'
require 'nn'
require 'nnet'
require 'sys'


local function mutate(mlp_input, sigma)
    local mlp = mlp_input:clone() 
    
    local mutation = function(x)    
        return x + sigma * (torch.randn(1):squeeze())
    end
    local found = false
    repeat 
        local i = torch.random(1, mlp:size())

        local layer  = mlp:get(i)
        if layer.weight and layer.bias then
            -- mutate one of the weights or biases with sigma
            if torch.rand(1):squeeze() >= 0.5 then
                local m = torch.random(1, layer.weight:size(1))
                local n = torch.random(1, layer.weight:size(2))
                layer.weight[m][n] = mutation(layer.weight[m][n])
            else
                local m = torch.random(1, layer.bias:size(1))
                layer.bias[m] = mutation(layer.bias[m])
            end
            found = true
        end
    until found
    return mlp
end

local function main()
    local options = nnet.parse_arg(arg)
    
    options = nnet.set_options(options)
  
    nnet.init_experiment(options)

    local samples = nnet.get_data(options)
   

    local best_mlp 
    local best_mlp_loss = math.huge
    local mlp
    
    local sigma_start = 1
    local sigma = sigma_start
    local sigma_decay = 1e-4
    local epoch = 1 

    while true do
        mlp = mlp or nnet.get_net(options)  
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
        print(string.format('Epoch %3d: Best MLP: %.4f\tCurrent MLP: %.4f\tSigma: %.3f', epoch, best_mlp_loss, mlp_loss, sigma))
       
        if best_mlp then
            table.insert(funs,
                            function(x) 
                                return best_mlp:forward(torch.Tensor(1):fill(x)):squeeze() 
                            end)
        end

        nnet.plot(samples, funs, options)
        --os.execute('sleep 2')
        mlp = mutate(best_mlp, sigma) 
        sigma = sigma_start / (1 + sigma_decay * epoch)
        epoch = epoch + 1
    end
end

main()
    
