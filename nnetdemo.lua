require 'torch'
require 'nn'
require 'nnet'
require 'sys'
require 'optim'



function train_network(model, samples, config, options)
    local criterion = nn.MSECriterion()
    local parameters, gradParameters = model:getParameters()
    local shuffle = torch.randperm(options.size)


    local inputs = torch.Tensor(options.size, 1):copy(samples.x)
    local targets = torch.Tensor(options.size, 1):copy(samples.y)

    local feval = function(x) 
        local f = 0
        gradParameters:zero()

        for t = 1,options.size do
            local i = shuffle[t]
            local pred = model:forward(inputs[i])
            loss = criterion:forward(pred, targets[i])
            f = f + loss
            local df_do = criterion:backward(pred, targets[i])
            
            model:backward(inputs[i], df_do)
            
        end
        -- average parameters
        gradParameters:div(options.size)

        return f, gradParameters
    end

    optim.sgd(feval, parameters, config)
end



local function main()
    local options = nnet.parse_arg(arg)
    
    options = nnet.set_options(options)
  
    nnet.init_experiment(options)

    local samples = nnet.get_data(options)
   

    local best_mlp 
    local best_mlp_loss = math.huge
    local mlp
    
    local config = {learningRate        = options.learningRate,
                    momentum            = options.momentum,
                    weightDecay         = options.weightDecay,
                    learningRateDecay   = options.learningRateDecay }

    
    local epoch = 0 

    while true do
        mlp = mlp or nnet.get_net(options)  
        
        local funs = {  function(x)  
                            return torch.abs(x) * torch.sin(x) 
                        end, 
                        function(x) 
                            return mlp:forward(torch.Tensor(1):fill(x)):squeeze() 
                        end
                    }

        train_network(mlp, {x=samples, y=samples:clone():apply(funs[1])}, config, options)

       
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
        print(string.format('Epoch %3d: Best MLP: %.4f\tCurrent MLP: %.4f', epoch, best_mlp_loss, mlp_loss))
        
        --os.execute('sleep 2')
        epoch = epoch + 1
    end
end

main()
    
