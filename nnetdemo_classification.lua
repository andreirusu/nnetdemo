#!/usr/bin/env torch
require 'torch'
require 'torch-env'
require 'nn'
require 'sys'
require 'optim'
require 'dataset'
require 'dataset/TableDataset'

require 'nnet_classification'


require 'nnd'
require 'experiment'
require 'xae'



local function classes(ds)
    local classes = {}
    for i=1,ds:size() do
        if not classes[ds.dataset.class[i]] then 
            classes[ds.dataset.class[i]] = ds.dataset.class[i] 
        end
    end
    return classes
end

--[[
function test_network(model, test_ds, config, options)
    local criterion = nn.ClassNLLCriterion()
    
    local samples = test_ds:random_mini_batch({size=test_ds:size()})

    local inputs    =   samples.data
    local targets   =   samples.class

    print(inputs:size())
    print(targets:size())

    local pred = model:forward(inputs:narrow(1,1,1000))
    local loss = criterion:forward(pred, targets)


    local confusion = optim.ConfusionMatrix(classes(test_ds))
    for i=1,pred:size(1) do
        confusion:add(pred[i], targets[i])
    end
    return loss, confusion
end
--]]


function setEnableDropout(mlp, state)
    print('Configuring MLP:')
    for i = 1,mlp:size() do
        local module = mlp:get(i)
        if torch.typename(module) == 'nnd.Dropout' then
            module:setEnabled(state)
            print(mlp:get(i))
        end
    end
end



function test_network(model, test_ds, config, options, limit)
    local criterion = nn.ClassNLLCriterion()
    local parameters, gradParameters = model:getParameters()
    local data = test_ds.dataset.data
    local class = test_ds.dataset.class

    local confusion = optim.ConfusionMatrix(classes(test_ds))
    local shuffle = torch.randperm(test_ds:size())


    setEnableDropout(model, false)
   
    if not limit then limit = test_ds:size() end
    
    local mb_size = 1000

    local loss = 0
    for t = 1, limit-1, mb_size do
        local inputs = torch.Tensor(mb_size, options.input)
        local targets = torch.Tensor(mb_size)
        
        for i=1,mb_size,1 do
            if (t+i) <= test_ds:size() then 
                inputs[i]:copy(data[shuffle[t+i]]:resize(options.input))
                targets[i] = class[shuffle[t+i]]
            else 
                inputs = inputs:narrow(1,1, i-1)
                targets = targets:narrow(1,1, i-1)
                break
            end
        end

        
        local preds = model:forward(inputs)
        loss = loss + criterion:forward(preds, targets)
        
        for i=1,preds:size(1) do
            confusion:add(preds[i], targets[i])
        end

    end

        
    return loss, confusion
end


function train_network(model, train_ds, config, options)
    local criterion = nn.ClassNLLCriterion()
    local parameters, gradParameters = model:getParameters()
    local data = train_ds.dataset.data
    local class = train_ds.dataset.class

    local shuffle = torch.randperm(train_ds:size())

    setEnableDropout(model, true)

    for t = 1, train_ds:size()-1, options.szMinibatch do
        local inputs = torch.Tensor(options.szMinibatch, options.input)
        local targets = torch.Tensor(options.szMinibatch)
        
        for i=1,options.szMinibatch,1 do
            if (t+i) <= train_ds:size() then 
                inputs[i]:copy(data[shuffle[t+i]]:resize(options.input))
                targets[i] = class[shuffle[t+i]]
            else 
                inputs = inputs:narrow(1,1, i-1)
                targets = targets:narrow(1,1, i-1)
                break
            end
        end

        
        local feval = function(x) 
            gradParameters:zero()
            local preds = model:forward(inputs)
            local loss = criterion:forward(preds, targets)
            local df_do = criterion:backward(preds, targets)
            model:backward(inputs, df_do)
            gradParameters:div(inputs:size(1)) 
            return loss, gradParameters
        end

        optim.sgd(feval, parameters, config)

        -- check max time 
        if (os.clock() - start_time) > options.maxTime then
            break
        end
    end
end


local function eval_network(mlp, samples, epoch, options) 
    local train_mlp_loss, train_confusion = test_network(mlp, samples.train, config, options, 5000)
    local test_mlp_loss, test_confusion = test_network(mlp, samples.test, config, options, 5000)
    train_confusion:updateValids()
    test_confusion:updateValids()
    print(string.format('\nEpoch %3d: \tMLP Train loss: %.4f\t Train accuracy: %.4f%%\n%s\n\n', epoch, 
                                                                        train_mlp_loss, 
                                                                        train_confusion.totalValid*100,
                                                                        tostring(train_confusion)))
    print(string.format('Epoch %3d: \tMLP Test loss: %.4f\t Test accuracy: %.4f%%\n%s\n\n\n', epoch, 
                                                                        test_mlp_loss, 
                                                                        test_confusion.totalValid*100,
                                                                        tostring(test_confusion)))
end


local function main()
    local options = nnet.parse_arg(arg, true)
    
    options = nnet.set_options(options)
  
    nnet.init_experiment(options)

    local samples = nnet.get_data(options)

    print(samples)
  

    local mlp = nnet.get_model(options) 


    local config = {learningRate        = options.learningRate,
                    momentum            = options.momentum,
                    weightDecay         = options.weightDecay,
                    learningRateDecay   = options.learningRateDecay }
    
    local epoch = 0 
    while true do
        if epoch % options.saveEvery == 0 then 
            nnet.save_network({network=mlp}, options)
        end
        if epoch % options.reportEvery == 0 then 
            eval_network(mlp, samples, epoch, options) 
        end
                
        train_network(mlp, samples.train, config, options)
        -- check max time 
        if (os.clock() - start_time) > options.maxTime then
            print('MaxTime reached...')
            eval_network(mlp, samples, epoch, options) 
            nnet.save_network({network=mlp}, options)
            break
        end

        epoch = epoch + 1
    end

end



do
    main()
end

