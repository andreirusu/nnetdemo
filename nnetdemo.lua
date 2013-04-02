require 'torch'
require 'nn'


local function set_options()
    local options = {}
    options.input = {}

    options.input.size  = 100
    options.input.min   = -5
    options.input.max   = 5
    options.input.cols  = 1

    options.seed = 0

    options.n_units = {options.input.cols, 5,  0, 1}

    torch.manualSeed(options.seed)

    return options
end

local function get_net(options)
    local mlp = nn.Sequential()
    local n_old 
    for _, v in pairs(options.n_units) do
        if v ~= 0 then
            if n_old then 
                mlp:add(nn.Linear(n_old, v))
            end
            n_old = v
        else 
            mlp:add(nn.Tanh())
        end
    end
    mlp:get(1).weight:fill(1)
    mlp:get(1).bias:fill(0)
    mlp:get(3).weight:fill(1)
    mlp:get(3).bias:fill(0)

    return mlp
end


local function plot(samples, funs, options)
    if type(funs) == 'function' then
        funs = {funs}
    end

    local tab = {}
    for _,fun in pairs(funs) do
        table.insert(tab, {samples, samples:clone():apply(fun), '+-'})
    end
    gnuplot.plot(tab)
end

local function get_data(options)
    return (torch.rand(options.input.size) * (options.input.max - options.input.min) + options.input.min):sort()
end

local function main()
    local options = set_options()
   
    local samples = get_data(options)
    
    mlp = get_net(options)  
    print(mlp)
    
    local funs = {  function(x) 
                        return torch.abs(x) * torch.sin(x) 
                    end, 
                    function(x) 
                        return mlp:forward(torch.Tensor(1):fill(x)):squeeze() 
                    end
                }

    plot(samples, funs, options)
end


main()

