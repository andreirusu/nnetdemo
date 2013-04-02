require 'torch'
require 'nn'

nnet = {}

function nnet.set_options()
    local options = {}
    options.input = {}

    options.input.size  = 1000
    options.input.min   = -5
    options.input.max   = 5
    options.input.cols  = 1

    options.seed = os.time()

    options.n_units = {options.input.cols, 100,  0, 1}

    torch.manualSeed(options.seed)

    return options
end

function nnet.get_net(options)
    local mlp = nn.Sequential()

    local init_weight = function(x) 
        return torch.randn(1):mul(1):squeeze()
    end

    local init_bias = function(x) 
        return torch.randn(1):mul(0.1):squeeze()
    end

    local n_old 
    for _, v in pairs(options.n_units) do
        if v ~= 0 then
            if n_old then 
                local lin = nn.Linear(n_old, v)
                lin.weight:apply(init_weight)
                lin.bias:apply(init_bias)
                mlp:add(lin)
            end
            n_old = v
        else 
            mlp:add(nn.Tanh())
        end
    end

    

    
    return mlp
end


function nnet.plot(samples, funs, options)
    if type(funs) == 'function' then
        funs = {funs}
    end

    local tab = {}
    for _,fun in pairs(funs) do
        table.insert(tab, {samples, samples:clone():apply(fun), '~'})
    end
    gnuplot.plot(tab)
end

function nnet.get_data(options)
    return (torch.rand(options.input.size) * (options.input.max - options.input.min) + options.input.min):sort()
end

function nnet.eval_net(samples, funs, options)
    --return torch.mean(torch.abs(samples:clone():apply(funs[1]):add(-samples:clone():apply(funs[2])))) -- L1
    return torch.mean(torch.pow(samples:clone():apply(funs[1]):add(-samples:clone():apply(funs[2])),2)) -- L2
end


