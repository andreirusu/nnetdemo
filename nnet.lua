require 'torch'
require 'nn'
require 'PNL'
require 'util'
require 'util/arg'
require 'sys'


nnet = {}


function nnet.parse_arg(arg)
    local dname, fname = sys.fpath()
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Non-linear Regression with Neural Networks - Demo Tool')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-save',             'demo',                 'subdirectory to save/log experiments in')
    cmd:option('-network',          '',                     'reload pretrained network')
    cmd:option('-visualize',        false,                  'visualize input data and weights during training')
    cmd:option('-seed',             0,                      'fixed input seed for repeatable experiments')
    cmd:option('-szMinibatch',      10,                     'mini-batch size (1 = pure stochastic)')
    cmd:option('-learningRate',     0.1,                      'learning rate at t=0')
    cmd:option('-weightDecay',      0,                      'weight decay (SGD only)')
    cmd:option('-learningRateDecay',1e-3,                   'learning rate decay (SGD only)')
    cmd:option('-momentum',         0,                      'momentum (SGD only)')
    cmd:option('-threads',          1,                      'nb of threads to use')
    cmd:option('-maxEpochs',        math.huge,              'maximum number of epochs to train')
    cmd:option('-maxTime',          math.huge,              'maximum time to train (seconds)')
    cmd:option('-noplot',           false,                  'disable plotting')
    cmd:option('-double', 	        true,            	    'set default tensor type to double')
    cmd:option('-min', 	            -5,            	        'set minimum value of samples on x')
    cmd:option('-max', 	             5,            	        'set maximum value of samples on x')
    cmd:option('-cols',             1,            	        'set number of columns in representation')
    cmd:option('-size',             100,            	    'set number of samples')
    cmd:option('-h1',               100,            	    'set number of units in the first hidden layer')
    cmd:option('-saveEvery',        100,            	    'set number of epochs between saves')
    
    cmd:text()
    return cmd:parse(arg)
end



function nnet.NL(NL, gradNL, a, b, c, d, e)  
    a = a or 1  
    b = b or 1       
    c = c or 0       
    d = d or 0       
    e = e or 0       

    return function(x) 
                    --return a*(math.tanh(x + b) + math.tanh(x - b)) 
                    return a * NL(b * x + c) + d * x + e
                end,
           function(x) 
                return a * b * NL(b * x + c) * gradNL(b * x + c) + d 
            end
end




function nnet.get_model(options)
    -- get a model; default behavior is to load, otherwise create
    local ret = {} 
    if options.network == '' then
        print('Creating new model...')
        ret.network = nnet.get_net(options)
    else
        print('Loading previously trained model: ' .. options.network)
        ret = torch.load(options.network)
    end
    local model = ret.network
    print(model)
    
    print('Done\n')
    return model
end



function nnet.save_network(options, t)
    -- save/log current net
    local filename = paths.concat(options.save, 'mlp.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if sys.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end

    print('<nnetdemo> saving network to '..filename)
    torch.save(filename, t)
end

function nnet.init_experiment(options)
    -- set random seed
    torch.manualSeed(options.seed)
    -- set number of threads 
    os.execute('export OMP_NUM_THREADS='..options.threads)
    torch.setnumthreads(options.threads)
    -- change tensor type to float, unless double flag is raised
    if options.double then
    	torch.setdefaulttensortype('torch.DoubleTensor')
    else
    	torch.setdefaulttensortype('torch.FloatTensor')
    end
    print(options)
end



function nnet.set_options(options)
    local options = options or {}


    options.n_units = {options.cols, options.h1,  0, 1}

    return options
end

function nnet.get_net(options)
    local mlp = nn.Sequential()

    local init_weight = function(x) 
        return torch.randn(1):mul(1):squeeze()
    end

    local init_bias = function(x) 
        return torch.randn(1):mul(1):squeeze()
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
            mlp:add(nnd.PNL(options.NL, options.gradNL))
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
        table.insert(tab, {samples, samples:clone():apply(fun), '+-'})
    end
    gnuplot.plot(tab)
end

function nnet.get_data(options)
    return (torch.rand(options.size) * (options.max - options.min) + options.min):sort()
end

function nnet.eval_net(samples, funs, options)
    --return torch.mean(torch.abs(samples:clone():apply(funs[1]):add(-samples:clone():apply(funs[2])))) -- L1
    return torch.mean(torch.pow(samples:clone():apply(funs[1]):add(-samples:clone():apply(funs[2])),2)) -- L2
end


