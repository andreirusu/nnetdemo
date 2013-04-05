require 'torch'
require 'nn'
require 'PNL'
require 'util'
require 'util/arg'
require 'sys'


nnet = {}



function nnet.parse_arg(arg, initNLparams)
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
    cmd:option('-h1',               10,            	        'set number of units in the first hidden layer')
    cmd:option('-saveEvery',        1000,            	    'set number of epochs between saves')
    cmd:option('-reportEvery',      100,            	    'set number of epochs between saves')
    
    if initNLparams then
        cmd:option('-a',                1,            	        'NL parameter a')
        cmd:option('-b',                1,            	        'NL parameter b')
        cmd:option('-c',                0,            	        'NL parameter c')
        cmd:option('-d',                0,              	    'NL parameter d')
        cmd:option('-e',                0,            	        'NL parameter e')
    else
        cmd:option('-a',                math.huge,     	        'NL parameter a')
        cmd:option('-b',                math.huge,     	        'NL parameter b')
        cmd:option('-c',                math.huge,     	        'NL parameter c')
        cmd:option('-d',                math.huge,         	    'NL parameter d')
        cmd:option('-e',                math.huge,         	    'NL parameter e')
    end
    cmd:option('-obj', 
                    'return function(x) return math.abs(x) * math.sin(x) end',            	        
                    'objective function for non-linear regression')

    cmd:text()
    return cmd:parse(arg)
end



function nnet.set_options(options)
    local options = options or {}
     
    options.n_units = {options.cols, options.h1,  0, 1}

    return options
end




function nnet.updatePNLParameters(mlp, options, suppressPrinting) 
    for i = 1,mlp:size(),1 do
        local layer = mlp:get(i)
        if torch.typename(layer) == 'nnd.PNL' then
            if not suppressPrinting then
                print('Updating parameters of: ', layer)
            end
            layer:updateStaticParameters(options, suppressPrinting)
        end
    end
end


-- return a Sequential module which
-- implements a*NL(b*x + c) + d*x + e
function nnet.NL(nl, sizesLongStorage, options)  
    -- construct the module
    
    local Seq1 = nn.Sequential()
    Seq1:add(nn.Mul(sizesLongStorage))
    Seq1:add(nn.Add(sizesLongStorage, c))
    Seq1:add(nl())
    Seq1:add(nn.Mul(sizesLongStorage))

    local Seq2 = nn.Sequential()
    Seq2:add(nn.Mul(sizesLongStorage))
    Seq2:add(nn.Add(sizesLongStorage, e))

    local p = nn.ConcatTable()
    p:add(Seq1)
    p:add(Seq2)

    local NL = nn.Sequential()
    NL:add(p)
    NL:add(nn.CAddTable())

    function NL:updateStaticParameters(options)
        -- set the parameters
        self:get(1):get(1):get(1).weight[1] = options.b
        self:get(1):get(1):get(2).bias[1]   = options.c
        self:get(1):get(1):get(4).weight[1] = options.a
        self:get(1):get(2):get(1).weight[1] = options.d
        self:get(1):get(2):get(2).bias[1]   = options.e
    end

    NL:updateStaticParameters(options)

    return NL
end

--return (a1*tanh(b1*x + c1)  + d1*x + e1) + (a2*tanh(b2*x + c2) + d2*x + e2)
function nnet.DoubleNL(nl, sizesLongStorage, options1, options2)
    local p = nn.ConcatTable()
    p:add(nnet.NL(nl, sizesLongStorage, options1))
    p:add(nnet.NL(nl, sizesLongStorage, options2))

    local DoubleNL = nn.Sequential()
    DoubleNL:add(p)
    DoubleNL:add(nn.CAddTable())

    function DoubleNL:updateStaticParameters(options1, options2)
        self:get(1):get(1):updateStaticParameters(options1)
        self:get(1):get(2):updateStaticParameters(options2)
    end

    DoubleNL:updateStaticParameters(options1, options2)

    return DoubleNL
end

function nnet.get_net(options)

    --[[
    options.NL = nnd.PNL(nnet.DoubleNL(nn.Tanh, 
                    torch.LongStorage({options.h1}), 
                    options,  
                    {a = options.a,  
                     b = options.b, 
                     c = -options.c, 
                     d = options.d, 
                     e = options.e } ))
    --]]

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
            mlp:add(options.NL)
        end
    end
    
    return mlp
end

function nnet.eval_obj(options)
    if type(options.objectiveFunction) == 'function' then
        return
    end
    assert(type(options.obj) == 'string', 'Objective function string error...')

    print('Loading objective function from string: ', options.obj)
    options.objectiveFunction = loadstring(options.obj)()
    assert(options.objectiveFunction, 'Error in loading objective function string...')
end 




function nnet.get_model(options, suppressPrinting)
    -- get a model; default behavior is to load, otherwise create
    local ret = {} 
    if not options.network or options.network == '' then
        if not suppressPrinting then
            print('Creating new model...')   
        end
        options.NL = nnd.PNL(nnet.NL(nn.Tanh, torch.LongStorage({options.h1}), options))
        ret.network = nnet.get_net(options)
    else
        if not suppressPrinting then
            print('Loading previously trained model: ' .. options.network)
        end
        ret = torch.load(options.network)
        
        -- always load objective function from saved net file
        options.obj = ret.options.obj

        -- load NL parameters from saved net file only if undefined in command line
        if options.a == math.huge then
            options.a = ret.options.a
        end

        if options.b == math.huge then
            options.b = ret.options.b
        end
        
        if options.c == math.huge then
            options.c = ret.options.c
        end
        
        if options.d == math.huge then
            options.d = ret.options.d
        end
        
        if options.e == math.huge then
            options.e = ret.options.e
        end
        
        options.NL = nnd.PNL(nnet.NL(nn.Tanh, torch.LongStorage({options.h1}), options))
    end
    local model = ret.network
    nnet.eval_obj(options)

    
    nnet.updatePNLParameters(model, options, suppressPrinting) 

    if not suppressPrinting then
        print(options)
        print(model)
        print('Done')
    end
    return model
end



function nnet.save_network(t, options)
    -- save/log current net
    local filename = paths.concat(options.save, 'mlp.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if sys.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    -- make sure options have been saved with the mlp
    t.options = options

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
end


function nnet.plot(samples, funs, options)
    if options.noplot then
        return
    end
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


