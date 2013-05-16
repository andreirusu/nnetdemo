#!/usr/bin/env torch-qlua
require 'torch'
require 'torch-env'
require 'dataset'
require 'dataset/TableDataset'
require 'dataset/whitening'
require 'util'
require 'util/arg'



local function parse_arg(arg)
    local dname, fname = sys.fpath()
    local cmd = torch.CmdLine()
    cmd:text('Options:')
    cmd:option('-input',        '',         'input table dataset which will be ZCA-whitened')
    cmd:option('-output',       '',         'output table dataset')
    cmd:option('-params',       '',         'output table dataset')

    cmd:text()
    return cmd:parse(arg)
end




local function main()
    local options = parse_arg(arg)
    
    assert(options.input ~= '')
    assert(options.output ~= '')


    ds = torch.load(options.input)
    print(ds.dataset)
    
    if options.params == '' then 
    	print('Estimating parameters ...')
        ds.dataset.data, ds.means, ds.P, ds.invP = unsup.zca_whiten(ds.dataset.data)
        assert(ds.means)
        assert(ds.P)
        assert(ds.invP)
    else
    	print('Loading parameters...')
        local params_ds = torch.load(options.params)
        assert(params_ds.means)
        assert(params_ds.P)
        assert(params_ds.invP)
        ds.dataset.data, ds.means, ds.P, ds.invP = unsup.zca_whiten(ds.dataset.data, params_ds.means, params_ds.P, params_ds.invP)
    end
    print(ds)
    torch.save(options.output, ds)
end


do 
    main()
end
