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
    cmd:option('-params',       '',         'params dataset')
    cmd:option('-corr',         false,      'display data and correlation coefficients before and after whitening')

    cmd:text()
    return cmd:parse(arg)
end



function display_corr(ds) 

    local data = ds.dataset.data:squeeze() -- :narrow(1, 1, 100)

    print(data:size())

    image.display({image=data, symmetric=true, min=-1, max=1})

    print('Min data:', torch.min(data))
    print('Mean data:', torch.mean(data))
    print('Max data:', torch.max(data))

    local corr = torch.Tensor(data:size(2), data:size(2))

    local means = torch.mean(data, 1):squeeze()
    local stds = torch.std(data, 1):squeeze()

    for i=1,data:size(2) do
        for j=i,data:size(2) do
            corr[j][i] = ((data[{{}, i}] - means[i]) * (data[{{}, j}] - means[j])) / stds[i] / stds[j]/data:size(1)
            corr[i][j] = corr[j][i]
        end
    end

    print('Min corr:', torch.min(corr))
    print('Mean corr:', torch.mean(corr))
    print('Max corr:', torch.max(corr))


    image.display({image=corr, symmetric=true, min=-1, max=1, legend=filename..' input dimension correlation coefficient'})
end



local function main()
    local options = parse_arg(arg)
    

    assert(options.input ~= '')


    ds = torch.load(options.input)
    print('Input dataset: ', ds)

    if options.corr then 
        display_corr(ds)
    end
    
    if options.params == '' then 
    	print('Estimating parameters ...')
        ds.dataset.data, ds.means, ds.P, ds.invP = unsup.zca_whiten(ds.dataset.data)
        assert(ds.means)
        assert(ds.P)
        assert(ds.invP)
        print('Done.')
    else
    	print('Loading parameters...')
        local params_ds = torch.load(options.params)
        assert(params_ds.means)
        assert(params_ds.P)
        assert(params_ds.invP)
        ds.dataset.data, ds.means, ds.P, ds.invP = unsup.zca_whiten(ds.dataset.data, params_ds.means, params_ds.P, params_ds.invP)
        assert(ds.means)
        assert(ds.P)
        assert(ds.invP)
        print('Done.')
    end
    print('Whitened dataset: ', ds)

    if options.corr then 
        display_corr(ds)
    end

    if options.output ~= '' then 
       torch.save(options.output, ds)
    end

end


do 
    main()
end
