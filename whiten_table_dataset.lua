#!/usr/bin/env torch-qlua
require 'torch'
require 'torch-env'
require 'dataset'
require 'dataset/TableDataset'
require 'dataset/whitening'
require 'util'
require 'util/arg'
require 'VerySparseRandomProjections'



local function parse_arg(arg)
    local dname, fname = sys.fpath()
    local cmd = torch.CmdLine()
    cmd:text('Options:')
    cmd:option('-input',        '',         'input table dataset which will be ZCA-whitened')
    cmd:option('-project',      0,          'use sparse random projections of the input table dataset before whitening')
    cmd:option('-seed',         0,          'random number generator seed')
    cmd:option('-output',       '',         'output table dataset')
    cmd:option('-params',       '',         'params dataset')
    cmd:option('-corr',         false,      'display data and correlation coefficients before and after whitening')
    cmd:option('-quick',        false,      'do not compute correlation coefficients before and after whitening')

    cmd:text()
    return cmd:parse(arg)
end


function project(ds, proj, options)
    ds.dataset.data = proj:forward(ds.dataset.data):clone()
end

function display_corr(ds, options) 

    local data = ds.dataset.data:squeeze() -- :narrow(1, 1, 100)

    if options.corr then 
        image.display({image=data, symmetric=true, min=-1, max=1})
    end

    print('Dataset stats:')
    print({ Min     =   torch.min(data), 
            Mean    =   torch.mean(data),
            Max     =   torch.max(data)     })

    if options.quick then
        return 
    end

    local corr = torch.Tensor(data:size(2), data:size(2))

    local means = torch.mean(data, 1):squeeze()
    local stds = torch.std(data, 1):squeeze()

    for i=1,data:size(2) do
        for j=i,data:size(2) do
            corr[j][i] = ((data[{{}, i}] - means[i]) * (data[{{}, j}] - means[j])) / stds[i] / stds[j]/data:size(1)
            corr[i][j] = corr[j][i]
        end
    end

    print('Dataset stats:')
    print({ Min_corr     =   torch.min(corr),
            Mean_corr    =   torch.mean(corr),
            Max_corr     =   torch.max(corr)    })


    if options.corr then 
        image.display({image=corr, symmetric=true, min=-1, max=1, legend='input dimension correlation coefficient'})
    end
end



local function main()
    local options = parse_arg(arg)
   
    torch.manualSeed(options.seed)

    assert(options.input ~= '')


    ds = torch.load(options.input)
    print('Input dataset: ', ds)


    
    if options.params == '' then 
        if options.project > 0 then
            local data = ds.dataset.data
    	    print('Creating very sparse random projection matrix ...')
            ds.rand_projection_layer = nn.VerySparseRandomProjections(data:nElement()/data:size(1), options.project)
            
            print('Projection matrix column std stats:')
            print({ Min     =   torch.min(torch.std(ds.rand_projection_layer.weight, 1)), 
                    Mean    =   torch.mean(torch.std(ds.rand_projection_layer.weight, 1)),
                    Max     =   torch.max(torch.std(ds.rand_projection_layer.weight, 1))     })

            if options.corr then
                
                image.display({image=torch.cat(ds.rand_projection_layer.weight, 
                                                ds.rand_projection_layer.bias, 2), 
                                                    symmetric=true, min=-1, max=1, legend='very sparse random projection matrix'})
            end
            print('Done.')
    	    
            print('Projecting on '..options.project..' dimensions ...')
            ds.dataset.data = ds.rand_projection_layer:forward(data:resize(data:size(1), 
                                        data:nElement()/data:size(1))):resize(data:size(1), 1, 1, options.project):clone()
            print('Done.')
        end

        display_corr(ds, options)

    	print('Estimating whitening parameters ...')
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

        if options.project > 0 then
            local data = ds.dataset.data
            print('Loading sparse random projection matrix ...')
            ds.rand_projection_layer = params_ds.rand_projection_layer:clone()
            options.project = params_ds.rand_projection_layer.weight:size(1)
 
            print('Projection matrix column std stats:')
            print({ Min     =   torch.min(torch.std(ds.rand_projection_layer.weight, 1)), 
                    Mean    =   torch.mean(torch.std(ds.rand_projection_layer.weight, 1)),
                    Max     =   torch.max(torch.std(ds.rand_projection_layer.weight, 1))     })


            if options.corr then
                image.display({image=torch.cat(ds.rand_projection_layer.weight, 
                                                ds.rand_projection_layer.bias, 2), 
                                                    symmetric=true, min=-1, max=1, legend='very sparse random projection matrix'})

            end
            print('Done.')
    	    
            print('Projecting on '..options.project..' dimensions ...')
            ds.dataset.data = ds.rand_projection_layer:forward(data:resize(data:size(1), 
                                        data:nElement()/data:size(1))):resize(data:size(1), 1, 1, options.project):clone()
            print('Done.')
        end

        display_corr(ds, options)

        ds.dataset.data, ds.means, ds.P, ds.invP = unsup.zca_whiten(ds.dataset.data, params_ds.means, params_ds.P, params_ds.invP)
        assert(ds.means)
        assert(ds.P)
        assert(ds.invP)
        print('Done.')
    end
    print('Whitened dataset: ', ds)

    display_corr(ds, options)

    if options.project > 0 then 
        print('Projection matrix column std stats:')
        print({ Min     =   torch.min(torch.std(ds.rand_projection_layer.weight, 1)), 
                Mean    =   torch.mean(torch.std(ds.rand_projection_layer.weight, 1)),
                Max     =   torch.max(torch.std(ds.rand_projection_layer.weight, 1))     })

    end

    if options.output ~= '' then 
        print('Saving output to: '..options.output)
        torch.save(options.output, ds)
        print('Done.')
    end

end


do 
    main()
end
