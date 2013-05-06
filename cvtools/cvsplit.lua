require 'torch'
require 'nn'
require 'util'
require 'util/arg'
require 'sys'
require 'dataset'
require 'dataset/TableDataset'

require 'cv'

local function permute_rows(data, perm)
    if not data then return nil end
    local pdata = torch.Tensor(data:size())
    if pdata:nDimension() == 1 then
        for i = 1,data:size(1) do
            pdata[perm[i]] = data[i]
        end
    else
        for i = 1,data:size(1) do
            pdata[perm[i]]:copy(data[i])
        end
    end
    assert(torch.sum(torch.abs(torch.mean(data, 1) 
                                - torch.mean(pdata, 1))) < 1e-10)
    return pdata
end

function cv.KFolds(_ds, options) 
    local ds = _ds.dataset
    
    if ds.cass then 
        perm_class = torch.Tensor(ds.class:size())
    end

    local perm = torch.randperm(ds.data:size(1))
    
    local perm_data = permute_rows(ds.data, perm)
    local perm_class = permute_rows(ds.class, perm)

    local folds = {}
    
    assert(not paths.dirp('kfolds'))
    local fold_size = math.abs(ds.data:size(1)/options.k)
    local root_path = './kfolds/'
    print(fold_size)
    -- first fold
    do
        local i = 1
        os.execute('mkdir -p '..root_path..i)
        do
            -- save test
            local ds_table = {}
            ds_table.data=perm_data:narrow(1, 1, fold_size):clone() 
            if perm_class then ds_table.class=perm_class:narrow(1, 1, fold_size):clone() end 
            local fold_ds = dataset.TableDataset(ds_table)
            print(fold_ds)
            torch.save(paths.concat(paths.concat(root_path, i), 'test.th7'), fold_ds)
        end
        
        do
            -- save train
            local ds_table = {}
            ds_table.data=perm_data:narrow(1, fold_size+1, perm_data:size(1) - fold_size):clone() 
            if perm_class then 
                ds_table.class=perm_class:narrow(1, fold_size+1, perm_class:size(1) - fold_size):clone() 
            end 
            local fold_ds = dataset.TableDataset(ds_table)
            print(fold_ds)
            torch.save(paths.concat(paths.concat(root_path, i), 'train.th7'), fold_ds)
        end 
    end
    -- intermediate folds
    for i=2,options.k-1 do
        os.execute('mkdir -p '..root_path..i)
        
        do
            -- save test
            local ds_table = {}
            ds_table.data=perm_data:narrow(1, (i-1)*fold_size+1, fold_size):clone() 
            if perm_class then ds_table.class=perm_class:narrow(1, (i-1)*fold_size+1, fold_size):clone() end 
            local fold_ds = dataset.TableDataset(ds_table)
            print(fold_ds)
            torch.save(paths.concat(paths.concat(root_path, i), 'test.th7'), fold_ds)
        end

        do
            -- save train
            local ds_table = {}
            ds_table.data=torch.cat(perm_data:narrow(1, 1, (i-1)*fold_size),
                                    perm_data:narrow(1, i*fold_size+1, perm_data:size(1) - i*fold_size), 1):clone() 
            if perm_class then 
                ds_table.class=torch.cat(perm_class:narrow(1, 1, (i-1)*fold_size),
                                            perm_class:narrow(1, i*fold_size+1, perm_class:size(1) - i*fold_size), 1):clone() 
            
            end 
            local fold_ds = dataset.TableDataset(ds_table)
            print(fold_ds)
            torch.save(paths.concat(paths.concat(root_path, i), 'train.th7'), fold_ds)
        end 
    end
    -- last fold
    do
        local i = options.k
        os.execute('mkdir -p '..root_path..i)
        do
            -- save test
            local ds_table = {}
            ds_table.data=perm_data:narrow(1, (i-1)*fold_size+1, perm_data:size(1) - (i-1)*fold_size):clone() 
            if perm_class then ds_table.class=perm_class:narrow(1, (i-1)*fold_size+1, perm_class:size(1) - (i-1)*fold_size):clone() end 
            local fold_ds = dataset.TableDataset(ds_table)
            print(fold_ds)
            torch.save(paths.concat(paths.concat(root_path, i), 'test.th7'), fold_ds)
        end

        do
            -- save train
            local ds_table = {}
            ds_table.data=perm_data:narrow(1, 1, (i-1)*fold_size):clone() 
            if perm_class then 
                ds_table.class=perm_class:narrow(1, 1, (i-1)*fold_size):clone() 
            end 
            local fold_ds = dataset.TableDataset(ds_table)
            print(fold_ds)
            torch.save(paths.concat(paths.concat(root_path, i), 'train.th7'), fold_ds)
        end 
    end
    -- test folds
    for i=1,options.k do
        print('Checking fold ', i)
        local train_ds = torch.load(paths.concat(paths.concat(root_path, i), 'train.th7'))
        local test_ds = torch.load(paths.concat(paths.concat(root_path, i), 'test.th7'))

        print(test_ds)
        print(train_ds)

        assert(torch.std(torch.cat(train_ds.dataset.data, test_ds.dataset.data, 1)) 
                                     - torch.std(perm_data) < 1e-10)
        assert(torch.std(torch.cat(train_ds.dataset.class, test_ds.dataset.class, 1)) 
                                     - torch.std(perm_class) < 1e-10)
    end
    
    return folds
end

local function main()
    local options = cv.parse_arg(arg, true)
    
    options = cv.set_options(options)
  
    cv.init_experiment(options)
    
    ds = cv.getTableDataset(options.dataset)

    cv.KFolds(ds, options) 
end



do 
    main()
end

