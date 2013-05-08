require 'torch'
require 'nn'
require 'util'
require 'util/arg'
require 'sys'
require 'dataset'
require 'dataset/TableDataset'

cv = {}

function cv.parse_arg(arg, initNLparams)
    local dname, fname = sys.fpath()
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Cross-validation tool for splitting torch-datasets in folds which are saved in the current directory')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-dataset',      '',         'saved TableDataset which will be split up')
    cmd:option('-k',            4,          'number of folds')
    cmd:option('-seed',         0,          'fixed input seed for repeatable experiments')
    cmd:option('-resample',     50,         'fixed input seed for repeatable experiments')
    
    cmd:text()
    return cmd:parse(arg)
end


function cv.init_experiment(options)
    -- set random seed
    torch.manualSeed(options.seed)
end


function cv.wrapFunction(str)
    return loadstring('return '..str)()
end

function cv.getTableDataset(filepath)
    local ret = torch.load(filepath) 
    print(ret)
    return ret
end


function cv.set_options(options)
    local options = options or {}
    assert(options.dataset, 'A dataset path must be specified!') 
    assert(paths.filep(options.dataset), '-dataset path must be a valid TableDataset!')
    return options
end


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
                                - torch.mean(pdata, 1))) < 1e-3)
    return pdata
end


local function check_ds(ds1, ds2)
    local data_std1 = torch.std(torch.cat(ds1[1].dataset.data, ds1[2].dataset.data, 1))
    local data_std2 = torch.std(torch.cat(ds2[1].dataset.data, ds2[2].dataset.data, 1))
    assert(torch.abs(data_std1 - data_std2) < 1e-2, 'Missmatch data stds: '..tostring(data_std1)..' ~= '..tostring(data_std2))

    local class_std1 = torch.std(torch.cat(ds1[1].dataset.class, ds1[2].dataset.class, 1))                              
    local class_std2 = torch.std(torch.cat(ds2[1].dataset.class, ds2[2].dataset.class, 1))          
    assert(class_std1 == class_std2, 'Missmatch class stds: '..tostring(class_std1)..' ~= '..tostring(class_std2))

    print('Data std abs. difference: '..tostring(torch.abs(data_std1 - data_std2)))
    print('Class std difference: '..tostring(torch.abs(class_std1 - class_std2)))
    print('OK!')
end


local function write_ds_file(filename, trainPath, testPath)

    local ds_file_string = "require 'dataset/TableDataset' \n\n" .. 
        "return { \n" ..
        "    train = torch.load('"..trainPath.."'),\n" ..
        "    test = torch.load('"..testPath.."')\n" ..
        "}\n" 

    local f = io.open(filename, 'w')
    f:write(ds_file_string)
    f:close()

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
            fold_ds = cv.resample(fold_ds, options)
            fold_ds.source = options.dataset
            fold_ds.perm = perm
            fold_ds.testIdx = {1, fold_size}
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
            fold_ds = cv.resample(fold_ds, options)
            fold_ds.source = options.dataset
            fold_ds.perm = perm
            fold_ds.trainIdx = {fold_size+1, perm_data:size(1)}
            print(fold_ds)
            torch.save(paths.concat(paths.concat(root_path, i), 'train.th7'), fold_ds)
        end
        
        write_ds_file(paths.concat(paths.concat(root_path, i), 'fold'), 
                        paths.concat(paths.concat(root_path, i), 'train.th7'), 
                        paths.concat(paths.concat(root_path, i), 'test.th7'))

    end
    collectgarbage()
    -- intermediate folds
    for i=2,options.k-1 do
        os.execute('mkdir -p '..root_path..i)
        
        do
            -- save test
            local ds_table = {}
            ds_table.data=perm_data:narrow(1, (i-1)*fold_size+1, fold_size):clone() 
            if perm_class then ds_table.class=perm_class:narrow(1, (i-1)*fold_size+1, fold_size):clone() end 
            local fold_ds = dataset.TableDataset(ds_table)
            fold_ds = cv.resample(fold_ds, options)
            fold_ds.source = options.dataset
            fold_ds.perm = perm
            fold_ds.testIdx = {(i-1)*fold_size+1, i*fold_size}
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
            fold_ds = cv.resample(fold_ds, options)
            fold_ds.source = options.dataset
            fold_ds.perm = perm
            fold_ds.trainIdx = {{1, (i-1)*fold_size}, {i*fold_size+1, perm_data:size(1)}}
            print(fold_ds)
            torch.save(paths.concat(paths.concat(root_path, i), 'train.th7'), fold_ds)
        end

        write_ds_file(paths.concat(paths.concat(root_path, i), 'fold'), 
                        paths.concat(paths.concat(root_path, i), 'train.th7'), 
                        paths.concat(paths.concat(root_path, i), 'test.th7'))

        collectgarbage()
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
            fold_ds = cv.resample(fold_ds, options)
            fold_ds.source = options.dataset
            fold_ds.perm = perm
            fold_ds.testIdx = {(i-1)*fold_size+1, perm_data:size(1)}
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
            fold_ds = cv.resample(fold_ds, options)
            fold_ds.source = options.dataset
            fold_ds.perm = perm
            fold_ds.trainIdx = {1, (i-1)*fold_size}
            print(fold_ds)
            torch.save(paths.concat(paths.concat(root_path, i), 'train.th7'), fold_ds)
        end
      
        write_ds_file(paths.concat(paths.concat(root_path, i), 'fold'), 
                        paths.concat(paths.concat(root_path, i), 'train.th7'), 
                        paths.concat(paths.concat(root_path, i), 'test.th7'))
    end
    do
        -- test datasets
        local train_ref_ds = torch.load(paths.concat(paths.concat(root_path, 1), 'train.th7'))
        local test_ref_ds = torch.load(paths.concat(paths.concat(root_path, 1), 'test.th7'))

        assert(train_ref_ds)
        assert(test_ref_ds)

        for i=2,options.k do
            print('Checking fold '..i..' against fold '..1)
            do
                local train_ds = torch.load(paths.concat(paths.concat(root_path, i), 'train.th7'))
                local test_ds = torch.load(paths.concat(paths.concat(root_path, i), 'test.th7'))
                check_ds({train_ds, test_ds}, {train_ref_ds, test_ref_ds})
            end
            collectgarbage()
        end
    end
    return folds
end




local function compute_class_index(ds)
    local class = ds.dataset.class
    local class_index = {}
    for i=1,class:size(1) do
        if not class_index[class[i]] then class_index[class[i]] = {} end
        table.insert(class_index[class[i]], i)
    end
    return class_index
end

----- resample options.resample x size-of-dataset times making sure that classes are ballanced
function cv.resample(ds, options)
    local data = ds.dataset.data
    local class = ds.dataset.class
    local newdata
    local newclass
        
    local class_index = compute_class_index(ds)
    print('Class count: ', #class_index)
    for k, v in pairs(class_index) do
        print(k, #v)
    end
    local per_class_count = math.floor(options.resample * class:size(1) / #class_index)
    print('New sample count per class: ', per_class_count)
    local dims = data:size()
    dims[1] = #class_index * per_class_count 
    newdata = torch.Tensor(dims)
    newclass = torch.Tensor(#class_index * per_class_count)
    
    for i=1,#class_index do
        for j=1,per_class_count do
            local ind = torch.random(1, #class_index[i])
            newdata[(i-1)*per_class_count + j]:copy(data[class_index[i][ind]])           
            newclass[(i-1)*per_class_count + j] = class[class_index[i][ind]]
            assert(i == newclass[(i-1)*per_class_count + j])
        end
    end
    
    local newds = dataset.TableDataset({data = newdata, class = newclass})

    return newds
end

