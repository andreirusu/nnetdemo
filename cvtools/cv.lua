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



