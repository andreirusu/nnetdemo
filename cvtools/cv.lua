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
    cmd:option('-dataset',      '',         'subdirectory to save/log experiments in')
    cmd:option('-k',            4,                     'reload pretrained network')
    cmd:option('-seed',         0,                      'fixed input seed for repeatable experiments')
    
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

function cv.getTableDatasets(options)
    return {train=torch.load(paths.concat(options.dataset, 'train.th7')), 
                test=torch.load(paths.concat(options.dataset, 'test.th7'))}
end


function cv.set_options(options)
    local options = options or {}
    assert(options.dataset, 'A dataset path must be specified!') 
    assert(paths.dirp(options.dataset), '-dataset path must be a directory containing two valid th7 files: train.th7 & test.th7!')
    assert(paths.filep(paths.concat(options.dataset, 'train.th7')), '-dataset path must be a directory containing a valid train.th7 file!')
    assert(paths.filep(paths.concat(options.dataset, 'test.th7')), '-dataset path must be a directory containing a valid test.th7 file!')
    return options
end



