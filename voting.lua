#!/bin/env torch

require 'torch'
require 'torch-env'
require 'util/arg'
require 'sys'

voting = {}


function voting.parse_arg(arg, initNLparams)
    local dname, fname = sys.fpath()
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Voting aggregation of several output files')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-input',            '',                     'input files')
    cmd:option('-output',           '',                     'name of output file')
    cmd:option('-nclasses',         0,                      'number of classes')
    cmd:option('-seed',             0,                      'fixed input seed for repeatable experiments')
    
    cmd:text()
    return cmd:parse(arg)
end



function voting.init_experiment(options)
    -- set random seed
    torch.manualSeed(options.seed)
end

function voting.read_files(options)
    local files = {}
    ---TODO: implement: read all files in memory as strings; we will subsequently read from these strings; alternatively, make sure file reading works properly.    
    return file_strings
end


local function main()
    print(arg)
    local options = voting.parse_arg(arg, true)

    
    voting.init_experiment(options)

    print(options.input)

    --local file_strings = voting.open_files(options)
   
    --TODO: implement voting output
    
end


do 
    main()
end

