require 'torch'
require 'nn'
require 'util'
require 'util/arg'
require 'sys'
require 'dataset'
require 'dataset/TableDataset'

require 'cv'


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

