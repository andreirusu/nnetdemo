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
    
    data = cv.getTableDatasets(options)

    print(data.train)
    print(data.test)
     
end



do 
    main()
end

