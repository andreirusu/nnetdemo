require 'torch'
require 'dataset'
require 'experiment'


local function main()
    local ret = experiment.loadDataset({dataset=paths.cwd()})
    torch.save('train.th7', ret.train)
    torch.save('test.th7', ret.test)
end

do 
    main()
end

