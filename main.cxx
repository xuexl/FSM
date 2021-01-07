
//#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include "torch/torch.h"

#include <cuda.h>

#include"data.h"
#include"config.h"
#include"net.h"

#include"cout.h"


int main()
{    
    
    auto data = load_data_from_folder(Config::datasetPath);
    std::vector<std::string> list_images = data.first;
    std::vector<int> list_labels = data.second;
    
    auto stoneDataset = StoneDataset(list_images, list_labels).map(torch::data::transforms::Stack<>());
    
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                std::move(stoneDataset),  
                torch::data::DataLoaderOptions().batch_size(Config::trainBatchSize).workers(8));    
    
    auto net = std::make_shared<Net>();
    net->train();
    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-3));
    
    for(size_t epoch = 0; epoch < Config::epochs; ++epoch)
    {
        for(auto &batch: *data_loader)
        {            
            auto data = batch.data;
            auto target = batch.target.squeeze();

            data = data.to(torch::kF32);
            target = target.to(torch::kInt64);
            
            optimizer.zero_grad();
            auto output = net->forward(data);
            auto loss = torch::nll_loss(output, target);
            loss.backward();
            optimizer.step();   // Update the parameters
            
            std::cout << "Train Epoch:  " << epoch<<"，Loss:  "<< loss.item<float>() << std::endl;
        }
    }
    
    torch::save(net, "fsm.pt");
    
    
    return 0;
}
