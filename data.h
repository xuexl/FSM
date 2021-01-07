#ifndef DATA_H
#define DATA_H

#include <torch/torch.h>
#include <vector>
#include <string>


/* This function returns a pair of vector of images paths (strings) and labels (integers) */
std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(std::string path);


class StoneDataset: public torch::data::datasets::Dataset<StoneDataset>
{
public:
    StoneDataset(std::vector<std::string> list_images, std::vector<int> list_labels);
    
    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override;
    
    // Return the length of data
    torch::optional<size_t> size() const override;
    
private:
    std::vector<torch::Tensor> images, labels;
    
};

#endif // DATA_H
