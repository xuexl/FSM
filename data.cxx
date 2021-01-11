#include "data.h"

#include <tuple>
#include <opencv2/opencv.hpp>

#include<io.h>
#include"cout.h"


/* Convert and Load image to tensor from location argument */
torch::Tensor read_data(std::string loc)
{
    cv::Mat img = cv::imread(loc, 0);
    cv::resize(img, img, cv::Size(200, 200), cv::INTER_CUBIC);
//    std::cout << "Sizes: " << img.size() << std::endl;
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 1}, torch::kByte);
    img_tensor = img_tensor.permute({2,0,1});   // Channels x Height x Width
    
    return img_tensor.clone();
}

/* Converts label to tensor type in the integer argument */
torch::Tensor read_label(int label)
{    
//    torch::Tensor label_tensor = torch::full({1}, label);    
    torch::Tensor label_tensor = torch::ones({1});    
    return label_tensor.clone();
}

/* Loads images to tensor type in the string argument */
std::vector<torch::Tensor> process_images(std::vector<std::string> list_images)
{
    std::vector<torch::Tensor> states;

    for(auto &it: list_images)
    {
        torch::Tensor img = read_data(it);
        states.emplace_back(img);
    }
    
    return states;
}

/* Loads labels to tensor type in the string argument */
std::vector<torch::Tensor> process_labels(std::vector<int> list_labels)
{
    std::vector<torch::Tensor> labels;
    
    for(auto &it: list_labels)
    {
        torch::Tensor label = read_label(it);
        labels.emplace_back(label);
    }
    
    return labels;
}

std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(std::string path)
{
    std::vector<std::string> list_images;
    std::vector<int> list_labels;
    
    intptr_t hFile = 0;
    struct _finddata_t fileinfo;
    std::string tmp;    
    if((hFile = _findfirst(tmp.assign(path).append("\\*.jpg").c_str(), &fileinfo)) != -1)
    {                        
        do
        {
            if(fileinfo.attrib & _A_ARCH)
            {                
                list_images.emplace_back(tmp.assign(path).append("\\").append(fileinfo.name));
                list_labels.emplace_back(1);
            }
        }while(_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
    
    return std::make_pair(list_images, list_labels);
}

/********************/
StoneDataset::StoneDataset(std::vector<std::string> list_images, std::vector<int> list_labels)
{
    this->images = process_images(list_images);    
    this->labels = process_labels(list_labels);    
}

torch::data::Example<> StoneDataset::get(size_t index) 
{
    torch::Tensor sample_img = this->images.at(index);
    torch::Tensor sample_label = this->labels.at(index);
    return {sample_img.clone(), sample_label.clone()};
}

torch::optional<size_t> StoneDataset::size() const
{
    return this->labels.size();
}
