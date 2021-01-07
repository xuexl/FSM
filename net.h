#ifndef NET_H
#define NET_H

#include <torch/torch.h>

class Net: public torch::nn::Module
{
public:
    Net();
    
    // Implement Algorithm
    torch::Tensor forward(torch::Tensor x);
    
private:
    // Declare layers
    torch::nn::Conv2d conv1_1{nullptr};
    torch::nn::Conv2d conv1_2{nullptr};
    torch::nn::Conv2d conv2_1{nullptr};
    torch::nn::Conv2d conv2_2{nullptr};
    torch::nn::Conv2d conv3_1{nullptr};
    torch::nn::Conv2d conv3_2{nullptr};
    torch::nn::Conv2d conv3_3{nullptr};
    torch::nn::Conv2d conv4_1{nullptr};
    torch::nn::Conv2d conv4_2{nullptr};
    torch::nn::Conv2d conv4_3{nullptr};
    torch::nn::Conv2d conv5_1{nullptr};
    torch::nn::Conv2d conv5_2{nullptr};
    torch::nn::Conv2d conv5_3{nullptr};
    
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
    
};

#endif // NET_H
