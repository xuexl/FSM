#include "net.h"

#include"cout.h"

Net::Net()
{
    // Initialize the network
    // On how to pass strides and padding: https://github.com/pytorch/pytorch/issues/12649#issuecomment-430156160
    conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 3).padding(1)));
    conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 3).padding(1)));
    // Insert pool layer
    conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(20, 30, 3).padding(1)));
    conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(30, 40, 3).padding(1)));
    // Insert pool layer
    conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(40, 50, 3).padding(1)));
    conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(50, 60, 3).padding(1)));
    conv3_3 = register_module("conv3_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(60, 70, 3).padding(1)));
    // Insert pool layer
    conv4_1 = register_module("conv4_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(70, 80, 3).padding(1)));
    conv4_2 = register_module("conv4_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(80, 90, 3).padding(1)));
    conv4_3 = register_module("conv4_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(90, 100, 3).padding(1)));
    // Insert pool layer
    conv5_1 = register_module("conv5_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(100, 110, 3).padding(1)));
    conv5_2 = register_module("conv5_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(110, 120, 3).padding(1)));
    conv5_3 = register_module("conv5_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(120, 130, 3).padding(1)));
    // Insert pool layer
    fc1 = register_module("fc1", torch::nn::Linear(130*6*6, 2000));
    fc2 = register_module("fc2", torch::nn::Linear(2000, 1000));
    fc3 = register_module("fc3", torch::nn::Linear(1000, 100));
    fc4 = register_module("fc4", torch::nn::Linear(100, 2));
}

torch::Tensor Net::forward(torch::Tensor x) 
{ 
    x = conv1_1->forward(x);
    x = torch::relu(x);
    
    x = torch::relu(conv1_2->forward(x));
    x = torch::max_pool2d(x, 2);

    x = torch::relu(conv2_1->forward(x));
    x = torch::relu(conv2_2->forward(x));
    x = torch::max_pool2d(x, 2);

    x = torch::relu(conv3_1->forward(x));
    x = torch::relu(conv3_2->forward(x));
    x = torch::relu(conv3_3->forward(x));
    x = torch::max_pool2d(x, 2);

    x = torch::relu(conv4_1->forward(x));
    x = torch::relu(conv4_2->forward(x));
    x = torch::relu(conv4_3->forward(x));
    x = torch::max_pool2d(x, 2);

    x = torch::relu(conv5_1->forward(x));
    x = torch::relu(conv5_2->forward(x));
    x = torch::relu(conv5_3->forward(x));
    x = torch::max_pool2d(x, 2);


    x = x.view({-1, 130*6*6});

    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = torch::relu(fc3->forward(x));
    x = fc4->forward(x);
        
    return torch::log_softmax(x, 1);    
}
