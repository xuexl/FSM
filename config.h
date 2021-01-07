#ifndef CONFIG_H
#define CONFIG_H

#include<torch/torch.h>

namespace Config
{
    constexpr size_t trainBatchSize = 4;
    constexpr size_t testBatchSize = 100;
    constexpr size_t epochs = 100;
    constexpr size_t logInterval = 20;
    constexpr char datasetPath[] = "C:\\study\\check_stone\\stone\\train";
    constexpr torch::DeviceType device = torch::kCUDA;
}

#endif // CONFIG_H
