import torch

flag = torch.cuda. is_available()
print(flag) # 返网true为安装成功

ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())

import torch

# Check CUA version

cuda_version = torch.version.cuda
print("CUDA Version:", cuda_version)
# Check CuM version
cudnn_version = torch.backends.cudnn.version()
print("cuDNN Version:", cudnn_version)