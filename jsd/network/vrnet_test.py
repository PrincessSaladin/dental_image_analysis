import torch

from jsd.network.vrnet import RegressionNet


def test_vrnet():
  batch_size, in_channel, dim_z, dim_y, dim_x = 2, 1, 64, 32, 32
  input_tensor = torch.randn([batch_size, in_channel, dim_z, dim_y, dim_x])
  
  input_channel, output_channel = 1, 276
  network = RegressionNet([dim_z, dim_y, dim_x], input_channel, output_channel)
  output_tensor = network(input_tensor)
  print(output_tensor.shape)


if __name__ == '__main__':
  
  test_vrnet()