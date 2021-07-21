import torch
import numpy as np

def sparse_coo_mul(tensor1, tensor2):
  tensor1_indices = tensor1.coalesce().indices()
  tensor2_indices = tensor2.coalesce().indices()
  tensor1_values = tensor1.coalesce().values()
  tensor2_values = tensor2.coalesce().values()

  def swap_two_rows(tensor):
    index = [1, 0]
    tensor[index,] = tensor.clone()
    return tensor

  test_1 = tensor1_indices.t()
  test_2 = swap_two_rows(tensor2_indices).t()

  mul_i_j = []

  final = {}

  final_list = []

  for i, val_i in enumerate(test_1):
    for j, val_j in enumerate(test_2):

      if torch.equal(val_i[1], val_j[1]):
        mul_i_j.append([val_i[0].item(), val_j[0].item(), (tensor1_values[i]*tensor2_values[j]).item()])

  for ele1, ele2, ele3 in mul_i_j:
    if (ele1, ele2) not in final:
      final[(ele1, ele2)] = ele3
    else:
      final[(ele1, ele2)] = final[(ele1, ele2)] + ele3

  for key, value in final.items():
    temp = [key[0], key[1] ,value]
    final_list.append(temp)

  final_list = np.array(final_list).T

  new_index = final_list[0:2,]
  new_value = final_list[2,]

  new_sparse_coo_tensor = torch.sparse_coo_tensor(new_index, new_value)
  
  return new_sparse_coo_tensor
