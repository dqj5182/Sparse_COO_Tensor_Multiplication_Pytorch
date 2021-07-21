# Sparse_COO_Tensor_Multiplication_Pytorch

Sparse matrix is everywhere including social network, polymer structure, and road network. As much as dense matrix is widely used, sparse matrix also has potential to be widely used in every domain. However, there are lack of Sparse COO Tensor Multiplication library or even features in Pytorch. In this github, I will introduce a function "sparse_coo_mul()" that takes two Pytorch Sparse COO Tensors and outputs one Pytorch Sparse COO Tensor (result of matrix multiplication of two inputted Pytorch Sparse COO Tensors)

**This is not element wise matrix multiplication, but regular matrix multiplication for two sparse coo tensors implemented in Pytorch**
