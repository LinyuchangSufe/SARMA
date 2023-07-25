'''
3-dim tensor matrization and inverse operation
----------------------------------------------
fold: matrix 2 tensor
unfold: tensor 2 matrix
'''

import torch


class tensor_op:

    def fold(matrix, shape, mode):
        if matrix.dtype == 'float64':
            matrix = torch.from_numpy(matrix)
        if mode == 0:
            shape = [shape[0], shape[2], shape[1]]
            tensor = matrix.reshape(shape).permute(0, 2, 1)
        elif mode == 1:
            shape = [shape[1], shape[2], shape[0]]
            tensor = matrix.reshape(shape).permute(2, 0, 1)
        elif mode == 2:
            shape = [shape[2], shape[1], shape[0]]
            tensor = matrix.reshape(shape).permute(2, 1, 0)
        return tensor

    def unfold(tensor, mode):
        if tensor.dtype == 'float64':
            tensor = torch.from_numpy(tensor)
        shape = tensor.shape
        if mode == 0:
            matrix = tensor.permute(0, 2, 1).reshape(shape[0], -1)
        elif mode == 1:
            matrix = tensor.permute(1, 2, 0).reshape(shape[1], -1)
        elif mode == 2:
            matrix = tensor.permute(2, 1, 0).reshape(shape[2], -1)
        return matrix

    # def tensorize(vector, shape, mode):
    #     if vector.dtype == 'float64':
    #         vector = torch.from_numpy(vector)
    #     if mode == 0:
    #         tensor = vector.reshape(shape[2], shape[1], shape[0]).permute(2, 1, 0)
    #     elif mode == 1:
    #         tensor = vector.reshape(shape[2], shape[0], shape[1]).permute(1, 2, 0)
    #     elif mode == 2:
    #         tensor = vector.reshape(shape[1], shape[0], shape[2]).permute(1, 0, 2)
    #     return (tensor)

    # def vectorize(tensor, mode):
    #     if tensor.dtype == 'float64':
    #         tensor = torch.from_numpy(tensor)
    #     matrix = tensor_op.unfold(tensor, mode)
    #     vector = matrix.permute(1,0).reshape(-1)
    #     return vector

    # def kron(A, B):
    #     AB = torch.einsum("ab,cd->acbd", A, B).contiguous()  # contiguous 需要加在view前面
    #     AB = AB.view(A.size(0) * B.size(0), A.size(1) * B.size(1))
    #     return AB