import torch
import torch.nn as nn
from torch_sparse import spmm


class MeshUnpool(nn.Module):
    def __init__(self, unroll_target):
        super(MeshUnpool, self).__init__()
        self.unroll_target = unroll_target
        self.result = None

    def __call__(self, features, meshes):
        return self.forward(features, meshes)

    def pad_groups(self, group, unroll_start):
        start, end = group.shape
        padding_rows =  unroll_start - start
        padding_cols = self.unroll_target - end
        if padding_rows != 0 or padding_cols !=0:
            padding = nn.ConstantPad2d((0, padding_cols, 0, padding_rows), 0)
            group = padding(group)
        return group

    def sparse_pad_groups(self, group, unroll_start):
        if not group.is_coalesced():
            group = group.coalesce()
        return torch.sparse.FloatTensor(group.indices(), group.values(), (unroll_start, self.unroll_target)).coalesce()

    def pad_occurrences(self, occurrences):
        padding = self.unroll_target - occurrences.shape[0]
        if padding != 0:
            padding = nn.ConstantPad1d((0, padding), 1)
            occurrences = padding(occurrences)
        return occurrences

    def olforward(self, features, meshes):
        batch_size, nf, edges = features.shape
        groups = [mesh for mesh in meshes]
        groups = [self.sparse_pad_groups(mesh.get_groups().to_sparse(), edges) for mesh in groups]


    def forward(self, features, meshes):
        batch_size, nf, edges = features.shape
        groups = [mesh.get_groups() for mesh in meshes]
        #groups = [self.pad_groups(group, edges) for group in og_groups]
        #unroll_mat = torch.cat(groups, dim=0).view(batch_size, edges, -1)
        og_occu = [mesh.get_occurrences() for mesh in meshes]
        occurrences = [self.pad_occurrences(mesh) for mesh in og_occu]
        occurrences = torch.cat(occurrences, dim=0).view(batch_size, 1, -1)
        occurrences = occurrences.expand((batch_size, edges, self.unroll_target))
        #unroll_mat = unroll_mat / occurrences
        #unroll_mat = unroll_mat.to(features.device)


        groups = [self.sparse_pad_groups(mesh, edges) for mesh in groups]
        indices = torch.cat([torch.cat([torch.ones((1, g.indices().shape[-1]), dtype=torch.int64).to(features.device) * idx, g.indices()], dim=0) for
                                                            idx, g in enumerate(groups)], dim=1)
        values = torch.cat([g.values() for g in groups], dim=0)

        values = values / occurrences[indices[0,:], indices[1,:], indices[2,:]]
        #groups = torch.sparse.FloatTensor(indices, values, (batch_size, edges, self.unroll_target)).coalesce()

        #return torch.matmul(features, unroll_mat)

        if self.result is None or self.result.shape != (batch_size, features.shape[1], self.unroll_target):
            result = torch.zeros((batch_size, features.shape[1], self.unroll_target), device=features.device)



        # transpose
        b, row, col = indices
        indices = torch.stack([b, col, row], dim=0)
        transposed_features = features.transpose(1,2)

        for b_idx in range(batch_size):
            mask = indices[0,:] == b_idx
            #tmp = torch.sparse.FloatTensor(indices[1:, mask], values[mask], (self.unroll_target, edges))
            #result[b_idx, :, :] = torch.sparse.mm(tmp, transposed_features[b_idx, :, :]).T
            result[b_idx, :, :] = spmm(indices[1:, mask], values[mask], self.unroll_target, edges, transposed_features[b_idx, :, :]).T

        for mesh in meshes:
            mesh.unroll_gemm()

        return result