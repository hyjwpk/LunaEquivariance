from functools import cache
from math import prod
from types import MethodType

import torch
from .gather_rotation.interface import rotate_gather_order_degree
from .inverse_rotation_reduction import aggregate_sum
from .recombination_activation.interface import gated_activation
from .utils.graph import GraphCSRRuntime
from .utils.layout import get_angular_layout


@cache
def configure_gcvm():
    from triton.backends.amd.compiler import HIPBackend

    original = HIPBackend.make_amdgcn

    def make_amdgcn(source, metadata, options):
        compatible = source.replace("getelementptr inbounds nuw ", "getelementptr inbounds ")
        return original(compatible, metadata, options)

    HIPBackend.make_amdgcn = staticmethod(make_amdgcn)


def rotmat_and_wigner(self, edge_vectors):
    frames, wigner, inverse = type(self).get_rotmat_and_wigner(self, edge_vectors)
    retained = wigner.flatten(1).index_select(1, self.luna_wigner_indices)
    return frames, retained, inverse


def linear_products(convolution, sectors, scalar_prefix=0):
    edges = sectors[0].shape[0]
    scalar = convolution.fc_m0(sectors[0].flatten(1))
    order_zero = scalar[:, scalar_prefix:].unflatten(1, (-1, convolution.m_output_channels))
    products = [order_zero]
    for value, layer in zip(sectors[1:], convolution.so2_m_conv, strict=True):
        width = prod(value.shape[1:]) // 2
        product = layer.fc(value.reshape(edges * 2, width))
        products.append(product.reshape(edges, 2, product.shape[-1]))
    return scalar[:, :scalar_prefix], tuple(products)


def edgewise(self, x, x_edge, edge_distance, edge_index, wigner, wigner_inv, node_offset=0):
    graph = self.luna_graph
    index = self.luna_block_index
    try:
        metadata = graph.consume(index, edge_index, x.shape[0], node_offset)
        first, second = self.so2_conv_1, self.so2_conv_2
        sectors = rotate_gather_order_degree(
            x, edge_index, first.rad_func(x_edge), wigner,
            lmax=self.lmax, mmax=self.mmax, metadata=metadata,
        )
        gate, projected = linear_products(first, sectors, first.extra_m0_output_channels)
        activated = gated_activation(
            gate, projected[0], projected[1:], lmax=self.lmax, mmax=self.mmax
        )
        _, projected = linear_products(second, activated)
        output = aggregate_sum(
            projected, self.envelope(edge_distance / self.cutoff), wigner,
            edge_index[1] - node_offset, x.shape[0], lmax=self.lmax, mmax=self.mmax,
            row_ptr=metadata[2],
        )
        graph.finish(index)
        return output
    except Exception:
        graph.reset()
        raise


def optimize(model):
    backbone = model.backbone
    if getattr(backbone.get_rotmat_and_wigner, "__func__", None) is rotmat_and_wigner:
        return model

    edges = tuple(block.edge_wise for block in backbone.blocks)
    configure_gcvm()
    graph = GraphCSRRuntime(len(edges))
    layout = get_angular_layout(lmax=backbone.lmax, mmax=backbone.mmax, channels=1)
    indices = [
        row * layout.full_components + column
        for degree in range(backbone.lmax + 1)
        for row, row_degree in zip(layout.dense_rows, layout.reduced_degrees, strict=True)
        if row_degree == degree
        for column in range(degree**2, (degree + 1)**2)
    ]
    backbone.register_buffer(
        "luna_wigner_indices",
        torch.tensor(indices, dtype=torch.long, device=backbone.Jd_0.device),
        persistent=False,
    )
    backbone.get_rotmat_and_wigner = MethodType(rotmat_and_wigner, backbone)
    for index, edge in enumerate(edges):
        edge.luna_graph = graph
        edge.luna_block_index = index
        edge.forward = MethodType(edgewise, edge)
    return model
