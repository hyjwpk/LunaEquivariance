import torch

__all__ = ["GraphCSRRuntime"]


def validate_receivers(receivers, nodes, *, sorted=False):
    if receivers.ndim != 1 or receivers.dtype not in (torch.int32, torch.int64):
        raise ValueError("receivers must be a one-dimensional integer tensor")
    if not 0 <= nodes < 2**31 or receivers.numel() >= 2**31:
        raise ValueError("graph sizes must fit the 32-bit CSR representation")
    torch._assert_async(
        ((receivers >= 0) & (receivers < nodes)).all(),
        "receiver indices must lie within the node range",
    )
    if sorted:
        torch._assert_async(
            (receivers[1:] >= receivers[:-1]).all(),
            "CSR aggregation requires edges sorted by receiver",
        )


def validate_edge_index(edge_index, nodes, *, sorted_receivers=False):
    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.int64
    ):
        raise ValueError("edge_index must have shape (2, E) and dtype int64")
    validate_receivers(edge_index[1], nodes, sorted=sorted_receivers)
    torch._assert_async(
        ((edge_index[0] >= 0) & (edge_index[0] < nodes)).all(),
        "source indices must lie within the node range",
    )


def prefix_csr_metadata(edge_index, nodes):
    validate_edge_index(edge_index, nodes, sorted_receivers=True)
    source_order = torch.argsort(edge_index[0], stable=True)
    source_sorted = edge_index[0].index_select(0, source_order)
    boundaries = torch.arange(
        nodes + 1, device=edge_index.device, dtype=edge_index.dtype
    )
    source_row_ptr = torch.searchsorted(source_sorted, boundaries).to(torch.int32)
    receiver_row_ptr = torch._convert_indices_from_coo_to_csr(edge_index[1], nodes).to(
        torch.int32
    )
    return source_order.to(torch.int32), source_row_ptr, receiver_row_ptr


class GraphCSRRuntime:

    def __init__(self, block_count: int):
        self.block_count = block_count
        self.reset()

    def reset(self):
        self.edge_index = None
        self.metadata = None
        self.next_index = 0

    def __deepcopy__(self, memo):
        result = type(self)(self.block_count)
        memo[id(self)] = result
        return result

    def consume(self, index, edge_index, nodes, node_offset):
        try:
            if node_offset != 0:
                raise ValueError("the CSR runtime requires zero node_offset")
            if index == 0:
                self.reset()
                self.edge_index = edge_index
                self.metadata = prefix_csr_metadata(edge_index, nodes)
                self.next_index = 0
            if self.metadata is None or self.edge_index is not edge_index:
                raise RuntimeError("edgewise blocks must share one graph tensor")
            if index != self.next_index:
                raise RuntimeError(
                    f"edgewise order changed: expected {self.next_index}, got {index}"
                )
            self.next_index += 1
            return self.metadata
        except Exception:
            self.reset()
            raise

    def finish(self, index):
        if index == self.block_count - 1:
            self.reset()
