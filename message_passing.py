import torch
import torch_geometric


def image_to_graph(
    image: torch.Tensor, conv2d: torch.nn.Conv2d | None = None
) -> torch_geometric.data.Data:
    """
    Converts an image tensor to a PyTorch Geometric Data object.
    each pixel becomes a node, edges connect neighbors in the kernel window,
    edge attributes encode the kernel position index.

    Arguments:
    ----------
    image : torch.Tensor
        Image tensor of shape (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None
        Is used to determine the size of the receptive field.

    Returns:
    --------
    torch_geometric.data.Data
        Graph representation of the image.
    """
    assert image.dim() == 3, f"Expected 3D tensor, got {image.dim()}D tensor."

    C, H, W = image.shape

    # get conv hyperparameters
    if conv2d is not None:
        kh, kw = conv2d.kernel_size
        pad_h, pad_w = conv2d.padding
    else:
        kh, kw = 5, 5
        pad_h, pad_w = 2, 2

    # node features: flatten pixels, each node has C features
    x = image.reshape(C, H * W).T  # (H*W, C)

    # build edges with vectorized meshgrid
    rows = torch.arange(H)
    cols = torch.arange(W)
    ky_range = torch.arange(kh)
    kx_range = torch.arange(kw)

    # all combinations of (dst_row, dst_col, kernel_y, kernel_x)
    r, c, ky, kx = torch.meshgrid(rows, cols, ky_range, kx_range, indexing='ij')

    # source pixel positions
    r_src = r + ky - pad_h
    c_src = c + kx - pad_w

    # keep only valid source pixels (inside image bounds)
    valid = (r_src >= 0) & (r_src < H) & (c_src >= 0) & (c_src < W)

    # compute node indices and edge attributes
    src_nodes = (r_src[valid] * W + c_src[valid]).long()
    dst_nodes = (r[valid] * W + c[valid]).long()
    edge_attrs = (ky[valid] * kw + kx[valid]).long()

    edge_index = torch.stack([src_nodes, dst_nodes], dim=0)

    return torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attrs)


def graph_to_image(
    data: torch.Tensor, height: int, width: int, conv2d: torch.nn.Conv2d | None = None
) -> torch.Tensor:
    """
    Converts a graph representation of an image to an image tensor.
    just reshapes the node features (N, C) back to (C, H, W).

    Arguments:
    ----------
    data : torch.Tensor
        Graph data representation of the image.
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None

    Returns:
    --------
    torch.Tensor
        Image tensor of shape (C, H, W).
    """
    assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D tensor."

    # transpose (N, C) to (C, N) then reshape to (C, H, W)
    return data.T.reshape(-1, height, width)


class Conv2dMessagePassing(torch_geometric.nn.MessagePassing):
    """
    A Message Passing layer that simulates a given Conv2d layer.
    """

    def __init__(self, conv2d: torch.nn.Conv2d):
        # sum aggregation to match convolution behavior
        super().__init__(aggr='add')
        # store reference to conv weights
        self.conv_weight = conv2d.weight  # (C_out, C_in, kh, kw)
        self.conv_bias = conv2d.bias      # (C_out,) or None
        self.kw = conv2d.kernel_size[1]

    def forward(self, data):
        self.edge_index = data.edge_index

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # add bias if present
        if self.conv_bias is not None:
            out = out + self.conv_bias

        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Computes the message to be passed for each edge.
        For each edge e = (u, v) in the graph indexed by i,
        the message trough the edge e (ie from node u to node v)
        should be returned as the i-th line of the output tensor.
        (The message is phi(u, v, e) in the formalism.)
        To do this you can access the features of the source node
        in x_j[i] and the attributes of the edge in edge_attr[i].

        Arguments:
        ----------
        x_j : torch.Tensor
            The features of the souce node for each edge (of size E x in_channels).
        edge_attr : torch.Tensor
            The attributes of the edge (of size E x edge_attr_dim).

        Returns:
        --------
        torch.Tensor
            The message to be passed for each edge (of size E x out_channels)
        """
        # flatten weight: (C_out, C_in, kh*kw)
        C_out, C_in = self.conv_weight.shape[:2]
        w_flat = self.conv_weight.reshape(C_out, C_in, -1)

        # gather the weight slice for each edge based on kernel position
        # w_flat[:, :, edge_attr] -> (C_out, C_in, E)
        w = w_flat[:, :, edge_attr.long()]

        # multiply weights with source features and sum over C_in
        # w: (C_out, C_in, E), x_j.T: (C_in, E) -> result: (C_out, E) -> (E, C_out)
        messages = torch.einsum('oie,ie->eo', w, x_j.T)

        return messages