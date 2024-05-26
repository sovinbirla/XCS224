import torch
import io
import pickle

if __name__ == "__main__":
    with open('acm.pkl', 'rb') as f:
        data = pickle.load(f)




# class HeteroGNNConv(pyg_nn.MessagePassing):
#     def __init__(self, in_channels_src, in_channels_dst, out_channels):
#         super(HeteroGNNConv, self).__init__(aggr="mean")

#         self.in_channels_src = in_channels_src
#         self.in_channels_dst = in_channels_dst
#         self.out_channels = out_channels

#         # To simplify implementation, please initialize both self.lin_dst
#         # and self.lin_src out_features to out_channels
#         self.lin_dst = None
#         self.lin_src = None

#         self.lin_update = None

#         ############# Your code here #############
#         ## (~3 lines of code)
#         ## Note:
#         ## 1. Initialize the 3 linear layers.
#         ## 2. Think through the connection between the mathematical
#         ##    definition of the update rule and torch linear layers!
#         # pass
#         # self.lin = torch.nn.Linear(self.in_channels_src + self.in_channels_dst, self.out_channels)
#         self.lin_dst = torch.nn.Linear(in_channels_dst, out_channels)
#         self.lin_src = torch.nn.Linear(in_channels_src, out_channels)
#         self.lin_update = torch.nn.Linear(out_channels * 2, out_channels)


#         ##########################################

#     def forward(
#         self,
#         node_feature_src,
#         node_feature_dst,
#         edge_index,
#         size=None
#     ):
#         ############# Your code here #############
#         ## (~1 line of code)
#         ## Note:
#         ## 1. Unlike Colab 3, we just need to call self.propagate with
#         ## proper arguments.
#         # pass
#         out = self.propagate(edge_index, x=(node_feature_src, node_feature_dst))

#         # Then apply a final linear transformation.
#         out = self.lin_update(out)

#         return out
#         ##########################################

#     def message_and_aggregate(self, edge_index, node_feature_src):

#         ############# Your code here #############
#         ## (~1 line of code)
#         ## Note:
#         ## 1. Different from what we implemented in Colab 3, we use message_and_aggregate
#         ##    to combine the previously seperate message and aggregate functions.
#         ##    The benefit is that we can avoid materializing x_i and x_j
#         ##    to make the implementation more efficient.
#         ## 2. To implement efficiently, refer to PyG documentation for message_and_aggregate
#         ##    and sparse-matrix multiplication:
#         ##    https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
#         ## 3. Here edge_index is torch_sparse SparseTensor. Although interesting, you
#         ##    do not need to deeply understand SparseTensor represenations!
#         ## 4. Conceptually, think through how the message passing and aggregation
#         ##    expressed mathematically can be expressed through matrix multiplication.
        
#         src, dst = edge_index
#         # slightly unsure about this
#         msg_src = self.lin_src(src)
#         msg_dst = self.lin_dst(dst)
#         msg = torch.cat([msg_src[src], msg_dst[dst]], dim=1)
#         out = matmul(msg, SparseTensor(row=edge_index[0], col=edge_index[1]))
#         ##########################################

#         return out

#     def update(self, aggr_out, node_feature_dst):

#         ############# Your code here #############
#         ## (~4 lines of code)
#         ## Note:
#         ## 1. The update function is called after message_and_aggregate
#         ## 2. Think through the one-one connection between the mathematical update
#         ##    rule and the 3 linear layers defined in the constructor.
#         # pass
#         aggr_out = self.lin_update(torch.cat([aggr_out, node_feature_dst], dim=1))

#         ##########################################

#         return aggr_out

# class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
#     def __init__(self, convs, args, aggr=None):
#         super(HeteroGNNWrapperConv, self).__init__(convs, None)
#         self.aggr = aggr

#         # Map the index and message type
#         self.mapping = {}

#         # A numpy array that stores the final attention probability
#         self.alpha = None

#         self.attn_proj = None

#         if self.aggr == "attn":
#             ############# Your code here #############
#             ## (~1 line of code)
#             ## Note:
#             ## 1. Initialize self.attn_proj, where self.attn_proj should include
#             ##    two linear layers. Note, make sure you understand
#             ##    which part of the equation self.attn_proj captures.
#             ## 2. You should use nn.Sequential for self.attn_proj
#             ## 3. nn.Linear and nn.Tanh are useful.
#             ## 4. You can model a weight vector (rather than matrix) by using:
#             ##    nn.Linear(some_size, 1, bias=False).
#             ## 5. The first linear layer should have out_features as args['attn_size']
#             ## 6. You can assume we only have one "head" for the attention.
#             ## 7. We recommend you to implement the mean aggregation first. After
#             ##    the mean aggregation works well in the training, then you can
#             ##    implement this part.
#             # pass
#             self.attn_proj = nn.Sequential(
#                 nn.Linear(args['attn_size'], args['attn_size'], bias=False),
#                 nn.Tanh()
#             )
#             ####################
#             ##########################################

#     def reset_parameters(self):
#         super(HeteroGNNWrapperConv, self).reset_parameters()
#         if self.aggr == "attn":
#             for layer in self.attn_proj.children():
#                 layer.reset_parameters()

#     def forward(self, node_features, edge_indices):
#         message_type_emb = {}
#         for message_key, edge_index in edge_indices.items():
#             src_type, edge_type, dst_type = message_key
#             node_feature_src = node_features[src_type]
#             node_feature_dst = node_features[dst_type]
#             message_type_emb[message_key] = (
#                 self.convs[message_key](
#                     node_feature_src,
#                     node_feature_dst,
#                     edge_index,
#                 )
#             )
#         node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
#         mapping = {}
#         for (src, edge_type, dst), item in message_type_emb.items():
#             mapping[len(node_emb[dst])] = (src, edge_type, dst)
#             node_emb[dst].append(item)
#         self.mapping = mapping
#         for node_type, embs in node_emb.items():
#             if len(embs) == 1:
#                 node_emb[node_type] = embs[0]
#             else:
#                 node_emb[node_type] = self.aggregate(embs)
#         return node_emb

#     def aggregate(self, xs):
#         # TODO: Implement this function that aggregates all message type results for one node type.
#         # Here, xs is a list of tensors (embeddings) with respect to message
#         # type aggregation results.

#         # Useful dimensions from xs - particularly for `attn` aggregation
#         N = xs[0].shape[0] # Number of nodes for the given node type
#         M = len(xs) # Number of message types for the given node type

#         if self.aggr == "mean":

#             ############# Your code here #############
#             ## (~2 lines of code)
#             ## Note:
#             ## 1. Explore the function parameter `xs`!
#             # pass
#             return torch.mean(torch.stack(xs), dim=0)
#             ##########################################

#         elif self.aggr == "attn":
#             ############# Your code here #############
#             ## (~10 lines of code)
#             ## Note:
#             ## 1. Try to map out how the equations can be translated into code.
#             ## 2. N and M defined above may be useful at least to understand.
#             ## 3. Work first to compute the un-normalized attention weights e
#             ##    for each message type - try to vectorize this!
#             ## 4. torch.softmax and torch.cat are useful.
#             ## 5. It might be useful to reshape and concatenate tensors using the
#             ##    `view()` function https://pytorch.org/docs/stable/tensor_view.html
#             ##    and `torch.cat()`https://pytorch.org/docs/stable/generated/torch.cat.html
#             ## 6. Store the value of attention alpha (as a numpy array) to self.alpha,
#             ##    which has the shape (len(xs), ) self.alpha will be not be used
#             ##    to backpropagate etc. in the model. We will use it to see how much
#             ##    attention the layer pays on different message types.
#             # pass
#             # Compute the un-normalized attention weights
#             e = self.attn_proj(torch.cat(xs, dim=1))
#             # Compute the normalized attention weights
#             alpha = torch.softmax(e, dim=0)
#             self.alpha = alpha.detach().cpu().numpy()
#             # Update the node embedding
#             return torch.sum(alpha.view(N, M, 1) * torch.stack(xs), dim=1)
#             ####################
#             ##########################################

# def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
#     # TODO: Implement this function that returns a dictionary of `HeteroGNNConv`
#     # layers where the keys are message types. `hetero_graph` is deepsnap `HeteroGraph`
#     # object and the `conv` is the `HeteroGNNConv`.

#     convs = {}

#     ############# Your code here #############
#     ## (~9 lines of code)
#     ## Note:
#     ## 1. See the hints above!
#     ## 2. conv is of type `HeteroGNNConv`
#     # pass
#     for edge_type in hetero_graph.edge_types:
#         src_type, dst_type = edge_type
#         if first_layer:
#             convs[edge_type] = conv(hetero_graph.node_feature[src_type].shape[1], hetero_graph.node_feature[dst_type].shape[1], hidden_size)
#         else:
#             convs[edge_type] = conv(hidden_size, hidden_size, hidden_size)
#     ##########################################

#     return convs

# class HeteroGNN(torch.nn.Module):
#     def __init__(self, hetero_graph, args, aggr="mean"):
#         super(HeteroGNN, self).__init__()

#         self.aggr = aggr
#         self.hidden_size = args['hidden_size']

#         self.convs1 = None
#         self.convs2 = None

#         self.bns1 = nn.ModuleDict()
#         self.bns2 = nn.ModuleDict()
#         self.relus1 = nn.ModuleDict()
#         self.relus2 = nn.ModuleDict()
#         self.post_mps = nn.ModuleDict()

#         ############# Your code here #############
#         ## (~10 lines of code)
#         ## Note:
#         ## 1. For self.convs1 and self.convs2, call generate_convs at first and then
#         ##    pass the returned dictionary of `HeteroGNNConv` to `HeteroGNNWrapperConv`.
#         ## 2. For self.bns, self.relus and self.post_mps, the keys are node_types.
#         ##    `deepsnap.hetero_graph.HeteroGraph.node_types` will be helpful.
#         ## 3. Initialize all batchnorms to torch.nn.BatchNorm1d(hidden_size, eps=1).
#         ## 4. Initialize all relus to nn.LeakyReLU().
#         ## 5. For self.post_mps, each value in the ModuleDict is a linear layer
#         ##    where the `out_features` is the number of classes for that node type.
#         ##    `deepsnap.hetero_graph.HeteroGraph.num_node_labels(node_type)` will be
#         ##    useful.
#         # pass
#         self.convs1 = generate_convs(hetero_graph, HeteroGNNWrapperConv, self.hidden_size, first_layer=True)
#         self.convs2 = generate_convs(hetero_graph, HeteroGNNWrapperConv, self.hidden_size)
#         ##########################################

#     def forward(self, node_feature, edge_index):
#         # TODO: Implement the forward function. Notice that `node_feature` is
#         # a dictionary of tensors where keys are node types and values are
#         # corresponding feature tensors. The `edge_index` is a dictionary of
#         # tensors where keys are message types and values are corresponding
#         # edge index tensors (with respect to each message type).

#         x = node_feature

#         ############# Your code here #############
#         ## (~7 lines of code)
#         ## Note:
#         ## 1. `deepsnap.hetero_gnn.forward_op` can be helpful for
#         ##    the bn, relu, and post_mp ops.
#         # pass
#         for conv in self.convs1.values():
#             x = deepsnap.hetero_gnn.forward_op(conv, x, edge_index)
#         for bn, relu in zip(self.bns1.values(), self.relus1.values()):
#             x = bn(x)
#             x = relu(x)
#         for conv in self.convs2.values():
#             x = deepsnap.hetero_gnn.forward_op(conv, x, edge_index)
#         for bn, relu in zip(self.bns2.values(), self.relus2.values()):
#             x = bn(x)
#             x = relu(x)
#         for post_mp in self.post_mps.values():
#             x = post_mp(x)
#         ##########################################

#         ##########################################

#         return x

#     def loss(self, preds, y, indices):

#         loss = 0
#         loss_func = F.cross_entropy

#         ############# Your code here #############
#         ## (~3 lines of code)
#         ## Note:
#         ## 1. For each node type in preds, accumulate computed loss to `loss`
#         ## 2. Loss need to be computed with respect to the given index
#         ## 3. preds is a dictionary of model predictions keyed by node_type.
#         ## 4. indeces is a dictionary of labeled supervision nodes keyed
#         ##    by node_type
#         # pass
#         for node_type in preds:
#             loss += loss_func(preds[node_type][indices[node_type]], y[node_type][indices[node_type]])
#         ##########################################

#         return loss

# import pandas as pd

# def train(model, optimizer, hetero_graph, train_idx):
#     model.train()
#     optimizer.zero_grad()
#     preds = model(hetero_graph.node_feature, hetero_graph.edge_index)

#     loss = None

#     ############# Your code here #############
#     ## Note:
#     ## 1. Compute the loss here
#     ## 2. `deepsnap.hetero_graph.HeteroGraph.node_label` is useful
#     # pass
#     loss = model.loss(preds, hetero_graph.node_label, train_idx)

#     ##########################################

#     loss.backward()
#     optimizer.step()
#     return loss.item()

# def test(model, graph, indices, best_model=None, best_val=0, save_preds=False, agg_type=None):
#     model.eval()
#     accs = []
#     for i, index in enumerate(indices):
#         preds = model(graph.node_feature, graph.edge_index)
#         num_node_types = 0
#         micro = 0
#         macro = 0
#         for node_type in preds:
#             idx = index[node_type]
#             pred = preds[node_type][idx]
#             pred = pred.max(1)[1]
#             label_np = graph.node_label[node_type][idx].cpu().numpy()
#             pred_np = pred.cpu().numpy()
#             micro = f1_score(label_np, pred_np, average='micro')
#             macro = f1_score(label_np, pred_np, average='macro')
#             num_node_types += 1

#         # Averaging f1 score might not make sense, but in our example we only
#         # have one node type
#         micro /= num_node_types
#         macro /= num_node_types
#         accs.append((micro, macro))

#         # Only save the test set predictions and labels!
#         if save_preds and i == 2:
#           print ("Saving Heterogeneous Node Prediction Model Predictions with Agg:", agg_type)
#           print()

#           data = {}
#           data['pred'] = pred_np
#           data['label'] = label_np

#           df = pd.DataFrame(data=data)
#           # Save locally as csv
#           df.to_csv('ACM-Node-' + agg_type + 'Agg.csv', sep=',', index=False)

#     if accs[1][0] > best_val:
#         best_val = accs[1][0]
#         best_model = copy.deepcopy(model)
#     return accs, best_model, best_val

# # Please do not change the following parameters
# args = {
#     'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#     'hidden_size': 64,
#     'epochs': 100,
#     'weight_decay': 1e-5,
#     'lr': 0.003,
#     'attn_size': 32,
# }

# import numpy as np
# import pickle


# def set_seed(seed=224):

#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.backends.cudnn.benchmark = False

#     # Load the data
#     with open("acm.pkl", "rb") as f:
#         data = pickle.load(f)

#     edge_index[message_type_2] = data["psp"]

#     # Dictionary of node features
#     node_feature = {}
#     node_feature["paper"] = data["feature"]

#     # Dictionary of node labels
#     node_label = {}
#     node_label["paper"] = data["label"]

#     # Load the train, validation and test indices
#     train_idx = {"paper": data["train_idx"].to(args["device"])}
#     val_idx = {"paper": data["val_idx"].to(args["device"])}
#     test_idx = {"paper": data["test_idx"].to(args["device"])}

#     # Construct a deepsnap tensor backend HeteroGraph
#     hetero_graph = HeteroGraph(
#         node_feature=node_feature,
#         node_label=node_label,
#         edge_index=edge_index,
#         directed=True,
#     )

#     print(
#         f"ACM heterogeneous graph: {hetero_graph.num_nodes()} nodes, {hetero_graph.num_edges()} edges"
#     )

#     # Node feature and node label to device
#     for key in hetero_graph.node_feature:
#         hetero_graph.node_feature[key] = hetero_graph.node_feature[key].to(
#             args["device"]
#         )
#     for key in hetero_graph.node_label:
#         hetero_graph.node_label[key] = hetero_graph.node_label[key].to(args["device"])

#     # Edge_index to sparse tensor and to device
#     for key in hetero_graph.edge_index:
#         edge_index = hetero_graph.edge_index[key]
#         adj = SparseTensor(
#             row=edge_index[0],
#             col=edge_index[1],
#             sparse_sizes=(
#                 hetero_graph.num_nodes("paper"),
#                 hetero_graph.num_nodes("paper"),
#             ),
#         )
#         hetero_graph.edge_index[key] = adj.t().to(args["device"])
#     print(hetero_graph.edge_index[message_type_1])
#     print(hetero_graph.edge_index[message_type_2])

# import numpy as np


# def set_seed(seed=224):

#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.backends.cudnn.benchmark = False


# if "IS_GRADESCOPE_ENV" not in os.environ:
#     best_model = None
#     best_val = 0

#     set_seed()

#     model = HeteroGNN(hetero_graph, args, aggr="mean").to(args["device"])

#     # Disable compile as this does not seem to work yet in PyTorch 2.0.1/PyG 2.3.1
#     # try:
#     #   model = torch_geometric.compile(model)
#     #   print(f"HeteroGNN Model compiled")
#     # except Exception as err:
#     #   print(f"Model compile not supported: {err}")

#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
#     )

#     for epoch in range(args["epochs"]):
#         loss = train(model, optimizer, hetero_graph, train_idx)
#         accs, best_model, best_val = test(
#             model, hetero_graph, [train_idx, val_idx, test_idx], best_model, best_val
#         )
#         print(
#             f"Epoch {epoch + 1}: loss {round(loss, 5)}, "
#             f"train micro {round(accs[0][0] * 100, 2)}%, train macro {round(accs[0][1] * 100, 2)}%, "
#             f"valid micro {round(accs[1][0] * 100, 2)}%, valid macro {round(accs[1][1] * 100, 2)}%, "
#             f"test micro {round(accs[2][0] * 100, 2)}%, test macro {round(accs[2][1] * 100, 2)}%"
#         )
#     best_accs, _, _ = test(
#         best_model,
#         hetero_graph,
#         [train_idx, val_idx, test_idx],
#         save_preds=True,
#         agg_type="Mean",
#     )
#     print(
#         f"Best model: "
#         f"train micro {round(best_accs[0][0] * 100, 2)}%, train macro {round(best_accs[0][1] * 100, 2)}%, "
#         f"valid micro {round(best_accs[1][0] * 100, 2)}%, valid macro {round(best_accs[1][1] * 100, 2)}%, "
#         f"test micro {round(best_accs[2][0] * 100, 2)}%, test macro {round(best_accs[2][1] * 100, 2)}%"
#     )
