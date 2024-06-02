import torch
from typing import Optional, List
from abc import ABC
from GESS.utils.config_process import Union, CommonArgs, Munch
from GESS.models.models.base_model import BaseModel
from torch_geometric.data import Batch
from torch import Tensor
from torch_geometric.utils import degree
import numpy as np
from .base_algo import BaseAlgo
from GESS import register


@register.algorithm_register
class DIR(BaseAlgo):
    r"""
    Original Paper:
    @inproceedings{wu2021discovering,
      title={Discovering Invariant Rationales for Graph Neural Networks},
      author={Wu, Yingxin and Wang, Xiang and Zhang, An and He, Xiangnan and Chua, Tat-Seng},
      booktitle={International Conference on Learning Representations},
      year={2021}
      https://github.com/Wuyxin/DIR-GNN
    }
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DIR, self).__init__(config)
        self.alpha = config.algo.extra.alpha
        self.ratio = config.algo.coeff
        self.conf_optimizer: torch.optim.Adam = None
        self.model_optimizer: torch.optim.Adam = None

    def setup_optimizer(self,):
        pred_lr = self.config.train.lr
        pred_wd = self.config.train.wd

        self.conf_optimizer = torch.optim.Adam(self.model.classifier_conf.parameters(), lr=pred_lr, weight_decay=pred_wd)
        self.model_optimizer = torch.optim.Adam(
            list(self.model.emb_model.parameters()) +
            list(self.model.extractor.parameters()) + 
            list(self.model.GDLencoder.parameters()) + 
            list(self.model.classifier.parameters()),
            lr=pred_lr, weight_decay=pred_wd
        )

    def loss_backward(self, loss: List[Tensor]):
        conf_loss, batch_loss = loss[0], loss[1]
        
        self.conf_optimizer.zero_grad()
        conf_loss.backward()
        self.conf_optimizer.step()

        self.model_optimizer.zero_grad()
        batch_loss.backward()
        self.model_optimizer.step()
    
    def forward_pass(self, data: Batch, epoch: int, phase: str):
        # input: data batch
        alpha_prime = self.alpha * (epoch ** 1.6)
        # generate causal & non-causal part
        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch, causal_pos), \
        (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch,
         conf_pos), pred_edge_weight = self.rationale_generator(data)

        # causal repr
        # need: x, pos, edge_attr, edge_index, data.batch, edge_attn=edge_attn

        causal_rep = self.model.forward_pass(
            causal_x, causal_pos, causal_edge_attr, causal_edge_index, causal_batch, causal_edge_weight)
        # NOTE: torch.Size([127, 64])
        # self.clf.causal_out --> self.clf.clf_out
        causal_out = self.model.clf_out(causal_rep)
        # causal_out is what we need
        conf_rep = self.model.forward_pass(
            conf_x, conf_pos, conf_edge_attr, conf_edge_index, conf_batch, conf_edge_weight).detach()
        conf_out = self.model.conf_out(conf_rep)
        is_labeled = data.y == data.y
        # torch.Size([127, 1]) torch.Size([128, 1])
        causal_loss = self.criterion(
            causal_out.to(torch.float32)[is_labeled],
            data.y.to(torch.float32)[is_labeled]
        )
        if phase != 'train':
            return causal_loss, {'pred': causal_loss.item()}, causal_out
        conf_loss = self.criterion(
            conf_out.to(torch.float32)[is_labeled],
            data.y.to(torch.float32)[is_labeled]
        )
        env_loss = torch.tensor([]).to(self.device)
        for conf in conf_out:
            rep_out = self.get_comb_pred(causal_out, conf)
            tmp = self.criterion(rep_out.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])  # [1]
            env_loss = torch.cat([env_loss, tmp.unsqueeze(0)])

        DIR_loss = (env_loss.mean() + torch.var(env_loss * conf_rep.size(0)))
        batch_loss = causal_loss + alpha_prime * DIR_loss
        # optimize batch_loss and conf_loss.
        loss_dict = {'conf_loss': conf_loss.item(), 'pred': causal_loss.item(), 'DIR_loss': DIR_loss.item()}
        # return logits
        return (conf_loss, batch_loss), loss_dict, causal_out

    def get_comb_pred(self, causal_pred: Tensor, conf_pred: Tensor):
        conf_pred_tmp = conf_pred.detach()
        return torch.sigmoid(conf_pred_tmp) * causal_pred

    def rationale_generator(self, data: Batch):

        # self.clf.add_geo_feature(data)
        x, _ = self.model.get_emb(data)
        # data.edge_index & data.edge_attr & data.pos is valued

        # calculate edge weight
        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)  # torch.Size([256100, 128])
        pred_edge_weight = self.model.extractor(edge_rep).view(-1)

        causal_edge_index = torch.LongTensor([[], []]).to(x.device)
        causal_edge_weight = torch.tensor([]).to(x.device)
        causal_edge_attr = torch.tensor([]).to(x.device)
        conf_edge_index = torch.LongTensor([[], []]).to(x.device)
        conf_edge_weight = torch.tensor([]).to(x.device)
        conf_edge_attr = torch.tensor([]).to(x.device)

        edge_indices, num_nodes, _, num_edges, cum_edges = split_batch(data)

        for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
            n_reserve = int(self.ratio * N)
            edge_attr = data.edge_attr[C:C + N]
            single_mask = pred_edge_weight[C:C + N]
            single_mask_detach = pred_edge_weight[C:C + N].detach().cpu().numpy()
            rank = np.argpartition(-single_mask_detach, n_reserve)

            # idx_reverse: causal edge; idx_drop: non_causal edge
            idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]

            causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
            conf_edge_index = torch.cat([conf_edge_index, edge_index[:, idx_drop]], dim=1)

            causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
            # NOTE: -1 * single_mask[idx_drop]
            conf_edge_weight = torch.cat([conf_edge_weight, -1 * single_mask[idx_drop]])
            causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
            conf_edge_attr = torch.cat([conf_edge_attr, edge_attr[idx_drop]])
        causal_x, causal_edge_index, causal_batch, causal_pos = relabel(x, causal_edge_index, data.batch, data.pos)
        conf_x, conf_edge_index, conf_batch, conf_pos = relabel(x, conf_edge_index, data.batch, data.pos)

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch, causal_pos), \
               (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch, conf_pos), \
               pred_edge_weight


def relabel(x, edge_index, batch, pos=None):
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    # print(sub_nodes)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    # print(batch)
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    torch.set_printoptions(profile="full")
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])
    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges
