# -*- coding: utf-8 -*-
"""
W-GATCR-CL 模型 - 训练与评估 (融合图对比学习 - 自适应噪声版)


功能:
1.  加载预处理好的带权图和物品特征。
2.  构建 W_GATCR 模型，集成LayerNorm和残差连接。
3.  引入图对比学习(GCL)辅助任务。
4.  采用与节点度数相关的自适应噪声增强策略。
5.  实现一个包含BPR损失和InfoNCE对比损失的联合学习框架。
6.  完整保留了所有命令行参数控制的数据集划分和早停机制。
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import dgl
import json
from datetime import datetime
import matplotlib.pyplot as plt


class WeightedGATLayer(nn.Module):
    """自定义的GAT层，用于融合边权重"""

    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0., attn_drop=0., negative_slope=0.2, activation=None):
        super(WeightedGATLayer, self).__init__()
        self._num_heads = num_heads
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.lambda_w = nn.Parameter(torch.tensor(1.0))

    def forward(self, g, feat, edge_weights):
        with g.local_scope():
            h = self.feat_drop(feat)
            feat_transformed = self.fc(h).view(-1, self._num_heads, self._out_feats)
            el = (feat_transformed * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_transformed * self.attn_r).sum(dim=-1).unsqueeze(-1)
            g.ndata.update({'ft': feat_transformed, 'el': el, 'er': er})
            g.apply_edges(dgl.function.u_add_v('el', 'er', 'e'))
            edge_weights_broadcast = edge_weights.view(-1, 1).expand(-1, self._num_heads)
            g.edata['e'] = g.edata['e'].squeeze(-1) + self.lambda_w * edge_weights_broadcast
            e = self.leaky_relu(g.edata.pop('e'))
            g.edata['a'] = self.attn_drop(dgl.ops.edge_softmax(g, e)).unsqueeze(-1)
            g.update_all(dgl.function.u_mul_e('ft', 'a', 'm'), dgl.function.sum('m', 'ft'))
            rst = g.ndata['ft']
            if self.activation:
                rst = self.activation(rst)
            return rst


class W_GATCR_CL(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout,
                 node_degrees, cl_epsilon=0.1, cl_alpha=0.5):
        super(W_GATCR_CL, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.cl_epsilon = cl_epsilon
        self.cl_alpha = cl_alpha


        # 预计算每个节点的自适应噪声幅度，并注册为缓冲区
        if node_degrees is not None:
            max_degree = torch.max(node_degrees)
            normalized_degrees = node_degrees / max_degree if max_degree > 0 else torch.zeros_like(node_degrees)
            adaptive_epsilons = cl_epsilon * (1 + cl_alpha * normalized_degrees)
            self.register_buffer('adaptive_epsilons', adaptive_epsilons.unsqueeze(1))
        else:
            self.register_buffer('adaptive_epsilons', torch.tensor(cl_epsilon))


        self.layers.append(WeightedGATLayer(in_dim, hidden_dim, num_heads, dropout, dropout, activation=F.elu))
        self.layer_norms.append(nn.LayerNorm(hidden_dim * num_heads))
        for _ in range(num_layers - 2):
            self.layers.append(
                WeightedGATLayer(hidden_dim * num_heads, hidden_dim, num_heads, dropout, dropout, activation=F.elu))
            self.layer_norms.append(nn.LayerNorm(hidden_dim * num_heads))
        self.layers.append(WeightedGATLayer(hidden_dim * num_heads, out_dim, 1, dropout, dropout, activation=None))


    def _add_noise(self, x):
        """为嵌入添加基于节点度数的自适应随机噪声"""
        if not self.training or self.cl_epsilon == 0:
            return x

        random_noise = torch.rand_like(x, device=x.device)
        noise_direction = random_noise * torch.sign(x)

        norm = torch.linalg.norm(noise_direction, dim=1, keepdim=True) + 1e-8
        normalized_noise = noise_direction / norm

        scaled_noise = normalized_noise * self.adaptive_epsilons

        return x + scaled_noise



    def forward(self, g, features):
        """执行两次前向传播，一次带噪声，一次不带"""
        h_noisy = features
        edge_weights = g.edata['weight']
        for i, layer in enumerate(self.layers):
            h_noisy = self._add_noise(h_noisy)
            h_prev_noisy = h_noisy
            h_noisy = layer(g, h_noisy, edge_weights)
            if i < len(self.layers) - 1:
                h_noisy = h_noisy.flatten(1)
                h_noisy = self.layer_norms[i](h_noisy)
                if h_prev_noisy.shape == h_noisy.shape:
                    h_noisy = h_noisy + h_prev_noisy
            else:
                h_noisy = h_noisy.squeeze(1)

        h_clean = features
        for i, layer in enumerate(self.layers):
            h_prev_clean = h_clean
            h_clean = layer(g, h_clean, edge_weights)
            if i < len(self.layers) - 1:
                h_clean = h_clean.flatten(1)
                h_clean = self.layer_norms[i](h_clean)
                if h_prev_clean.shape == h_clean.shape:
                    h_clean = h_clean + h_prev_clean
            else:
                h_clean = h_clean.squeeze(1)

        return h_clean, h_noisy

    def predict(self, final_embeds, item_pairs):
        embeds1 = final_embeds[item_pairs[:, 0]]
        embeds2 = final_embeds[item_pairs[:, 1]]
        return torch.sum(embeds1 * embeds2, dim=1)


def calculate_cl_loss(embeds1, embeds2, tau=0.2):
    """计算InfoNCE对比损失"""
    embeds1 = F.normalize(embeds1, p=2, dim=1)
    embeds2 = F.normalize(embeds2, p=2, dim=1)
    pos_score = torch.sum(embeds1 * embeds2, dim=1)
    all_scores = torch.mm(embeds1, embeds2.t())
    log_prob = pos_score - torch.log(torch.sum(torch.exp(all_scores / tau), dim=1))
    loss = -torch.mean(log_prob)
    return loss


class PairDataset(Dataset):
    def __init__(self, pos_pairs, num_items):
        self.pos_pairs, self.num_items = pos_pairs, num_items
        self.pos_pairs_set = {tuple(p) for p in pos_pairs}
        self.pos_pairs_set.update({(p[1], p[0]) for p in pos_pairs})

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        pos_u, pos_v = self.pos_pairs[idx]
        while True:
            neg_v = np.random.randint(0, self.num_items)
            if (pos_u, neg_v) not in self.pos_pairs_set: break
        return torch.tensor([pos_u, pos_v]), torch.tensor([pos_u, neg_v]), torch.tensor([pos_u])


def evaluate(model, g, features, test_pairs, all_item_ids, ks=[5, 10]):
    model.eval()
    hits = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    mrrs = []

    with torch.no_grad():
        final_embeds, _ = model(g, features)
        for source_item, true_item in tqdm(test_pairs, desc="评估", leave=False):
            items_to_exclude = [true_item, source_item]
            neg_pool = np.setdiff1d(all_item_ids, items_to_exclude, assume_unique=True)
            neg_items = np.random.choice(neg_pool, min(99, len(neg_pool)), replace=False)
            candidates = torch.tensor([true_item] + list(neg_items))
            sources = torch.full_like(candidates, fill_value=source_item)
            test_batch = torch.stack([sources, candidates], dim=1).to(features.device)
            scores = model.predict(final_embeds, test_batch)
            rank = (scores.argsort(descending=True) == 0).nonzero(as_tuple=True)[0].item()
            mrrs.append(1 / (rank + 1))
            for k in ks:
                if rank < k:
                    hits[k].append(1)
                    ndcgs[k].append(1 / np.log2(rank + 2))
                else:
                    hits[k].append(0)
                    ndcgs[k].append(0)

    results = {'MRR': np.mean(mrrs)}
    for k in ks:
        results[f'HR@{k}'] = np.mean(hits[k])
        results[f'NDCG@{k}'] = np.mean(ndcgs[k])
    return results


def plot_loss_curve(losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"训练损失曲线图已保存至: {save_path}")


def run_training(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data_dir)

    print("--- 步骤 1: 加载预处理数据 ---")
    features = torch.from_numpy(np.load(data_path / "item_features.npy")).float().to(DEVICE)
    nx_g = nx.read_weighted_edgelist(data_path / "weighted_graph.edgelist", nodetype=int)

    nx_digraph = nx_g.to_directed()
    g = dgl.from_networkx(nx_digraph, edge_attrs=['weight']).to(DEVICE)
    g.edata['weight'] = g.edata['weight'].float()
    g = dgl.add_self_loop(g)
    g.edata['weight'][g.edata['weight'].shape[0] - g.num_nodes():] = torch.ones(g.num_nodes()).to(DEVICE)

    num_items, in_dim = features.shape
    all_item_ids = np.arange(num_items)

    print("--- 步骤 2: 划分训练、验证、测试集 ---")
    pos_pairs = np.array(list(nx_g.edges()))
    np.random.shuffle(pos_pairs)

    train_size = int(len(pos_pairs) * args.train_ratio)
    val_size = int(len(pos_pairs) * args.val_ratio)

    train_pos_pairs = pos_pairs[:train_size]
    val_pos_pairs = pos_pairs[train_size: train_size + val_size]
    test_pos_pairs = pos_pairs[train_size + val_size:]

    train_dataset = PairDataset(train_pos_pairs, num_items)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"训练集大小: {len(train_pos_pairs)}, 验证集大小: {len(val_pos_pairs)}, 测试集大小: {len(test_pos_pairs)}")

    print("\n--- 步骤 3: 初始化模型和优化器 ---")

    # 计算节点度数，并传递给模型以实现自适应噪声
    node_degrees = g.in_degrees().float().to(DEVICE)
    model = W_GATCR_CL(in_dim, args.hidden_dim, args.out_dim, args.num_heads,
                       args.num_layers, args.dropout, node_degrees=node_degrees,
                       cl_epsilon=args.cl_epsilon, cl_alpha=args.cl_alpha).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    print("\n--- 步骤 4: 开始训练与评估 ---")
    best_val_ndcg, best_epoch, epochs_without_improvement = 0.0, 0, 0
    train_losses = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for pos_batch, neg_batch, cl_batch_nodes in tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}"):
            optimizer.zero_grad()
            final_embeds_clean, final_embeds_noisy = model(g, features)
            pos_scores = model.predict(final_embeds_clean, pos_batch.to(DEVICE))
            neg_scores = model.predict(final_embeds_clean, neg_batch.to(DEVICE))
            rec_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            cl_nodes = torch.unique(cl_batch_nodes).to(DEVICE)
            cl_embeds_clean = final_embeds_clean[cl_nodes]
            cl_embeds_noisy = final_embeds_noisy[cl_nodes]
            cl_loss = calculate_cl_loss(cl_embeds_clean, cl_embeds_noisy, args.cl_tau)
            loss = rec_loss + args.cl_lambda * cl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} - 训练损失: {avg_loss:.4f} (BPR: {rec_loss.item():.4f}, CL: {cl_loss.item():.4f})")

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_results = evaluate(model, g, features, val_pos_pairs, all_item_ids)
            print_str = f"  验证集结果 -> "
            for metric, value in val_results.items(): print_str += f"{metric}: {value:.4f} | "
            print(print_str[:-2])

            val_ndcg10 = val_results.get('NDCG@10', 0)
            if val_ndcg10 > best_val_ndcg:
                best_val_ndcg, best_epoch = val_ndcg10, epoch + 1
                torch.save(model.state_dict(), data_path / 'w_gatcr_cl_best_model.pth')
                print(f"  *** 新的最佳模型已保存 (验证集 NDCG@10: {best_val_ndcg:.4f}) ***")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"\n早停机制触发。")
            break

    print("\n--- 训练完成 ---")
    print(f"加载第 {best_epoch} 个epoch保存的最佳模型进行最终评估...")
    model.load_state_dict(torch.load(data_path / 'w_gatcr_cl_best_model.pth'))
    final_test_results = evaluate(model, g, features, test_pos_pairs, all_item_ids)

    final_report = {"hyperparameters": vars(args),
                    "best_metrics_on_test_set": {"evaluated_at_best_epoch": best_epoch, **final_test_results}}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name_str = Path(args.data_dir).name.replace('w_gatcr_', '')
    result_dir = data_path / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    results_filename = result_dir / f"final_results_{dataset_name_str}_CL_Adaptive_{timestamp}.json"
    with open(results_filename, 'w') as f:
        json.dump(final_report, f, indent=4)
    print(f"最终测试结果与超参数已保存至: {results_filename}")

    plot_filename = result_dir / f"loss_curve_{dataset_name_str}_CL_Adaptive_{timestamp}.png"
    plot_loss_curve(train_losses, plot_filename)
    return final_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="W-GATCR-CL 模型训练与评估脚本 (自适应噪声版)")
    parser.add_argument('--data_dir', type=str, default="./processed_data/w_gatcr_Grocery_and_Gourmet_Food")
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)#
    parser.add_argument('--num_layers', type=int, default=2)#
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)#改学习率
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50) # 150
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--cl_lambda', type=float, default=0.05, help="对比损失的权重")
    parser.add_argument('--cl_tau', type=float, default=0.2, help="对比损失的温度系数")#0.15
    parser.add_argument('--cl_epsilon', type=float, default=0.1, help="噪声增强的基础扰动幅度")

    parser.add_argument('--cl_alpha', type=float, default=0.5, help="自适应噪声中的热门度影响因子")

    args = parser.parse_args()
    run_training(args)
