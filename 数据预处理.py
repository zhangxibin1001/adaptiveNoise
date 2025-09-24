# -*- coding: utf-8 -*-
"""
W-GATCR 模型 - 数据预处理脚本
版本: 1.3 (最终版)

功能:
1.  加载并过滤亚马逊数据集。
2.  使用BERT+ReLU提取物品初始特征。
3.  构建原始同购图。
4.  实现图加权模块：为边计算内在互补性分数，并将其作为权重。
5.  保存带权重的图、特征和ID映射，并确保三者完美对齐。
"""
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
from transformers import BertModel, BertTokenizer


class TextDataset(Dataset):
    """用于BERT批处理的文本数据集"""

    def __init__(self, texts, tokenizer, max_len):
        self.texts, self.tokenizer, self.max_len = texts, tokenizer, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[item]), add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}


class GraphWeighter:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_name = self.args.dataset_name
        self.dataset_path = Path(self.args.input_dir) / f"meta_{self.dataset_name}.json"
        self.output_dir = Path(self.args.base_output_dir) / f"w_gatcr_{self.dataset_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- 初始化图加权器 ---")
        print(f"使用设备: {self.device}, 输入: {self.dataset_path}, 输出至: {self.output_dir}")

    def _load_and_filter_data(self):
        print("\n--- 步骤 1: 加载并过滤原始数据 ---")
        if not self.dataset_path.exists(): raise FileNotFoundError(f"错误: 文件未找到于 '{self.dataset_path}'.")
        data = [json.loads(line) for line in open(self.dataset_path, 'r', encoding='utf-8')]
        df = pd.DataFrame(data).rename(columns={'also_buy': 'copurchase'})
        df = df.dropna(subset=['asin', 'title', 'copurchase', 'category'])
        df = df[df['copurchase'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        df = df[df['category'].apply(lambda x: isinstance(x, list) and len(x) > 1)]
        df = df.drop_duplicates(subset=['asin']).reset_index(drop=True)

        all_asins = set(df['asin'])
        temp_g = nx.Graph()
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="构建临时图以过滤"):
            for neighbor_asin in row['copurchase']:
                if neighbor_asin in all_asins: temp_g.add_edge(row['asin'], neighbor_asin)

        valid_nodes = {node for node, degree in temp_g.degree() if degree >= self.args.min_degree}
        df_filtered = df[df['asin'].isin(valid_nodes)].copy().reset_index(drop=True)

        # 临时的ID映射，仅用于特征提取
        self.temp_asin_to_id = {asin: i for i, asin in enumerate(df_filtered['asin'])}

        print(f"数据加载与过滤完成。保留物品数量: {len(df_filtered)}")
        return df_filtered

    def _clean_categories(self, df):
        print("\n--- 步骤 2: 清洗类别数据 ---")
        noise_words = {'brands', 'stores', 'characters', 'see', 'all'}
        df['category_clean'] = df['category'].apply(
            lambda cats: list(dict.fromkeys(
                [c.lower().strip() for c in cats if c.lower().strip() not in noise_words]
            ))
        )
        return df

    def _get_bert_relu_features(self, df):
        print(f"\n--- 步骤 3: 使用 {self.args.bert_model} 提取物品特征 ---")
        tokenizer = BertTokenizer.from_pretrained(self.args.bert_model)
        model = BertModel.from_pretrained(self.args.bert_model).to(self.device)
        model.eval()
        texts = [f"{row.get('title', '')} . {' '.join(row.get('description', []))}" for _, row in df.iterrows()]
        data_loader = DataLoader(TextDataset(texts, tokenizer, 128), batch_size=self.args.bert_batch_size)
        all_features = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="生成特征向量"):
                outputs = model(input_ids=batch['input_ids'].to(self.device),
                                attention_mask=batch['attention_mask'].to(self.device))
                all_features.append(F.relu(outputs.last_hidden_state[:, 0, :]).cpu().numpy())
        del model, tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return np.vstack(all_features)

    def _score_and_weight_edges(self, g, df, item_features):
        print("\n--- 步骤 4: 向量化计算边权重 ---")
        edges = np.array(list(g.edges()))
        u_asins, v_asins = edges[:, 0], edges[:, 1]

        u_ids = np.array([self.temp_asin_to_id[asin] for asin in u_asins])
        v_ids = np.array([self.temp_asin_to_id[asin] for asin in v_asins])

        s_centroid = F.cosine_similarity(torch.from_numpy(item_features[u_ids]),
                                         torch.from_numpy(item_features[v_ids])).numpy()

        s_lca, s_div, s_dist = [], [], []
        for u_asin, v_asin in tqdm(zip(u_asins, v_asins), total=len(u_asins), desc="计算层次化分数"):
            u_row = df[df['asin'] == u_asin].iloc[0]
            v_row = df[df['asin'] == v_asin].iloc[0]
            paths = [u_row['category_clean'], v_row['category_clean']]
            lca_depth, avg_dist = self._get_lca_depth_and_distance(paths)
            s_lca.append(lca_depth)
            s_div.append(len({p[-1] for p in paths if p}) / 2.0)
            s_dist.append(1 / (1 + avg_dist))

        s_h = (self.args.w_lca * minmax_scale(s_lca) + self.args.w_div * minmax_scale(
            s_div) + self.args.w_dist * minmax_scale(s_dist))
        s_c = minmax_scale(s_centroid)
        final_scores = self.args.alpha * s_c + (1 - self.args.alpha) * s_h

        return dict(zip(map(tuple, edges), final_scores))

    def _get_lca_depth_and_distance(self, paths):
        paths = [p for p in paths if p]
        if not paths or len(paths) < 2: return 0, 0
        lca_depth = 0
        for i in range(min(len(p) for p in paths)):
            if all(p[i] == paths[0][i] for p in paths):
                lca_depth += 1
            else:
                break
        return lca_depth, np.mean([len(p) - lca_depth for p in paths])

    def run(self):
        df_filtered = self._load_and_filter_data()
        df_cleaned = self._clean_categories(df_filtered)
        item_features_all = self._get_bert_relu_features(df_cleaned)

        g = nx.Graph()
        valid_asins = set(df_cleaned['asin'])
        for _, row in tqdm(df_cleaned.iterrows(), total=df_cleaned.shape[0], desc="构建原始图"):
            for neighbor_asin in row['copurchase']:
                if neighbor_asin in valid_asins: g.add_edge(row['asin'], neighbor_asin)

        edge_scores = self._score_and_weight_edges(g, df_cleaned, item_features_all)
        nx.set_edge_attributes(g, edge_scores, 'weight')

        print("\n--- 步骤 5: 生成最终对齐的数据 ---")
        final_nodes_asin = sorted(list(g.nodes()))
        final_asin_to_id = {asin: i for i, asin in enumerate(final_nodes_asin)}

        original_indices = [self.temp_asin_to_id[asin] for asin in final_nodes_asin]
        item_features_final = item_features_all[original_indices]

        id_g = nx.relabel_nodes(g, final_asin_to_id)

        print("\n--- 步骤 6: 保存最终输出文件 ---")
        np.save(self.output_dir / "item_features.npy", item_features_final)
        nx.write_weighted_edgelist(id_g, self.output_dir / "weighted_graph.edgelist")
        with open(self.output_dir / "item_id_map.json", 'w') as f:
            json.dump(final_asin_to_id, f)

        print(f"\n预处理完成！所有文件已保存至目录: {self.output_dir}")
        print(f"最终物品数: {len(final_nodes_asin)}, 特征矩阵形状: {item_features_final.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="W-GATCR 数据预处理脚本")
    parser.add_argument('--dataset_name', type=str, default="Grocery_and_Gourmet_Food", help="亚马逊数据集的名称")
    parser.add_argument('--input_dir', type=str, default="./amazon_data", help="输入数据集的根目录")
    parser.add_argument('--base_output_dir', type=str, default="./processed_data", help="处理后数据的根目录")
    parser.add_argument('--min_degree', type=int, default=5, help="物品在同购图中的最小度数")
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help="预训练BERT模型名称")
    parser.add_argument('--bert_batch_size', type=int, default=32, help="BERT特征提取的批次大小")
    parser.add_argument('--alpha', type=float, default=0.5, help="S_centroid的融合权重")
    parser.add_argument('--w_lca', type=float, default=0.333, help="S_hierarchy中LCA深度的权重")
    parser.add_argument('--w_div', type=float, default=0.333, help="S_hierarchy中多样性的权重")
    parser.add_argument('--w_dist', type=float, default=0.333, help="S_hierarchy中距离的权重")

    args = parser.parse_args()
    refiner = GraphWeighter(args)
    refiner.run()
