# src/run_loo_experiments.py

import argparse
import torch
import pandas as pd
import numpy as np
import os
import random
import json
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from modules import MoEClassifier as MoEIRTClassifier

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TextMF(nn.Module):
    def __init__(self, question_embeddings, model_embedding_dim, alpha, num_models, num_prompts, text_dim=768, num_classes=2):
        super(TextMF, self).__init__()
        self.P = nn.Embedding(num_models, model_embedding_dim)
        self.Q = nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        self.Q.weight.data.copy_(question_embeddings)
        self.text_proj = nn.Linear(text_dim, model_embedding_dim)
        self.alpha = alpha
        self.classifier = nn.Linear(model_embedding_dim, num_classes)

    def forward(self, model, prompt, test_mode=False):
        p = self.P(model)
        q = self.Q(prompt)
        if not test_mode:
            q += torch.randn_like(q) * self.alpha
        q = self.text_proj(q)
        return self.classifier(p * q)

class LooDataset(Dataset):
    def __init__(self, df, model_map, prompt_map, is_embedllm=False):
        self.model_ids = torch.tensor(df['model_id'].map(model_map).values, dtype=torch.long)
        self.prompt_ids = torch.tensor(df['prompt_id'].map(prompt_map).values, dtype=torch.long)
        if is_embedllm:
            self.labels = torch.tensor(df['label'].values, dtype=torch.long)
        else:
            self.labels = torch.tensor(df['label'].values, dtype=torch.float)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.model_ids[idx], self.prompt_ids[idx], self.labels[idx]

def evaluate_temp_model(model, val_loader, device, is_embedllm=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for model_ids, prompt_ids, labels in val_loader:
            model_ids, prompt_ids, labels = model_ids.to(device), prompt_ids.to(device), labels.to(device)
            logits = model(model_ids, prompt_ids, test_mode=True) if is_embedllm else model(model_ids, prompt_ids)
            preds = (torch.sigmoid(logits) > 0.5).long() if not is_embedllm else torch.argmax(logits, dim=1)
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

def train_temp_moe_irt(args, train_df, val_df, model_map, prompt_map, prompt_embeddings, device):
    num_models, num_prompts = len(model_map), len(prompt_map)
    model = MoEIRTClassifier(
        num_models=num_models, num_prompts=num_prompts, num_experts=args.num_experts,
        prompt_embeddings=prompt_embeddings, model_embed_dim=args.moe_model_embed_dim, 
        top_k_experts=args.num_experts, expert_hidden_dim=args.moe_expert_hidden_dim, 
        expert_output_dim=args.moe_expert_output_dim, shared_expert_hidden_dim=args.moe_shared_hidden_dim, 
        dropout_rate=0.5, embedding_noise=0.05
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(LooDataset(train_df, model_map, prompt_map), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(LooDataset(val_df, model_map, prompt_map), batch_size=args.batch_size, shuffle=False)
    best_val_acc, best_model_state = 0.0, None
    for epoch in tqdm(range(args.loo_epochs), desc="Training MoE-IRT"):
        model.train()
        for model_ids, prompt_ids, labels in train_loader:
            model_ids, prompt_ids, labels = model_ids.to(device), prompt_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            a_q, b_q, gating_logits, _ = model.analyze_prompt(prompt_ids)
            theta = model.model_embedder(model_ids)
            logit = (torch.sum(a_q * theta, dim=1, keepdim=True) - b_q).squeeze(-1)
            loss = loss_fn(logit, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                mean_expert_logits = gating_logits.mean(dim=0)
                ideal_logit_value = mean_expert_logits.mean()
                bias_update = torch.zeros_like(model.routed_moe.bias)
                bias_update.masked_fill_(mean_expert_logits > ideal_logit_value, -args.bias_update_speed)
                bias_update.masked_fill_(mean_expert_logits < ideal_logit_value, args.bias_update_speed)
                model.routed_moe.bias.add_(bias_update)
        val_acc = evaluate_temp_model(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc, best_model_state = val_acc, model.state_dict()
    if best_model_state: model.load_state_dict(best_model_state)
    return model

def train_temp_embedllm(args, train_df, val_df, model_map, prompt_map, prompt_embeddings, device):
    num_models, num_prompts = len(model_map), len(prompt_map)
    model = TextMF(question_embeddings=prompt_embeddings, model_embedding_dim=args.embedllm_dim, alpha=0.05, num_models=num_models, num_prompts=num_prompts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(LooDataset(train_df, model_map, prompt_map, is_embedllm=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(LooDataset(val_df, model_map, prompt_map, is_embedllm=True), batch_size=args.batch_size, shuffle=False)
    best_val_acc, best_model_state = 0.0, None
    for epoch in tqdm(range(args.loo_epochs), desc="Training EmbedLLM"):
        model.train()
        for model_ids, prompt_ids, labels in train_loader:
            model_ids, prompt_ids, labels = model_ids.to(device), prompt_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(model_ids, prompt_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        val_acc = evaluate_temp_model(model, val_loader, device, is_embedllm=True)
        if val_acc > best_val_acc:
            best_val_acc, best_model_state = val_acc, model.state_dict()
    if best_model_state: model.load_state_dict(best_model_state)
    return model

def predict_benchmark_accuracy(model, model_map, target_df, prompt_map, device, is_embedllm=False):
    target_prompt_ids = target_df['prompt_id'].unique()
    target_prompt_indices = torch.tensor([prompt_map[pid] for pid in target_prompt_ids if pid in prompt_map], dtype=torch.long).to(device)
    predicted_accuracies = {}
    model.eval()
    with torch.no_grad():
        for model_id, model_idx in model_map.items():
            if model_id not in target_df['model_id'].unique(): continue
            current_model_indices = torch.tensor([model_idx], dtype=torch.long).to(device).repeat(len(target_prompt_indices))
            logits = model(current_model_indices, target_prompt_indices, test_mode=True) if is_embedllm else model(current_model_indices, target_prompt_indices)
            probs = torch.softmax(logits, dim=1)[:, 1] if is_embedllm else torch.sigmoid(logits)
            predicted_accuracies[model_id] = probs.mean().item()
    return predicted_accuracies

def main():
    parser = argparse.ArgumentParser(description="Run LOO experiments for benchmark score prediction.")
    parser.add_argument("--train_data_path", type=str, default="../data/train.csv")
    parser.add_argument("--val_data_path", type=str, default="../data/val.csv")
    parser.add_argument("--test_data_path", type=str, default="../data/test.csv")
    parser.add_argument("--prompt_embedding_path", type=str, default="../data/question_embeddings.pth")
    parser.add_argument("--results_save_path", type=str, default="../reports/benchmark_score_results.json")
    parser.add_argument("--moe_model_embed_dim", type=int, default=232)
    parser.add_argument("--num_experts", type=int, default=39)
    parser.add_argument("--moe_expert_hidden_dim", type=int, default=512)
    parser.add_argument("--moe_shared_hidden_dim", type=int, default=512)
    parser.add_argument("--moe_expert_output_dim", type=int, default=256)
    parser.add_argument("--embedllm_dim", type=int, default=232)
    parser.add_argument("--loo_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--bias_update_speed", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    CACHE_DIR = "../cache/embedllm_benchmark_preds/"
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_save_path), exist_ok=True)


    all_data_df = pd.concat([
        pd.read_csv(args.train_data_path),
        pd.read_csv(args.val_data_path),
        pd.read_csv(args.test_data_path)
    ])

    all_models = all_data_df['model_id'].unique(); model_map = {mid: i for i, mid in enumerate(all_models)}
    all_prompts_df = all_data_df[['prompt_id', 'prompt']].drop_duplicates('prompt_id'); prompt_map = {pid: i for i, pid in enumerate(all_prompts_df['prompt_id'].tolist())}
    prompt_embeddings = torch.load(args.prompt_embedding_path, map_location='cpu')
    
    main_datasets = ["mmlu", "gpqa", "truthfulqa", "logiqa", "medmcqa", "mathqa", "piqa", "gsm8k"]
    # main_datasets = ["asdiv", "mmlu", "gpqa", "truthfulqa", "logiqa", "medmcqa", "mathqa", "piqa", "social_iqa", "gsm8k"]
    def find_dataset(category):
        for ds in main_datasets:
            if ds in str(category): return ds
        return None
    all_data_df['main_dataset'] = all_data_df['category'].apply(find_dataset)

    all_results = {}
    
    for held_out_dataset in main_datasets:
        print(f"\n===== LOO ROUND: Holding out '{held_out_dataset}' =====")
        
        loo_full_train_df = all_data_df[all_data_df['main_dataset'] != held_out_dataset]
        target_df = all_data_df[all_data_df['main_dataset'] == held_out_dataset]
        
        if loo_full_train_df.empty or target_df.empty:
            print(f"Skipping '{held_out_dataset}' due to empty data split.")
            continue
        
        loo_train_df, loo_val_df = train_test_split(loo_full_train_df, test_size=0.1, random_state=42)

        true_accuracies = target_df.groupby('model_id')['label'].mean().to_dict()

        temp_moe_irt = train_temp_moe_irt(args, loo_train_df, loo_val_df, model_map, prompt_map, prompt_embeddings, device)
        moe_irt_preds = predict_benchmark_accuracy(temp_moe_irt, model_map, target_df, prompt_map, device, is_embedllm=False)
        
        cache_file = os.path.join(CACHE_DIR, f"embedllm_preds_{held_out_dataset}.json")
        embedllm_preds = None
        if os.path.exists(cache_file):
            try:
                print(f"--- cache_file: {cache_file} ---")
                with open(cache_file, 'r') as f:
                    loaded_preds = json.load(f)
                    embedllm_preds = {int(k): v for k, v in loaded_preds.items()}
            except (json.JSONDecodeError, TypeError):
                embedllm_preds = None

        if embedllm_preds is None:
            temp_embedllm = train_temp_embedllm(args, loo_train_df, loo_val_df, model_map, prompt_map, prompt_embeddings, device)
            embedllm_preds = predict_benchmark_accuracy(temp_embedllm, model_map, target_df, prompt_map, device, is_embedllm=True)
            preds_to_save = {str(k): v for k, v in embedllm_preds.items()}
            with open(cache_file, 'w') as f:
                json.dump(preds_to_save, f)

        all_results[held_out_dataset] = {
            'true_scores': true_accuracies,
            'moe_irt_pred_scores': moe_irt_preds,
            'embedllm_pred_scores': embedllm_preds
        }

    results_to_save = {}
    for dataset, data in all_results.items():
        results_to_save[dataset] = {
            'true_scores': {str(k): v for k, v in data['true_scores'].items()},
            'moe_irt_pred_scores': {str(k): v for k, v in data['moe_irt_pred_scores'].items()},
            'embedllm_pred_scores': {str(k): v for k, v in data['embedllm_pred_scores'].items()}
        }

    with open(args.results_save_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)
if __name__ == "__main__":
    main()
