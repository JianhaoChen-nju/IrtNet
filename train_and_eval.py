# src/train_and_eval.py

import torch
import pandas as pd
import os
import argparse
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from modules import MoEClassifier
import torch.nn.functional as F

torch.manual_seed(42)

class ClassificationDataset(Dataset):
    def __init__(self, df, model_map, prompt_map):
        self.model_ids = torch.tensor(df['model_id'].map(model_map).values, dtype=torch.long)
        self.prompt_ids = torch.tensor(df['prompt_id'].map(prompt_map).values, dtype=torch.long)
        self.labels = torch.tensor(df['label'].astype(float).values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.model_ids[idx], self.prompt_ids[idx], self.labels[idx]

def create_router_dataloader(test_df, model_map, prompt_map):
    label_lookup = {}
    for _, row in test_df.iterrows():
        p_id = prompt_map.get(row['prompt_id'])
        m_id = model_map.get(row['model_id'])
        if p_id is not None and m_id is not None:
             label_lookup[(p_id, m_id)] = row['label']


    unique_prompts = [pid for pid in test_df['prompt_id'].unique() if pid in prompt_map]
    unique_models = [mid for mid in test_df['model_id'].unique() if mid in model_map]


    router_prompts, router_models, router_labels = [], [], []

    for prompt_id_str in tqdm(unique_prompts, desc="routing data"):
        p_id = prompt_map[prompt_id_str]
        for model_id_str in unique_models:
            m_id = model_map[model_id_str]
            label = label_lookup.get((p_id, m_id), 0.0)
            router_prompts.append(p_id)
            router_models.append(m_id)
            router_labels.append(label)

    batch_size = len(unique_models)
    router_dataset = TensorDataset(
        torch.tensor(router_prompts, dtype=torch.long),
        torch.tensor(router_models, dtype=torch.long),
        torch.tensor(router_labels, dtype=torch.float)
    )
    router_dataloader = DataLoader(router_dataset, batch_size=batch_size, shuffle=False)

    return router_dataloader


def train(args, device):
    train_df = pd.read_csv(args.train_data_path); val_df = pd.read_csv(args.val_data_path); test_df = pd.read_csv(args.test_data_path)
    all_models = pd.concat([train_df['model_id'], val_df['model_id'], test_df['model_id']]).unique(); model_map = {mid: i for i, mid in enumerate(all_models)}; num_models = len(model_map)
    all_prompts_df = pd.concat([train_df[['prompt_id', 'prompt']], val_df[['prompt_id', 'prompt']], test_df[['prompt_id', 'prompt']]]).drop_duplicates('prompt_id').sort_values('prompt_id'); prompt_map = {pid: i for i, pid in enumerate(all_prompts_df['prompt_id'].tolist())}
    if not os.path.exists(args.prompt_embedding_path):
        sbert_model = SentenceTransformer('all-mpnet-base-v2', device=device); prompt_embeddings = torch.tensor(sbert_model.encode(all_prompts_df['prompt'].tolist(), show_progress_bar=True)); torch.save(prompt_embeddings, args.prompt_embedding_path)
    else:
        prompt_embeddings = torch.load(args.prompt_embedding_path, map_location='cpu')

    train_dataset = ClassificationDataset(train_df, model_map, prompt_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = ClassificationDataset(val_df, model_map, prompt_map)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MoEClassifier(
        num_models=num_models, 
        num_prompts=len(prompt_map), 
        num_experts=args.num_experts,
        prompt_embeddings=prompt_embeddings, 
        model_embed_dim=args.model_embed_dim, 
        top_k_experts=args.top_k_experts, 
        expert_hidden_dim=args.expert_hidden_dim, 
        expert_output_dim=args.expert_output_dim, 
        shared_expert_hidden_dim=args.shared_expert_hidden_dim, 
        dropout_rate=args.dropout_rate, 
        embedding_noise=args.embedding_noise
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    best_val_accuracy, epochs_no_improve = 0.0, 0

    for epoch in range(args.epochs):
        model.train()
        for model_ids, prompt_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training"):
            model_ids, prompt_ids, labels = model_ids.to(device), prompt_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            
            a_q, b_q, gating_logits, indices = model.analyze_prompt(prompt_ids)
            theta = model.model_embedder(model_ids)
            ability_term = torch.sum(a_q * theta, dim=1, keepdim=True)
            logit = (ability_term - b_q).squeeze(-1)
            
            loss = loss_fn(logit, labels)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                mean_expert_logits = gating_logits.mean(dim=0)
                
                ideal_logit_value = mean_expert_logits.mean()
                
                is_overloaded = mean_expert_logits > ideal_logit_value
                is_underloaded = mean_expert_logits < ideal_logit_value
                
                bias_update = torch.zeros_like(model.routed_moe.bias)
                bias_update.masked_fill_(is_overloaded, -args.bias_update_speed)
                bias_update.masked_fill_(is_underloaded, args.bias_update_speed)
                
                model.routed_moe.bias.add_(bias_update)

        model.eval()
        val_loss, val_labels, val_preds = 0, [], []
        with torch.no_grad():
            for model_ids, prompt_ids, labels in val_loader:
                model_ids, prompt_ids, labels = model_ids.to(device), prompt_ids.to(device), labels.to(device)
                logits = model(model_ids, prompt_ids)
                preds = (torch.sigmoid(logits) > 0.5).long()
                val_labels.extend(labels.cpu().tolist()); val_preds.extend(preds.tolist())
                val_loss += loss_fn(logits, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"\nEpoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        scheduler.step(avg_val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy, epochs_no_improve = val_accuracy, 0
            torch.save(model.state_dict(), args.model_save_path)
            print(f"model_saved {args.model_save_path}")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= args.early_stopping_patience:
            break
    
    return args.model_save_path, model_map, prompt_map, best_val_accuracy


def evaluate(args, best_model_path, model_map, prompt_map, best_val_accuracy, device):
    train_df = pd.read_csv(args.train_data_path)
    test_df = pd.read_csv(args.test_data_path)
    num_models = len(model_map)
    prompt_embeddings = torch.load(args.prompt_embedding_path, map_location='cpu')

    model = MoEClassifier(
        num_models=num_models, 
        num_prompts=len(prompt_map), 
        num_experts=args.num_experts,
        prompt_embeddings=prompt_embeddings, 
        model_embed_dim=args.model_embed_dim,
        top_k_experts=args.top_k_experts, 
        expert_hidden_dim=args.expert_hidden_dim, 
        expert_output_dim=args.expert_output_dim, 
        shared_expert_hidden_dim=args.shared_expert_hidden_dim, 
        dropout_rate=args.dropout_rate, 
        embedding_noise=args.embedding_noise
    ).to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_labels, test_preds = [], []
    correctness_test_loader = DataLoader(ClassificationDataset(test_df, model_map, prompt_map), batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for model_ids, prompt_ids, labels in tqdm(correctness_test_loader, desc="correctness prediction"):
            model_ids, prompt_ids, labels = model_ids.to(device), prompt_ids.to(device), labels.to(device)
            logits = model(model_ids, prompt_ids)
            preds = (torch.sigmoid(logits) > 0.5).long()
            test_labels.extend(labels.cpu().tolist())
            test_preds.extend(preds.tolist())
    test_accuracy = accuracy_score(test_labels, test_preds)
    
    
    main_datasets = ["asdiv", "mmlu", "gpqa", "truthfulqa", "logiqa", "medmcqa", "mathqa", "piqa", "social_iqa", "gsm8k"]
    
    def find_dataset(category):
        for ds in main_datasets:
            if ds in category:
                return ds
        return None

    all_prompts_info = pd.concat([
        train_df[['prompt_id', 'category']],
        test_df[['prompt_id', 'category']]
    ]).drop_duplicates('prompt_id')
    all_prompts_info['main_dataset'] = all_prompts_info['category'].apply(find_dataset)
    prompt_id_to_dataset_map = all_prompts_info.set_index('prompt_id')['main_dataset'].to_dict()
    
    idx_to_prompt_id = {idx: pid for pid, idx in prompt_map.items()}

    successful_routes_by_dataset = {ds: 0 for ds in main_datasets}
    total_prompts_by_dataset = {ds: 0 for ds in main_datasets}
    
    router_loader = create_router_dataloader(test_df, model_map, prompt_map)


    with torch.no_grad():
        for prompts, models, labels in tqdm(router_loader, desc="evaluate routing"):
            logits = model(models.to(device), prompts.to(device))
            best_model_idx = torch.argmax(logits)
            
            prompt_idx = prompts[0].item()
            prompt_id = idx_to_prompt_id.get(prompt_idx)
            main_dataset = prompt_id_to_dataset_map.get(prompt_id)

            if main_dataset:
                total_prompts_by_dataset[main_dataset] += 1
                if labels[best_model_idx] == 1.0:
                    successful_routes_by_dataset[main_dataset] += 1


    dataset_accuracies = []
    for dataset in main_datasets:
        total = total_prompts_by_dataset[dataset]
        if total > 0:
            accuracy = successful_routes_by_dataset[dataset] / total
            dataset_accuracies.append(accuracy)
            print(f"  - {dataset}: = {accuracy:.4f} ({successful_routes_by_dataset[dataset]}/{total})")
        else:
            print(f"  - {dataset}: no test data.")

    macro_average = sum(dataset_accuracies) / len(dataset_accuracies) if dataset_accuracies else 0.0
    total_successful_routes = sum(successful_routes_by_dataset.values())
    total_prompts = sum(total_prompts_by_dataset.values())
    micro_average = total_successful_routes / total_prompts if total_prompts > 0 else 0.0
    

    print("\n\n================ final report ================")
    if isinstance(best_val_accuracy, str):
        print(f"best_val_accuracy: {best_val_accuracy}")
    else:
        print(f"best_val_accuracy: {best_val_accuracy:.4f}")
    print(f"test_accuracy: {test_accuracy:.4f}")
    print("---------------------------------------------")
    print(f"  -(Macro-average): {macro_average:.4f}")
    print(f"  -(Micro-average): {micro_average:.4f}")
    print("=============================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and Evaluate the Decoupled MoE-IRT Classifier.")
    parser.add_argument("--eval_only", action="store_true", help="only eval")
    parser.add_argument("--train_data_path", type=str, default="../data/train.csv")
    parser.add_argument("--val_data_path", type=str, default="../data/val.csv")
    parser.add_argument("--test_data_path", type=str, default="../data/test.csv")
    parser.add_argument("--prompt_embedding_path", type=str, default="../data/question_embeddings.pth")
    parser.add_argument("--model_save_path", type=str, default="../models/best_decoupled_moe_irt.pth")
    parser.add_argument("--model_embed_dim", type=int, default=232)
    parser.add_argument("--num_experts", type=int, default=39, help="routed experts num")
    parser.add_argument("--top_k_experts", type=int, default=39)
    parser.add_argument("--bias_update_speed", type=float, default=0.01, help="Update rate of the bias term in unassisted lossy load balancing (gamma)")
    parser.add_argument("--expert_hidden_dim", type=int, default=512)
    parser.add_argument("--shared_expert_hidden_dim", type=int, default=512)
    parser.add_argument("--expert_output_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--embedding_noise", type=float, default=0.05)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    args = parser.parse_args()
    
    # ... (the rest of the __main__ block remains the same) ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.eval_only:
        if not os.path.exists(args.model_save_path):
            exit()

        train_df = pd.read_csv(args.train_data_path)
        val_df = pd.read_csv(args.val_data_path)
        test_df = pd.read_csv(args.test_data_path)
        all_models = pd.concat([train_df['model_id'], val_df['model_id'], test_df['model_id']]).unique()
        model_map = {mid: i for i, mid in enumerate(all_models)}
        all_prompts_df = pd.concat([train_df[['prompt_id', 'prompt']], val_df[['prompt_id', 'prompt']], test_df[['prompt_id', 'prompt']]]).drop_duplicates('prompt_id').sort_values('prompt_id')
        prompt_map = {pid: i for i, pid in enumerate(all_prompts_df['prompt_id'].tolist())}
        
        evaluate(args, args.model_save_path, model_map, prompt_map, "N/A", device)

    else:
        best_model_path, model_map, prompt_map, best_val_acc = train(args, device)
        evaluate(args, best_model_path, model_map, prompt_map, best_val_acc, device)
