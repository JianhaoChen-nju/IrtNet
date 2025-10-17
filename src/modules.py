# src/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
torch.manual_seed(42)
random.seed(42)

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class SparseMoE(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, expert_hidden_dim, expert_output_dim, dropout_rate):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gating_network = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([Expert(input_dim, expert_hidden_dim, expert_output_dim, dropout_rate) for _ in range(num_experts)])
        self.register_buffer("bias", torch.zeros(1, num_experts))

    def forward(self, x):
        gating_logits = self.gating_network(x)
        biased_logits = gating_logits + self.bias
        _, indices = torch.topk(biased_logits, self.top_k, dim=-1)

        weights_for_gating = F.softmax(gating_logits.gather(1, indices), dim=-1)
        batch_size, _ = x.shape
        flat_x = x.repeat_interleave(self.top_k, dim=0)
        flat_indices = indices.view(-1)
        expert_outputs = torch.zeros(batch_size * self.top_k, self.experts[0].net[-1].out_features, device=x.device)
        for i in range(self.num_experts):
            mask = (flat_indices == i)
            if mask.any():
                expert_input = flat_x[mask]
                expert_outputs[mask] = self.experts[i](expert_input)

        expert_outputs = expert_outputs.view(batch_size, self.top_k, -1)
        weighted_outputs = expert_outputs * weights_for_gating.unsqueeze(-1)
        final_output = weighted_outputs.sum(dim=1)

        return final_output, gating_logits, indices

class MoEClassifier(nn.Module):
    def __init__(self, num_models, num_prompts, num_experts, prompt_embeddings,
        model_embed_dim, top_k_experts,
        expert_hidden_dim, expert_output_dim,
        shared_expert_hidden_dim, dropout_rate, embedding_noise):
        
        super().__init__()
        prompt_embed_dim = prompt_embeddings.shape[1]
        self.model_embedder = nn.Embedding(num_models, model_embed_dim)
        self.prompt_embedder = nn.Embedding(num_prompts, prompt_embed_dim, _weight=prompt_embeddings)
        self.prompt_embedder.requires_grad_(False)
        moe_input_dim = prompt_embed_dim
        self.shared_expert = Expert(moe_input_dim, shared_expert_hidden_dim, expert_output_dim, dropout_rate)
        print(f"Initializing model with a fixed number of {num_experts} routed experts.")
        self.routed_moe = SparseMoE(
            input_dim=moe_input_dim,
            num_experts=num_experts,
            top_k=top_k_experts,
            expert_hidden_dim=expert_hidden_dim,
            expert_output_dim=expert_output_dim,
            dropout_rate=dropout_rate
        )
        self.difficulty_predictor = nn.Linear(expert_output_dim, 1)
        self.discrimination_predictor = nn.Linear(expert_output_dim, model_embed_dim)
        self.embedding_noise = embedding_noise
        
    def analyze_prompt(self, prompt_ids):
        prompt_vecs = self.prompt_embedder(prompt_ids)
        moe_input = prompt_vecs
        if self.training and self.embedding_noise > 0:
            moe_input = moe_input + torch.randn_like(moe_input) * self.embedding_noise
        shared_expert_output = self.shared_expert(moe_input)
        routed_experts_output, gating_logits, indices = self.routed_moe(moe_input)
        h_q = shared_expert_output + routed_experts_output
        b_q = self.difficulty_predictor(h_q)
        a_q = self.discrimination_predictor(h_q)
        return a_q, b_q, gating_logits, indices

    def forward(self, model_ids, prompt_ids):
        a_q, b_q, _, _ = self.analyze_prompt(prompt_ids)
        theta = self.model_embedder(model_ids)
        ability_term = torch.sum(a_q * theta, dim=1, keepdim=True)
        logit = ability_term - b_q
        return logit.squeeze(-1)


