import torch

from src.dataloader import create_dataloader, load_raw_tokens
from src.tokenization import tokenizer
from src.modules import GPT2Ripoff, TransformerBlock

torch.manual_seed(42)

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embed_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


print(tokenizer.decode([15496, 11, 7062, 15500, 0]))

# text_tokens = load_raw_tokens(1000)
dataloader = create_dataloader(batch_size=2, max_length=4, stride=2)

print()
for i, (input_ids, target_ids) in enumerate(dataloader):
    batches = input_ids.shape[0]
    print(f"=== Sample {i} ===")
    for batch in range(batches):
        decoded_in  = tokenizer.decode(input_ids[batch].tolist())
        decoded_out = tokenizer.decode(target_ids[batch].tolist())

        print(f"--> Batch {batch}")
        print(input_ids[batch].tolist())
        print(target_ids[batch].tolist())

        print(f"{decoded_in}\n{decoded_out}\n")
    if i >= 3:
        break

x = torch.rand((2, 4, 768))
print(f"Input shape: {x.shape}")
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print(f"Output shape: {output.shape}")

model = GPT2Ripoff(GPT_CONFIG_124M)
print(f"Total number of parameters: {sum(p.numel() for p in model.parameters()):,}")

t = torch.tensor([[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]], dtype=torch.long)
output = model(t)
print(f"Output shape: {output.shape}")
