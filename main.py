from src.dataloader import create_dataloader, load_raw_tokens
from src.tokenization import tokenizer


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
