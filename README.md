# DIY-LLM

Just a pet project of mine to learn the inner workings of LLMs and PyTorch. The goal is to end up with a small LLM I've trained myself (however bad it may be).

## Data Download

Because I only have so much RAM, the data is downloaded from Dolma, extracted, and split into individual files of size `--max-file-size-mb` (default of 512MB).

```bash
python src/download_training_data.py --data-dir data/ --extract --limit 5 --random --max-file-size-mb 512
```

This will create a directory structure like this:

```
data/
├── <filename>_part001.json
├── <filename>_part002.json
├── ...
└── <filename>_partNNN.json
```

During training, the dataloader in `src/dataloader.py` loads these files and tokenizes them iteratively, inheriting from `IterableDataset` to hopefully keep memory overhead low.
