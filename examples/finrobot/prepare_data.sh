mkdir -p data1
wget https://huggingface.co/datasets/FinGPT/fingpt-fiqa_qa/resolve/main/data/train-00000-of-00001-ab79bf9300210e98.parquet -O data/fiqa.parquet
wget https://huggingface.co/datasets/FinGPT/fingpt-headline/resolve/main/data/train-00000-of-00001-b8e635bd2f11110b.parquet -O data/headline.parquet
wget https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train/resolve/main/data/train-00000-of-00001-dabab110260ac909.parquet -O data/sentiment.parquet
wget https://huggingface.co/datasets/FinGPT/fingpt-finred/resolve/main/data/train-00000-of-00001-4af4d9c6fa204dbc.parquet -O data/relation.parquet