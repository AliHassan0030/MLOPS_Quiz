# Run preprocessing script [cite: 35]
preprocess:
	python src/preprocess.py

# Run training script [cite: 43]
train:
	python src/train.py

# Full pipeline: preprocess -> train [cite: 47, 49]
all: preprocess train
	@echo "Pipeline finished. Check terminal for accuracy."