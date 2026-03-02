# Training Your Own Models

Sentio has two independent model systems. Each can be trained separately.

---

## 1. NLU Ensemble (Browser-based)

**No setup required.** Open the app, go to `/train`, and click Train. The 5-classifier ensemble trains in 1-3 seconds on your browser.

### Adding your own intents

1. Go to `/intents`
2. Create new intents with at least 2 examples each (30+ recommended)
3. Go to `/train` and train
4. Test on `/test`

The model persists in localStorage. Export via the Train page in 7 formats (Rasa, Dialogflow, Lex, LUIS, Wit.ai, CSV, JSON).

### Python pipeline (large datasets)

For 50K+ examples or when you want GPU-accelerated embeddings:

```bash
cd training
pip install -r requirements.txt

# Quick: TF-IDF + 5 classifiers (~30 seconds)
python train_nlu.py --input training_data.json --output results/ --quick

# Full: Optuna hyperparameter optimization (~15 minutes)
python train_nlu.py --input training_data.json --output results/ --optimize --n-trials 100

# With embeddings + knowledge distillation
python train_nlu.py --input training_data.json --output results/ --optimize --embeddings --distill
```

### Synthetic data generation

Generate training examples using Google Gemini:

```bash
export GOOGLE_API_KEY=your-key
python generate_synthetic.py \
  --input training_data.json \
  --output augmented_data.json \
  --provider gemini \
  --model gemini-2.0-flash \
  --count 50
```

Cost: ~$0.02 for 600 examples.

---

## 2. Vulnerability Classifier (PyTorch)

The vulnerability classifier is a 9.7M-parameter ResNet-MLP trained on 224K+ real CVEs.

### Architecture

```
TF-IDF (50K features)
  → Linear(50K, 192) + BatchNorm + GELU
  → Linear(192, 192) + BatchNorm + GELU + skip connection
  → Linear(192, 96) + BatchNorm + GELU
  → Linear(96, 349 CWE classes)
```

### Training on GCP

The model requires ~16 GB RAM and trains in ~2 hours on a CPU VM.

```bash
cd training/vuln-classifier

# Set up a GCP spot VM (~$0.07 for 2 hours)
bash setup_vm.sh

# SSH in and start training
gcloud compute ssh vuln-trainer --zone=asia-south2-a --tunnel-through-iap
cd /opt/vuln-trainer
source venv/bin/activate
nohup python train.py > training.log 2>&1 &
```

Training automatically:
1. Downloads the CVE-CWE dataset from HuggingFace (224K records)
2. Fits a TF-IDF vectorizer (50K features)
3. Trains for 50 epochs with focal loss and cosine annealing
4. Saves checkpoints every 5 epochs
5. Exports weights for JavaScript inference

### Exporting weights for the web app

After training, export the weights with BatchNorm fusion:

```bash
python export_weights.py --checkpoint output/best_model.pt --output output/weights.json
```

This produces:
- `weights.json` (~87 MB) — fused weight matrices for pure JS inference
- `labels.json` — CWE class mapping
- `tfidf_vocab.json` — vocabulary for TF-IDF vectorization
- `metrics.json` — accuracy and performance metrics

Copy these to `public/models/vuln-classifier/` in the web app.

### Training on your own data

To train on a different dataset:

1. Prepare a CSV/JSON with columns: `text` (description) and `label` (class)
2. Modify `train.py` to load your dataset instead of the HuggingFace one
3. Adjust `TFIDF_FEATURES` and architecture constants as needed
4. Run training and export

### Requirements

```
torch>=2.0
scikit-learn>=1.5
datasets>=2.0
pandas>=2.0
numpy>=1.26
```

See `training/vuln-classifier/requirements.txt` for the full list.

---

## Choosing the Right Approach

| Scenario | Approach | Time | Cost |
|----------|----------|------|------|
| Quick intent classifier | Browser training | 3 seconds | Free |
| Custom NLU with 50K+ examples | Python pipeline | 15-30 min | Free |
| NLU with embeddings | Python + GPU VM | 30-60 min | ~$0.12 |
| Vulnerability classifier | PyTorch + CPU VM | ~2 hours | ~$0.07 |
| Custom classifier (new domain) | Fork + modify train.py | Varies | Varies |
