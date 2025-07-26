
# 🚀 GPT-2: Training from Scratch & Fine-Tuning for Text Classification

This project demonstrates a complete NLP pipeline with two stages:

1. Training a **GPT-2 model from scratch** using a custom tokenizer and the C4 dataset.
2. Fine-tuning that model for **text classification** on the IMDB sentiment dataset.

---

## 📁 Project Structure

```
├── gpt-small-c4/                       # Trained GPT-2 model checkpoints
├── gpt-tokenizer/                      # Custom BPE tokenizer
├── logs/                               # Training logs (HuggingFace Trainer)
├── Trained_gpt2_from_scrath.ipynb     # Notebook: train GPT-2 from scratch
├── Text_classification_Pretrained.ipynb # Notebook: fine-tune for sentiment analysis
├── gpt_tokenizer                       # Tokenizer artifact (JSON)
└── README.md                           # This file
```

---

## 🧠 Part 1: Training GPT-2 from Scratch

📘 Notebook: `Trained_gpt2_from_scrath.ipynb`

### What it does:
- Loads a clean dataset: `c4-filter-small` (text-only).
- Trains a **Byte Pair Encoding (BPE)** tokenizer using `tokenizers`.
- Wraps it with `PreTrainedTokenizerFast` and saves it.
- Initializes a GPT-2 model using `GPT2LMHeadModel`.
- Trains the model from scratch using `Trainer`.

### Output:
- `gpt-tokenizer/`: Trained tokenizer.
- `gpt-small-c4/`: GPT-2 checkpoints trained on C4.

---

## 🔤 Part 2: Fine-tuning for Text Classification

📘 Notebook: `Text_classification_Pretrained.ipynb`

### Task:
Binary classification (positive/negative) on IMDB movie reviews.

### What it does:
- Loads the IMDB dataset from Hugging Face.
- Uses the previously trained GPT-2 tokenizer.
- Tokenizes the text with truncation and padding.
- Loads `GPT2ForSequenceClassification` with 2 output labels.
- Fine-tunes the model and evaluates on test set.

### Metrics:
- Accuracy
- Loss
- (Optionally) F1-score, Precision, Recall

---

## 🛠 Installation

Install the required Python libraries:

```bash
pip install transformers datasets tokenizers scikit-learn
```

---

## ⚙️ Run on Google Colab

To run this project on Google Colab:

1. Open the `.ipynb` files in Colab.
2. Mount Google Drive when prompted.
3. Run `Trained_gpt2_from_scrath.ipynb` to train the model and tokenizer.
4. Then run `Text_classification_Pretrained.ipynb` to fine-tune and evaluate.

---

## ☁️ Upload to Hugging Face Hub

You can publish the trained model and tokenizer with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model.push_to_hub("your-username/gpt2-from-scratch")
tokenizer.push_to_hub("your-username/gpt-tokenizer")
```

---

## 🖼 Project Illustration

![Folder Structure](./c7719961-03c6-478c-aced-7b0e3d734d18.png)

---

## 👤 Author

**Nguyễn Tiến Anh**  
*Feel free to add contact info or GitHub/Hugging Face links here*

---

## 📚 References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)
