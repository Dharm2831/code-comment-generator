# AI-Based Code Comment Generator (Python / Java)

A deep learning project that reads a Python or Java function and writes a one-line comment
describing what it does. It fine-tunes **CodeT5-small**, a Transformer encoder-decoder, on the
**CodeXGLUE code-to-text** dataset, and serves the result as a web app.

```
code-comment-generator/
├── src/
│   ├── config.py           all settings (dataset size, epochs, learning rate, ...)
│   ├── preprocessing.py    download, clean and tokenise the dataset
│   ├── model.py            load the Transformer, describe its layers
│   ├── training.py         training loop, saves graphs and numbers
│   ├── evaluation.py       test-set scores, model comparison, sample predictions
│   └── prediction.py       find functions in pasted code, generate comments
├── app.py                  web app (Gradio)
├── check_setup.py          checks that PyTorch can use your GPU
├── deploy.py               puts the model and the app online
├── dataset/                dataset is downloaded here automatically
├── models/                 your trained model is saved here
├── results/                graphs and numbers for the report
├── requirements.txt
└── space_requirements.txt  requirements used by the online app
```

---

## 1. How it works (the idea in one minute)

1. **Data.** The dataset holds real open-source functions paired with the comment their author
   wrote. That pair is one training example: code in, comment out.
2. **Cleaning.** The comment is removed from the code (otherwise the model would just copy it),
   only its first sentence is kept as the target, and bad pairs are dropped (too short,
   non-English, duplicates).
3. **Tokenisation.** Text is split into sub-word tokens and turned into numbers. Input is cut
   at 256 tokens, output at 48.
4. **Model.** CodeT5-small is a Transformer: 6 encoder layers read the code, 6 decoder layers
   write the comment one token at a time. It was already pre-trained on a lot of code, so you
   only **fine-tune** it (transfer learning) instead of training from zero.
5. **Training.** For each batch: forward pass → cross-entropy loss between predicted and real
   comment tokens → backpropagation → AdamW updates the weights. After every epoch the model
   is checked on validation data and the best version is saved.
6. **Evaluation.** On unseen test functions, generated comments are compared with the human
   ones using **BLEU** and **chrF** (0-100, higher is better).
7. **App.** The web page splits pasted code into functions, asks the model for a comment for
   each, and inserts it above the function.

---

## 2. What you need

| Item | Why |
|---|---|
| Python 3.10 - 3.13 | run `python --version` to check |
| VS Code + the "Python" extension | editing and running |
| Up-to-date NVIDIA driver | the RTX 5050 needs a recent driver (NVIDIA App or nvidia.com) |
| PyTorch built for CUDA 12.8 or newer | RTX 50-series cards do not work with older builds |
| Git | to put the code on GitHub (git-scm.com) |
| Free Hugging Face account | hosts the model and the live app (huggingface.co) |
| Free GitHub account | the report asks for a repository link |
| About 10 GB free disk and internet | PyTorch, the dataset and the model are downloaded once |

You do **not** need to install CUDA or cuDNN separately. The PyTorch package includes them.

---

## 3. Setup (once)

Open the project folder in VS Code, then open a terminal (`Terminal → New Terminal`).
Commands below are for Windows PowerShell.

```powershell
# 1. create and activate a private Python environment for this project
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activate script, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

```powershell
# 2. PyTorch with GPU support (large download)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. everything else
pip install -r requirements.txt

# 4. check the GPU
python check_setup.py
```

`check_setup.py` must end with **"OK: the GPU works"**. If it does not, open
<https://pytorch.org/get-started/locally/>, choose *Stable / Windows / Pip / Python / the newest
CUDA*, and run the command it shows after `pip uninstall -y torch`.

In VS Code press `Ctrl+Shift+P → Python: Select Interpreter` and pick the one inside `.venv`.

---

## 4. Train

```powershell
# a) smoke test: tiny data, one epoch. Confirms downloads and the GPU work.
python -m src.training --quick

# b) the real run (30,000 Python + 30,000 Java functions, 3 epochs)
python -m src.training
```

The first run downloads the dataset into `dataset/` and the base model.
The quick test shows how fast your laptop goes, so you can judge how long the real run will
take. Keep the laptop plugged in.

Useful options:

```powershell
python -m src.training --samples 60000 --epochs 4     # more data, better comments, longer
python -m src.training --batch-size 8                 # if you see "CUDA out of memory"
```

After training you have:

- `models/comment-generator/` - the trained model
- `results/training_curves.png` - training vs validation loss and accuracy
- `results/training_history.json` - the same numbers per epoch
- `results/run_config.json` - every hyperparameter, dataset sizes, layer counts, training time

---

## 5. Evaluate

```powershell
python -m src.evaluation
```

This scores the model on 1,000 Python + 1,000 Java test functions it never saw and writes:

- `results/test_metrics.json` - BLEU, chrF, test loss, token accuracy, inference time per
  function, scores per language, and the same scores for the model **before fine-tuning**
- `results/sample_predictions.csv` - code, human comment, generated comment (open in Excel)

What to expect: comment generation is a hard task. In the CodeT5 paper the small model scores
about 20 BLEU on this dataset after training on all of it (using a slightly different BLEU
formula, so your number will not match exactly). A score in the teens or low twenties is a
normal result. Many generated comments are correct but worded differently from the human one,
which BLEU counts as wrong.

---

## 6. Run the app on your laptop

```powershell
python app.py
```

Open <http://127.0.0.1:7860>, paste code (or click an example) and press **Generate comments**.
Take your screenshots for the report here.

`python app.py --share` gives a temporary public link that works while your laptop is on.

---

## 7. Put it live

### a) Live app on Hugging Face (free)

1. Create an account at huggingface.co.
2. `Settings → Access Tokens → Create new token`, type **Write**. Copy it.
3. Run:

```powershell
python deploy.py
```

Paste the token when asked. The script uploads your model, creates a public Space
(the web app), and prints both links. The Space takes a few minutes to build the first time.
It runs on a free CPU machine, so each request takes a little longer than on your GPU, and
it goes to sleep when unused (the first visit after that wakes it up).

The **Live app** link is your "Deployed Link" for Appendix A.

### b) Code on GitHub

Create an empty repository on github.com, then:

```powershell
git init
git add .
git commit -m "AI code comment generator"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/code-comment-generator.git
git push -u origin main
```

`.gitignore` keeps the dataset and the trained model out of GitHub (they are too large; the
model lives on Hugging Face instead).

---

## 8. Where each report chapter gets its content

Everything below comes from your own run, so the report describes the actual implementation.

| Report section | Take it from |
|---|---|
| 3.2 Hardware / software requirements | Section 2 above, `requirements.txt`, `pip list` |
| 4.1 System architecture diagram | The 7 steps in Section 1 (draw them as boxes and arrows) |
| 4.2 Model architecture, layer table | `results/run_config.json` → `model`, and `python -m src.model` |
| 4.3 Activation functions | ReLU inside each feed-forward block, softmax at the output |
| 4.4 Loss function | Cross-entropy over the vocabulary (`src/training.py`) |
| 4.5 Optimizer | AdamW, learning rate 5e-5, linear warm-up then decay |
| 4.6 / Appendix C Model configuration | `results/run_config.json` |
| 5.1 - 5.3 Dataset description, statistics | `src/config.py`, `results/run_config.json`, `results/test_metrics.json` |
| 5.4 Preprocessing | `src/preprocessing.py` (cleaning, duplicate removal, tokenisation, truncation) |
| 5.5 Train / validation / test split | The dataset's own official splits; counts in the result files |
| 5.6 Data augmentation | Not performed, so leave this section out |
| 5.7 Sample dataset | First rows of `results/sample_predictions.csv` (code + human comment) |
| 6 Methodology | Section 1 above + the numbered comments in `src/training.py` |
| 7 Implementation, code snippets | Short pieces of `src/*.py`; the folder tree at the top of this file |
| 7.6 User interface | Screenshots of the running app |
| 8.1 Training results, graphs | `results/training_curves.png`, `results/training_history.json` |
| 8.2 / 8.3 Test cases | Paste five or more functions into the app; record input, expected comment, actual comment |
| 8.4 Confusion matrix | Not applicable: this is text generation, not classification |
| 9.1 Performance metrics | `results/test_metrics.json` (BLEU, chrF, token accuracy, loss, inference time) |
| 9.3 Model comparison | `before_fine_tuning` vs `fine_tuned_model` in `results/test_metrics.json` |
| 10 Challenges and limitations | See below, plus whatever you actually ran into |
| Appendix A | Your GitHub link and the Hugging Face Space link |

Honest limitations to mention: one summary line per function, not line-by-line comments;
functions longer than 256 tokens are cut off; trained on a subset of the data; the model can
write a fluent comment that is wrong; Java methods with unusual formatting may not be detected.

---

## 9. Papers and sources to cite

1. A. Vaswani et al., "Attention Is All You Need," NeurIPS, 2017. (Transformer)
2. C. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text
   Transformer," JMLR, 2020. (T5)
3. Y. Wang, W. Wang, S. Joty, S. C. H. Hoi, "CodeT5: Identifier-aware Unified Pre-trained
   Encoder-Decoder Models for Code Understanding and Generation," EMNLP, 2021. (the model used)
4. Z. Feng et al., "CodeBERT: A Pre-Trained Model for Programming and Natural Languages,"
   Findings of EMNLP, 2020.
5. W. U. Ahmad et al., "A Transformer-based Approach for Source Code Summarization," ACL, 2020.
6. X. Hu et al., "Deep Code Comment Generation," ICPC, 2018. (LSTM-based DeepCom)
7. S. Iyer et al., "Summarizing Source Code using a Neural Attention Model," ACL, 2016. (CodeNN)
8. H. Husain et al., "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search,"
   arXiv:1909.09436, 2019. (source of the data)
9. S. Lu et al., "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and
   Generation," NeurIPS Datasets and Benchmarks, 2021. (the dataset used)
10. K. Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation," ACL, 2002.

Dataset page: <https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text>
Model page: <https://huggingface.co/Salesforce/codet5-small>

Check each reference against the original paper before it goes into the report.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `check_setup.py` says PyTorch cannot see the GPU | You have the CPU build. `pip uninstall -y torch`, then reinstall with the command in Section 3 |
| "no kernel image is available" / "sm_120 is not compatible" | PyTorch build too old for the RTX 5050. Reinstall as above |
| `CUDA out of memory` | Add `--batch-size 8`; close games and browsers using the GPU |
| `No module named 'src'` | Run commands from the project folder, with `python -m src.training` (not `python src/training.py`) |
| "No trained model found" | Train first. The app loads `models/comment-generator` |
| Download fails or is slow | Check the connection and run the same command again; downloads resume |
| Warning about symlinks on Windows | Harmless, ignore it |
