# Group 8 — Sentiment Analysis of Reviews with Negation

**Natural Language Processing | Section 3 | Option 2 | Academic Year 2026–2027**

## Overview

This project classifies short English reviews as **Positive (1)** or **Negative (0)** and investigates whether a traditional text-classification approach or a pretrained language-model approach performs better on reviews containing **negation**, including *not, never, don't,* and *wasn't*.

**Experimental question:** Which method performs better on reviews containing negation: TF-IDF with Logistic Regression or a pretrained language model approach?

## Methods

1. **Traditional:** TF-IDF word features + Logistic Regression.
2. **Pretrained:** `sentence-transformers/all-MiniLM-L6-v2` sentence embeddings + Logistic Regression classifier.

Both models use the same training, development, and held-out test splits. Classifier hyperparameters are selected on the development set, not the test set.

## Dataset

**UCI Sentiment Labelled Sentences** — reviews from Amazon, IMDb, and Yelp.

- Raw records loaded in the recorded Colab run: **2,748** (Amazon 1,000; IMDb 748; Yelp 1,000).
- After cleaning and removing 21 duplicate rows: **2,727**.
- Train: **1,908**; development: **409**; test: **410**.
- Test reviews flagged as containing negation: **94**.
- Class labels: **0 = Negative**, **1 = Positive**.

The executed download yielded fewer IMDb rows than the approximately 3,000 reviews anticipated in the proposal. The results below reflect the records actually loaded and evaluated, not the proposal's estimated dataset size.

## Recorded Results

These scores come from the **November 18, 2026 Google Colab run**.

| Evaluation | TF-IDF + LR | MiniLM + LR |
|---|---:|---:|
| Overall test accuracy | 0.8317 | **0.8902** |
| Overall test macro F1 | 0.8317 | **0.8902** |
| Negation subset accuracy (n=94) | **0.8085** | 0.7872 |
| Negation subset macro F1 (n=94) | 0.6908 | **0.6934** |

**Interpretation:** MiniLM performs better overall, but the negation-only comparison is mixed. TF-IDF has slightly higher negation accuracy, while MiniLM has only a marginally higher negation macro F1. The negation subset is small, so this is not evidence of a decisive advantage for either method on negation.

## How to Run (Google Colab)

1. Open [`Group8.ipynb`](Group8.ipynb) in Google Colab. (If your notebook has a different filename, open that `.ipynb` file instead.)
2. Use **Runtime → Run all**. An internet connection is required to download the UCI data and pretrained MiniLM model.
3. Allow the setup cell to install the necessary Python packages.
4. Review the test-set evaluation and negation analysis outputs.
5. Use the **Try it** cell to classify an unfamiliar review without retraining. Empty input is handled with a helpful message.
6. The notebook writes its exported scores, predictions, and other evidence to a `results/` folder; download this folder if you want to preserve the run outputs.

### Dependencies

- Python 3 (tested in Google Colab with Python **3.13.18**)
- `sentence-transformers`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `joblib`, `requests`

The notebook contains its own installation cell. For a local environment, use the same installation commands from that cell and ensure the machine can download the pretrained encoder.

## Error Analysis

The investigation includes error cases involving negation and mixed wording. For example, *“I promise they won't disappoint”* is a positive review, but both models predicted negative in the recorded test run. More examples and their predicted labels are exported by the notebook.

## References and Attribution

- [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) — source dataset.
- [all-MiniLM-L6-v2 on Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — pretrained sentence encoder.
- A. L. Maas et al., “Learning Word Vectors for Sentiment Analysis,” ACL-HLT, 2011.
- N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” EMNLP-IJCNLP, 2019.
- Implemented with Python and open-source NLP and ML libraries; ChatGPT was used for drafting and coding assistance. Reported metrics are from the executed Colab notebook, not generated estimates.

## Submission Notes

Course deliverables include the executable notebook, report (seven required sections), presentation, and team appendix. The appendix must record **verified, dated individual technical contributions** and **two original labelled examples per member**. Do not treat a planned role as evidence of completed work.
