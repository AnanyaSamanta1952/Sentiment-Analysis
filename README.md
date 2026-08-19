# <b>Cross-Domain Explainable Sentiment Analysis</b>

Sentiment classifiers trained on one product category often fail on another — a model trained on Book reviews doesn't automatically understand Kitchen appliance reviews. This project fine-tunes BERT to generalize across domains, benchmarks a domain-adversarial (DANN) approach against it, and adds LIME-based explainability to interpret individual predictions.

<b>Dataset</b>: Multi-Domain Sentiment Dataset — Books, DVD, Electronics, Kitchen & Housewares.

# Notebooks
1. <b>01_bert_finetune_crossdomain.ipynb</b> — main result: fine-tune + cross-domain eval
2. <b>02_pretrained_baseline_4fold.ipynb</b> — zero-shot baseline, 4-fold rotation
3. <b>03_dann_experiment.ipynb</b> — domain-adversarial training experiment
4. <b>04_explainability_lime.ipynb</b> — LIME explanations for individual predictions

<b>Setup</b>
  ```
    pip install -r requirements.txt
  ```
<b>Requirements</b>
1. torch>=2.0.0
2. transformers>=4.30.0
3. scikit-learn>=1.2.0
4. nltk>=3.8
5. lime>=0.2.0.1
6. matplotlib>=3.7.0
7. seaborn>=0.12.0
8. numpy>=1.24.0
