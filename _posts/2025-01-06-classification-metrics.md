---
layout: post
title: "Metrics for Classification Problems Explained: Accuracy, Precision, Recall, and F1-Score"
date: 2025-01-06T18:50:00+11:00
categories:
  - blog
tags:
  - datascience
  - fundamentals
---
In the world of machine learning, evaluating a model's performance is as crucial as building the model itself. Imagine developing a spam email filter or a disease diagnostic tool without a clear idea of how well it works. That’s where metrics for machine learning problems come into play. Today, we’ll break down four key metrics (for classification problems) — accuracy, precision, recall, and F1-score — to understand what they mean, when to use them, and why.

### Accuracy: The Simplest Measure

Accuracy is perhaps the first metric that comes to mind when evaluating a model. It represents the proportion of correct predictions out of all predictions:

$$Accuracy = \frac{n_{TP} + n_{TN}}{n_{TP} + n_{TN} + n_{FP} + n_{FN}} = \frac{n_{TP} + n_{TN}}{n_{Total}}$$

where:
- $$n_{TP}$$: number of True Positives (i.e., the instances where the model correctly predicts the positive class)
- $$n_{TN}$$: number of True Negatives (i.e., the instances where the model correctly predicts the negative class)
- $$n_{FP}$$: number of False Positives (i.e., the instances where the model incorrectly predicts the positive class)
- $$n_{FN}$$: number of False Negatives (i.e., the instances where the model incorrectly predicts the negative class)
- $$n_{Total}$$: total number of instances

For example, if our model correctly predicts whether an email is spam or not for 90 out of 100 emails, the accuracy is 90%.

#### When Accuracy Shines

Accuracy works well when the dataset is balanced, meaning the classes are equally represented. For instance, if we have a 50-50 split between spam and non-spam emails, accuracy gives us a good sense of performance.

#### The Pitfall of Accuracy

However, accuracy can be misleading in imbalanced datasets. Imagine a medical diagnostic tool where only 1% of patients have a specific disease. If the model predicts "no disease" for everyone, it achieves 99% accuracy — yet it’s utterly useless for diagnosing the actual disease. This is where other metrics come into play.

### Precision: The Focus on Positives
Precision answers the question: "Of all the instances the model predicted as positive, how many were actually positive?":

$$\text{Precision} = \frac{n_{TP}}{n_{TP} + n_{FP}}$$

For example, in the context of spam filtering, precision measures how many emails flagged as spam are truly spam.

#### Why Precision Matters
Higher precision means lower $$n_{FP}$$, which is crucial in scenarios where false positives are costly. Think about email filters. Marking an important email as spam can lead to missed deadlines or opportunities. 

### Recall: The Hunt for True Positives
Recall (or sensitivity) answers the question: "Of all the actual positives, how many did the model correctly identify?":

$$\text{Recall} = \frac{n_{TP}}{n_{TP} + n_{FN}}$$

#### Why Recall Matters
Higher precision means lower $$n_{FN}$$, which is critical in scenarios where false negatives are unacceptable. In disease diagnosis, for instance, failing to detect a disease (a false negative) could have severe consequences for patients.

### F1-Score: The Balance Between Precision and Recall
Often, there’s a trade-off between precision and recall. The F1-score combines them into a single metric, offering a harmonic mean:

$$\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### When to Use the F1-Score
The F1-score is particularly useful when the dataset is imbalanced and both false positives and false negatives carry significant weight.
Imagine building a fraud detection system where missing fraudulent transactions (false negatives) and flagging legitimate ones (false positives) are equally problematic. The F1-score provides a balanced view of our model’s performance in such cases.

### Choosing the Right Metric
Each metric serves a specific purpose, and the choice depends on our problem:
- Use accuracy for balanced datasets.
- Focus on precision when false positives are costly (e.g., spam filters).
- Prioritize recall when false negatives are critical (e.g., disease detection).
- Opt for F1-score when we need a balance between precision and recall, especially with imbalanced datasets.

## Summary and Extension
Understanding and selecting the right metric is a cornerstone of building reliable machine learning models. Each metric—accuracy, precision, recall, and F1-score—offers unique insights into our model’s performance. By choosing wisely based on our specific application, we ensure that our model not only performs well but also meets the real-world demands of our use case.
