# Predicting Effective Argument

**Problem Statement:**

This is a Kaggle Competition hosted by Georgia State University and the Learning Agency Lab to improve the effective writing skills of students. The current education system did not put much emphasis in persuasive writing, which may hinder critical thinking development of the students. The task is to build an automated writing feedback tools.

**Objective:**

* Build a model to classify argumentative elements in student writing as “effective”, “adequate” or “ineffective”.

* Create a front-end interface for students to submit their writings for evaluations.

**Summary:**

The model was able to perform well in classifying "Effective" and "Adequate", however, the model failed to identify "Ineffective" class. Ensembling technique was able to lower the log loss and also increase the f1-score, however, it the results was not significant.

**Data:**

* Information of the data used can be found here: https://www.kaggle.com/competitions/feedback-prize-effectiveness/data

* Pretrained model used:

    1. https://huggingface.co/distilbert-base-uncased
    2. https://huggingface.co/Intel/bert-large-uncased-sparse-90-unstructured-pruneofa
    3. https://huggingface.co/microsoft/deberta-v3-base

* Finetuned pretrained model:

    1. https://huggingface.co/kitkeat/bert-large-uncased-sparse-90-unstructured-pruneofa-argumentativewriting
    2. https://huggingface.co/kitkeat/deberta-v3-base-argumentativewriting
    3. https://huggingface.co/kitkeat/distilbert-based-uncased-argumentativewriting

**Approach:**

* 3 NLP pretrained models were further finetuned using the given data.
* Ensemble technique was used by calculating a weighted average of the each predicted output to get the final results

**Metric**

* Two metrics were used to evaluate the model:

    1. F1-Score (per class & weighted average)
        * F1-Score was used to evaluate the prediction as the data is imbalance, thus accuracy will not be very representative.

    2. Cross Entropy Loss
        
        * Since this is a multiclass classification problem and neural networks were used for modelling, a loss function that is differentiable is required, thus cross entropy loss were used.

**Future Work**

1. Review hyperparameter

2. Try batch normalization to reduce loss fluctuation during training

3. Review bert-large-uncased-sparse-90-unstructured-pruneofa model to ensure correct implementation

4. Include weights in loss function to counter data imbalance


Reference:

1. https://www.kaggle.com/competitions/feedback-prize-effectiveness

