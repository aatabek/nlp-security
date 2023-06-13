# nlp-security
Enhancing Bias via Backdoor Attacks in Traditional and Transformer Text Classification Models

The robustness of NLP models has been under investigation by researchers, encompassing their susceptibility to adversarial attacks, which involve manipulating input data with the intent to trick models into producing inaccurate predictions or classifications. This article aims to unravel vulnerabilities introduced by backdoor attacks from a fairness perspective. It demonstrates how adversarial attacks can be used to target certain demographic groups or generate biased outputs, perpetuating or even amplifying the bias in the model's predictions. To test how NLP models are susceptible to backdoor attacks on amplifying bias, two main pipelines are  implemented in which NLP models are compared with baseline conventional machine learning models using the same adversarial attack strategy by injecting poisoned bias-enhancing triggers and analyzing the effect of such backdoor attack. To evaluate the success of backdoor attacks on NLP models, a newly proposed metric called Backdoor Bias Success Rate (BBSR) is introduced. This metric is used to assess the effectiveness of injecting bias-enhancing triggers. Finally, we conclude that it is important to consider both robustness and fairness in designing and evaluating NLP models to ensure ethical, secure, and effective applications.

## BERT AND ROBERTA BASED PIPELINES CODE EXPLANATION:
### Enhancing Bias via Backdoor Attacks in Traditional and Transformer Text Classification Models

The robustness of NLP models has been under investigation by researchers, encompassing their susceptibility to adversarial attacks, which involve manipulating input data with the intent to trick models into producing inaccurate predictions or classifications. This article aims to unravel vulnerabilities introduced by backdoor attacks from a fairness perspective. It demonstrates how adversarial attacks can be used to target certain demographic groups or generate biased outputs, perpetuating or even amplifying the bias in the model's predictions. To test how NLP models are susceptible to backdoor attacks on amplifying bias, two main pipelines are implemented in which NLP models are compared with baseline conventional machine learning models using the same adversarial attack strategy by injecting poisoned bias-enhancing triggers and analyzing the effect of such backdoor attack. To evaluate the success of backdoor attacks on NLP models, a newly proposed metric called Backdoor Bias Success Rate (BBSR) is introduced. This metric is used to assess the effectiveness of injecting bias-enhancing triggers. Finally, we conclude that it is important to consider both robustness and fairness in designing and evaluating NLP models to ensure ethical, secure, and effective applications.

Method: create and inject bias enhancing trigger pattern into train data

Strategy: claim the label of backdoor samples

Goal: leverage at test time by presenting trigger pattern

In this notebook, you will:

- Load the IMDB dataset
- Inject poisoned samples into training dataset
- Load BERT or RoBERTa from Hugging Face library.
- Build a simple model by combining pretrained BERT or RoBERTa with a classifier
- Train the built classifier model, fine-tuning BERT or RoBERTa as part of that
- Use the model to classify sentences

FOR SETUP, we recommend working on Google Colab since finetuning BERT and RoBERTa would require GPU. No external files, data etc. is required to run this code. We expect 60-80 mins for the entire file to execute (using V100 or A100 GPU).

CONFIG:
- model_type: (str) choose 'roberta' or 'bert'
- dataset: (str) choose 'imdb' or 'rotten_tomatoes'
- backdoor_type: (str) choose 'word' or 'sentence' or None for healthy
- poison_rate_proxy: (float) choose 0.2 or 0.6 (called proxy since half of it would be equal to overall poison rate)

REFERENCES:
- https://huggingface.co/roberta-base
- https://huggingface.co/bert-base-cased
- https://huggingface.co/docs/transformers/training
- https://huggingface.co/datasets/imdb

