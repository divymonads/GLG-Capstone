## GLG Topic Modelling

### Gerson Lehrman Group sponsored project on NLP

Team Members: Surnjani Djoko, Divy Gandhi, Prithvi Nuthanakalva

Description:
Given an article of text, this application will return a distribution of probabilities of what topic it could be and a subject matter expert associated with the topic.

It also does metadata tagging through Named Entity Recognition.

Essentially, this is a light weight UI that fronts a fine-tuned BERT model for NER and a custom LDA model that was trained on 2.2 million articles.

Basic Components:
- Web app in streamlit for UI
- Flask container for LDA model (LDA Multicore using gensim)
- Flask container for NER model (Fine-tuned BERT model)

Both of the model containers were deployed to Sagemaker.


![Image of AWS architecture using Sagemaker and Elastic Beanstalk](./architecture.png)


References:
https://ai.stanford.edu/~ang/papers/nips01-lda.pdf
Text Analytics with Python: A Practitioner’s Guide to Natural Language Processing by Dipanjan Sarkar
Blueprints for Text Analytics Using Python by Jens Albrecht, sidharth Ramachandran and Christian Winkler
Practical Natural Language Processing by Sowmya Vajjala, Bodhisattwa Majumder, Anuj Gupta, Harshit Surana
https://github.com/philschmid/huggingface-sagemaker-workshop-series
https://github.com/aws/amazon-sagemaker-examples/tree/main/advanced_functionality/scikit_bring_your_own/container
https://towardsdatascience.com/how-to-deploy-a-semantic-search-engine-with-streamlit-and-docker-on-aws-elastic-beanstalk-42ddce0422f3
