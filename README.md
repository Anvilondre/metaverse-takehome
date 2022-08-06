# Take-home assignment for Metaverse Mind 

**!! Note:** I don't think someone will actually clone this repo, so didn't include the requirements.txt, but the project mainly relies on sklearn, torch, transformers and nltk.

## Overall time spent
1 evening + a bit in the morning, I'd estimate to 4-5h, which is more than was expected in the test task. I'd say fit-predicts were like 1h tops for me, the rest was data exploration and iteration. I didn't go further with the data drift thingy exactly because I've noticed my time spent was much greater than requested.

## General notes

I tried to approach this problem data-first, and couldn't find anything crucial in the data provided. No class imbalance, doesn't look like there's a data drift, or any outliers. Only several things caught my eye:
- Dataset contains multiple languages (at least Russian, French and English), and I tried to deal with that
- Valid/Test accuracy is too different. I tried plotting the TSNE of document's TF-IDfs, but didn't look like any significant data drift is present. I believe more investigation should be done in this direction
- Fake news seem to be longer
- All texts are **very** long, up to 20k characters
- Title actually turned out to be a **much** better predictor

## Files and solution

### EDA.html
Even though it was discouraged, I did the EDA in Jupyter, because that's the most easy and natural way for me. You don't need to look inside, but if you will - I've included the HTML conversion for your convenience

### DataDrift.html
Same as the previous. My attempt quickly see if there are any noticeable data drifts between the train and test data. That's just `simple cleaning -> sklearn Tf-IDf -> sklearn TSNE` 

### BERT_Multilingual.py
When I encountered a huge vocabulary and different languages, the first thing that came to my mind was to just train a Multilingual BERT on raw text. Easy and fast baseline, that normally works well.  
However, it failed miserably in comparison to the Tf-IDf approach, which can be easily interpreted with texts by the length of articles. BERT would trim text to 512 tokens, which doesn't seem to be enough here. With the titles I'd explain it as just keywords mean a lot, and we don't need more. 

**!! Note:** this script was trained on a machine from a computing cluster, with 2 RTX6000 GPUs in multiprocessing. I'm attaching `slurm_job.sh` I used just for the sake of reproducibility.

## LogReg.py
That's my second model. Basically `simple cleaning -> sklearn Tf-IDf -> filter out very frequent and infrequent words -> sklearn Logistic Regression`.

## Results
Files with Accuracy, F1 and Confusion Matrix:
- results_bert_text.json contains metrics for BERT model trained on texts
- results_logreg_text.json contains results for LogisticRegression model trained on texts
- results_bert_title.json contains metrics for BERT model trained on titles
- results_logreg_title.json contains results for LogisticRegression model trained on titles

|              | Accuracy | F1    |
|--------------|----------|-------|
| BERT_Text    | 0.632    | 0.647 |
| LogReg_Text  | 0.644    | 0.652 |
| BERT_Title   | 0.637    | 0.653 |
|**LogReg_Title** | **0.703**    | **0.740** |

## Potential improvements
- Explore the train/test drift and find why metrics are so different
- Deal with large texts
  - Use something like LongFormer (still not necessarily enough) for texts
  - Compute the final class by voting of BERT-classified subsets, or by embedding every text subset and passing to another network   
- Utilize both Text, Author (e.g. graph embeddings for authors), and Title, as it may provide a **vast** insight into the article, and it's credibility
  - Just do joint embeddings
- Try simpler models with feature engineering like:
    - Percentage of fakes posted by the specific author
    - Number of words
    - Tf-IDf separated by label (semantically that would be to see text place in separate latent spaces of fakes and non-fakes)
    - Sentiment
    - Some topic features
    
![](http://www.quickmeme.com/img/e5/e52e67332c7592c3f88d26dc641168d50297b1d850013b766c92da39dc935901.jpg)