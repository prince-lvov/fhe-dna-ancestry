# Encrypted DNA ancestry using Concrete ML
Over 30 million people have taken DNA tests to determine their ancestry through computer genetic genealogy. By processing the digitized sequences of DNA bases, sophisticated computer algorithms can identify if one’s ancestors came from a number of ethnic groups. 

DNA is a sensitive personal data as it can identify an individual uniquely.<br>
Fully Homomorphic Encryption can make DNA ancestry identification secure for users, as it allows for the use of encrypted DNA sequences during models training and predictions. 

## DNA ancestry prediction approache
DNA ancestry identification is a complex process that involves multiple steps. First, DNA phasing assigns alleles (the As, Cs, Ts and Gs in DNA strands) to the paternal and maternal chromosomes. Second, ancestry can be determined by referencing specific segments of the DNA with large databases of DNA of known ancestry. An alternative is to use machine learning to classify each such segment and, finally, to aggregate the ancestry of each individual segment into a final classification.

## Gnomix model (Non-FHE)
[G-Nomix](https://github.com/AI-sandbox/gnomix/blob/main/README.md) - fast, scalable, and accurate local ancestry method for DNA ancestry prediction.

We chose it to be our **Non-FHE** model, because it is simple and very effective.<br>
You can read about G-Nomix advantages in [paper](https://www.biorxiv.org/content/10.1101/2021.09.19.460980v1.full#xref-ref-39-1).

## ConcreteGnomix model (FHE)
Out **FHE model** for this project has two stages: a classifier (base model) that performs an initial estimate of the ancestry probabilities within genomics windows, and a second stage consisting of another module (smoother) that learns to combine and refine these estimates, significantly increasing our accuracy. 

For the classifier (base model), we use Logistic Regression as a more accurate solution according to the benchmark.
For the smoother we use XGBoost as the most successful smoother, surpassing alternatives like linear convolutional filters and conditional random fields (CRFs).<br>
Since it is an **FHE model**, we use a Concrete ML versions of the base and smooth models.

So our **FHE model** is basically a fork of the **Gnomix** model, which we named the **ConcreteGnomix**.

## Training data
Training dataset is generated based on **query file** from 1000 genomes project.<br> 
In order to download it to **/data** directory just execute corresponding cell in **main.ipynb**

## Results
For comparison, we tried three options:
1) Gnomix model (Non-FHE)
2) ConcreteGnomix model, which uses **FHE simulation** at both stages (Sim-FHE)
3) ConcreteGnomix model, which uses **FHE** only at the first stage (Half-FHE)


|  | Non-FHE | Sim-FHE | Half-FHE |
|----------------|----------------|----------------|-----------------------|
| Accuracy   | 97.75          | 97.26          | 97.26                  |
| Inference time  |  0.912712         | 19.036097         | 826.388379                  |

### Accuracy

We got almost the same accuracy for both models.<br>
This is an expected outcome, as the models are very similar.

### Inference time comparison

Half-FHE's inference time is **three orders of magnitude greater** than Non-FHE time.<br>
Similar results were obtained in [Season 4 bounty project](https://github.com/zama-ai/bounty-and-grant-program/issues/79) submissions.

## Challenges
**Compiled Concrete models cannot be pickled**<br>
This leads to the following problems:
1) We can't use multiprocessing for Logistic Regression
2) There is no easy way to save/dump model

**Long inference/prediction time**<br>
Even in a half-FHE mode, it takes around 15 minutes (on our server) to get a prediction on one query.
Because of this:
1) We didn't get metrics for the full FHE model (we stopped prediction on one query after 1000 minutes)
2) We don't see much value in the client-server deployment approach for this project

## Conclusion
Based on the results of our work, we believe that Fully Homomorphic Encryption can make DNA ancestry identification secure for users.<br>
Usage of Concrete ML models (instead of non-FHE) does not impact accuracy. And even a relatively long prediction time does not matter much, since the user's DNA sequence processing takes days anyway. 

## Using the Repository

1) Build docker container using files from **/docker**
2) Clone this repo
3) Run **main.ipynb**
4) Change constants and/or config.yaml (if needed)

**NOTE:** Model trainnig is a very RAM intensive task. You need at least 100Gb of RAM to run main.ipynb with default params





