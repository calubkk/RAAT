<div align="center">
<h1>Enhancing Noise Robustness of Retrieval-Augmented Language Models
with Adaptive Adversarial Training</h1>

arXiv: [Abstract](https://arxiv.org/abs/2405.20978) / [PDF](https://arxiv.org/pdf/2405.20978)

</div>

# 📣 News
- **[16/May/2024]** 🎉 Our paper is accepted by **ACL 2024 Main Conference(The 62nd Annual Meeting of the Association for Computational Linguistics)**!

# ✨ Abstract
Large Language Models (LLMs) exhibit substantial capabilities yet encounter challenges, including hallucination, outdated knowledge, and untraceable reasoning processes. Retrieval-augmented generation (RAG) has emerged as a promising solution, integrating knowledge from external databases to mitigate these challenges. However, inappropriate retrieved passages can potentially hinder the LLMs' capacity to generate comprehensive and high-quality responses. Prior RAG studies on the robustness of retrieval noises often confine themselves to a limited set of noise types, deviating from real-world retrieval environments and limiting practical applicability. In this study, we initially investigate retrieval noises and categorize them into three distinct types, reflecting real-world environments. We analyze the impact of these various retrieval noises on the robustness of LLMs. Subsequently, we propose a novel RAG approach known as Retrieval-augmented Adaptive Adversarial Training (RAAT). RAAT leverages adaptive adversarial training to dynamically adjust the model's training process in response to retrieval noises. Concurrently, it employs multi-task learning to ensure the model's capacity to internally recognize noisy contexts. Extensive experiments demonstrate that the LLaMA-2 7B model trained using RAAT exhibits significant improvements in F1 and EM scores under diverse noise conditions.

# ✨ The overview of RAAT
<div align="center"><img src="resources\overview.jpg" style="zoom:100%"></div>

# 💪 Dataset
## Data Preparation
We provide the RAG-Bench for training and testing, available at 
[https://drive.google.com/file/d/1u1XNg2Hk0vE8kJkogwXNNjbcDiiY50e1/view?usp=sharing]

### retrieval_robustness_benchmark(RAG-Bench)
- `train.json`
- `dev.json`
- `test.json`

1. train_data: 4500 samples.
2. dev_data: 300 samples.
3. test_data: 3000 samples.

__Description of keys in training data__ : 
* _"best_ctx"_ means golden retrieval. 
* _"ad_ctx"_ includes three types of retrieval noises, it is a list object, and the three types of noise in the list correspond to the following arrangement:[relevant retrieval noise,counterfactual retrieval noise,irrelevant retrieval noise].

__How we use test_data to construct different retrieval contexts__ : 

* Golden retrieval : choose from "best_ctx".

* Relevant retrieval noise : choose from "ctxs"(without answer entity).

* Irrelevant retrieval noise: choose from other quires' retrieval context.

* Counterfactual retrieval noise: choose from "counter_fac".

__Description of benchmark_cache__: 

* ```o.json```:golden retrieval
* ```p.json```: golden retrieval + relevant retrieval noise
* ```c.json```:golden retrieval + irrelevant retrieval noise
* ```f.json```:golden retrieval + counterfactual retrieval noise

> The testing data used in the paper(cache): ```RAAT\benchmark_cache```.

> The training data used in the paper(cache):```RAAT\tuner\data\temp.json```.

You can download temp.json with the following link: https://drive.google.com/file/d/109CVe8KWiYdpZLkz4nZjDZklYdUjxaZ2/view?usp=sharing

> What is the difference between the training(or testing) data we used in the paper(cache) and RAG-Bench?

The training and test data utilized in our paper are subsets of RAG-Bench, as this dataset provides multiple noisy retrieval contexts for each query. In RAG-Bench, each type of retrieval noise may be associated with more than one context. However, for both testing and training purposes, it is not necessary to use all available contexts. To mitigate the randomness introduced by the selection of retrieval contexts and to ensure reproducibility of our results, we have cached the specific test and training data that we selected. If you wish to reproduce the results presented in our paper, it is recommended to use our cached data selection.

> Regarding the classification of retrieved passages as golden retrieval context and noisy retrieval context (relevant retrieval noise,counterfactual retrieval noise,irrelevant retrieval noise), is this labeling done by the language model (LLM) or manually annotated?

It is manually annotated. Golden retrieval is required to be text that is somewhat related to the query and contains the answer entity (determined by regular matching). Relevant Retrieval Noise is required to be text that is highly related to the query but does not contain the answer entity. Irrelevant Retrieval Noise is required to be text that is completely unrelated to the query; we directly utilize retrieval contexts from other queries for this. Counterfactual Noise is a variant of Golden retrieval, where we change the answer entity in the Golden retrieval to a counterfactual answer entity (this counterfactual answer entity is constructed by ChatGPT based on the correct answer entity). You don't need to worry about Counterfactual Noise being too similar to Golden retrieval because we ensure that each query in the dataset has at least two Golden retrievals.
  

# 💪 Usage
## Train
We provide the training scripts for training the model. For example, you can run the following commands to train the model:
```
cd RAAT
pip install -r requirements.txt
mkdir checkpoints
mkdir logs
cp -r path_to_retrieval_robustness_benchmark  ./tuner/data/
cp path_to_temp ./tuner/data/
cd scripts
bash train.sh
```
The scripts can be easily modified to train LLMs with different datasets. 
> **Note:** Before running, the ```model_name_or_path```  has to be specified. Additionally, please download RAG-Bench and temp.json

## Test
The following command can be used to test the model:
```
cd RAAT
cd scripts
bash test.sh
```
> **Note:** Before running, the ```test_model_name_or_path```  has to be specified.

# 🔓 Citation
If this work is helpful to you, welcome to cite our paper as:
```
@article{fang2024enhancing,
  title={Enhancing Noise Robustness of Retrieval-Augmented Language Models with Adaptive Adversarial Training},
  author={Fang, Feiteng and Bai, Yuelin and Ni, Shiwen and Yang, Min and Chen, Xiaojun and Xu, Ruifeng},
  journal={arXiv preprint arXiv:2405.20978},
  year={2024}
}
```
