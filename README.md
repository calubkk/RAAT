<div align="center">
<h1>Enhancing Noise Robustness of Retrieval-Augmented Language Models
with Adaptive Adversarial Training</h1>

arXiv: [Abstract](https://arxiv.org/abs/2405.20978) / [PDF](https://arxiv.org/pdf/2405.20978)

</div>

## 📣 News
- **[16/May/2024]** 🎉 Our paper is accepted by **ACL 2024 Main Conference(The 62nd Annual Meeting of the Association for Computational Linguistics)**!

## ✨ Abstract
Large Language Models (LLMs) exhibit substantial capabilities yet encounter challenges, including hallucination, outdated knowledge, and untraceable reasoning processes. Retrieval-augmented generation (RAG) has emerged as a promising solution, integrating knowledge from external databases to mitigate these challenges. However, inappropriate retrieved passages can potentially hinder the LLMs' capacity to generate comprehensive and high-quality responses. Prior RAG studies on the robustness of retrieval noises often confine themselves to a limited set of noise types, deviating from real-world retrieval environments and limiting practical applicability. In this study, we initially investigate retrieval noises and categorize them into three distinct types, reflecting real-world environments. We analyze the impact of these various retrieval noises on the robustness of LLMs. Subsequently, we propose a novel RAG approach known as Retrieval-augmented Adaptive Adversarial Training (RAAT). RAAT leverages adaptive adversarial training to dynamically adjust the model's training process in response to retrieval noises. Concurrently, it employs multi-task learning to ensure the model's capacity to internally recognize noisy contexts. Extensive experiments demonstrate that the LLaMA-2 7B model trained using RAAT exhibits significant improvements in F1 and EM scores under diverse noise conditions.

## ✨ The overview of RAAT
<div align="center"><img src="resources\overview.jpg" style="zoom:100%"></div>

## 💪 Dataset
### Data Preparation
We provide the RAG-Bench for training and testing, available at 
https://drive.google.com/file/d/1i4umieNgG3dctNqdTMI3Rj5tsrR5JvnM/view?usp=sharing

#### retrieval_robustness_benchmark
- `train.json`
- `dev.json`
- `test.json`


1. train_data: 4500 samples.  

   best_ctx: golden retrieval

   ad_ctx: [relevant retrieval noise,counterfactual retrieval noise,irrelevant retrieval noise] 

2. dev_data: 300 samples.

3. test_data: 3000 samples.

   Golden retrieval : choose from "best_ctx".

   Relevant retrieval noise : choose from "ctxs".(without answers)

   Irrelevant retrieval noise: choose from other samples.

   Counterfactual retrieval noise: choose from "counter_fac".

> The test data used in the paper: ```RAAT\benchmark_cache```.

   golden retrieval：```o.json```

   relevant retrieval noise：```p.json```

   irrelevant retrieval noise：```c.json```

   counterfactual retrieval noise：```f.json```

> The training data used in the paper:```RAAT\tuner\data\temp.json```.

You download temp.json with the following link: https://drive.google.com/file/d/109CVe8KWiYdpZLkz4nZjDZklYdUjxaZ2/view?usp=sharing

> What is the difference between the training and test data we used in the paper and RAG-Bench?

The training and test data we used in the paper are subsets of RAG-Bench because RAG-Bench provides multiple noise retrieval samples for different retrieval noises. However, in testing or training, we only need to use one noise retrieval sample for each type of retrieval noise. To control the randomness brought by the selection of retrieval samples on the results, we cache the selected test and training data. If you want to reproduce the results in the paper, it is best to use our selected data cache.
  

## 💪 Usage
### Train
We provide the training scripts for training the model. For example, you can run the following commands to train the model:
```
cd RAAT
pip install -r requirements.txt
mkdir checkpoints
mkdir logs
cp -r path_to_retrieval_robustness_benchmark  ./data/
cp path_to_temp ./data/
cd scripts
bash train.sh
```
The scripts can be easily modified to train LLMs with different datasets. 
> **Note:** Before running, the ```model_name_or_path```  has to be specified. Additionally, please download RAG-Bench and temp.json

### Test
The following command can be used to test the model:
```
cd RAAT
cd scripts
bash test.sh
```
> **Note:** Before running, the ```test_model_name_or_path```  has to be specified.

## 🔓 Citation
If this work is helpful to you, welcome to cite our paper as:
```
@article{fang2024enhancing,
  title={Enhancing Noise Robustness of Retrieval-Augmented Language Models with Adaptive Adversarial Training},
  author={Fang, Feiteng and Bai, Yuelin and Ni, Shiwen and Yang, Min and Chen, Xiaojun and Xu, Ruifeng},
  journal={arXiv preprint arXiv:2405.20978},
  year={2024}
}
```
