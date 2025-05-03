<!-- 标题部分 -->
<div style="width: 100%; height: 100px; text-align: center; background-color: #f4f4f4; padding: 20px 0;">
   <h1 style="font-size: 50px; font-weight: bold; color: black; line-height: 100px;">
       CroFinben: Multilingual Benchmarking of LLMs for Cross Mainstream and Low-resource Finance
   </h1>
</div>
   Welcome to here, let's get to know CroFinben together. </br>
   we introduce CroFinBen, the first multilingual benchmark specifically designed to bridge the language-resource gap between mainstream and low-resource finance. It includes 4 key financial tasks (FinSA, FinSP, FinTS, and FinTC) in both high-resource languages (English and Chinese) and low-resource SEA languages (Indonesian, Malaysian, Thai, Filipino, and Vietnamese), comprising 50k samples from 16 datasets to ensure a comprehensive and balanced evaluation. It enhances the fairness and robustness of FinLLMs, providing strong support for improving performance  in global financial scenarios.
<!-- 作者部分 -->

   
<!--<h1 align="left">Main Contributors</h1>

<div align="left" style="margin-top: 40px;">

<div style="font-size: 18px; margin-bottom: 20px;margin-top:20px">
   <img src="https://github.com/qqgzi/SeaFBen/blob/master/asset/%E4%B8%93%E5%AE%B6%E6%95%99%E6%8E%88.svg?raw=true" 
   alt="专家" style="float: left; width: 25px; height: 25px; margin-right: 10px;">
    <a target='_blank' style="color: #2980B9; text-decoration: none; font-weight: bold;">Gang Hu</a><br />
    <span style="font-size: 16px; color: #555;">Cross-lingual Intelligent Information Processing, Yunnan University</span>
</div>
    

<div style="font-size: 18px; margin-bottom: 20px;margin-top:20px">
   <img src="https://github.com/qqgzi/SeaFBen/blob/master/asset/%E5%AD%A6%E7%94%9F.svg?raw=true" 
         alt="学生" style="float: left; width: 25px; height: 25px; margin-right: 10px;">
   <a target='_blank' style="color: #2980B9; text-decoration: none; font-weight: bold;">Siqi Lv</a><br />
   <span style="font-size: 16px; color: #555;">Graduate Student at the School of Information, Yunnan University</span>
</div>

<div style="font-size: 18px; margin-bottom: 20px;margin-top:20px">
  <img src="https://github.com/qqgzi/SeaFBen/blob/master/asset/%E5%AD%A6%E7%94%9F.svg?raw=true" 
         alt="学生" style="float: left; width: 25px; height: 25px; margin-right: 10px;">
  <a target='_blank' style="color: #2980B9; text-decoration: none; font-weight: bold;">Kang Wang</a><br />
   <span style="font-size: 16px; color: #555;">Graduate Student at the School of Information, Yunnan University</span>
 </div>   
  
<div style="font-size: 18px; margin-bottom: 20px;margin-top:20px">
   
   <img src="https://github.com/qqgzi/SeaFBen/blob/master/asset/%E5%AD%A6%E7%94%9F.svg?raw=true" 
         alt="学生" style="float: left; width: 25px; height: 25px; margin-right: 10px;">
    <a target='_blank' style="color: #2980B9; text-decoration: none; font-weight: bold;">Ke Qin</a><br />
    <span style="font-size: 16px; color: #555;">Graduate Student at the School of Information, Yunnan University</span>
    
</div>
-->
<!-- 底部图片部分
<div align="center" style="margin-top: 40px;">
    <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbx3AQWiMhxwOvFb7r1PH-h_i5-b3H9xsGVKnkQwbFlA&s' alt='Yunnan University' height='100px' style="margin-right: 30px;">
    <img src='https://i.postimg.cc/CLtkBwz7/57-EDDD9-FB0-DF712-F3-AB627163-C2-1-EF15655-13-FCA.png' alt='Wuhan University Logo' height='100px'>
</div>
 -->

-----------------

<!-- ![](https://img.shields.io/badge/pixiu-v0.1-gold) -->
![](https://black.readthedocs.io/en/stable/_static/license.svg)
[![Discord](https://img.shields.io/discord/1146837080798933112)](https://discord.gg/HRWpUmKB)
<!-- 
[Pixiu Paper](https://arxiv.org/abs/2306.05443) | [FinBen Leaderboard](https://huggingface.co/spaces/finosfoundation/Open-Financial-LLM-Leaderboard)
 -->



**Languages**

- high-resource languages: - [EN] - [ZH]
- low-rescoure languages: -[Tha] - [Ind]  - [Vie]  - [May]  - [Fil]

**Evaluations**:


FinSA, FinSP, FinTS, and FinTC

> Financial Sentiment Analysis (FinSA):
- [FE (ZH)]
- [FPB (EN)]
- [FinSent (Ind)]
- [MayFPB (May)]

> Financial Stock Prediction (FinSP):
- [StockA (ZH)]
- [ACL18 (EN)]
- [IndCIKM18 (Ind)]
- [FilACL18 (Fil)]

> Financial Text Classification (FinTC):
- [NA (ZH)]
- [EDTSUM (EN)]
- [URLSum (Ind)](https://huggingface.co/datasets/SeaFinAI/URLSum)
- [ThaNA (Tha)]

> Financial Text Classification (FinTC):

- [NL (ZH)]
- [Headlines (EN)]
- [AppRews (Ind)](https://huggingface.co/datasets/SeaFinAI/AppRevs)
- [VieNL (Vie)]


### Key Breakthroughs


- **Breaking high- and low-resource cross-lingual barrier**: CroFinBen is the first benchmark supporting cross-mainstream and low-resource finance, particularly SEA languages (Ind, Tha, Fil, Vie, and May), filling the gap in low-resource financial tasks.
- **Highly localized annotation for low-resource dataset**: Unlike other benchmarks that rely on direct translations, CroFinBen works with SEA domain experts to refine financial terms, removing finance- and culture-specific elements to better meet local needs. 
- **Fairness in high- and low-resource task consistency**: Ensures fair evaluation via consistent NLP main tasks across both high- and low-resource languages, eliminating biases that arise from differing task complexities in existing methods.
- **Comprehensive evaluation of multi-language tasks**: Conducts an in-depth evaluation of 23 representative general, SEA LLMs and FinLLMs, revealing their strengths and limitations in adapting to both high- and low-resource financial environments.

---

## CroFinben Evalution Benchmark result: The evaluation results of 23 representative large models on CroFinben.


### Tasks


| FinTask | Language | Dataset   | Specific task           | Evaluation | Evaluation                                       | Source                    | types          | Textual Hierarchy | License      | NLP Task             |
|---------|----------|-----------|-------------------------|------------|--------------------------------------------------|---------------------------|----------------|-------------------|--------------|----------------------|
| FinSA     | ZH       | FE        | sentiment analysis      | 2020       | F1, Accuracy, Macro F1                           | social texts              | raw collection | Sentence          | Public       | Classification (CLS) |
|         | EN       | FPB       | sentiment analysis      | 970        | F1, Accuracy, Macro F1                           | economic news             | raw collection | Sentence          | CC BY-SA 3.0 |                      |
|         | Ind      | FinSent   | sentiment analysis      | 2000       | F1, Accuracy, Macro F1                           | News Headlines            | raw collection | Sentence          | Apache-2.0   |                      |
|         | May      | MayFPB    | sentiment analysis      | 970        | F1, Accuracy, Macro F1                           | Economic News             | expert review  | Paragraph         | MIT License  |                      |
| FinSP      | ZH       | StockA    | stock prediction        | 1477       | F1, Accuracy, Macro F1                           | news,historical prices    | raw collection | Paragraph         | Public       | Reasoning (REA)      |
|         | EN       | ACL18     | stock prediction        | 3720       | F1, Accuracy, Macro F1                           | tweets, historical prices | raw collection | Paragraph         | MIT License  |                      |
|         | Ind      | IndCIKM18 | stock prediction        | 1139       | F1, Accuracy, Macro F1                           | tweets Stock Prices       | expert review  | Paragraph         | Public       |                      |
|         | Fil      | FilACL18  | stock prediction        | 2000       | F1, Accuracy, Macro F1                           | Tweets, Stock prices      | expert review  | Paragraph         | MIT License  |                      |
| FinTS     | ZH       | NA        | text summarization      | 3600       | Rouge-1, Rouge-2, Rouge-L,  BERTScore, BARTScore | news, announcements       | raw collection | Paragraph         | Public       | Generation (GEN)     |
|         | EN       | EDTSUM    | text summarization      | 2000       | Rouge-1, Rouge-2, Rouge-L,  BERTScore, BARTScore | news articles             | raw collection | Paragraph         | Public       |                      |
|         | Ind      | URLSum    | url summarization       | 2834       | Rouge-1, Rouge-2, Rouge-L,  BERTScore, BARTScore | Indonesian News URLs      | raw collection | Paragraph         | Public       |                      |
|         | Tha      | ThaNA     | text summarization      | 2000       | Rouge-1, Rouge-2, Rouge-L,  BERTScore, BARTScore | news, announcements       | expert review  | Paragraph         | Public       |                      |
| FinTC     | ZH       | NL        | news classification     | 884        | F1, Accuracy, Macro F1                           | news articles             | raw collection | Paragraph         | Public       | Classification (CLS) |
|         | EN       | Headlines | headline classification | 20547      | Avg F1                                           | news headlines            | raw collection | Sentence          | CC BY-SA 3.0 |                      |
|         | Ind      | AppRevs   | financial review        | 1999       | F1, Accuracy, Macro F1                           | Mandiri App Reviews       | raw collection | Sentence          | CC BY-NC 4.0 |                      |
|         | Vie      | VieNL     | news classification     | 2000       | F1, Accuracy, Macro F1                           | news articles             | expert review  | Paragraph         | Public       |                      |
|         |          |           |                         |            |                                                  |                           |                |                   |              |                      |








| Model        |Finanical Task |        |        |        | NLP Task |        |        | language |        |        | overall |
|--------------|----------------|--------|--------|--------|----------|--------|--------|----------|--------|--------|---------|
|              | FinSA            | FinSP     | FinTS    | FinTC      | CLS      | GEN    | REA    | EN       | ZH     | SEA    |         |
| LLaMa2       | 0.262          | 0.333  | 0.219  | 0.322  | 0.292    | 0.219  | 0.333  | 0.357    | 0.195  | 0.299  | 0.284   |
| LLaMa2-Chat  | 0.356          | 0.365  | 0.236  | 0.314  | 0.335    | 0.236  | 0.365  | 0.413    | 0.269  | 0.271  | 0.318   |
| Llama3       | 0.276          | 0.298  | 0.237  | 0.314  | 0.295    | 0.237  | 0.298  | 0.410    | 0.144  | 0.290  | 0.281   |
| Gemma        | 0.328          | 0.332  | 0.248  | 0.377  | 0.352    | 0.248  | 0.332  | 0.432    | 0.224  | 0.308  | 0.321   |
| BLOOM        | 0.284          | 0.324  | 0.162  | 0.335  | 0.310    | 0.162  | 0.324  | 0.376    | 0.309  | 0.144  | 0.276   |
| ChatGPT(3.5) | 0.625          | 0.400  | 0.245  | 0.586  | 0.605    | 0.245  | 0.400  | 0.538    | 0.391  | 0.463  | 0.464   |
| Qwen         | 0.485          | 0.370  | 0.193  | 0.359  | 0.422    | 0.193  | 0.370  | 0.479    | 0.274  | 0.301  | 0.352   |
| Qwen2        | 0.532          | 0.431  | 0.332  | 0.318  | 0.425    | 0.332  | 0.431  | 0.409    | 0.349  | 0.452  | 0.403   |
| Baichuan     | 0.248          | 0.297  | 0.145  | 0.274  | 0.261    | 0.145  | 0.297  | 0.355    | 0.177  | 0.191  | 0.241   |
| Baichuan2    | 0.050          | 0.270  | 0.144  | 0.219  | 0.134    | 0.144  | 0.270  | 0.275    | 0.033  | 0.204  | 0.171   |
| ChatGLM2     | 0.000          | 0.168  | 0.116  | 0.201  | 0.101    | 0.116  | 0.168  | 0.252    | 0.032  | 0.081  | 0.121   |
| ChatGLM3     | 0.482          | 0.372  | 0.269  | 0.397  | 0.440    | 0.269  | 0.372  | 0.537    | 0.376  | 0.227  | 0.380   |
| DeepSeek     | 0.333          | 0.303  | 0.245  | 0.348  | 0.340    | 0.245  | 0.303  | 0.420    | 0.226  | 0.275  | 0.307   |
| Internlm     | 0.304          | 0.313  | 0.218  | 0.324  | 0.314    | 0.218  | 0.313  | 0.393    | 0.274  | 0.202  | 0.290   |
| SeaLLM-v2    | 0.460          | 0.429  | 0.333  | 0.446  | 0.453    | 0.333  | 0.429  | 0.443    | 0.366  | 0.442  | 0.417   |
| SeaLLM-v2.5  | 0.538          | 0.422  | 0.339  | 0.440  | 0.489    | 0.339  | 0.422  | 0.447    | 0.396  | 0.461  | 0.435   |
| SeaLLM-v3    | 0.552          | 0.426  | 0.326  | 0.343  | 0.447    | 0.326  | 0.426  | 0.408    | 0.369  | 0.458  | 0.412   |
| Polylm       | 0.000          | 0.236  | 0.038  | 0.200  | 0.100    | 0.038  | 0.236  | 0.255    | 0.012  | 0.088  | 0.118   |
| Typhoon      | 0.335          | 0.382  | 0.202  | 0.309  | 0.322    | 0.202  | 0.382  | 0.396    | 0.251  | 0.274  | 0.307   |
| PhoGPT       | 0.082          | 0.258  | 0.206  | 0.265  | 0.174    | 0.206  | 0.258  | 0.296    | 0.135  | 0.178  | 0.203   |
| FinMA        | 0.467          | 0.267  | 0.211  | 0.279  | 0.373    | 0.211  | 0.267  | 0.406    | 0.227  | 0.284  | 0.306   |
| CFGPT        | 0.202          | 0.254  | 0.152  | 0.236  | 0.219    | 0.152  | 0.254  | 0.338    | 0.152  | 0.143  | 0.211   |
| DISC-FinLLM  | 0.559          | 0.409  | 0.260  | 0.308  | 0.433    | 0.260  | 0.409  | 0.504    | 0.310  | 0.338  | 0.384   |
|              |                |        |        |        |          |        |        |          |        |        |         |



### Evaluation

#### Preparation

##### Locally install
```bash

cd SeaFBen
pip install -r requirements.txt
cd src/financial-evaluation
pip install -e .[multilingual]
```




<!--
#### Automated Task Assessment
Before evaluation, please download [BART checkpoint](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) to `src/metrics/BARTScore/bart_score.pth`.

 For automated evaluation, please follow these instructions:

1. Huggingface Transformer

   To evaluate a model hosted on the HuggingFace Hub, use this command:

```bash
python eval.py \
    --model "hf-causal-llama" \
    --model_args "use_accelerate=True,pretrained=PoLylm-13B,tokenizer=PoLylm-13B,use_fast=False" \
    --tasks "CroFinben_NL"
```

More details can be found in the [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) documentation.
-->
1. Commercial APIs


Please note, for tasks such as NA, the automated evaluation is based on a specific pattern. This might fail to extract relevant information in zero-shot settings, resulting in relatively lower performance compared to previous human-annotated results.

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python eval.py \
    --model chatgpt \
    --tasks CroFinben_NA
```

2. Self-Hosted Evaluation

To run inference backend:

```bash
bash scripts/run_interface.sh
```

Please adjust run_interface.sh according to your environment requirements.

To evaluate:

```bash
python data/*/evaluate.py
```

### Create new tasks

Creating a new task for CroFinben involves creating a Huggingface dataset and implementing the task in a Python file. This guide walks you through each step of setting up a new task using the CroFinben framework.

#### Creating your dataset in Huggingface

Your dataset should be created in the following format:

```python
{
    "query": "...",
    "answer": "...",
    "text": "..."
}
```

In this format:

- `query`: Combination of your prompt and text
- `answer`: Your label
<!--
For **Multi-turn** tasks (such as )

For **Classification** tasks , additional keys should be defined:

- `choices`: Set of labels
- `gold`: Index of the correct label in choices (Start from 0)



For **abstractive Summarization** tasks (such as [EDTSUM (FinBen_edtsum)](https://huggingface.co/datasets/TheFinAI/flare-edtsum)), no additional keys should be defined
#### Implementing the task

Once your dataset is ready, you can start implementing your task. Your task should be defined within a new class in flare.py or any other Python file located within the tasks directory.

To cater to a range of tasks, we offer several specialized base classes.

For instance, if you are embarking on a classification task, you can directly leverage our `IT` base class. This class allows for efficient and intuitive task creation. To better demonstrate this, let's delve into an example of crafting a task named FPB using the `Classification` base class:

```python
class Sea(Classification):
    DATASET_PATH = "flare-fpb"
```

And that's it! Once you've created your task class, the next step is to register it in the `src/tasks/__init__.py` file. To do this, add a new line following the format `"task_name": module.ClassName`. Here is how it's done:

```python
TASK_REGISTRY = {
    "flare_fpb": flare.FPB,
    "your_new_task": your_module.YourTask,  # This is where you add your task
}
```

-->
#### Predefined task metrics

| Task                                     | Metric                                 | Illustration                                                 |
| ---------------------------------------- | -------------------------------------- | ------------------------------------------------------------ |
| Classification                           | Accuracy                               | This metric represents the ratio of correctly predicted observations to total observations. It is calculated as (True Positives + True Negatives) / Total Observations. |
| Classification                           | F1 Score                               | The F1 Score represents the harmonic mean of precision and recall, thereby creating an equilibrium between these two factors. It proves particularly useful in scenarios where one factor bears more significance than the other. The score ranges from 0 to 1, with 1 signifying perfect precision and recall, and 0 indicating the worst case. Furthermore, we provide both 'weighted' and 'macro' versions of the F1 score. |
| Classification                           | Macro F1 |The Macro F1 is a metric that evaluates the overall performance of a multi-class classification model by computing the unweighted average of the F1 scores for each class. It provides a balanced measure of precision and recall across all classes, with a score ranging from 0 to 1. A score of 1 indicates perfect precision and recall for all classes, while a score of 0 reflects poor performance, with the model failing to correctly identify any class. |
| Extractive and Abstractive Summarization | Rouge-N                                | This measures the overlap of N-grams (a contiguous sequence of N items from a given sample of text) between the system-generated summary and the reference summary. 'N' can be 1, 2, or more, with ROUGE-1 and ROUGE-2 being commonly used to assess unigram and bigram overlaps respectively. |
| Extractive and Abstractive Summarization | Rouge-L                                | This metric evaluates the longest common subsequence (LCS) between the system and the reference summaries. LCS takes into account sentence level structure similarity naturally and identifies longest co-occurring in-sequence n-grams automatically. |
| Extractive and Abstractive Summarization | BERTScore | BERTScore measures the semantic similarity between the system-generated summary and the reference summary by computing contextualized embeddings with BERT. It evaluates precision, recall, and F1 score based on token similarity, providing a nuanced assessment of semantic content beyond exact match. |
| Extractive and Abstractive Summarization | BARTScore | BARTScore assesses the quality of generated summaries by estimating the likelihood of the candidate given the reference (or vice versa) using a pretrained BART model. It captures fluency and relevance through probability scores, effectively evaluating the coherence and informativeness of the summaries. |

>  Additionally, you can determine if the labels should be lowercased during the matching process by specifying `LOWER_CASE` in your class definition. This is pertinent since labels are matched based on their appearance in the generated output. For tasks like examinations where the labels are a specific set of capitalized letters such as 'A', 'B', 'C', this should typically be set to False.

---
<!--
## FIT: Financial Instruction Dataset

Our instruction dataset is uniquely tailored for the domain-specific LLM, FinMA. This dataset has been meticulously assembled to fine-tune our model on a diverse range of financial tasks. It features publicly available multi-task and multi-modal data derived from the multiple open released financial datasets.

The dataset is multi-faceted, featuring tasks including sentiment analysis, news headline classification, named entity recognition, question answering, and stock movement prediction. It covers both textual and time-series data modalities, offering a rich variety of financial data. The task specific instruction prompts for each task have been carefully degined by domain experts.

### Modality and Prompts

The table below summarizes the different tasks, their corresponding modalities, text types, and examples of the instructions used for each task:

| **Task**                     | **Modalities**    | **Text Types**        | **Instructions Examples**                                    |
| ---------------------------- | ----------------- | --------------------- | ------------------------------------------------------------ |
| Sentiment Analysis           | Text              | news headlines,tweets | "Analyze the sentiment of this statement extracted from a financial news article.Provide your answer as either negative, positive or neutral. For instance, 'The company's stocks plummeted following the scandal.' would be classified as negative." |
| News Headline Classification | Text              | News Headlines        | "Consider whether the headline mentions the price of gold. Is there a Price or Not in the gold commodity market indicated in the news headline? Please answer Yes or No." |
| Named Entity Recognition     | Text              | financial agreements  | "In the sentences extracted from financial agreements in U.S. SEC filings, identify the named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required answer format is: 'entity name, entity type'. For instance, in 'Elon Musk, CEO of SpaceX, announced the launch from Cape Canaveral.', the entities would be: 'Elon Musk, PER; SpaceX, ORG; Cape Canaveral, LOC'" |
| Question Answering           | Text              | earnings reports      | "In the context of this series of interconnected finance-related queries and the additional information provided by the pretext, table data, and post text from a company's financial filings, please provide a response to the final question. This may require extracting information from the context and performing mathematical calculations. Please take into account the information provided in the preceding questions and their answers when formulating your response:" |
| Stock Movement Prediction    | Text, Time-Series | tweets, Stock Prices  | "Analyze the information and social media posts to determine if the closing price of *\{tid\}* will ascend or descend at *\{point\}*. Please respond with either Rise or Fall." |

### Dataset Statistics

The dataset contains a vast amount of instruction data samples (136K), allowing FinMA to capture the nuances of the diverse financial tasks. The table below provides the statistical details of the instruction dataset:

| Data      | Task                         | Raw    | Instruction | Data Types                | Modalities        | License      | Original Paper |
| --------- | ---------------------------- | ------ | ----------- | ------------------------- | ----------------- | ------------ | -------------- |
| FPB       | sentiment analysis           | 4,845  | 48,450      | news                      | text              | CC BY-SA 3.0 | [1]            |
| FiQA-SA   | sentiment analysis           | 1,173  | 11,730      | news headlines, tweets    | text              | Public       | [2]            |
| Headline  | news headline classification | 11,412 | 11,412      | news headlines            | text              | CC BY-SA 3.0 | [3]            |
| NER       | named entity recognition     | 1,366  | 13,660      | financial agreements      | text              | CC BY-SA 3.0 | [4]            |
| FinQA     | question answering           | 8,281  | 8,281       | earnings reports          | text, table       | MIT License  | [5]            |
| ConvFinQA | question answering           | 3,892  | 3,892       | earnings reports          | text, table       | MIT License  | [6]            |
| BigData22 | stock movement prediction    | 7,164  | 7,164       | tweets, historical prices | text, time series | Public       | [7]            |
| ACL18     | stock movement prediction    | 27,053 | 27,053      | tweets, historical prices | text, time series | MIT License  | [8]            |
| CIKM18    | stock movement prediction    | 4,967  | 4,967       | tweets, historical prices | text, time series | Public       | [9]            |

1. Pekka Malo, Ankur Sinha, Pekka Korhonen, Jyrki Wallenius, and Pyry Takala. 2014. Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology 65, 4 (2014), 782–796.
2. Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018. Www’18 open challenge: financial opinion mining and question answering. In Companion proceedings of the the web conference 2018. 1941–1942
3. Ankur Sinha and Tanmay Khandait. 2021. Impact of news on the commodity market: Dataset and results. In Advances in Information and Communication: Proceedings of the 2021 Future of Information and Communication Conference (FICC), Volume 2. Springer, 589–601
4. Julio Cesar Salinas Alvarado, Karin Verspoor, and Timothy Baldwin. 2015. Domain adaption of named entity recognition to support credit risk assessment. In Proceedings of the Australasian Language Technology Association Workshop 2015. 84–90.
5. Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge, et al . 2021. FinQA: A Dataset of Numerical Reasoning over Financial Data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 3697–3711.
6. Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. 2022. Convfinqa: Exploring the chain of numerical reasoning in conversational finance question answering. arXiv preprint arXiv:2210.03849 (2022).
7. Yejun Soun, Jaemin Yoo, Minyong Cho, Jihyeong Jeon, and U Kang. 2022. Accurate Stock Movement Prediction with Self-supervised Learning from Sparse Noisy Tweets. In 2022 IEEE International Conference on Big Data (Big Data). IEEE, 1691–1700.
8. Yumo Xu and Shay B Cohen. 2018. Stock movement prediction from tweets and historical prices. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 1970–1979.
9. Huizhe Wu, Wei Zhang, Weiwei Shen, and Jun Wang. 2018. Hybrid deep sequential modeling for social text-driven stock prediction. In Proceedings of the 27th ACM international conference on information and knowledge management. 1627–1630.

### Generating Datasets for FIT

When you are working with the Financial Instruction Dataset (FIT), it's crucial to follow the prescribed format for training and testing models.

The format should look like this:

```json
{
    "id": "unique id",
    "conversations": [
        {
            "from": "human",
            "value": "Your prompt and text"
        },
        {
            "from": "agent",
            "value": "Your answer"
        }
    ],
    "text": "Text to be classified",
    "label": "Your label"
}
```

Here's what each field means:

- "id": a unique identifier for each example in your dataset.
- "conversations": a list of conversation turns. Each turn is represented as a dictionary, with "from" representing the speaker, and "value" representing the text spoken in the turn.
- "text": the text to be classified.
- "label": the ground truth label for the text.


The first turn in the "conversations" list should always be from "human", and contain your prompt and the text. The second turn should be from "agent", and contain your answer.

---

## FinMA v0.1: Financial Large Language Model

We are pleased to introduce the first version of FinMA, including three models FinMA-7B, FinMA-7B-full, FinMA-30B, fine-tuned on LLaMA 7B and LLaMA-30B. FinMA-7B and FinMA-30B are trained with the NLP instruction data, while FinMA-7B-full is trained with the full instruction data from FIT covering both NLP and prediction tasks. 

FinMA v0.1 is now available on [Huggingface](https://huggingface.co/TheFinAI/finma-7b-nlp) for public use. We look forward to the valuable contributions that this initial version will make to the financial NLP field and encourage users to apply it to various financial tasks and scenarios. We also invite feedback and shared experiences to help improve future versions.

### How to fine-tune a new large language model using PIXIU based on FIT?

Coming soon.

---

## FinMem: A Performance-Enhanced LLM Trading Agent

FinMem is a novel LLM-based agent framework devised for financial decision-making, encompasses three core modules: Profiling, to outline the agent's characteristics; Memory, with layered processing, to aid the agent in assimilating realistic hierarchical financial data; and Decision-making, to convert insights gained from memories into investment decisions. Currently, FinMem can trade single stocks with high returns after a simple mode warm-up. Below is a quick start for a dockerized version framework, with TSLA as sample input.

Step 1: Set environmental variables
in `.env` add HUGGINGFACE TOKEN and OPENAI API KEY as needed.
```bash
OPENAI_API_KEY = "<Your OpenAI Key>"
HF_TOKEN = "<Your HF token>"
```

Step 2: Set endpoint URL in `config.toml`
Use endpoint URL to deploy models based on the model of choice (OPENAI, Gemini, open source models on HuggingFace, etc.). For open-source models on HuggingFace, one choice for generating TGI endpoints is through RunPod. 
```bash
[chat]
model = "tgi"
end_point = "<set the your endpoint address>"
tokenization_model_name = "<model name>"
...
```

Step 3: Build Docker Image and Container
```bash
docker build -t test-finmem .devcontainer/. 
```
start container:
```bash
docker run -it --rm -v $(pwd):/finmem test-finmem bash
```

Step 4: Start Simulation!
```bash
 Usage: run.py sim [OPTIONS]                                                                                                                
                                                                                                                                            
 Start Simulation                                                                                                                           
                                                                                                                                            
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --market-data-path    -mdp      TEXT  The environment data pickle path [default: data/06_input/subset_symbols.pkl]                       │
│ --start-time          -st       TEXT  The training or test start time [default: 2022-06-30 For Ticker 'TSLA']                                                               │
│ --end-time            -et       TEXT  The training or test end time [default: 2022-10-11]                                                                 │
│ --run-model           -rm       TEXT  Run mode: train or test [default: train]                                                           │
│ --config-path         -cp       TEXT  config file path [default: config/config.toml]                                                     │
│ --checkpoint-path     -ckp      TEXT  The checkpoint save path [default: data/10_checkpoint_test]                                             │
│ --result-path         -rp       TEXT  The result save path [default: data/11_train_result]                                               │
│ --trained-agent-path  -tap      TEXT  Only used in test mode, the path of trained agent [default: None. Can be changed to data/05_train_model_output OR data/06_train_checkpoint]                                  │
│ --help                                Show this message and exit.                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                              
```
Example Usage:
```bash
python run.py sim --market-data-path data/03_model_input/tsla.pkl --start-time 2022-06-30 --end-time 2022-10-11 --run-model train --config-path config/tsla_tgi_config.toml --checkpoint-path data/06_train_checkpoint --result-path data/05_train_model_output
```

There are also checkpoint functionalities. For more details please visit [FinMem Repository](https://github.com/pipiku915/FinMem-LLM-StockTrading) directly. 

---

## Citation

If you use PIXIU in your work, please cite our paper.

```
@misc{xie2023pixiu,
      title={PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance}, 
      author={Qianqian Xie and Weiguang Han and Xiao Zhang and Yanzhao Lai and Min Peng and Alejandro Lopez-Lira and Jimin Huang},
      year={2023},
      eprint={2306.05443},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{xie2024FinBen,
      title={The FinBen: An Holistic Financial Benchmark for Large Language Models}, 
      author={Qianqian Xie and Weiguang Han and Zhengyu Chen and Ruoyu Xiang and Xiao Zhang and Yueru He and Mengxi Xiao and Dong Li and Yongfu Dai and Duanyu Feng and Yijing Xu and Haoqiang Kang and Ziyan Kuang and Chenhan Yuan and Kailai Yang and Zheheng Luo and Tianlin Zhang and Zhiwei Liu and Guojun Xiong and Zhiyang Deng and Yuechen Jiang and Zhiyuan Yao and Haohang Li and Yangyang Yu and Gang Hu and Jiajia Huang and Xiao-Yang Liu and Alejandro Lopez-Lira and Benyou Wang and Yanzhao Lai and Hao Wang and Min Peng and Sophia Ananiadou and Jimin Huang},
      year={2024},
      eprint={2402.12659},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

PIXIU is licensed under [MIT]. For more details, please see the [MIT](LICENSE) file.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=The-FinAI/PIXIU&type=Date)](https://star-history.com/#The-FinAI/PIXIU&Date)

-->
