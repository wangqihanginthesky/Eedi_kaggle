# 2nd place solution

## Preprocess

We found a useful subject\_metadata.csv of which the SubjectId and SubjectName are Identical to this competition from the past eedi competition hosted on [NeurlPS 2022](https://codalab.lisn.upsaclay.fr/competitions/5626#learn_the_details-get_starting_kit). The subject\_metadata.csv contains the parent subject, so we made a vector db with this metadata to add the parent subject information to both train.csv and test.csv. 

## Synthetic data

### synthetic questions

We generated synthetic data 3 times, let’s call them generation1, generation2, generation3.  
The base approach is:

- provide LLM a misconception with few examples and let it generate questions  
- use qwen-math to solve the question to get the correct answer  
- use qwen-math to solve the question under the constraint of misconception to get the wrong answer  
- use gpt-4o-mini to score the quality of the question and choose those with score larger than 2 (max is 5\)

The difference between each generation is as follows.  
Generation1:

- few shot examples: randomly sampled from train.csv

Generation2:

- few shot examples: sample the question with the same misconception from train.csv and Generation1

Generation3:

- The prompt of generation 3 is based on the prompt of [this tech blog](https://tech.preferred.jp/ja/blog/llm-synthetic-dataset-for-math/)  
- few shot examples: randomly sample 2 questions from train.csv, Generation1 and Generation2

### misconception augmentation

The misconception only contains a short sentence. In order to make the embedding of the misconception more meaningful, we used LLM to generate explanation for each misconception. Since we don’t need to run inference for misconception in submission, the approach costs nothing in submission.  
The prompt is as follows. The explanation of llama3.1-70b-Instruct and qwen2.5-72b-Instruct out-perform gpt-4o-mini in training retriever.  
\`\`\`  
system\_prompt\_template \= 'You are an excellent math teacher about to teach students of year group 1 to 14\. The subject of your lesson includes Number, Algebra, Data and Statistics, Geometry and Measure. You will be provided a misconception that your students may have. Please explain the misconception in detail and provide some short cases when the misconception will occur. No need to provide the correct approach. The explanation should be in format of "Explanation: {explanation}"'

user\_prompt\_template \= 'Misconception: {misconception}'  
\`\`\`

## Chain of thought

We used qwen2.5-32B-Instruct-AWQ to generate chain-of-thought as additional input for the following retrieve and rerank. The prompt is as follows:  
\`\`\`  
system\_prompt\_template \= "You are an excellent math teacher about to teach students of year group 1 to 14\. The detail of your lesson is as follows. Subject:{first\_subject}, Topic: {second\_subject}, Subtopic {third\_subject}. Your students have made a mistake in the following question. Please explain the mistake step by step briefly and describe the misunderstanding behind the wrong answer at conceptual level. No need to provide the correct way to achieve the answer."

user\_prompt\_template \= "Question: {question\_text}\\nCorrect Answer: {correct\_text}\\nWrong Answer of your students: {answer\_text}\\n\\nExplanation: \\nMisunderstanding: "  
\`\`\`

## Retrieve

We trained retriever with 2 different pipeline.  
Pipeline1:

- backbone: Linq-AI-Research/Linq-Embed-Mistral  
- loss: Arcface  
- use Chain of thought as additional input in training and inference

Pipeline2:

- backbone: Qwen/Qwen2.5-14B, Qwen/Qwen2.5-32B, Qwen/QwQ-32B-Preview  
- loss: MultipleNegativesRankingLoss  
- w/o Chain of thought (due to inference time)

Single model performance is as follows.   
|retriever|synthetic data|Private LB|Public LB|Inference time|  
|:----|:----|:----|:----|:----|  
|Linq-AI-Research/Linq-Embed-Mistral|Generation123|0.461|0.484|50 min|  
|Qwen/Qwen2.5-14B|Generation12|0.479|0.507|45 min|  
|Qwen/Qwen2.5-14B|Generation123|0.485|0.492|45 min|  
|Qwen/Qwen2.5-32B|Generation123|0.495|0.536|140 min|  
|Qwen/QwQ-32B-Preview|Generation123|0.500|0.531|140 min|  
Our best submission used an ensemble of Mistral and 2 x qwen2.5-14B to give enough time to 72b rerank, the private LB and public LB is 0.513, 0.530.

\#\#\# Key Factors for Retriever Improvements

1\. \*\*Using Large Models\*\*    
   As is shown in the table above, the larger the better, GPU and Credit Card is all you need to get Power\!\!\! Never open the billing page during the competition.

- qwen2.5-14B with Generation12: H100 about 2days  
- qwen2.5-14B with Generation123: H100 about 3days  
- qwen2.5-32B with Generation123(sampled): H100 about 5days  
- QWQ with Generation123(sampled): H100 about 5days

2\. \*\*Synthetic question\*\*    
   I believe most of the participants used synthetic questions, the more high quality questions, the better performance. For our team, using gpt-4o-mini to filter high quality questions is the key.

3\. \*\*Misconception augmentation\*\*    
   Using misconception augmentation significantly boosted retriever performance by about 2-4%.

4\. \*\*Chain of Thought\*\*    
   CoT is also useful. But for 14B and 32B models, adding CoT to the prompt will double the inference time.

5\. \*\*Pooling Selection\*\*  
   We found that last token pooling achieved better performance than mean pooling in the Qwen model.

## Rerank

\#\# Rerank

We used a listwise reranker to refine the ranking of retrieved candidates. Our reranking process employed a sliding window approach: first, we used a lightweight LLM to reorder candidates ranked between 8th and 17th. Then, we leveraged larger models to finalize the rankings for the top 10 candidates.

The LLMs for reranking were fine-tuned on a combination of synthetic and training data.

\- \*\*Window 1 (8th \~ 17th)\*\*    
  \- Qwen2.5-14B-Instruct    
\- \*\*Window 2 (1st \~ 10th)\*\*    
  \- Qwen2.5-72B-Instruct    
  \- Llama-3.3-70B-Instruct  

\#\#\# Key Factors for Reranking Improvements

1\. \*\*Using Large Models\*\*    
   We found that larger models (e.g., 72B parameters) consistently delivered stronger validation scores compared to smaller ones like 14B or 32B models. However, these larger models initially performed worse on the Public LB, leading to some concerns. Despite this, we trusted the validation scores and included the 72B model in our final submissions (special thanks to the three-submission rule\!). Ultimately, the 72B model produced outstanding Private Leaderboard scores, helping us secure a prize.

2\. \*\*Chain of Thought\*\*    
   The above CoT prompts greatly improved reranking performance.

3\. \*\*Sliding Window\*\*    
   Instead of increasing the number of candidates for reranking, applying the sliding window strategy multiple times to refine the top-10 rankings proved to be more effective.  
[画像ファイル](https://private-user-images.githubusercontent.com/28046594/395771272-b3345de9-939b-4d7b-8e77-0acb8ffed879.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzQxNzEwNzMsIm5iZiI6MTczNDE3MDc3MywicGF0aCI6Ii8yODA0NjU5NC8zOTU3NzEyNzItYjMzNDVkZTktOTM5Yi00ZDdiLThlNzctMGFjYjhmZmVkODc5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMTQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjE0VDEwMDYxM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWRiMDA3MzYxMDZiNTkwNzJiYmNjNjIyNjAwOGVlOTU0MzZmMmQxODE0NWI4MzUyYTY3MmIxMTIzNGU0NWU1YTgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.X370Y06rkD3pTgXeoM1oO-gpMK6y6JERNA1cNy9K-2A)  
![][image1]

4\. \*\*Test-Time Augmentation\*\*    
   During inference, we used TTA with some models by generating prompts in reverse order and averaging their scores with those from standard prompts. This technique provided a slight boost in accuracy.

\#\#\# Training the Rerank Models    
We developed the QLoRA training code for our LLM rerankers based on the first-place solution from [atmacup17](https://www.guruguru.science/competitions/24/discussions/21027ff1-2074-4e21-a249-b2d4170bd516/). Special thanks to [@kcotton21](https://www.kaggle.com/kcotton21) for sharing such excellent solutions.

Here are some tips that proved effective during the reranking model training:  

1\. \*\*Randomizing Listwise Choices\*\*    
   Instead of always using the top-10 candidates for prompts, we created prompts with a variety of top-N combinations such as top-3, top-5, top-15, and top-25.  

2\. \*\*Synthetic Data\*\*    
   The synthetic data used for training the retriever was also helpful for training the reranker. In total, we trained on 8,000 records of training data (2 epochs) and 14,000 synthetic records, resulting in a combined dataset of 22,000 records.  

3\. \*\*Negative Sample Mining\*\*    
   To mine negative samples, we used a hybrid retriever combining the fine-tuned \`dunzhang/stella\_en\_400M\_v5\` model and TF-IDF. Each of the 22,000 positive samples had corresponding negative samples mined using this setup.  

\#\#\#\# Training Time  
Qwen2.5-14B: \~2 hours on H100    
Qwen2.5-72B: \~8 hours on H100  

\#\#\# Quantization    
We used the [intel/auto-round](https://github.com/intel/auto-round) library for quantizing the LLM rerankers. Compared to AutoGPTQ and AutoAWQ, this library was easier to use and caused minimal accuracy loss (typically less than 2%). Additionally, it could produce models compatible with vLLM.

qwen2.5-72b-Instruct have some issues to run on multi GPU due to its intermediate\_size(29568). Following the workaround provided by the [document of gptq](https://qwen.readthedocs.io/en/latest/quantization/gptq.html#troubleshooting), we padded the weights to 29696 and then performed quantization.

For calibration, we used the training dataset. Below are the quantization parameters:  

\`\`\`python  
bits, group\_size, sym \= 4, 128, True  
autoround \= AutoRound(    
    model, tokenizer, bits=bits, group\_size=group\_size, sym=sym, dataset=calib\_prompts, seqlen=256,    
    nsamples=512,    
    iters=500,    
)  
\`\`\`

\#\#\# Inference  
We use vLLM for inference.

By setting the \`enabling\_prefix\_cache\` to \`True\`, we were able to save approximately 10% of the inference time.

[jagatkiran](https://www.kaggle.com/jagatkiran) shared his insights on performing  [inference with a 72B LLM model](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/550223) .In this competition , larger models tend to perform better, which has been very helpful for us.

We implemented the reranker using \`logits\_processors\` and \`logprobs\` by assigning a weight of \+100 to specific tokens. This approach helped us establish the framework for the ranker. We also tried using a classification head directly, but the results were not satisfactory, and it was not easy to perform inference with vLLM. We believe this method could become a paradigm in future competitions.

\#\#\# Ablation  
|Baseline (retriever)|Qwen14B|Qwen72B|Llama70B|8-17th Qwen14B|Private LB|Public LB|  
|:----|:----|:----|:----|:----|:----|:----|  
|✅| | | | |0.513|0.530|  
|✅|✅| | | |0.568|0.583|  
|✅| |✅| | |0.593|0.582|  
|✅|✅|✅|✅| |0.596|0.609|  
|✅|✅|✅|✅|✅|\*\*0.604\*\*|0.622|

On the final day, we tried fine-tuning Nexusflow/Athene-V2-Chat instead of Llama70B.  Unfortunately, the submission with this model missed the deadline due to timeout issues, but it showed highly impressive performance on the leaderboard: 0.609.

\#\# Reference  
[LLMにおける合成データセットによる数学推論タスクの精度向上の検討](https://tech.preferred.jp/ja/blog/llm-synthetic-dataset-for-math/)  
[NeurIPS 2022 CausalML Challenge: Causal Insights for Learning Paths in Education](https://codalab.lisn.upsaclay.fr/competitions/5626#learn_the_details-overview)  
[A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models](https://arxiv.org/pdf/2310.09497)  
[Qwen2.5-72B & Llama 3.3-70B on 2xT4(questionable performance)](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/550223)  
[Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs](https://arxiv.org/abs/2309.05516)  
[1st place solution from atmacup17](https://www.guruguru.science/competitions/24/discussions/21027ff1-2074-4e21-a249-b2d4170bd516/)  
[Winning Amazon KDD Cup'24](https://openreview.net/forum?id=sv0E1mBhu8)  
[Qwen GPTQ Troubleshooting](https://qwen.readthedocs.io/en/latest/quantization/gptq.html#troubleshooting)

[image1]: <data:image/png;base64,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>