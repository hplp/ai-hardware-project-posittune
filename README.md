[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Buol6fpg)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=16837588)

# AI_Hardware_Project

https://myuva-my.sharepoint.com/:p:/g/personal/yg9bq_virginia_edu/EbjDJD3vRmFHr43dp7wxsugB0Ji3ZrClOwHXlz48IbN-lg?e=XdtVJc 

## Team Name: 
6501_project_group 8 

## Team Members:
- Yimin Gao
- Shan He
- Zhenghong Chen

## Project Title:
PositTune: Design space exploration of mixed-precision posit quantization

## Project Description:
This project aims to explore the design space of mixed-precision quantization using posit arithmetic. We will begin by examining uniform post-training quantization across various ML inference models, comparing accuracy results among formats like FP32, FP16, INT16, INT8, posit<8,1>, posit<8,0>, and posit<6,1>. Next, we’ll analyze layer-wise and channel-wise quantization sensitivity in a selected ML model (e.g., GPT-2) and develop an algorithm to fine-tune the post-training quantization scheme, determining the optimal precision and quantization type for each layer or channel. If time allows, we will also explore hardware development for an energy-efficient versatile posit arithmetic unit supporting fused MAC operations with configurable bit precision (offering flexible total bitwidth and regime bitwidth). 
(Provide a short description of the problem you're addressing)

## Key Objectives:
- Evaluate Uniform Post-Training Quantization: Assess the accuracy of various quantization formats (e.g., FP32, FP16, INT16, INT8, posit<8,1>, posit<8,0>, posit<6,1>) across multiple ML inference models.
- Analyze Quantization Sensitivity by Layer/Channel: Investigate layer-wise and channel-wise sensitivity to posit quantization in a selected ML model (e.g., GPT-2) to identify optimal posit precision settings.
- Develop a Fine-Tuning Algorithm: Create an algorithm for post-training quantization, enabling dynamic selection of precision and quantization type per layer or channel for improved model performance.
- (If time allows) Explore Hardware Development for Posit Arithmetic (time permitting): Design an energy-efficient, versatile posit arithmetic unit with support for fused MAC operations and configurable bit precision, adaptable in total and regime bitwidth.

## Technology Stack:
Hardware platform: cuda with NVIDIA 2080 Super
Software tools: Qtorch plus framework
Languages: Python

## Expected Outcomes:
- We expect posit quantization to achieve comparable accuracy to FP32 and FP16 quantization while using a lower bitwidth across various ML models.
- We expect that the selected ML model (e.g., GPT-2) will exhibit varying sensitivities across layers and channels, enabling us to fine-tune quantization by adjusting posit configurations. This includes using the same total bitwidth with varying exponent bits or different total bitwidths altogether to achieve the best possible accuracy.
- We expect that an automated algorithm for posit quantization will offer significant advantages and may advance the state of the art in LLM quantization by potentially outperforming integer and low-bitwidth floating-point quantization in certain applications.

## Timeline:
- Evaluate Uniform Post-Training Quantization in various data format on GPT-2 (e.g., FP32, FP16, INT16, INT8, posit<8,1>, posit<8,0>, posit<6,1>)  11/4-11/10
- Analyze Quantization Sensitivity by Layer/Channel 11/11-11/22
- Develop a fine-tuning algorithm to find the best quantization scheme which balances the model accuracy and computation cost 11/23-11/29
- (If time allows) Explore Hardware Development for Posit Arithmetic: Design an energy-efficient, versatile posit arithmetic unit with support for fused MAC operations and configurable bit precision, adaptable in total and regime bitwidth

## Milestone Report on 11/26:
[PositTune Milestone](https://docs.google.com/document/d/1pTAWPl2mSRFma4lOWRw0iVCUqy6DBjIWGkL-NMCtweE/edit?usp=sharing)


## Final Report:
[Final Report](https://docs.google.com/document/d/1P3ssoJj-iGFYuIgr8NX16kJbeMJYuJoSXZm8dU3G1Wc/edit?usp=sharing)

Here are the jupyter notebooks we used to benchmark three models with layer-wise adaptive scaling during posit quantization: 

- [GPT2](https://colab.research.google.com/drive/1FIOMbVFmk1wJLa8ySeniGPl5ZKkwioO9?usp=sharing)
- [Yolo](https://colab.research.google.com/drive/1lXEbmaDumhhqBa-TqXaIc0qEOPpt7y0X?usp=sharing)
- [ALBERT](https://colab.research.google.com/drive/1W14j33hOgq_tM71EWNQv63YRhlo8PnWY?usp=sharing)

(All the other codes for this project](https://drive.google.com/drive/folders/1N5s3er1LL_gCWstWVO4nBSjxM2iwgJFw?usp=sharing)

