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
# PositTune Final Report  
**Yimin Gao, Zhenghong Chen, Shan He**

---

## Introduction  
PositTune explores the design space of mixed-configuration posit quantization on ML models using Qtorch Plus as the implementation platform. Posit, a relatively new numerical representation, has no system-level hardware implementation yet. However, the promising results of mixed-configuration posit quantization from this project could motivate the development of a reconfigurable/versatile posit arithmetic unit in future work.

The quantization process, which transforms equations from floating point to posit, requires GPUs to improve efficiency. For this project, we utilized an NVIDIA 2080 Super GPU.

Qtorch, a PyTorch implementation of posit quantization, allows simulating the accuracy of posit-quantized ML models without actual posit hardware. The quantized numbers (weights/activation) retain floating-point representation but closely follow posit arithmetic equations. While posit<32,0> emulation offers higher accuracy than 32-bit floating point, this project focuses on low-bitwidth posit quantization (e.g., 8-bit or 4-bit), where this limitation is not a concern.

Posit is a dynamic numerical representation that provides better accuracy near 1 (2^0) and a wider range than floating point at the same bitwidth. Unlike floating point, which maintains uniform accuracy across a wide range, posit optimizes accuracy around 1. This characteristic is particularly beneficial for ML applications, where weights/activations often fall within this region. To handle weights/activations outside this range, we applied layer-wise adaptive scaling.

### Weight Distribution Example  
The weight distribution of GPT2's first layer demonstrates the effectiveness of adaptive scaling. The distribution, centered at 2^(-2.72), is shifted to center around 1 for optimal posit accuracy. This method reduces quantization error, as shown in the quantization error distributions.
![Weight Distribution of GPT2](https://github.com/hplp/ai-hardware-project-posittune/blob/main/imgs/distribution.png)

---

## PositTune Tutorial  
Below are Jupyter notebooks for benchmarking three models using layer-wise adaptive scaling during posit quantization:  

- **GPT2:** [Notebook](https://colab.research.google.com/drive/1FIOMbVFmk1wJLa8ySeniGPl5ZKkwioO9?usp=sharing)  
- **YOLO:** [Notebook](https://colab.research.google.com/drive/1lXEbmaDumhhqBa-TqXaIc0qEOPpt7y0X?usp=sharing)  
- **ALBERT:** [Notebook](https://colab.research.google.com/drive/1W14j33hOgq_tM71EWNQv63YRhlo8PnWY?usp=sharing)  

These notebooks apply Qtorch's posit quantization function `posit_quantize()` with parameters for bitwidth (`nsize`), exponent bitwidth (`es`), and scaling factor (`scale`). Layer-wise adaptive scaling adjusts the center bin of each layer's weight distribution before quantization.

---

## Evaluation Results  

### GPT2 (Generative Pre-trained Transformer 2)  
GPT2 generates coherent text by predicting the next word in a sequence. Using the Wikitext dataset, we evaluated perplexity under various data types. Posit(8,1) achieved higher accuracy than 8-bit floating point, even without adaptive scaling. Layer-wise adaptive scaling further reduced perplexity, particularly in narrow configurations like Posit(6,0), by aligning data with posit’s optimal accuracy range.

### YOLO (You Only Look Once)  
YOLOv5s, a compact object detection model with 7.2M parameters and 16.4 GFLOPs, balances accuracy and speed for edge computing. Comparing results under different quantization schemes, adaptive scaling improved accuracy by automatically determining optimal scale values, enhancing precision in Posit(8,0) and Posit(6,0).

### ALBERT (A Lite BERT)  
ALBERT evaluates quantization performance through F1 scores in question-answering tasks. Posit<6,1> with adaptive scaling outperformed 8-bit floating point, achieving strong results with lower bitwidth. This demonstrates that adaptive scaling maintains precision while reducing computational resources.

---

## Conclusion and Lessons Learned  
Posit arithmetic, even without adaptive scaling, provides an efficient alternative for edge inference. Layer-wise adaptive scaling further enhances accuracy at reduced bitwidth, demonstrating its potential in edge AI. We hope this work inspires the development of a runtime-configurable posit arithmetic unit, advancing energy-efficient AI training and inference.

---


## Milestone Report on 11/26:
[PositTune Milestone](https://docs.google.com/document/d/1pTAWPl2mSRFma4lOWRw0iVCUqy6DBjIWGkL-NMCtweE/edit?usp=sharing)


## Final Report:
[Final Report](https://docs.google.com/document/d/1P3ssoJj-iGFYuIgr8NX16kJbeMJYuJoSXZm8dU3G1Wc/edit?usp=sharing)

Here are the jupyter notebooks we used to benchmark three models with layer-wise adaptive scaling during posit quantization: 

- [GPT2](https://colab.research.google.com/drive/1FIOMbVFmk1wJLa8ySeniGPl5ZKkwioO9?usp=sharing)
- [Yolo](https://colab.research.google.com/drive/1lXEbmaDumhhqBa-TqXaIc0qEOPpt7y0X?usp=sharing)
- [ALBERT](https://colab.research.google.com/drive/1W14j33hOgq_tM71EWNQv63YRhlo8PnWY?usp=sharing)

[All the codes we used for this project](https://drive.google.com/drive/folders/1N5s3er1LL_gCWstWVO4nBSjxM2iwgJFw?usp=sharing)

