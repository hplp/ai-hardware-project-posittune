[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Buol6fpg)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=16837588)
# PositTune Final Report  
**Yimin Gao, Zhenghong Chen, Shan He**

---

## Introduction  
PositTune is a project that explores the design space of mixed-configuration posit quantization on ML models using Qtorch plus implementation as our platform. Since posit is a relatively new concept, there is no real system-level hardware implementation in this project, but we wish with promising results on mixed-configuration posit quantization, we can further motivate and prove the necessity of the development of a reconfigurable/versatile posit arithmetic unit in future work. GPUs are required to speed up the quantization process (equation transform from floating point to posit), or train a model entirely in posit. (a NVIDIA 2080 Super GPU is used for this project)  

[QPytorch](https://github.com/minhhn2910/QPyTorch) is a Pytorch implementation of posit quantization so we can simulate the accuracy of the posit-quantized ML model without the need of having real posit hardware. The numbers (weights/activation) after quantizing to posit are still floating point in the hardware, but have the same/similar values to posit arithmetic following posit’s equation. However, the limit there is how you could emulate say posit<32,0> since it has a better accuracy than 32-bit floating point. But since we are only implementing low-bitwidth posit quantization such as 8-bit or 4-bit, that’s not a concern to this project.  

[Posit](https://posithub.org/docs/Posits4.pdf) is this dynamic numerical representation that has the best accuracy near 1 (2^0), and a wider range compared to floating point of the same bitwidth. As shown in the figure below, floating point arithmetic shows the same level of accuracy across a wide range of numbers. However, this is usually redundant for most of the applications such as an ML inference where most of the weights/activation lie in a relatively small region. Posit, on the other hand, offers a more efficient design by demonstrating better accuracy around 1 and a wider range compared to floating point. There is this sweet spot around 1 where posit outperforms floating point in terms of accuracy and this is usually where we care the most for ML applications. However, what if the weight/activations for certain ML models go out of this region? Our solution is that we could dynamically shift the weights for each layer to where posit is the most accurate.  
<img src="https://github.com/hplp/ai-hardware-project-posittune/blob/main/imgs/distribution.png" alt="Accuracy distribution of various data format" width="650">

As an example, the figure below shows the weight distribution of the first layer of GPT2 model. As we can see, the center bin of the distribution is at 2^(-2.72). To benefit the most from posit arithmetic, we can shift this distribution (in log 2 scale) to the right where the distribution is centered around 1 (where posit is the most accurate), as shown in the following figure.  

<img src="https://github.com/hplp/ai-hardware-project-posittune/blob/main/imgs/embed0.png" alt="Distribution of the first layer in GPT2 model" width="600">
<img src="https://github.com/hplp/ai-hardware-project-posittune/blob/main/imgs/embed1.png" alt="Distribution of the first layer in GPT2 model after adaptive scaling" width="600">


---

## PositTune Tutorial  
Here are the jupyter notebooks we used to benchmark three models with layer-wise adaptive scaling during posit quantization:  

- **GPT2:** [Notebook](https://colab.research.google.com/drive/1FIOMbVFmk1wJLa8ySeniGPl5ZKkwioO9?usp=sharing)  
- **YOLO:** [Notebook](https://colab.research.google.com/drive/1lXEbmaDumhhqBa-TqXaIc0qEOPpt7y0X?usp=sharing)  
- **ALBERT:** [Notebook](https://colab.research.google.com/drive/1W14j33hOgq_tM71EWNQv63YRhlo8PnWY?usp=sharing)  

While the detailed implementation of these posit quantization methods can be found in the original Qtorch paper, the same quantization method is applied in all three benchmarks - The weight and the activation of each layer for the selected model is quantized to posit arithmetic using a predefined quantization function from the qtorch_plus library: `posit_quantize()`. The function allows you to choose the total bitwidth (`nsize`), exponent bitwidth (`es`) and the scaling factor (`scale`) for the quantization process. In each of the three benchmark notebooks, there is a step where the program iterates through all the linear layers of the selected ML model to quantize the weights and the activation using the defined `linear_weight` and `register_forward_prehook` functions.  

<img src="https://github.com/hplp/ai-hardware-project-posittune/blob/main/imgs/code.png" width="700">
This is where we made our changes to apply layer-wise adaptive scaling to the weights during posit quantization. To take GPT2 as an example, below is the revised version of the code where we first find the center bin of the weight distribution for each layer (`x_with_max_frequency`) and apply this scaling factor (2^(-x_with_max_frequency)) in the posit quantization process.  
<img src="https://github.com/hplp/ai-hardware-project-posittune/blob/main/imgs/code2.png" alt="Adaptive scaling in the code" width="700">

---

## Evaluation Results  

### GPT2 (Generative Pre-trained Transformer 2)  
GPT2 is a large language model by OpenAI that generates coherent text by predicting the next word in a sequence. The Wikitext dataset is used for evaluating the perplexity of the quantized model, and the detailed perplexity result among various data types can be found in the table below. As shown in the table, even without adaptive scaling, posit(8,1) is already able to outperform 8-bit floating point in terms of accuracy. Then with the layer-wise adaptive scaling, the perplexity of the quantized model increases for each posit configuration. The improvement is more significant in certain configurations with a narrow accuracy distribution such as posit(6,0) due to that the scaling can help a lot to align the data to where this narrow posit configuration serves the best accuracy.  
<img src="https://github.com/hplp/ai-hardware-project-posittune/blob/main/imgs/gpt.png" width="700">

### YOLO (You Only Look Once)  
YOLOv5s is a deep neural network model used for real-time object detection. It is widely recognized in computer vision for its speed and efficiency, allowing it to detect objects in images or videos while predicting their class and location. We chose the v5s model, which has a size of 14MB, making it suitable for edge computing devices. Generally speaking, as the model size increases, the latency for processing each image also increases, but the average precision (AP) on Common Objects in Context (COCO) improves as well. YOLOv5s is a compact and efficient model designed specifically for speed and low resource consumption. It consists of 213 layers and approximately 7.2 million parameters. Although the model is small, it achieves a good balance between accuracy and speed in many real-world object detection tasks. With a computational complexity of only 16.4 GFLOPs, it is highly suitable for edge computing and real-time applications.  
<img src="https://github.com/hplp/ai-hardware-project-posittune/blob/main/imgs/yolo.png" width="700">

In the project, we determined the confidence value by comparing the results of running YOLOv5s with those of the larger YOLOv5x model and 32-bit floating-point computations. This value directly represents the model's accuracy. Additionally, we compared the results of processing a single image under different quantization schemes within the project. The model uses 16-bit floating-point precision, which we quantized into two different Posit schemes: 8-bit and 6-bit. By manually adjusting the scaling scheme and employing adaptive scaling, the algorithm automatically determined the optimal scale value, which improved the results and refined the conclusions.  
<img src="https://github.com/hplp/ai-hardware-project-posittune/blob/main/imgs/bert.png" width="700">

### ALBERT (A Little BERT)  
ALBERT is a smaller, faster version of BERT that reduces model size and improves efficiency by sharing parameters across layers and using optimized embedding techniques. The task we implement to evaluate quantization performance of ALBERT is answering arbitrary questions according to given context. The result is valued as F1 score, which measures overlap between the predicted and ground-truth answers.  

We tried to quantize linear weights with 32-bit floating point, 8-bit floating point, posit <6,1> without scaling, and posit <6,1> with the adaptive scaling method. As shown in the following table, the highest F1 score comes from 32-bit floating point quantization. However, the trade-off is more resources (32-bit operation) are needed. When we quantize with 8-bit floating point, the overall F1 score is slightly reduced from 86.19% to 85.36%, and the F1 score for questions with answers goes down from 86.32% to 82.86%. The F1 score for posit <6,1> without scaling dropped more. Nevertheless, if the adaptive scale was applied to posit <6,1>, the F1 score is higher than 8-bit floating point quantization. In conclusion, the use of posit quantization with the adaptive scaling method enables achieving strong performance even with a small bit size.  

---

## Conclusion and Lessons Learned  
Throughout this project, we have learned that posit arithmetic, even without adaptive scaling, offers a highly promising alternative for edge inference that balances efficiency and precision. Moreover, incorporating layer-wise adaptive scaling in posit inference enables achieving the same accuracy with reduced bitwidth or higher accuracy at the same bitwidth. We hope this motivates further research into developing a versatile posit arithmetic unit with runtime-configurable scaling to further enhance energy efficiency in edge AI training and inference.  

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

