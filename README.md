# CLEF2025-ParliamentaryDebateClassification


## 1. Overview
This project involves the classification of parliamentary debates, focusing on two primary tasks:

- **Political Ideology Classification:** Classification of a given speaker ideology into a ‘left’ or a ‘right’.

- **Political Orientation Classification:** Determining who is of the “governing party” or the “opposition party”.

To classify the speeches of parliaments, the project use fine-tuned multilingual masked language models and zero shot learning models. The results include comparisons off fine-tuning versus zero-shot inference models.

## 2. Installation and Usage
1.	**Clone the repository:** git clone https://github.com/Umyma-Shahzad/CLEF2025-ParliamentaryDebateClassification
2.	**Install dependencies:** pip install -r requirements.txt
3.	**Run the Fine tunning script:** python Fine_Tuning_Models.py
4.  **Run the Zero shot inference script:** python zero_shot_inference.py
   
## 3. Approach
For the tasks of political ideology classification and power identification, I adopted a two-step methodology:
### 3.1 Fine-tuning a multilingual masked language model:
- I selected **“xlm-roberta-base”** because it perform good on multiple languages.
- Fine-tuned the model separately for each task:
  - Task 1 used the **text_en (English translations)** of the speeches.
    
  - Task 2 used the **original text** in the chosen language.
### 3.2 Zero-shot inference using a multilingual causal language model:
- I used **“facebook/bart-large-mnli”** to evaluate performance on both text and text_en.
## 4. Dataset Statistics
The dataset for this assignment is from the **ParlaMint corpus**, performed classification of parliamentary debates from **France**. 

•	For Task 1 the dataset consisted of 3618 samples, with a class distribution of 30% for Left (0) and 70% for Right (1), observed to be imbalanced. 

•	For Task 2 the dataset included 9813 samples, with 60% labeled as Coalition (0) and 40% as Opposition (1), and the class distribution was also little imbalanced. 

•	The dataset was **stratified and split** into **90% Training Data** and **10% Test Data**.
## 5. Methodology
### 5.1 Preprocessing and EDA: 
I removed the special tags such as HTML tags, whitespaces, and the special characters. Filled missing translations (text_en) and original text (text) with a placeholder. Analyzed text length distributions and class balances.
#### 5.1.1 Exploratory Data Analysis of “Orientation”:
![image](https://github.com/user-attachments/assets/fc5e5e0f-3b92-42a4-a265-5a13f8786773)
![image](https://github.com/user-attachments/assets/ec123875-6d86-4800-8487-e146f71e8279)

  
#### 5.1.2 Exploratory Data Analysis of “Power”:
![image](https://github.com/user-attachments/assets/790e21eb-c31c-44e9-8ee7-6f67b9c654c7)
![image](https://github.com/user-attachments/assets/1ab0bb0f-c117-4639-bc99-d4ef14103b0b)


  
### 5.2 Data Augmentation
To address class imbalance, text augmentation techniques were applied:
-	**Synonym Replacement:** In this I replaced words with synonyms using WordNet.
-	**Random Insertion:** Randomly inserted existing words into sentences.
  
In these augmentation techniques the minority class was augmented by **1.5x**, ensuring balanced training data.
#### 5.2.1 Exploratory Data Analysis of “Orientation” After Augmentation:
![image](https://github.com/user-attachments/assets/d5cd3675-000f-4832-aef1-632ad83b7153)
![image](https://github.com/user-attachments/assets/9ab7f7c3-a797-4685-ae94-f65b6b3b7963)

  
#### 5.2.2 Exploratory Data Analysis of “Power” After Augmentation:
![image](https://github.com/user-attachments/assets/6c15309e-a4d1-42db-9adb-bf2f6291d37c)
![image](https://github.com/user-attachments/assets/672885dd-e797-407b-a5b9-6bff2c86e001)

  
### 5.3 Tokenization
Data was tokenized using **xlm-roberta-base** for efficient processing, truncating text to a maximum length of **512 tokens**.

## 6. Experimental Setup
### 6.1 Fine-Tuning Configuration:
•	**Model:** XLM-Roberta-base

•	**Learning Rate:** 2e-5

•	**Batch Size:** 8

•	**Epochs:** 3

•	**Optimizer:** AdamW

•	**Scheduler:** Linear scheduler with warm-up

•	**Evaluation Metric:** Accuracy

### 6.2 Zero-Shot Inference
**Model:** **facebook/bart-large-mnli** is used for zero-shot inference.

**Decoding:** For decoding I used top-k sampling with **k=50** and a temperature of **0.7**.
## 7. Results of Fine-Tuned Models
### 7.1 Task 1: Political Ideology Identification
The accuracy of fine-tuned model was pretty good and constituted **80%**. Class Left performed well and had a high precision, however, its recall was also low because the model fails to recognize all the instances of this class. On the other hand, Class Right had the high recall while having the slightly lower precision what suggests that there is some crossover and confusion between the two ideological classes. These results suggest the fact that it remains a difficult task to distinguish fine ideological differences within parliamentary debates.

![image](https://github.com/user-attachments/assets/36eabe39-dd6e-4700-836d-06daac7c451d)
![image](https://github.com/user-attachments/assets/d1f17426-02d7-4def-954d-0b2f2cb12dbf)
![image](https://github.com/user-attachments/assets/ded75e5e-0429-4545-95cc-7b67ef8f5153)



 
### 7.2 Task 2: Power Identification
The outcome of this task was a slight improvement from that of Task 1 with accuracy of **85.79%** and good performance metrics of the two classes. In general, the model established informative patterns that would help in identifying the membership of coalition and opposition parties with high recall and F1-scores. The class-wise analysing results show that the opposition class is slightly better than the governing class in terms of precision, while the overall analysis shows that the textual data is distinguishing between the governing and non-governing parties more sharply.

![image](https://github.com/user-attachments/assets/5f762926-942a-477f-b642-ba34f44c2de8)
![image](https://github.com/user-attachments/assets/a58856b9-3905-489c-9714-5c36af2c782b)
![image](https://github.com/user-attachments/assets/fffd7c87-89a7-4f32-9559-e681c9b29d3c)



 
## 8. Results Of Zero Shot Inference
### 8.1 Task 1: Political Ideology Identification
- **Using Text (Original Language):** From this classification, the model had higher accuracy with Class 1 (Right) at **56%** and a weighted F1-score of **0.56**. However, it also failed to perform well on Class 0 (Left) giving low precision and recall. 

- **Using Text_EN (English Translation):** Using the English translation increased recall for Class 0 (Left) and at the same time reduced the weighted F1-score to **0.50** and accuracy to **48%** with the major drawback of reduced recall for Class 1 (Right). 

**Comparison:** The F1 scores and accuracy of inference using the original text were found to be higher than those of the translated text. Thus the zero-shot inferences of the political ideology on the original text is more accurate than on English text.

  <img width="512" alt="image" src="https://github.com/user-attachments/assets/08ec2e20-ffb0-4c75-b9c0-0906d3f20968" />

![image](https://github.com/user-attachments/assets/1192567a-7b3d-4f65-ba4a-fcd62f9009ca)
![image](https://github.com/user-attachments/assets/58001465-6e4b-42af-a0b4-f3dca79f7c52)
![image](https://github.com/user-attachments/assets/223561fd-9d2c-4b43-9ea9-44f84600c32f)
![image](https://github.com/user-attachments/assets/ee913e1c-4c35-4bd4-90a5-376838693a86)


  
## 8.2 Task 2: Power Identification
-	**Using Text (Original Language):** The inference achieved the accuracy of **62%**. The model showed strong recall for Class 0 (Coalition) (0.94) but struggled with Class 1 (Opposition) (0.10), resulting in a weighted F1-score of 0.54.
  
-	**Using Text_EN (English Translation):** The inference accuracy using English text is **61%**. Performance for Class 0 (Coalition) was slightly lower (0.89), but recall for Class 1 (Opposition) improved marginally (0.13), maintaining a similar weighted F1-score of 0.54.
  
**Comparison:** Both text versions had similar F1-scores, with the original text performing slightly better for Class 0 (Coalition), while translation improved Class 1 (Opposition) recall. In this case accuracy of inference on original text is slight greater than on English translation.

<img width="512" alt="image" src="https://github.com/user-attachments/assets/b161cd6c-72af-48e5-8251-4eb2458ef453" />

![image](https://github.com/user-attachments/assets/a9ac7660-3d63-4c55-8f2a-aa09a2d540b2)
![image](https://github.com/user-attachments/assets/16cea25c-ed64-4ca5-bd0d-b57b5645244f)
![image](https://github.com/user-attachments/assets/a3186d3d-bfdf-4dd7-a9e1-1720d5145824)
![image](https://github.com/user-attachments/assets/22a57e3d-b72c-436c-a6b7-1bd050677104)

 

  
 
  
## 9. Comparative Results: Fine-Tuned vs Zero-Shot Causal Models
The fine-tuned models outperformed the zero-shot causal models in both tasks.
-	**Task 1:** Fine-tuning yielded balanced performance across both ideological classes, while the zero-shot model struggled, especially with the Left class.
-	**Task 2:** The fine-tuned model showed strong overall performance, particularly for Class Opposition, while the zero-shot model had significant recall issues for the Opposition class.
  
The fine-tuned models had better performance than zero shot inference based on the fact that these models had been trained on specific task data this would benefit them in understanding of patterns of political ideologies and party power relations better than the zero shot models. The zero shot models showed satisfactory performance, but did not have enough flexibility to give the comparable results.

<img width="535" alt="image" src="https://github.com/user-attachments/assets/d95b6a2a-6b2d-4829-9119-0c11227380a1" />


## 10. Conclusion
### 10.1 Best Performing Model
Therefore, this study shows that fine-tuned multilingual models can also be used efficiently when analyzing parliamentary debate. The fine-tuned models were much more accurate than the zero-shot ones, especially when translated text is used.
### 10.2 Limitations
  - **Class Imbalance:** Although augmentation techniques are used but still slight class imbalance still exists.
    
  -	**Translation Quality:** As the placeholder in used but it may introduce noise in data. 
  -	**Computation cost:** The training of multilingual models is quite computationally expensive.
### 10.3 Future Work:
  - Can try out more multilingual embeddings such as LaBSE for zero-shot inference
  -	Enhance the approaches to augmentation using paraphrasing models.
  -	Can try to deal with long-context dependencies by using the hierarchical attention mechanisms.

## Contact

umymashahzad@gmail.com
