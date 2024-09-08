### **OVERVIEW:** 

This repository is a practical/applied data science project from start to (near) finish. Working with the intake and outcome data publically available from a US animal shelter, I wanted to explore what makes a cat more or less likely to be adopted. I wrote a machine learning algorithm with 90% accuracy, which identified the following main factors: availability of the shelter on weekends, the cat's sex, the cat's appearance (coat color, pattern, and hair length), and whether the shelter had invested time into the cat's health (ex. performing an exam to confirm the sex, spaying/neutering the cat, etc.). 

---

#### **Cleaning:** 

The first step in my project was taking the messy input data and engineering it into something useable. I have a lot of future questions that might be answered with this same dataset, so my goal was to write a module that could be used to clean the data without species specificity. I did this in a [Jupyter Notebook](https://github.com/KathrynMercer/Austin-Animal-Center-Data/blob/main/Data%20Cleaning.ipynb) primarily with Pandas functionality. Then I retraced my steps and wrote a [self-contained module](https://github.com/KathrynMercer/Austin-Animal-Center-Data/blob/main/Data%20Cleaning%20Module.ipynb) for consistent future data cleaning. 

&emsp;Tasks: 

&emsp;&emsp; - merged the intake and outcome data sets, correctly joining to allow for repeat visits by the same animal 

&emsp;&emsp; - removed potential sensitive data (ie. names and intake locations)

&emsp;&emsp; - removed or modified null data

&emsp;&emsp; - converting ages to useable format using Regex

&emsp;&emsp; - separating the poorly defined "Nursing" condition into lactating adults and nursing juveniles

&emsp;&emsp; - splitting sex and reproductive status into separate features

&emsp;&emsp; - sanity checks for:
  
&emsp;&emsp;&emsp;  + negative age values
  
&emsp;&emsp;&emsp;  + conditions inappropriate to reported age (ex. old neonates, young seniors)
  
&emsp;&emsp;&emsp;  + conditions inappropriate for sex (ex. pregnant/nursing males)

---

#### Exploratory Data Analysis (EDA):

Here I refined the question (see the Notes file for some of the other ideas I considered) and looked at the data using that question to focus my perspective. I leaned heavily on being a subject matter expert to guide my feature engineering. I again worked in a [Jupyter Notebook](https://github.com/KathrynMercer/Austin-Animal-Center-Data/blob/main/Shelter%20Cat%20Outcome%20Influencers.ipynb), using Numpy, Pandas, MatPlotLib/Seaborn, YData Profiling, SciPy, Scikit-learn, regex, and imblearn. Below, I detail the data exploration (including early visualization), decision to pursue a machine learning algorithm, and preprocessing necessary before training a model.

**Key Question:**  What features of a cat or its circumstances might be associated with the cat being successfully adopted (vs another outcome)?

&emsp;Tasks: 

&emsp;&emsp; - refined the dataset to include only cats

&emsp;&emsp; - excluded cat's presented as "Euthanasia Request" or with "Lost/Missing" as an ultimate outcome

&emsp;&emsp; - Feature engineering: (see [Notes](https://github.com/KathrynMercer/Austin-Animal-Center-Data/blob/main/Notes.txt) file for details/further rationale)

&emsp;&emsp;&emsp; + added a "Duration of Stay" variable 

&emsp;&emsp;&emsp; + consolidated Outcome Types that were redundant for my question 

&emsp;&emsp;&emsp; + consolidated Intake Conditions that were similar

&emsp;&emsp;&emsp; + identified purebred status and breed components (when possible) using Regex

&emsp;&emsp;&emsp; + added a "Coat Length" variable using the provided breed information

&emsp;&emsp;&emsp; + tamed the free response "Color" feature into predominating colors and patterns

&emsp;&emsp; - outlier handling (for ages and Duration of Stay variables)

&emsp;&emsp; - designed a custom stacked bar chart and contingency table for examination of the categorical variable correlates

<div align="center"> <b> Fun Fact: </b> The "Color" feature engineering in this project inspired my <a href = "https://github.com/KathrynMercer/Cat-Coat-Classifier"> computer vision model </a> for describing cat coats from photos. </div><br>  

##### **Initial Findings:**   
After doing all that work, I reviewed some correlations in the data visually to get a sense for what might actually make good features for my model.

<figure>
  <div align="center"><img src="https://github.com/KathrynMercer/Austin-Animal-Center-Data/blob/main/EDA%20Outcome%20Repro%20Status.png" alt="Stacked Bar Chart of Outcome Reproductive Status vs Outcome Type" height = "550" width = "750"/><br>
  <figcaption>1. Outcome Reproductive Status vs Outcome Type.</figcaption></div>
</figure><br>
Here, we see the adoption rates of cats that are altered (ie. spayed or neutered) or intact. At first glance, it looks like people really prefer to adopt altered animals! However, this is a classic situation of subject matter influencing how you interpret the data and correlation not implying causation. In actuality, shelters rarely adopt out animals without first spaying/neutering them. More realistically, if someone wants to adopt a particular animal, that action/request triggers the shelter to alter the animal - not the other way around! A better interpretation here is that animals who get adopted may be more likely to have shelter investment (here, in the form of performing an examination and surgery on them, which costs precious resources). That also fits better with cats of "Unknown" reproductive status almost never getting adopted out; the shelter does not invest resources to figure out if they are intact or altered (possibly because they are already slated for transfer).<br>
<br>
<figure>
  <div align="center"><img src="https://github.com/KathrynMercer/Austin-Animal-Center-Data/blob/main/EDA%20Intake%20Repro%20Status.png" alt="Stacked Bar Chart of Intake Reproductive Status vs Outcome Type" height = "550" width = "750"/><br>
  <figcaption>2. Intake Reproductive Status vs Outcome Type.</figcaption></div>
</figure><br>
Above, we see that the adoption rates of cats that arrive at the shelter already altered and those of cats that arrive intact. The difference in their adoption rates is not nearly so large. But there does seem to be a difference between the "Return to Owner" rates of these cats. Again leaning on some domain knowledge, I suspect this difference has more to do with a confounding factor than a true difference from reproductive status. Microchips play a huge role in identifying the owner of a lost pet and getting them reunited. Many cats have microchips placed by their vet when they are spayed or neutered. So if a cat is brought to the shelter but has already been spayed/neutered, I suspect there is a higher chance that they have a microchip than if they are intact. But the presence or absence of a microchip is not part of this dataset, so we can't know for sure.<br>
<br>
<figure>
  <div align="center"><img src="https://github.com/KathrynMercer/Austin-Animal-Center-Data/blob/main/EDA%20Intake%20Condition%20Contingency.png" alt="Contingency Table of Intake Condition vs Outcome Type" height = "500" width = "750"/><br>
  <figcaption>3. Intake Condition vs Outcome Type.</figcaption></div>
</figure><br>
This contingency table shows the rates (expressed as percentages) of cats with various intake conditions having each outcome type. If cats with "Normal" intake conditions are considered the baseline, we see that most other conditions have lower rates of adoption, which makes logical sense. The non-healthy animals have higher rates of death, euthanasia, and transfer (presumably to a rescue better able to provide care). This is definitely a good candidate for a prediction model, whether as a simple Is_Normal boolean or keeping the granularity of the full range of abnormal intake conditions.<br>
<br>
<figure>
  <div align="center"><img src="https://github.com/KathrynMercer/Austin-Animal-Center-Data/blob/main/EDA%20Intake%20Age%20Violin.png" alt="Violin Plot of Intake Age vs Outcome Type" height = "500" width = "750"/><br>
  <figcaption>4. Intake Age vs Outcome Type.</figcaption></div>
</figure><br>
Logic suggests an older animal might have a harder time getting adopted. Because so many of the animals taken in by the shelter skew younger in age (<5.5 years old or 2000 days), most of the data is concentrated there. Still, you can see the median ages of euthanized pets and pets returned to their owner are older than the median adoption age and transfer ages. The IQR of pets who end up euthanized or returned to their owners is also much wider than for Transfer/Adoption/Death outcomes.<br>   
<br>
Additionally, the animals that have died skew younger than those adopted, which suggests the neonatal death rate is higher than for other groups. That's also supported by the contingency table above, which shows "nursing juvenile" cats have a 3% death rate (3x higher than "Normal" intake baseline). Age may be worth considering in a model, but this shelter seems to do a good job of placing animals regardless of their numeric age (whether that's in a new home or back with their previous owners).<br>
<br>
<figure>
  <div align="center"><img src="https://github.com/KathrynMercer/Austin-Animal-Center-Data/blob/main/EDA%20Base%20Color%20Contingency.png" alt="Contingency Table of Base Color vs Outcome Type" height = "600" width = "750"/><br>
  <figcaption>5. Base Color vs Outcome Type.</figcaption></div>
</figure><br>
This contingency table shows the rates (expressed as percentages) of cats with various base colors having each outcome type. The base adoption rate of any cat at this shelter is 51%, so that will be considered the "baseline" for comparison. We can see that the "flashy" cats like Torties, Calicos, and Cream/Off-White (which is the base color of Siamese-type cats) are adopted at a higher rate than average. Conversely, pure white cats, black cats, or brown cats have lower rates of adoption. The differences are not large, but this is a reasonable potential feature for modeling.<br>
<br>
<figure>
  <div align="center"><img src="https://github.com/KathrynMercer/Austin-Animal-Center-Data/blob/main/EDA%20Pattern%20Contingency.png" alt="Contingency Table of Coat Pattern vs Outcome Type" height = "350" width = "750"/><br>
  <figcaption>6. Coat Pattern vs Outcome Type.</figcaption></div>
</figure><br>
This contingency table shows the rates (expressed as percentages) of cats with various coat patterns having each outcome type. The base adoption rate of any cat at this shelter is 51%, so that will be considered the "baseline" for comparison. Again, the "flashy" cats have an advantage over more common patterns. Torties, Calicos, and Points (which is the pattern of Siamese-type cats) are adopted at a higher rate than average. Solid coated cats (ex. pure black) have a disadvantage. The adoption rate of Tabby cats matches the baseline 51%, but tabby is one of the most common patterns, so that makes sense whether or not the tabby pattern is desirable to owners. 

#### Machine Learning Preprocessing:
Since many of the features I identified as potentially significant above are categorical and high granularity (including the breed data I haven't shown here), I decided to train a machine learning model rather than focusing only on variables that could be relaxed into a regression. Before jumping to that, I needed to do some more preprocessing of my numerical values, normalizing, balancing, and encoding the categorical variables. 

This data is heavily unbalanced around Outcome Type; it's heavily skewed toward adoption and transfer (with return to owner, euthanasia, and died making up much smaller percentages). The easiest way to address this is to refine the question to "what features are associated with a cat getting adopted (vs not)" as this allows for binary classification, which would combine the large Transfer Outcome with all the smaller outcomes and yield a less unbalanced dataset. I also considered keeping all the Outcome Types but managing the imbalance by undersampling the overrepresented classes, which would ultimately have yielded a dataset containing 15k+ samples. Lastly, I considered weighting the underrepresented classes more heavily, so they would have more consideration by the model.

Ultimately, I decided to use a combination of oversampling the underrepresented groups and undersampling the overrepresented groups. This means my model will be partly trained on fabricated data. Because my data contains a mix of categorical & numerical data, I'm using SMOTE-NC, which will make sure the fabricated data does not become nonsensical by trying to find intermediate values between categories (ie. a new breed "between" Abyssinian and Siamese). Because I'm oversampling, it was important to split the data first, so I only oversampled on my training data. 

For now, I'm going to train a model to answer the binary question (still using SMOTE-NC to make sure the sampling for adopted and not-adopted is balanced), and I can expand the model to a multiclass classification architecture later if it underperforms or if I want to investigate the other outcomes more specifically.

&emsp;Tasks: 

&emsp;&emsp; - Converted month data from Timestamp to cyclically encoded numerical

&emsp;&emsp; - Engineered feature "is_weekend" to encode whether the intake/outcome occurred on a weekend (vs weekday) 

&emsp;&emsp; - Shuffled data

&emsp;&emsp; - Scaled numerical data

&emsp;&emsp; - Encoded Outcome Type into binary (Adopted or Not)

&emsp;&emsp; - Split data into training, validation, and test data sets

&emsp;&emsp; - SMOTE-NC oversampling to manage data imbalance

---
  
#### **Machine Learning:** 
[**First model:**](https://github.com/KathrynMercer/Austin-Animal-Center-Data/commit/7c0f2c73d01475438c5dc08a949941d075edd29d) I started the ML model with a basic **Pytorch linear regression model** using **One Hot Encoding** for my categorical variables, using a learning rate of 0.1. This achieved an 87% accuracy (on validation data). Examining the weights of coefficients reveals several positive contributors: outcome occurring on a weekend, cat being female, cat being tortie/calico. Interestingly, being a purebred cat didn't have as much impact as I was expecting it might. And negative contributors: cat being male, having the tabby coat pattern, or having brown or orange as a base color (brown and orange are primarily seen in tabby cats).

[**Second model:**](https://github.com/KathrynMercer/Austin-Animal-Center-Data/commit/63884e49ca15a69e5bd31b1ff3b46fc2aefcbf33) Because my data doesn't meet the requirements for linear regression (particularly multicollinearity considering that two of the most popular coat colors - Calico and Torties - are almost always female cats), I upgraded the model to a **neural network** with 10 hidden layers and settled on a learning rate of 0.087. Though I lost the ability to directly measure how each feature impacts a cat's likelihood of adoption, this model allows for more complication feature interaction, is less likely to be harmed by multicollinearity, and is more accurate (91% on validation data). Still, I think I can further improve upon this model by switching my categorical encoding from one-hot to embedding and/or by switching the model to a random forest/decision tree model (which would restore the interpretability of the weights). 

To be continued...

