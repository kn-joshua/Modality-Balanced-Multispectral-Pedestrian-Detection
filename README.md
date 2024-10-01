# Multimodal-Balanced-Multispectral-Pedestrian-Detection-Implementation
This repository is a implementation of the paper "Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems" by Kailai Zhou, Linsen Chen and Xun Cao published in the ECCV 2020 Conference (arXiv Link: https://arxiv.org/pdf/2008.03043v2). Further, I have summarized the paper for easier understanding, targeted to a beginner audience.

## Modalities in Multispectral Pedestrian Detection

To detect pedestrians both during the day and at night, we use two types of images: **RGB images** (which are regular images from a camera) and **thermal images** (which capture heat). This is where the term **"multispectral"** comes in, since we’re dealing with two different types of data — RGB and thermal. Each has its own strengths and weaknesses, and understanding these will help us see why using both is important.

### RGB Images

#### Pros

- RGB images provide clear visuals during the day, capturing color and textures that make it easy to see details like clothing, background, and objects when there’s good lighting.
- They offer high detail in well-lit areas, producing sharp, high-resolution images that help distinguish between different objects or people.
- RGB cameras are widely available and relatively inexpensive.

#### Cons

- However, RGB images struggle in low-light conditions; they can’t pick up much at night or in poor lighting since they rely on visible light.
- Weather and light changes, like fog, shadows, or glare from lights, can also interfere with what the RGB camera sees, complicating pedestrian detection.

### Thermal Images

#### Pros

- Thermal images work well in any lighting condition, capturing heat and allowing them to detect people even in complete darkness.
- They are less affected by bad weather; fog, smoke, or other visual obstructions don’t impact their performance as much.
- Thermal cameras make it easier to spot people, as humans emit heat and stand out clearly in thermal images.

#### Cons

- On the downside, thermal images lack detail; they don’t show textures or colors, making it harder to differentiate between different objects, such as a person versus a hot car engine.
- Thermal cameras typically have lower resolution, which means they don’t capture as much detail, especially when looking at things from a distance.
- If the environment is very hot, the contrast in thermal images can decrease, making it harder to spot people.

Both RGB and thermal images have their pros and cons. By using them together, we can get the best of both worlds. Multispectral pedestrian detection combines both types of images to ensure we can spot pedestrians in all kinds of environments.

## Problems in Balancing these Modalities

When combining RGB and thermal images for pedestrian detection, several challenges arise:

- The **foreground-to-background imbalance** is a significant issue. Typically, there are far more background examples than positive detections of pedestrians. This imbalance can make it difficult for the model to learn effectively since it may get overwhelmed by negative examples.

- **Illumination modality imbalance** is another challenge. RGB images perform well in well-lit conditions, capturing clear details of pedestrians. However, they struggle in low light or nighttime settings. Conversely, thermal images excel in darkness, detecting heat signatures regardless of lighting, but they provide less detail about the pedestrian’s appearance. This difference can lead to varying confidence scores and an uneven contribution to detection results.

- The **feature modality imbalance** emerges when the features extracted from RGB and thermal images do not align properly. For example, RGB images capture intricate details like skin tone and clothing, which are important for identifying pedestrians. Thermal images, however, lack these features, focusing instead on the shape and heat emitted by the person. This discrepancy can result in an uneven representation of features when the two modalities are combined.

- Misalignment between RGB and thermal images can also occur due to the fact that they are often captured at different times. This misalignment can lead to unbalanced feature representation, where the fixed receptive fields of convolutional kernels fail to capture essential details from both modalities.

## Failure of Existing Works

- The **baseline model F + T + THOG** is an initial approach that combines RGB (color) images with thermal images. It builds upon the Aggregated Channel Features (ACF) method by adding a thermal channel to detect pedestrians. However, this model struggles when lighting conditions change, such as in low-light situations.

- **CNN-based methods** refer to techniques that use Convolutional Neural Networks (CNNs), a type of deep learning model. These methods often use a two-stream architecture, where RGB and thermal feature maps are combined. While this approach can improve performance, it can also introduce redundant features, making it difficult for the model to clearly distinguish between the important information from each modality (RGB and thermal).

- The **Illumination-aware Faster R-CNN** is an advanced method that tries to adjust how it combines color and thermal images based on lighting conditions. Although this model merges the two types of data to generate confidence scores (which indicate how likely it is that a detected object is a pedestrian), it lacks effective ways to align the features from both types of images. This limitation means the model doesn't fully take advantage of the strengths of each modality.

- **Gated Fusion Units (GFUs)** are specialized components used in the **GFD-SSD** model, which learn how to combine features from the RGB and thermal images. However, GFUs do not effectively address the imbalance between the two modalities, which can lead to decreased accuracy in detecting pedestrians.

- The **Cross-Modality Interactive Attention Network (CIAN)** attempts to align the features from both RGB and thermal images at the middle level, meaning it tries to combine them after some processing has occurred. However, it does not fully resolve the misalignment issues between the two types of images, which negatively impacts the overall performance of pedestrian detection.

- Many existing models do not adequately integrate the diverse features from RGB and thermal images. This lack of integration means that the models may not perform well in real-world situations where lighting and visibility conditions vary, ultimately leading to less accurate pedestrian detection.

## Main Contributions of the Paper

- The paper identifies **modality imbalance problems** specific to multispectral pedestrian detection. Previous models struggled with effectively combining RGB and thermal images due to inconsistencies during optimization. This issue negatively affected the performance of existing detectors, especially in varying lighting conditions.

- To address these challenges, the paper introduces the **Modality Balance Network (MBNet)**, a one-stage detector designed to enhance the integration of features from different modalities. MBNet incorporates two key modules: the **Differential Modality Aware Fusion (DMAF)** module and the **Illumination Aware Feature Alignment (IAFA)** module. 

- The **DMAF module** explicitly integrates features from both RGB and thermal modalities, ensuring that the contributions of each modality are balanced and effectively utilized during detection. This approach helps overcome the limitations of previous models, which often led to the dominance of one modality over the other.

- The **IAFA module** aligns features from both modalities based on the illumination conditions, allowing the network to adaptively optimize its performance in different environments. This addresses the problem where previous methods inadequately merged information under varying lighting scenarios.

- The backbone of MBNet is built on **ResNet**, a popular architecture known for its efficiency in deep learning tasks. 

## Proposed Modality Balanced Model

The model proposed in the paper consists of two main modules. In this section, we shall describe the essential functioning of each module, without going into the mathematical details. The next section shall go into the details of the mathematics behind both of them.

### DMAF (Differential Modality Aware Fusion) Module

_TLDR: This module basically enhances our feature maps for both thermal and RGB maps through some computations such as taking the difference of both, followed by max-pooling, and passing it through an activation function. To the thermal feature map, the enhanced RGB map is added and vice versa: to the RGB map, the enhanced thermal map is added._

As we mentioned earlier, simply combining the features from the RGB (color) and thermal (heat) images can be problematic. This approach often leads to confusion in understanding how the two types of information complement each other. It mixes helpful signals with irrelevant noise and makes it hard to clearly identify important features.

To tackle the problem of feature modality imbalance, we introduce the **Differential Modality Aware Fusion (DMAF) module**. Here’s how it works, step by step:

1. **Enhancing Features**: The DMAF module improves features from one type of image by using information from the other type. It does this by looking at the differences between the RGB and thermal images.

2. **Separating Features**: The features from the RGB and thermal images are divided into two categories: common features (which both images share) and differential features (which are unique to each image). This separation allows the module to enhance the important features more effectively.

3. **Weighting by Importance**: The DMAF module gives different importance to each feature channel. This means that it can highlight the most useful features from both types of images, making sure the final output is more informative.

4. **Integration Throughout the Network**: The DMAF module is included in every part of the ResNet architecture. This allows the RGB and thermal features to interact continuously as the model processes the data, improving how the model learns.

5. **Reducing Redundancy**: By allowing the two types of images to work together, the DMAF module minimizes unnecessary repetition in the features the model learns. This results in a clearer understanding of both pedestrian and background features.

The inspiration for the DMAF module comes from differential amplifier circuits. In these circuits, common signals (similar signals from both images) are reduced, while different signals (the unique characteristics from each image) are boosted. This design helps preserve the original features while adjusting them based on the differences between the RGB and thermal images.

### IAFA (Illumination Aware Feature Alignment) Module

_TLDR: IAFA Module actually detects the pedestrians. It does through multiple stages: first to detect whether its day/night, then it predicts what is more trustable - RGB or Thermal based on the previous result, followed by aligning the two feature maps using bilinear interpolation. Finally, it detects the pedestrians through a 2 stage approach: the AP Stage and the IAFA Stage._

The **Illumination Aware Feature Alignment (IAFA) module** addresses the challenge of detecting pedestrians under varying lighting conditions. Different types of light can significantly affect how objects appear in images, which can lead to confusion for models that try to identify them. The IAFA module aims to align the features of RGB (color) and thermal images, ensuring that the model performs well regardless of the illumination present.

Here's how the IAFA module works:

1. **Problem Addressed**: The IAFA module is designed to adapt the model to different lighting conditions. Since pedestrian detection can be difficult when lighting changes (e.g., from day to night or in shadowy areas), this module helps ensure the model remains effective across these variations.

2. **Model Structure**: 
   - The IAFA module uses only RGB images to capture illumination values because thermal images are not as effective at showing environmental light conditions.
   - The RGB images are resized to 56 × 56 pixels, which reduces the amount of data to process while still providing necessary details.
   - A small neural network is utilized to analyze the illumination data. This network consists of:
     - **Two convolutional layers** that help extract relevant features from the RGB images.
     - **Three fully-connected layers** that further process these features, allowing the network to learn how to best identify illumination conditions.

3. **Feature Processing**: After the convolutional layers, a ReLU activation function is applied, followed by a max pooling layer. This setup helps compress the data and focus on the most important features that indicate lighting conditions.

4. **Training the Model**: The IAFA module is trained to minimize the difference between the predicted illumination values and the actual lighting conditions, enabling the model to learn and improve its ability to detect pedestrians effectively in different lighting scenarios.

## Mathematical Details 

In this section we will describe the mathematical equations that summarise the functioning of each module.

### DMAF (Differential Modality Aware Fusion) Module

To understand how the **Differential Modality Aware Fusion (DMAF)** module works, let's begin with the main equation that governs its operation:

$$ F_T' = F_T + F(F_T \oplus F_{RD}) = F_T + F(F_T \oplus (\sigma(GAP(F_D)) \odot F_R))  $$

$$ F_R' = F_R + F(F_R \oplus F_{TD}) = F_R + F(F_R \oplus (\sigma(GAP(F_D)) \odot  F_T)) $$

The goal of this equation is to improve the thermal feature map $$\( F_T \)$$ by incorporating useful information from the RGB feature map \( F_R \). The two modalities—RGB and thermal—offer complementary information that enhances feature representation in the model.

#### **What is $$\( F_T \)$$?**
- **$$\( F_T \)$$** is the **thermal feature map**. It captures essential heat-related information extracted from thermal images.

#### **What is $$\( F_R \)$$?**
- **$$\( F_R \)$$** is the **RGB feature map**. It represents critical information derived from color images, including color and texture.
  
#### **What is the function $$\( F \)$$?**
- **$$\( F \)$$** is a function that represents the residual learning mechanism. This function enhances the original feature maps by allowing them to learn additional information from the other modality.

#### **What are We Trying to Achieve?**
- We aim to enhance the thermal feature map $$\( F_T \)$$ by adding insightful information from the RGB feature map $$\( F_R \)$$. This allows the model to leverage the strengths of both modalities, improving overall performance.

#### **Step 1: The Basic Concept**
$$ F'_T = F_T + (\text{Useful information from RGB}) $$
- Here, $$\( F'_T \)$$ denotes the **improved thermal feature map**. By adding valuable RGB information to $$\( F_T \)$$, we aim to refine it.

#### **Step 2: Finding the Useful Information**
- The term $$\( F_T \oplus F_{RD} \)$$ indicates the combination of the original thermal features $$\( F_T \)$$ with RGB features $$\( F_{RD} \)$$, enhanced by the differences between the two modalities. The symbol $$\( \oplus \)$$ denotes this combination.

#### **Step 3: What’s $$\( F_{RD} \)$$ ?**
- **$$\( F_{RD} \)$$** represents the **differential information**—essentially, the unique aspects that distinguish the RGB features from the thermal features. 

#### **Step 4: Why Do We Need the Difference $$\( F_D \)$$?**
- **Complementary Information**: The RGB and thermal images capture different characteristics; RGB images convey color and texture while thermal images show heat signatures. Focusing on the differences allows us to integrate meaningful information from both modalities.
  
- **Guiding the Fusion Process**: The differences help determine how RGB features should influence the thermal features. This ensures that the enhancement is based on significant differences rather than arbitrary combinations.

#### **Step 5: Calculating $$\( F_{RD} \)$$**
$$
F_{RD} = \sigma(GAP(F_D)) \cdot F_R
$$
- **$$\( F_D \)$$** is the **difference between RGB and thermal features**. This difference helps capture the unique contributions of each modality.

- **$$GAP(F_D)$$** calculates the **average difference** across the entire image, providing a holistic view of the disparities.
  
- **$$\( \sigma \)$$**, the **tanh function**, normalizes this difference into a range from -1 to 1, highlighting the most crucial differences.

- Finally, these differences are multiplied by the RGB features $$\( F_R \)$$ to generate useful information for enhancing the thermal image.

#### **Step 6: Integrating Everything**
- After calculating the useful differences, we incorporate this information back into the original thermal feature map $$\( F_T \)$$. This integration boosts the thermal map, making it more effective by utilizing insights from both RGB and thermal images.

### IAFA (Illumination Aware Feature Alignment) Module

The **Illumination Aware Feature Alignment (IAFA)** module is designed to help the model adapt to different lighting conditions, such as day and night, by aligning RGB and thermal modalities and dynamically adjusting how much the model relies on each type of image based on illumination. This section will explain the module step-by-step, answering all relevant queries and addressing the underlying mathematical principles.

#### Predicting Illumination

The first task of the IAFA module is to determine whether the image is captured during the day or at night. Since thermal images do not convey much information about ambient light, the module relies on RGB images to assess the illumination condition.

A small neural network is used to make this prediction. It takes the RGB image, resizes it to 56x56 pixels, and processes it through two convolutional layers and three fully connected layers. The network then produces two values:

- $$\( w_d \)$$: the probability that it’s daytime.
- $$\( w_n \)$$: the probability that it’s nighttime.

These probabilities are normalized so that:

$$
w_d + w_n = 1
$$

The loss function for this prediction, called the **illumination loss** $$\( L_I \)$$, is calculated as follows:

$$
L_I = -w_{bd} \cdot \log(w_d) - w_{bn} \cdot \log(w_n)
$$

Here, $$\( w_{bd} \)$$ and $$\( w_{bn} \)$$ are the ground truth values (whether it was actually day or night). The network is trained to minimize this loss, improving its ability to differentiate between day and night over time.

#### Re-weighting RGB and Thermal Features

Once the system has predicted the illumination condition, it adjusts how much it relies on the RGB and thermal images. During the day, the RGB image provides more useful information, while at night, the thermal image is more reliable.

To achieve this, the IAFA module computes two weights:

1. **RGB weight $$\( w_r \)$$**
2. **Thermal weight $$\( w_t \)$$**

The formula for the RGB weight is:

$$
w_r = \left( \frac{w_d - w_n}{2} \right) \cdot \left( \alpha_w \cdot |w| + \gamma_w \right) + \frac{1}{2}
$$

Where:
- $$\( w_d \)$$ is the day probability.
- $$\( w_n \)$$ is the night probability.
- $$\( |w| \)$$ is the confidence of the system in its day/night prediction (the further from 0.5, the more confident it is).
- $$\( \alpha_w \)$$ and $$\( \gamma_w \)$$ are learnable parameters, adjusted during training.

This formula ensures that during the day $$(\( w_d > w_n \))$$, $$\( w_r \)$$ becomes larger, meaning the system trusts the RGB image more. At night (\( w_n > w_d \)), \( w_r \) decreases, and the system will rely more on the thermal image.

For the thermal image, the weight $$\( w_t \)$$ is simply:

$$
w_t = 1 - w_r
$$

This ensures that the total reliance on both modalities always sums to 1, maintaining balance between RGB and thermal features.

#### Alignment of RGB and Thermal Images

Since the RGB and thermal cameras may not always be perfectly aligned (due to differences in camera setup or calibration), the IAFA module corrects these misalignments using a small offset for each pixel.

The new pixel location is given by:

$$
(x', y') = (x + dx, y + dy)
$$

Where:
- $$\( (x, y) \)$$ is the original pixel location.
- $$\( (dx, dy) \)$$ is the offset predicted by the system to correct the misalignment.
- $$\( (x', y') \)$$ is the new, corrected location.

The system uses **bilinear interpolation** to smooth out these shifts. Bilinear interpolation works by taking the average of nearby pixels, making the corrected image look smooth without sharp edges or distortions. This ensures that the RGB and thermal features are well aligned before being used by the model for detection.

#### Why Is This Alignment Necessary?

When the RGB and thermal images are misaligned, they provide inconsistent features to the model, which can lead to incorrect predictions. Aligning them ensures that the features from both modalities are properly combined, making the model more accurate.

#### Two-Stage Detection: AP and IAFC Stages

The IAFA module works within a two-stage detection framework:

1. **Anchor Proposal (AP) Stage**: This stage makes an initial guess about where pedestrians might be located. These guesses are referred to as “anchors.”
2. **Illumination Aware Feature Complement (IAFC) Stage**: This stage refines the anchors based on the illumination condition and the alignment between RGB and thermal images.

After the refinement, the system calculates a final confidence score for each anchor:

$$
s_{final} = s_0 \times (w_r \cdot s_r + w_t \cdot s_t)
$$

Where:
- $$\( s_0 \)$$ is the confidence score from the AP stage.
- $$\( s_r \)$$ and $$\( s_t \)$$ are the confidence scores based on the RGB and thermal images.
- $$\( w_r \)$$ and $$\( w_t \)$$ adjust the influence of each modality based on the illumination.

For the final bounding box location, the system combines the predictions from both stages:

$$
t_{final} = t_0 + t_1
$$

Where:
- $$\( t_0 \)$$ is the initial bounding box prediction from the AP stage.
- $$\( t_1 \)$$ is the refined bounding box from the IAFC stage.

#### Classification and Regression Losses

To train the IAFA module, the system minimizes two types of loss functions:

1. **Classification Loss** $$\( L_{cls} \)$$, which measures how well the system differentiates between objects (pedestrians) and background.
2. **Regression Loss** $$\( L_{reg} \)$$, which measures how accurately the system predicts the location of objects.

The classification loss is applied at both the AP and IAFC stages:

$$
L_{cls} = -\alpha \sum_{i \in S+} (1 - s_i)^\gamma \log(s_i) - (1 - \alpha) \sum_{i \in S-} s_i^\gamma \log(1 - s_i)
$$

Where:
- $$\( S+ \)$$ is the set of positive samples (where pedestrians are present).
- $$\( S- \)$$ is the set of negative samples (background).

The regression loss is applied when pedestrians are detected:

$$
L_{reg} = \sum (b_{gt} - b_{pred})^2
$$

Where:
- $$\( b_{gt} \)$$ is the ground truth bounding box.
- $$\( b_{pred} \)$$ is the predicted bounding box.

#### Total Loss Function

The final loss function combines all the losses from both detection stages:

$$
L = L_I + L_{cls0} + L_{cls1} + [y=1] L_{reg0} + [y=1] L_{reg1}
$$

Where:
- $$\( L_I \)$$ is the illumination loss.
- $$\( L_{cls0} \)$$ and $$\( L_{cls1} \)$$ are the classification losses from the AP and IAFC stages.
- $$\( L_{reg0} \)$$ and $$\( L_{reg1} \)$$ are the regression losses, applied only when objects are detected (\( y=1 \)).


#### Queries I Had While Trying to Understand

Here are answers to some questions that arise when studying this module:

1. **Why do we add the bias of 1/2 in formula for $$\( w_r \)$$?**

   The bias ensures that the weight $$\( w_r \)$$ remains within a reasonable range (typically between 0 and 1). Without this, $$\( w_r \)$$ could become negative or too skewed toward RGB or thermal images. The $$\( \frac{1}{2} \)$$ ensures balance between the two modalities, preventing either from being overly dominant.

2. **What is bilinear interpolation?**

   Bilinear interpolation is a simple algorithm used to smooth out pixel values when they are shifted by the offsets $$\( dx \)$$ and $$\( dy \)$$. It estimates the value of a pixel by taking a weighted average of the surrounding pixels, ensuring that the image remains smooth even after misalignment correction.

3. **In the formula for $$\( w_r \)$$, what is $$\( w \)$$?**

   The term $$\( w \)$$ represents the confidence of the system in its day/night prediction. It measures how much the system’s prediction deviates from the neutral state (where $$\( w_d = w_n = 0.5 \)$$). The further $$\( w \)$$ is from 0.5, the more confident the system is in its day/night prediction.

4. **How do the two classification losses $$\( L_{cls0} \)$$ and $$\( L_{cls1} \)$$ work?**

   The two classification losses correspond to the two stages in the detection process. $$\( L_{cls0} \)$$ comes from the Anchor Proposal (AP) stage, which makes rough guesses about object locations. $$\( L_{cls1} \)$$ comes from the IAFC stage, which refines these predictions. Both losses are necessary because they help the system improve at both rough detection and refined detection.

5. **Why do we need both losses?**

   The AP stage provides initial estimates, and the IAFC stage refines these. Optimizing both stages allows the system to improve its initial guesses and then refine them for better accuracy. Without $$\( L_{cls0} \)$$ , the initial guesses could be poor, leading to more work for the IAFC stage. The reason we care about both is that the quality of the initial proposals (AP stage) affects the overall performance. Even if the IAFC stage improves the predictions, the model learns better if it also improves the initial guesses in the AP stage. Both stages contribute to learning.

6. **What does the term  $$\( [y=1] L_{reg0} \)$$ mean?**

   This notation means that the regression loss $$\( L_{reg0} \)$$ is only applied when an object (like a pedestrian) is detected. If there are no objects, the regression loss is not calculated. This avoids unnecessary computations and ensures that bounding box refinement only happens when there’s something to detect.

7. **What if there are multiple pedestrians?**

   If there are multiple pedestrians, the system calculates the regression loss for each predicted bounding box. The total regression loss is the sum of the individual losses for each detected object. This ensures that all objects are localized accurately.




