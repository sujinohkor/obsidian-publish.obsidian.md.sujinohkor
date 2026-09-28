* [EyeFormer: Predicting Personalized Scanpaths with Transformer-Guided Reinforcement Learning](https://yuejiang-nj.github.io/Publications/2024UIST_EyeFormer/project_page/main.html)
* [[Note]]
* [Colab: EyeFormer-0915](https://colab.research.google.com/drive/1A7al__HMSLrPm3jDUk5l_MoDsdCTDvWq?usp=drive_link)

# Paper Excerpts
## Abstract
From a visual-perception perspective, modern graphical user interfaces (GUIs) comprise a complex graphics-rich two-dimensional visuospatial arrangement of text, images, and interactive objects such as buttons and menus. 

While existing models can accurately predict regions and objects that are likely to attract attention “on average”, no scanpath model has been capable of predicting scanpaths for an individual. 

To close this gap, we introduce **EyeFormer,** which <u>utilizes a Transformer architecture as a policy network to guide a deep reinforcement learning algorithm that predicts gaze locations.</u> 

Our model offers the unique capability of <u>producing personalized predictions</u> when given a few user scanpath samples. <u>It can predict full scanpath information, including fixation positions and durations, across individuals and various stimulus types.</u> 

Additionally, we demonstrate applications in GUI layout optimization driven by our model.

## 1. Introduction
Prior work has focused primarily on **saliency maps,** which represent <u>eye-movement data via density maps</u> for the images [20]. <u>However, as static representations, these overlook temporal information.</u> In contrast, **scanpaths** <u>contain a wealth of information on fixations, retaining details of the order in which objects and regions are attended to, accompanied by the respective duration [7, 24, 41].</u> Scanpaths are, therefore, first-order models of human vision from which second-order measurements such as saliency maps can be derived, while the converse is not true.

To address this gap, we present **EyeFormer,** <u>a scanpath model for free-viewing tasks. It can accurately predict both population- and individual-level spatiotemporal characteristics of viewing behaviors across multiple stimulus types. We formulated fixations’ positioning as a reinforcement learning (RL) problem and used a Transformer architecture as a policy network guiding the selection of each sequence’s subsequent fixation.</u> Transformers have proven effective in various tasks, in fields from language to vision [6, 17, 18, 63]. Their capability of modeling long sequences [36] makes them especially suitable for scanpath prediction.

**EyeFormer’s Transformer-guided deep RL approach** was designed to address <u>three critical shortcomings of current approaches.</u>
* **Firstly,** <u>predicting the order of fixations from saliency maps via their probability distribution [10, 37] is inherently hard</u> because such representations <u>lack temporal information.</u>
* **The second issue** stems from <u>post-processing steps</u> implemented to prevent the excessive clustering of fixations that is characteristic of density-based approaches [24] (for instance, <u>inhibition of return [20] is applied to prevent repeated fixation at a previously identified position in a saliency map).</u> Because these steps are not learnable from data [13], one <u>cannot formulate proper loss terms</u> derived from them.
* **Thirdly,** though recent advances such as PathGAN [3] have brought progress toward handling fixation duration, accuracy in predicting fixation points remains limited since these techniques often generate points outside the areas of interest.

**EyeFormer is the first model to predict full scanpaths at both individual and population level, including fixations with coordinates and duration both.** It is unique in its ability to <u>predict an individual’s viewing behaviors when given a few sample scanpaths from such a viewer.</u> Moreover, we find that EyeFormer compares favorably to prior scanpath models by the vast majority of metrics for both GUIs and natural scenes in its population-level scanpath prediction. It accurately predicts both spatial order (“where”) and temporal (order and duration) characteristics of scanpaths with both of these scene types.

Further, we develop an application of personalized scanpath prediction for creating personalized GUI layouts by considering the viewing order and fixation density of GUI elements. In addition, we can generate a single optimized GUI layout that shows minimal variability across individuals, to attract attention to desired elements. We have made our code available at https://github.com/YueJiang-nj/EyeFormer-UIST2024.

**In summary, this paper makes the following contributions:**
1) We propose EyeFormer, a deep RL solution incorporating the Transformer architecture that <u>predicts both spatial and temporal characteristics of scanpaths,</u> thus yielding a comprehensive understanding of viewers’ viewing behaviors.
2) It shows how our model <u>generates personalized scanpaths</u> via only a few scanpath samples from the relevant viewer, whereby the model can capture and reflect each user’s viewing behaviors and preferences.
3) We present quantitative and qualitative evaluations demonstrating that the proposed model performs as well as or better than the state-of-the-art models at population-level scanpath prediction for GUIs and natural scenes.
4) We demonstrate an application of personalized GUI optimization facilitated by personalized scanpath prediction.

## 2. Related Work
**Scanpath models** <u>predict sequences of fixations for a given image.</u> **This task** is <u>more challenging</u> than predicting (dense) saliency maps <u>because the order of the (discrete) fixations must be predicted.</u> All previous research has concentrated on modeling scanpath patterns at population level (i.e., employing an “average-user” model), <u>while none has focused on the prediction of personalized scanpaths.</u> Therefore, we conducted a comprehensive review of the approaches to population-level scanpath prediction as groundwork for extending these techniques to individuals’ level by using a novel Transformer-based architecture.

**Prior work can be classed into three main groups** on the basis of how they have attempted to <u>derive sequential information:</u>
1) <u>computing it post hoc</u> from densities captured in saliency maps,
2) <u>directly predicting sequences,</u> and
3) <u>formulating this as a sequential control problem via RL.</u>

Our evaluation compares our EyeFormer model against several models mentioned next.

### 2.1. Saliency-Map-Based Scanpath Prediction
**Saliency maps,** <u>although they do not explicitly contain temporal information, can be used to estimate scanpaths.</u> Itti et al. [20] introduced an **Inhibition of Return (IOR)** mechanism to this end. <u>It samples a starting fixation and “discourages” future fixations from returning to it, thus producing a richer sequence.</u>

While several studies have refined the idea [2, 8, 33, 37, 48, 65, 68, 69], methods of this sort **still face challenges in three respects:**

1) they (i) <u>neglect some key temporal factors,</u> specifically <u>fixation duration;</u>
2) (ii) <u>lack a coherent ranking order</u> for the fixations; 
3) and (iii) <u>cannot serve in loss terms, since they are non-differentiable.</u>

### 2.2. Predicting Fixation Sequences
Some attempts to resolve these challenges have entailed sequentially sampling fixations from pre-generated Gaussian distributions and integrating well-designed supervised loss terms. This strategic choice enforces a meaningful order among the fixation points.

For instance, **IOR-ROI [8, 54], ScanpathNet [10], and Visual ScanPath Transformer [45]** <u>predict fixation distributions through a parameterized Gaussian mixture for generating such distributions.</u> **GazeFormer [41]** <u>incorporates a Transformer-based architecture for goal-oriented viewing tasks,</u> and **ScanDMM [53]** <u>utilizes a Markov model to represent fixation distributions.</u> **These models suffer from accumulation of error [46], whereby errors in previously generated points affect the prediction of the following points.**

Other models directly predict fixation sequences. **Verma and Sen [62]** <u>predicted fixations by means of a grid-based representation in which each fixation point is tied to a specific region.</u> **PathGAN [3] and ScanGAN [38],** in turn, <u>apply a GAN-based architecture to generate fixation sequences;</u> **however, GAN-based scanpath models show such limitations as clustering of fixation points toward the image’s center and reduced accuracy in predicting fixation points (which sometimes get placed outside the areas of interest).**

Also noteworthy is **NeVA [51],** which <u>addresses downstream visual tasks with unseen datasets by relying on existing pre-trained models for the task</u> rather than simulating human scanpaths.

### 2.3. Reinforcement Learning for Scanpath Prediction
Studies have examined RL’s potential in formulating scanpaths as a sequential control problem [5, 42]. 

For example, **Minut and Mahadevan [40]** proposed <u>an RL model for visual search tasks wherein an agent learns to focus on relevant areas to locate a target object in a cluttered environment.</u> **Ognibene et al. [43]** employed <u>RL using an eye-centered potential action map that accumulates possible target locations over fixations,</u> and **Yang et al. [72]** used <u>inverse RL for predicting the scanpaths involved in a visual search task.</u> In other work, **Xu et al. [71]** applied <u>deep RL specifically to predict head-movement-related scanpaths for panoramic videos.</u>

Recent work by **Chen et al. [7]** <u>discretized fixation positions by representing each image as a grid and predicting the cell in the grid corresponding to a particular fixation.</u> Inspired by policy gradient methods applied in discrete token generation for visual captioning [47], Chen et al. adopted their policy gradient to optimize for non-differentiable metrics in their discrete tokenizing. 

**The position discretization** offers <u>the advantage of optimizing a finite set of discrete actions rather than a continuous and hence infinite space.</u> However, **the artificiality of discretization** brings <u>coarser fixation representation, which leads in turn to loss of precision/information.</u> **The challenge with continuous control** is that <u>a continuous range of control encompasses an infinitude of feasible actions [57].</u> 

Against this backdrop, **we construct our fixation prediction as a continuous value generation task** and <u>turn to parametric functions for Gaussian distributions over actions, optimized by means of our designed rewards.</u>

## 3. Method
**EyeFormer performs scanpath predictions as sequential generation of fixation points, taking preceding fixations and the scene into account together as its state. The main challenges we tackle are related to**
1) <u>generating both spatial and temporal information on fixation points with parametric distributions,</u>
2) <u>optimizing a scanpath with non-differentiable objectives,</u> and 
3) <u>capturing individualspecific viewing differences to predict personalized scanpaths.</u>

**We propose a Transformer-guided RL approach (depicted in Figure 3) for three key reasons:** 
1) **The Transformer architecture** lets us <u>capture long-range sequential dependencies from previous fixations with Gaussian distributions [61].</u>
2) We find **RL preferable** to directly optimizing the loss since <u>some loss terms’ non-differentiable nature precludes direct optimization.</u> **The RL framework** enables <u>optimizing scanpaths with non-differentiable reward functions [56],</u> such as terms for <u>computing salient values with IOR.</u>
3) **Transformer-only models** suffer from the above-mentioned <u>error-accumulation issues: prediction errors from previously generated points propagate to subsequent predictions.</u> **During training, the model is fed the previous ground truth rather than its own predictions,** <u>so a mismatch arises during inference</u> when it must rely on those potentially inaccurate predictions. **We deal with this issue by using RL to train the model to generate sequences as it will during inference,** <u>optimizing its policies through continuous feedback and adjustments in line with cumulative rewards over time.</u>

### 3.1. Problem Formulation
Given **an image I,** our technique generates **a scanpath of length 𝑇:** <u>a sequence of ordered fixation points 𝑝1:𝑇 = (𝑝1, . . . , 𝑝𝑇 ) capturing the spatial and temporal information of the human gaze.</u> Each **fixation 𝑝𝑖 = (𝑥𝑖 , 𝑦𝑖 , 𝑡𝑖) is a three-dimensional vector** representing **the normalized point coordinates 𝑥𝑖 ∈ [0, 1] and 𝑦𝑖 ∈ [0, 1]** alongside the third dimension, **fixation duration expressed as 𝑡𝑖 ∈ (0, +∞).**

### 3.2. Environment, State, and Action
**Our predictive model acts as an agent that interacts with the environment,** where <u>the latter produces the state of both input image I and previous fixation points.</u> **The 𝜃 parameters** <u>dictate</u> **the policy, 𝜋𝜃,** whereby <u>the model generates an action</u> as **a prediction for the next fixation point 𝑝ˆ𝑖 ,** <u>sampled from the distribution produced by the policy model.</u> This process is formulated as **𝜋𝜃 (𝑝ˆ𝑖 |𝑝ˆ1:𝑖−1, I).**

### 3.3. Reward Function
<u>After each action, the agent receives a salient-value reward 𝑟sal that expresses that action’s contribution to the full scanpath.</u> Once the entire scanpath is generated, the agent is exposed to a reward 𝑟dtwd, calculated by means of the Dynamic Time Warping with duration (DTWD) metric discussed below. 

**The training’s objective** is to <u>minimize the negative expected reward, which is equivalent to maximizing a positive reward:</u>
$$
\mathcal{L}(\theta)
= -\mathbb{E}_{\hat{\mathbf{p}} \sim \pi_\theta}
r(\hat{\mathbf{p}})
$$

where pˆ = (𝑝ˆ1, ..., 𝑝ˆ𝑇 ) and where 𝑝ˆ𝑖 represents the 𝑖th fixation sampled from the model-generated distribution. 

The reward function combines the DTWD metric 𝑟dtwd, assessing the similarity between the predicted and the ground truth (GT) scanpath, with the summed salient-value reward for each fixation point along the scanpath generated 𝑟sal, thus:
$$
r(\hat{\mathbf{p}})
=
-r_{\mathrm{dtwd}}(\hat{\mathbf{p}})
+
\sum_{i=1}^{T} r_{\mathrm{sal}}(\hat{p}_i)
$$

#### 3.3.1. Dynamic Time Warping with Duration
**Dynamic Time Warping (DTW) is widely used for comparing two sequences that may differ in length [4, 50].** <u>It is useful for scanpaths because it finds an optimal alignment between the two scanpaths (ground truth and predicted ones) and computes the distance without missing any critical features.</u> 

We implement **DTW extended for duration** to <u>consider both spatial and temporal characteristics of scanpaths.</u> Specifically, <u>EyeFormer spatially aligns scanpaths by using fixation positions,</u> then <u>computes DTWD values as 3D vectors (𝑥, 𝑦, 𝑡) for fixations’ position and duration.</u> 

<u>By incorporating DTWD computations over the full scanpath into the reward function, we sought to generate scanpaths closer to the ground truth trajectories and duration.</u>

#### <u>3.3.2. Salient Values</u>
**EyeFormer applies rewards for salient values** to <u>encourage fixations in salient areas.</u> To avoid repeatedly fixating on the same location in the image, we implement an IOR mechanism to model the relevant tendency of the human visual system. 

We establish **inhibition areas (as regions of the saliency map)** for <u>all the previously predicted fixation points.</u> If the new predicted point falls within these areas, it does not elicit any additional reward; the reward corresponds to the salient value on the saliency map in all other cases. 

Importantly, predicted scanpaths can still return to an already-visited element, just as real-world ones may, since DTWD encourages fixations to revisit the most salient areas and our chosen IOR radius (explained next) is not so large as to preclude revisiting an element. 

We denote the display’s dimensions as 𝑊 ×𝐻. It sets the diameter of that display’s inhibition areas 𝑚display to be consistent with a human’s visual angle, the angle an object subtends at the eye (see Figure 2a). Our choices were informed by the diameter-setting suggested by Klein et al. [32] and further analyzed by Emami et al. [13]. 

Finally, we compute 𝑚orig, the diameter for the corresponding inhibition areas for the input image with size 𝑤I × ℎI (see Figure 2b):
$$
m_{\mathrm{orig}} = m_{\mathrm{display}} \min\left(\frac{W}{w_I}, \frac{H}{h_I}\right)
$$

Note that preparing the image for processing necessitates resizing it to𝑤inp×ℎinp, which corresponds to the size that the policy model requires for splitting the input image into patches. Using a square input image simplifies computations of this type [12]; accordingly, we resize the inhibition areas from circles to ellipses, while accounting for potential distortions (see Figure 2c). 

Any point (𝑥, 𝑦) in the image that satisfies the following condition gets inhibited (resulting in a salient-value reward of 0) and is omitted from the saliency map:
$$
\frac{(x-x_i)^2}{m_w^2} + \frac{(y-y_i)^2}{m_h^2} \leq 1,\quad \text{where}\quad
m_w = \frac{w_{\mathrm{inp}}}{w_I}m_{\mathrm{orig}},\quad
m_h = \frac{h_{\mathrm{inp}}}{h_I}m_{\mathrm{orig}}
$$

where (𝑥𝑖 , 𝑦𝑖) is the coordinates for the 𝑖th predicted fixation point. Hence, salient-value reward 𝑟sal at step 𝑖 is defined as the salient value of predicted fixation 𝑝ˆ𝑖 on the saliency map with IOR applied.

### 3.4. Policy Network
**A two-stage approach** characterizes the policy network for scanpath prediction. <u>The visual representation of any image I is learned through the image encoder E,</u> after which <u>a scanpath gets generated by means of the fixation decoder D.</u> 

**For population-level scanpath prediction,** <u>the visual embedding E (I) is taken as input to the decoder.</u> **For individual-level prediction,** <u>feeding the decoder this input along with a viewer embedding, 𝑒𝑢, allows the model to generate personalized scanpaths for separate viewers.</u>

#### <u>3.4.1. Vision Encoder</u>
We use a Vision Transformer (ViT) [6] network as the vision encoder. Specifically, the image is resized to a resolution of 𝑤inp ×ℎinp and split into 𝑛I non-overlapping patches for the vision encoder. 

Splitting functions mainly to speed up the model’s inference, capture local information, and obtain global information from relationships between patches. Next, a linear projection, a convolution layer, is applied to convert these patches into single-dimension embeddings 𝑒 𝑘 I ∈ R 𝑑I thus:
$$
\tilde{e}_I = [e_I^{\mathrm{CLS}}, e_I^1, \ldots, e_I^{n_I}] + e_{\mathrm{pos}}
$$

where 𝑒 𝐶𝐿𝑆 I is a learnable vector for the image context, [𝑒 𝐶𝐿𝑆 I , 𝑒1 I , . . . , 𝑒 𝑛I I ] is a matrix from concatenating the vectors 𝑒 𝐶𝐿𝑆 I , 𝑒1 I , . . . , 𝑒 𝑛I I , and 𝑒𝑝𝑜𝑠 ∈ R 𝑑I × (𝑛I+1) is the positional matrix reflecting the position context of the image patches. 

Finally, we apply a vision encoder E (·) based on a 12-layer version of the ViT model [12]. By employing per-patch convolution and using a Transformer to combine patch embeddings, the ViT model expresses the relationship for each patch and lets us derive the final image embedding, denoted as E (𝑒 I˜ ). 

We consider an alternative vision encoder, using a residual neural network (ResNet), also; the supplementary materials include comparison between it and the mechanism ultimately chosen.

#### <u>3.4.2 Fixation Decoder</u>
To generate fixation points, 𝑝ˆ𝑖 , we use a multi-layer Transformer decoder, D. It takes the image embedding E (𝑒 I˜ ) alongside the previously generated points denoted by 𝑝ˆ1:𝑖−1 as input to generate D (E (𝑒 I˜ ), 𝑝ˆ1:𝑖−1). 

This allows previous fixation points to influence points further along the scanpath. 

We set the first fixation to be at the center of the screen since the conditions behind most eye-tracking datasets involve asking participants to look at the center of the display before images get presented [24, 70]. 

For the given state (the previously predicted fixation points and the input image), the action (the next prediction for a fixation point) is
$$
\pi_{\theta}(\hat{p}_{1:T} \mid I)
=
\pi_{\theta}(\hat{p}_1 \mid I)
\prod_{i=2}^{T}
\pi_{\theta}(\hat{p}_i \mid \hat{p}_{1:i-1}, I)
$$

The policy 𝜋𝜃 is represented as a Gaussian distribution N (𝜇𝑖 , 𝜎𝑖). Alternatively, it could be represented as a mixed Gaussian distribution $\sum_{k=1}^{K} \lambda_{ik}\mathcal{N}(\mu_{ik}, \Sigma_{ik})$ with a total of 𝐾 Gaussian components, where 𝜆𝑖𝑘 denotes the weight of the 𝑘th Gaussian component, and Σ𝑖𝑘 denotes the covariance matrix specific to the component at step 𝑖. These variables for determining the distribution are sequentially generated by the decoder. 

We present more implementation details and a comparison between using a Gaussian and a mixed Gaussian distribution in supplementary materials.

### <u>3.5. Predicting Personalized Scanpaths</u>
To distinguish between individual viewers, we select a two-layer Transformer architecture as the viewer encoder E𝑢. This facilitates prediction of individual-level scanpaths considerably. 

The training process trains the model from the training users in the dataset. The viewer encoder is taught to allow each viewer’s distinct viewing behaviors to be encoded in a separate embedding space. 

In the test process, when given a new viewer, the model updates the viewer encoder with a few scanpaths from that viewer by backpropagating from the scanpath samples. Once the model has updated the viewer encoder, it can predict scanpaths specific to this unique viewer, thereby customizing its predictions for this individual’s viewing behaviors (note that this encoder is not applied for population-level predictions). 

Specifically, the image representation, E (𝑒 I˜ ), serves as the input query, while viewer embedding 𝑒𝑢 serves as the key and value in the cross-attention mechanism within the viewer encoder. The viewer embedding is a learnable matrix. For generation of fixations, the output of this encoder, E𝑢 (𝑒 I˜ , 𝑒𝑢), is directed to the fixation decoder.

### <u>3.6. Policy Gradient</u>
To compute the gradient of the objective function ∇𝜃 L (𝜃), our method employs the REINFORCE algorithm [47, 67], which offers a Monte Carlo variant of a policy-optimization technique commonly used in RL settings [56]. 

Under this algorithm, the agent accumulates samples from episodes by executing its current policy and utilizes those samples to update the policy’s parameters iteratively. The REINFORCE algorithm aims to maximize the cumulative expected reward across sequential actions by approximating the gradient of the expected reward for the current policy parameters. By adjusting these parameters iteratively in accordance with the gradient estimate, the algorithm attempts to enhance the policy’s performance over time. 

This algorithm is rooted in the insight that one can obtain the expected gradient of a non-differentiable reward function as follows:
$$
\nabla_{\theta}\mathcal{L}(\theta)
=
-\mathbb{E}_{\hat{p}\sim\pi_{\theta}}
\left[
r(\hat{p})\nabla_{\theta}\log\pi_{\theta}(\hat{p}\mid I)
\right]
$$

To approximate the expected gradient, we use a single MonteCarlo sample pˆ = (𝑝ˆ1, ..., 𝑝ˆ𝑇 ) from the policy 𝜋𝜃 for each training example in the minibatch:
$$
\nabla_{\theta}\mathcal{L}(\theta)
\approx
-r(\hat{p})\nabla_{\theta}\log\pi_{\theta}(\hat{p}\mid I)
$$

REINFORCE with a baseline. Our technique uses a baseline 𝑏 to assess the environment’s expected reward without any actions, thus generalizing the policy gradient obtained from REINFORCE. 

Applying this algorithm with a baseline allows us to estimate the advantage yielded by an action – i.e., the difference between the actual reward obtained and that expected from the baseline environment. By subtracting the baseline value, we reduce the variance of the gradient estimation, thereby arriving at a stabler optimization process. 

The gradient of the loss with respect to the 𝜃 policy parameters is then obtained as
$$
\nabla_{\theta}\mathcal{L}(\theta)
=
-\mathbb{E}_{\hat{p}\sim\pi_{\theta}}
\left[
\bigl(r(\hat{p})-b\bigr)\nabla_{\theta}\log\pi_{\theta}(\hat{p}\mid I)
\right]
$$

For each step in the training, our technique approximates the expected gradient with a single sample pˆ ∼ 𝜋𝜃 :
$$
\nabla_{\theta}\mathcal{L}(\theta)
\approx
-\bigl(r(\hat{p})-b\bigr)\nabla_{\theta}\log\pi_{\theta}(\hat{p}\mid I)
$$

In the discrete space, Rennie et al.’s conceptualization [47] serves as a foundational framework, wherein 𝑏 is estimated by means of the reward obtained from the policy’s greedy search. 

For operating in a continuous space, however, our approach diverges from theirs: at each step, the operation of our policy necessitates computation of 𝑏, defined as the reward associated with the mean of multiple samples drawn from the policy – in essence, the mean of the distribution generated by the policy. 

Consequently, the expected gradient is calculated as
$$
\nabla_{\theta}\mathcal{L}(\theta)
\approx
-\bigl(r(\hat{p})-r(\operatorname{sg}[\boldsymbol{\mu}])\bigr)
\nabla_{\theta}\log\pi_{\theta}(\hat{p}\mid I)
$$

where 𝝁 = (𝜇1, . . . , 𝜇𝑇 ) and 𝑠𝑔[·] constitute a stop-gradient operator having partial derivatives of 0.

## 4. Experiments
Our experiments attest to the new model’s unique capability of **producing personalized predictions when given a few user scanpath samples.**

### 4.1. Datasets
Both datasets in our experiments – **the GUI-oriented UEyes [24, 25]** and **OSIE [70], from natural scenes** – feature <u>multiple scanpaths for each image, from numerous viewers.</u> The two datasets were collected by <u>eye trackers that output fixation points and their durations,</u> rather than saccades. 

#### 4.1.1. GUIs and Information Graphics
**The UEyes** dataset provided us with <u>eye-tracking data (up to 7 s)</u> from <u>62 participants who viewed 1,980 images</u> drawn from four common types of GUI and information graphics (posters, desktop GUIs, mobile GUIs, and webpages). 

Collecting the data with an eye tracker in a laboratory setting guaranteed <u>precise fixation coordinates in the X–Y plane,</u> and <u>the coordinate values were subject to participant-specific calibration accounting for relevant human factors such as eye–display distance [35].</u> 

We used <u>the same training/test image split as Jiang et al. [24]:</u> 1,872 images in the training set and 108 in the test set, with the four GUI types distributed evenly within each set. 

In addition, we established <u>a training/test split for individual-level prediction,</u> randomly assigning <u>53 viewers to the training set (85%)</u> and <u>the remaining nine to the test set (15%).</u> **Our model** <u>was trained on the data collected from when the training viewers looked at the GUIs shown in the training images.</u> 

**Most scanpaths in UEyes** have <u>roughly 15 fixations (the average number of fixations per image is 15.3).</u> Further details of the dataset and implementation can be found in the supplementary materials. 

#### 4.1.2. Natural Scenes
The OSIE dataset, from <u>free viewing of natural scenes,</u> comprises 700 images with associated eye-movement data from <u>three seconds</u> of viewing by 15 participants. 

With OSIE, which has been widely used in previous research [7, 54], <u>we applied the same split used in prior work (80% training, 10% validation, and 10% testing data).</u> 

<u>We did not use datasets such as SALICON’s [21],</u> since they take mouse movements as a proxy for eye movements, <u>whereas EyeFormer is designed for replicating actual scanpaths recorded by eye trackers.</u>

### 4.2. Metrics
We assessed performance via metrics commonly employed for scanpath evaluation [1, 15]. <u>All experiments used coordinates 𝑥 ∈ [0, 1] and 𝑦 ∈ [0, 1], normalized for image size (px/px, dimensionless), and fixation duration 𝑡 ∈ [0, +∞) in milliseconds.</u> 

#### 4.2.1. Dynamic Time Warping (DTW)
**DTW** serves as <u>a standard metric for similarity between two temporal sequences even when they differ in length [4, 50].</u> It identifies the optimal match and calculates the distance between two scanpaths in a manner that preserves essential features. 

#### 4.2.2 Time Delay Embedding (TDE)
<u>By focusing on assessment of similarities at sub-scanpath level [59, 64],</u> TDE offers evaluation more nuanced than DTW’s, which attends only to overall comparison of entire scanpaths. 

#### 4.2.3. Eyenalysis
<u>Finding the closest mapping between fixation points on the two scanpaths,</u> Eyenalysis <u>takes each fixation point along the first scanpath and identifies the spatially closest fixation point on the second, and vice versa [39].</u> 

It then measures the average distances for all the closest fixation pairs, thereby <u>emphasizing evaluation of individual fixations</u> instead of the sequences. 

#### 4.2.4 Dynamic Time Warping with Duration (DTWD)
**Our extension of DTW to capture duration empowered considering fixations’ position and duration both.** 

<u>We align two scanpaths on the basis of their optimal match of (𝑥, 𝑦) coordinates and calculate the cumulative distance by computing, for each pair of aligned points, the distance between the two three-dimensional vectors (𝑥, 𝑦, 𝑡) representing the spatiotemporal information.</u>

#### 4.2.5 MultiMatch
With **MultiMatch metric [11],** five variants facilitate <u>assessing important aspects of fixations along scanpaths: shape, direction, length, position, and duration.</u> 

<u>While DTWD evaluates spatial and temporal characteristics, MultiMatch excels at capturing additional features such as shape, direction, and length and gives an overall evaluation based on all these features.</u>

## 5. Results
The results demonstrate that our model 
1) **predicts individual-level scanpaths** <u>when given a few viewing samples from the user;</u>
2) compares favorably with other models <u>for population-level scanpath prediction;</u> and 
3) **predicts both spatial and temporal characteristics of scanpaths** with stimuli that include GUI images, information graphics, and natural scenes.

### 5.1. Individual-Level Scanpath Prediction
Prior research has not addressed the challenge of predicting personalized individual-level scanpaths, partly because full re-training for each new viewer, with more data, is impractical. <u>Our model achieves a workable balance by generating scanpaths tailored to each person’s viewing behaviors and idiosyncrasies</u> **while still permitting a single model’s application for all viewers, without the burden of re-training.**

We verified our model’s ability to **generate personalized scanpaths by proceeding from a few scanpath samples from the individual,** thus confirming that <u>the model can effectively capture each viewer’s viewing preferences/behaviors and reflect them in its output.</u> 

When encountering a new viewer with a few samples available, the model updates the viewer embedding with 𝑛path scanpaths obtained from that viewer (in our experiments, 𝑛path = 50). Fine-tuning the model involves backpropagating from the scanpath samples so that it can predict scanpaths specific to this unique individual’s viewing behaviors. 

Since no established baseline method at present can function as a point of comparison for this personalization approach, we compared the model’s tailoring for the target test viewer with its tailoring for other test viewers to quantify its effectiveness in capturing the characteristics of individual viewers. 

The results (shown in Table 1) show that the errors in the former setting are smaller than those of personalization for other test viewers. We conclude, then, that the personalized model can better address individual-specific characteristics. 

Illustrative examples presented in Figure 4 capture the nature of the individual-level scanpath prediction qualitatively; in addition, the supplementary materials provide more results and explain the relationship between sample quantity and performance.

### 5.2. Population-Level Scanpath Prediction
To assess how well our model predicts the spatiotemporal information of scanpaths, we compared its performance with preexisting scanpath models’. We evaluated the model with both GUIs and natural scenes to check whether it can be generalized to different types of images. 

**For GUIs,** we compared to <u>Itti–Koch [20], DeepGaze III [33], DeepGaze++ [24], SaltiNet [2], UMSS [65], PathGAN [3], PathGAN++ [24], ScanGAN [38], ScanDMM [53], and the model of Chen et al. [7].</u>

**Comparisons for natural scenes** judged EyeFormer against models focused on such scenes: <u>Itti–Koch [20], SGC [55], the model by Wang et al. [64], Le Meur et al.’s model [34], STAR-FC [68], SaltiNet [2], PathGAN [3], IOR-ROI [54], GazeFormer [41], and Chen et al. [7].</u> 

While one of the baseline models, GazeFormer, is a Transformer-based method designed for visual search, directly comparing it with other methods is not possible because GazeFormer requires a pre-specified target, which freeviewing tasks do not provide. Therefore, we adapted GazeFormer to free-viewing tasks by providing a blank target as input. 

For a fair comparison, we trained all these models with the same dataset split. We fed the models every individual scanpath from all the viewers for each training image, helping the models learn the underlying scanpath distribution. 

Note that we did not combine the two datasets: all methods were trained on each dataset separately. Training and analysis too remained separate.

#### 5.2.1. Quantitative Evaluation
To account for variations in image sizes and minimize discrepancy-related errors, we normalized **the fixation points’ coordinates to the [0, 1] range.** 

Specifically for training on natural scenes, we used the **ResNet** instead of the ViT mechanism as the vision encoder, for better comparison to other baseline models since <u>prior work with training on the OSIE data [70] used a ResNet model as the encoder.</u> 

Table 2 presents a comprehensive comparison covering all the metrics. Our model proved at least as good as the baseline models by most metrics, for GUIs and natural scenes both. <u>The results indicate that it simulates scanpath trajectories more realistically.</u> 

Of the models tested, only <u>PathGAN, PathGAN++, SaltiNet, UMSS, IOR-ROI, GazeFormer, and Chen et al.’s technique can predict temporal information.</u> 

**Chen et al.,** which is one of the best baseline models, <u>predicts positions and duration separately; however, fixation positions and duration are highly correlated.</u> **GazeFormer,** by relying on a Transformer model to <u>generate an entire scanpath in a single step, overlooks the local dependencies and correlations between adjacent points.</u> 

The fact that **our model** <u>excels by the DTWD and MultiMatch Duration metrics attests to its capacity to yield more accurate results and also handle prediction of temporal information.</u>

#### 5.2.2. Qualitative Evaluation
Qualitative comparisons revealed that the predictions made by our model lie closer to the ground truth than those of the other models. **Figure 5** presents <u>population-level prediction results showcasing the performance of EyeFormer.</u> **Figure 6 and Figure 7** provide <u>comparison between our model and the baseline ones (more results are available in the supplementary materials).</u> 

While **PathGAN++ and ScanGAN** <u>generate realistic trajectories very well</u> (thanks to their discriminative component), <u>the points they predict often fall outside the salient areas and tend to lie in clusters.</u> 

In contrast, **DeepGaze++** <u>performs well in locating fixation points, by applying post-processing to density maps.</u> Nevertheless, it generates fixations in <u>incorrect order;</u> on account of the non-differentiable nature of the post-processing, the order is not optimized. 

**The Itti–Koch, SaltiNet, and UMSS** techniques generate scanpaths from saliency maps, encouraging fixations in salient areas, but they too <u>fail to optimize for correct fixation order.</u> 

**Chen et al.’s technique** tends to <u>generate several clusters of closely grouped points</u> since they improved the prediction of fixations <u>without addressing the need to spread consecutive points out more.</u> 

In additional analysis, <u>we computed the clustering-tendency error</u> via **the Laminarity metric [1].** The Laminarity value of our model is 73.137, and that of Chen et al.’s is 178.072 <u>(lower values are better).</u> . **Its higher score indicates that the model of Chen et al.** <u>produces fixation-clustering in locations where ground-truth fixations do not cluster.</u> Finally, **ScanDMM** focuses relatively <u>strongly on text elements.</u> 

Our model assigns fixations to salient areas and attends to the points’ order with greater precision. It accomplishes this by using <u>the salient-value reward (𝑟sal) to emphasize points that lie within areas of interest</u> and by employing <u>the DTWD reward (𝑟dtwd) to encourage more accurate trajectories.</u>

### 5.3. Ablation Study
Table 3 presents the results from an ablation study we performed on utilizing RL to produce both population- and individual-level scanpaths. 

The results reveal that a Transformer-only model does not yield satisfactory results and that **incorporating RL greatly enhances the prediction of fixations and their duration.**

**With its population-level prediction, our RL model** brings an improvement of 14.7% and 26.5%, respectively, by the TDE and the Eyenalysis metric. <u>This too is evidence that using RL increases the model’s capacity to generate realistic fixations for scanpaths.</u> 

As for fixation duration, applying RL has a positive influence on prediction accuracy, demonstrated by the 4.8% improvement shown by the DTWD metric. 

Similar effects are visible with the individual-specific predictions connected with training users (i.e., the trained model’s prediction of scanpaths for GUI images when given the IDs of particular training users). 

Additionally, the results highlight that the absence of either each type of reward or of inhibition of return leads to a decline in overall accuracy. Results from further ablation studies are included in the supplementary materials.

## 6. Application for Personalized Visual Flows
**EyeFormer enables handy prediction of individual-level scanpaths.** <u>Demonstrating this capability in practice,</u> **we applied it to the problem of personalizing visual flows.** 

<u>In model-assisted flow design, the designer identifies GUI elements intended to receive more attention than others [16].</u> **Our goal was to support this by controlling the flow of attention to selected elements.** While prior work has demonstrated model-assisted personalization of graphical layouts [60], <u>its focus has been solely on visual-search time, not visual flow.</u>

In our scenario, 
1) **the designer supplies a GUI layout** and 
2) **specifies the desired visiting order for three or more elements that should be fixated upon first (the most important ones).** 
3) After this, **our system outputs both population- and individual-optimized layouts.** 

**Generation of the individual-specific layouts is based on the personalized scanpath prediction results.** Specifically, <u>given a viewer with 𝑛path scanpath samples</u> (in our experiments, 𝑛path = 50), <u>EyeFormer generates corresponding layouts by proceeding from the predicted scanpaths at individual level for this particular viewer.</u>

### 6.1. Formulation of Optimization Problem
We expressed this application as **a constraint optimization problem [22, 23, 29–31]** that requires <u>ascertaining positions and sizes of elements for a GUI based on the predicted personalized scanpaths.</u> 

To address this problem, we built on **an integer-programming-based layout optimizer [9]** that <u>optimizes GUI layouts by considering their elements’ packing, alignment, and preferred positioning.</u> Additionally, we introduced **a constraint** requiring <u>adherence to the designer-specified fixation order,</u> along with **an objective score** <u>derived from EyeFormer’s predictions.</u>

#### 6.1.1. Fixation Order Constraint
We denote the order of the three most important elements, elem1, elem2, and elem3, which **should be fixated upon earliest,** as <u>[elem1, elem2, elem3].</u> Extending the list permits handling more elements, in a similar manner. 

Firstly, for **the predicted scanpath** <u>[𝑝ˆ1, 𝑝ˆ2, ..., 𝑝ˆ𝑇 ],</u> **the procedure identifies the GUI element receiving fixations, per fixation point,** denoted as <u>[elem𝑝ˆ1 , elem𝑝ˆ2 , ..., elem𝑝ˆ𝑇 ].</u> Secondly, <u>[elem1, elem2, elem3] is restricted to being a subset from the beginning of the deduplicated sequence [elem𝑝ˆ1 , elem𝑝ˆ2 , ..., elem𝑝ˆ𝑇 ];</u> **that is, the sequence [elem𝑝ˆ1 , elem𝑝ˆ2 , ..., elem𝑝ˆ𝑇 ] begins with repeated occurrences of elem1, followed by elem2 and subsequently elem3.** This constraint guarantees that **the required fixation order specified by the designer is honored.**

#### 6.1.2. Objective Term for Fixation Duration
To define optimality further, we applied **a fixation-duration objective term** for <u>GUI layouts that satisfy the required-order constraint.</u> **Where the fixations corresponding to the sub-sequence of repeated occurrences** of elem1 followed by elem2 and then by elem3, described above, are denoted as <u>[𝑝ˆ1, 𝑝ˆ2, ..., 𝑝ˆ𝑀 ]</u> (with 𝑝ˆ𝑀 being the final fixation before attention moves to other elements), **the objective is to select the layout whose fixation durations for these elements sum to the maximal value:** $\max \sum_{m=1}^{M} t_{\hat{p}_m}$.

### 6.2. Results
Given an original GUI design and an annotated sequence of the **(three)** most important GUI elements, **we generate both**
1) **the population-optimized layout** and 
2) **a layout personalized for each viewer.**

**The population-optimized layout relies on the population-level scanpath prediction,** which serves as <u>the best compromise across viewers,</u> while **the viewer-specific layouts are based on personalized scanpath prediction** such that <u>each viewer follows the desired order and devotes maximal time to the elements deemed important.</u>

Testing for 62 individual viewers yielded the following results for the designs shown in the figure:
* **For “Design 1”,** 56 viewers would follow the desired viewing order for **the designer-selected elements with the population-optimized layout,** <u>devoting 1.29 seconds of the seven-second viewing period to them, on average.</u> Shown the corresponding **personalized layout,** <u>all viewers would follow the desired order, with an average total duration of 1.86 seconds</u> (44.19% more than with population-level optimizing).
* **Given “Design 2”,** <u>46 viewers shown the populationoptimized layout would follow the fixation order desired,</u> with <u>an average duration sum of 2.75 s.</u> **With the personalized layout for Design 2,** <u>61 viewers would do so,</u> and <u>the average total duration is 3.19 seconds,</u> a sum 16% greater than that from the population-level layout.

**The results attest that personalized layouts can draw more of the viewer’s attention to the target elements than a populationoptimized layout does.**

## 7. Discussion and Future Work
**EyeFormer** is able to <u>cover both spatial and temporal characteristics of scanpaths across various stimulus types and factor in individualspecific viewing behaviors,</u> which are vital for <u>understanding visual attention.</u> It opens the door to <u>automated personalization of visual flows, which enables GUI software to respond better to each user’s behaviors and expectations.</u>

Personalized prediction is critical for practical developments. There is rather extensive variability in scanpaths across individuals; in fact, <u>averaged scanpath prediction may not be very meaningful – after all, it might be unlikely to match any actual user.</u> **From inputting example scanpaths of a single user, we have demonstrated that personalized layouts can be generated for that user.**

**Greater accessibility, through GUIs optimized for people with viewing difficulties,** is one of many possible application domains. Further research could also use subjective comparison studies to see whether users prefer GUIs personalized in accordance with scanpath predictions over the original interface.

### 7.1. Understanding Viewers
Future work could use <u>viewer clustering to enhance the interpretability of extensive sets of scanpaths for designers.</u> Clustering enabled by applying, for example, **𝐾-means** to the viewer embeddings in our model could help reveal <u>how viewers of various kinds interact with visual content,</u> thereby aiding designers in cultivating <u>aggregate-level insight beyond individual paths, for a broader perspective.</u> Further research could also yield better tools for visualizing and comprehending diverse viewer behaviors.

### 7.2. Practical Applications of Personalization
**By controlling the visual flow over GUIs, designers can encourage users to focus on the most important parts of the interface.** This improves usability and aids in reaching specific design goals, such as effective market funneling. 

**Personalized visual flows** can support optimal ad placement and related design such that key messages catch the attention of users and drive them toward such desired actions as clicking or buying. Prior work on visual-saliency analysis has highlighted that better visual flow can enhance users’ engagement and guide behaviors [14, 58, 66]. 

**The ability to predict individuals’ gaze patterns** could also support creating adaptive GUIs that respond dynamically to user interactions and preferences. Moreover, associated research addressing the correlation between design trends and user-interaction behaviors could prove fruitful; for instance, being able to fine-tune scanpath prediction in light of current user data could address the fact that individuals’ interactions with GUIs evolve over time. 

The potential advantages extend beyond GUIs. Education tools could benefit from adjusting visual content in line with the gaze patterns of each user, thus facilitating students’ improved comprehension of complex concepts. Similarly, training modules that adapt to users’ learning progress “on the fly” and focus on areas ripe for improvement might promote more efficient learning. 

Also, predicting users’ likely points of focus in augmented- and virtual-reality settings could encourage more immersive experiences through dynamic adjustment of visual content that helps users locate objects easily.

### 7.3. Ethics Concerns
A practical and ethics-related challenge remains, however, in **how to collect eye-tracking data from individuals.** We foresee two main options: 
1) using **Web cameras** or other commodity devices, with the user’s permission, and 
2) inferring patterns via proxy signals such as **mouse movements.** 

Since people with privacy concerns may be reluctant to share their gaze data, the applications developed – such as GUI layouts personalized on the basis of the user’s scanpaths – should be able to run locally; in the ideal case, sensitive gaze information should not be transmitted over the Internet. 

Another possibility is to compute **“sufficient statistic”** measurements and send these to a server that generates personalized GUI layouts. These approaches would help maintain user privacy while still offering the benefits of personalized scanpaths.

### 7.4. Limitations
**At present, the model is limited to fixed-length scanpaths,** since we considered a limited time window of free-viewing behaviors (based on the seven-second maximum span in the UEyes dataset [24], which permitted better comparison with earlier work). <u>However, it should be possible to output variable-length scanpaths by predicting the final state.</u> 

In addition, **our discussion concentrated on predicting fixation sequences.** We acknowledge that <u>viewing behaviors are far more complex, encompassing many other eye dynamics (blinks, vestibulo-ocular reflexes, post-saccadic oscillations, etc.),</u> which future studies could explore. 

Follow-up research could also investigate ways of reducing the number of scanpaths needed per viewer (from the current 50). 

Finally, the state-of-the-art scanpath-related metrics are designed primarily for natural scenes, so they may not fully capture the characteristics of scanpaths in GUI settings. Refining the metrics employed should afford deeper understanding of how models such as ours perform and thus enhance the development of more effective methods.

## 8. Our Conclusion
<u>EyeFormer is a Transformer-guided RL model, which predicts both population-level and individuals’ scanpaths well, using the Transformer architecture as the policy model offers a novel representation for accurately capturing variability in scanning patterns across stimuli and individuals.</u>

While the Transformer-guided design effectively captures long-range sequential dependencies on the basis of previous fixations, combining it with RL enhances the generation of fixation sequences through optimization that employs non-differentiable objectives, such as maximizing the salient values of fixations.

In addition to performing better than (or at least on par with) state-of-the-art models in the realm of populationlevel prediction, EyeFormer offers the first accurate modeling of <u>individual-to-individual variability in scanpaths, from only a few user samples.</u>

Its application for GUIs optimized in keeping with the personalized scanpath-prediction results marks another contribution offering a way forward.
