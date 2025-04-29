---
title: "Understanding Learning to Optimize: An Example from 6G Network Resource Allocation"
date: 2025-02-17
categories:
  - Data Science Projects
  - Research
tags:
  - Data Science
  - CNN
  - Optimization
---

In today's world of increasingly complex decision-making problems, optimization has become more critical than ever. Whether it's managing supply chains, allocating network resources, or designing smarter transportation systems, the ability to make better decisions faster defines the winners in nearly every industry.

Companies that can intelligently allocate resources, manage networks, or dynamically adjust their operations unlock major advantages. Higher efficiency, lower costs, improved customer satisfaction are just a few of the business impacts that effective optimization can deliver. In fast-moving, highly competitive markets, optimization isn't a luxury. It's a necessity.

Optimization touches a wide range of stakeholders. Business leaders depend on it for strategic decision-making. Technical teams use it to drive operational efficiency. End users benefit from smoother services, faster responses, and better experiences. Across all levels, better optimization leads to better outcomes.

This post introduces the concept of Learning to Optimize, which is a powerful new approach that combines machine learning with traditional optimization. We'll explore what Learning to Optimize means, why it matters, and when to use it. To make things concrete, we'll walk through a real-world example: using deep learning to optimize resource allocation in 6G wireless networks.

<img src="/assets/images/learn_to_optimize.png" alt="Learn to Optimize" width="700">

## What is Learning to Optimize?

Learning to Optimize is the process of training a machine learning model to replace or speed up complex optimization tasks. Instead of solving an optimization problem from scratch every time, we **"teach" a model how to find good solutions efficiently**. The model learns patterns in the problem structure and uses them to make fast, high-quality decisions.

**Intuition**: Think of it like teaching a chef how to cook a dish without needing to read the recipe every single time. Once the chef has practiced and internalized the best cooking patterns (e.g., the ingredients, techniques, and timing), they can cook efficiently and adapt to slight variations without starting from zero. Similarly, a model that learns to optimize can make decisions quickly and intelligently without re-solving the full problem every time.

**What Learning to Optimize is not**: It is not simply making a prediction or forecasting an outcome based on raw data, like predicting next month's sales figures. Rather, it focuses on learning the decision-making process itself, i.e., how to choose actions or allocations that optimize a given objective.

In simple terms, Learning to Optimize means shifting from repeatedly solving complex problems manually to creating a model that knows how to solve them automatically. It allows businesses to make better decisions faster, at scale, and with lower computational costs.

**Real examples**: 
- In logistics, instead of recalculating the best delivery routes each day with a traditional solver, a model trained through Learning to Optimize can instantly suggest near-optimal routes based on real-time traffic data. 
- In finance, instead of running complex portfolio optimization programs overnight, a trained model can quickly recommend asset allocations that maximize returns within risk constraints. 
- In wireless networks, Learning to Optimize can allocate frequencies and power levels dynamically, enabling faster and more efficient communication without manually solving optimization problems each time.

## Why is Learning to Optimize Important?

Learning to Optimize offers powerful business impacts that can redefine how organizations operate. One of its most significant benefits is the **dramatic acceleration of decision-making processes**. Traditional optimization methods often require solving complex problems from scratch, which can be time-consuming and computationally expensive. By contrast, a model trained to optimize can deliver near-instantaneous decisions, allowing businesses to respond quickly to changing conditions.

This ability to make real-time decisions is crucial for applications such as traffic control, dynamic pricing, and network optimization. When systems need to adapt immediately to new data or unexpected events, waiting for lengthy optimization computations is simply not an option. Learning to Optimize enables real-time operations, **making businesses more agile and resilient**.

Additionally, Learning to Optimize **significantly cuts computation costs**. Running complex solvers repeatedly can be expensive, both in terms of hardware and energy consumption. A learned model, once trained, can perform inference with minimal computational resources, leading to substantial cost savings over time.

Companies that master Learning to Optimize gain a **strong competitive advantage**. They can move faster, make smarter decisions, and operate more efficiently than competitors who rely on traditional methods. In highly competitive industries, the ability to optimize operations in real time can be the difference between leading the market and falling behind.

## Key Steps of a Learning-to-Optimize Framework

A Learning-to-Optimize framework follows a structured process to ensure that the machine learning model can effectively replace or accelerate the optimization solver. Below are the detailed steps:

### 1. Optimization Problem Formulation
The first and most critical step is to clearly define the original optimization problem. This requires a solid foundation in optimization theory, operations research, and domain-specific knowledge. The formulation must accurately capture the objectives, constraints, and operational realities of the problem. 

For example, in supply chain management, the goal might be to minimize total delivery time while respecting constraints like vehicle capacity limits, delivery time windows, and traffic conditions. A poor formulation can lead to models that optimize the wrong objective or violate critical real-world constraints, such as delivering goods late or overloading vehicles.

### 2. Building the Learning-to-Optimize Dataset
Once the problem is formulated, the next step is to build a dataset containing examples of problem instances and their corresponding solutions. The quality of the dataset directly impacts the model's performance. Ideally, this dataset consists of globally optimal solutions. However, for many complex optimization problems, finding global optima is computationally infeasible. In these cases, high-quality locally optimal solutions or feasible solutions that significantly outperform naive approaches are used. 

For example, in nonconvex mixed-integer optimization problems, heuristic or approximate algorithms can generate sufficiently good solutions for training.

### 3. Model Design: Learning the Mapping
The core of the framework is building a machine learning model that learns the mapping from input features (problem parameters) to optimal decisions. 

Depending on the problem structure, convolutional neural networks (CNNs), graph neural networks (GNNs), or other architectures might be chosen to exploit specific patterns, spatial structures, or relational information.

### 4. Training, Validation, and Testing
The model must be trained on the dataset with careful validation to avoid overfitting and to ensure generalization to unseen instances. 

Evaluation metrics are chosen based on how well the model approximates the true optimization solutions while respecting constraints. Testing on out-of-sample data verifies the model's practical utility.

### 5. Deployment and Inference
Finally, the trained model is deployed to predict optimized solutions for new problem instances in real time. Compared to solving each instance with a traditional solver, inference through the trained model offers significant speedups and computational savings.

Importantly, a Learning-to-Optimize framework requires **human-in-the-loop** involvement. Human expertise is crucial for both formulating the optimization problem correctly and building a high-quality solution dataset. Without domain knowledge and careful problem structuring, the model risks learning shortcuts that do not align with the real-world goals or constraints of the system.

## When to Use Learning to Optimize?

Learning to Optimize is most valuable in situations where traditional optimization methods struggle to keep up with the demands of the system. If solving **the optimization problem from scratch is too slow, too complex, or too costly to do frequently**, it may be a strong candidate for a learning-to-optimize approach. For example, large-scale scheduling, resource allocation, or supply chain management problems often fall into this category.

Another key scenario is when systems require **real-time decision-making**. In areas like autonomous driving, online recommendations, or dynamic network management, the environment can change rapidly, and decisions must be made immediately. Traditional solvers are often too slow for these use cases, whereas a trained model can make quick, approximate decisions that are good enough to meet operational needs.

Finally, Learning to Optimize is most effective when the underlying structure of the optimization problem remains relatively stable over time. If **the patterns and relationships in the data do not change dramatically**, a model trained once can continue to perform well across many different instances, saving time and computational resources compared to re-solving the optimization anew each time.

## Why Learning to Optimize is Still Needed Even with Machine Learning and LLMs

While machine learning models and large language models (LLMs) like ChatGPT, Claude, Grok, or Gemini have revolutionized many aspects of technology and decision-making, they are **not always efficient at solving constrained optimization problems**. For example, although LLMs can provide suggestions or approximate solutions, they are not designed to find precise, feasible solutions to problems that require strict adherence to mathematical constraints, such as those found in logistics planning or network resource allocation.

Constrained optimization problems typically need to be formulated mathematically and solved using specialized algorithms to ensure feasibility and optimality. While LLMs can assist in formulating these problems or suggesting approaches, **the real challenge lies in solving them quickly and effectively**. For standard problems like linear programming or convex optimization, it is possible for LLMs to call existing optimization solvers. However, the computational cost and solving time become significant as the problem size grows, especially when solutions are needed frequently or in real time.

Moreover, for complex problems with intricate constraints—such as nonconvex mixed-integer problems—LLMs often **struggle to produce high-quality solutions**. Their outputs may be infeasible, suboptimal, or computationally inefficient. In contrast, Learning to Optimize focuses specifically on training models that can generate solutions efficiently and effectively, making it a critical tool when fast, reliable decision-making is needed in complex, constraint-heavy environments.

## Detailed Example: Learning to Optimize 6G Network Resource Allocation

### Overview of 6G Technology and Problem Motivation

#### What is CFmMIMO and Why Does It Matter?

One of the promising technologies for future 6G wireless networks is called Cell-Free Massive MIMO, or CFmMIMO for short. Instead of using a single tower (or base station) to serve users in a fixed area like we do today, CFmMIMO uses many small access points spread out across a wide area. These access points work together to serve users, no matter where they are, creating a smoother and more consistent connection.

#### Benefits of CFmMIMO
- **Better coverage**: Users don’t have to worry about being far from a tower. Multiple access points can work together to serve them.
- **Faster data speeds**: By combining signals from different points, users can get better performance.
- **Lower energy use**: Because signals travel shorter distances, less power is needed.
- **Less delay**: Having access points closer to users reduces the time it takes for data to travel.

#### Challenges with CFmMIMO
While it sounds great, CFmMIMO isn't easy to manage. Here are some of the main challenges:
- **Limited power**: Each access point has a small power limit, so users have to share that power.
- **Signal interference**: If not managed carefully, signals meant for one user might interfere with others.
- **Limited data links**: The connection between access points and the main network can only handle so much data.
- **Minimum service requirements**: Every user expects a basic level of service, and we have to make sure they get it.
- **Access point limits**: Each access point can only serve a few users at once due to equipment limits.

#### Why User Association and Power Control Are Important

Wireless systems are complex because the environment is always changing. Signals can get weaker, bounce off buildings, or interfere with each other. Two of the most important decisions we need to make are:
- **Which access point should serve which user?** (This is called user association.)
- **How much power should each access point use for each user?** (This is called power control.)

If we make smart choices for these two things, we can make sure everyone gets good service while using the least amount of energy and avoiding interference.

### Business Impact
If we can solve these challenges, 6G networks will be faster, more reliable, and cheaper to run. This is good news for:
- **Telecom operators**, who can serve more users with fewer resources.
- **Equipment providers**, who can build smarter, more efficient systems.
- **Cloud providers**, who might help process the data involved.
- **Everyday users**, who will get better service.

### The Optimization Problem

To make all this work, we need to solve a math problem. The goal is to get the highest possible performance across the network (called spectral efficiency) while following all the rules and limits:
- Each access point can’t use more than a certain amount of power.
- The data connection to each access point can only carry so much.
- Every user should get a fair minimum amount of service.
- Each access point can only serve a certain number of users.

#### What Is Spectral Efficiency?
Spectral efficiency (SE) is just a fancy way of asking: “How much data can we send using a limited amount of bandwidth?” Imagine two highways with the same number of lanes. The one that moves more cars per hour is more efficient. In our case, the cars are data.

#### A Simple Version of the Problem
We want to **maximize** the total SE of all UEs in the network.

$$ \sum_{k=1}^K R_k (\{a_{mk}\}, \{p_{mk}\}) $$

where $$ R_k $$ is the SE received by user $$ k $$. 

**Variables**:
- **User Association**: A binary variable $$ a_{mk} $$ where $$ a_{mk} = 1 $$ if AP $$ m $$ serves UE $$ k $$, 0 otherwise.
- **Power Allocation**: continuous variable $$ p_{mk} $$ is the power allocated by AP $$ m $$ to UE $$ k $$.

Here, $$ R_k $$ is a complex logarithmic function that reflects how the wireless environment behaves. It depends on factors like signal strength and interference. The decision of which access points serve which users ($$a_{mk}$$) and how much power is allocated ($$ p_{mk} $$) will affect signal strength and interference, leading to the change in $$ R_k $$. 

In both theory and practice, $$ R_k $$ depends explicitly on user association and power control decisions. Theoretically, this relationship is modeled using clean, idealized formulas based on Shannon’s theory, signal power, and interference. In practice, however, real-world imperfections such as hardware limits and unpredictable interference make modeling the function
$$ R_k $$ more complex.

**Simplified Constraints**:
- $$ \sum_{k} p_{mk} \leq P_m^{\text{max}} $$ (Transmit power limit per AP)
- $$ \sum_{k} R_k \leq C_m^{\text{backhaul}} $$ (Backhaul capacity limit per AP)
- $$ R_k \geq R_k^{\text{target}} $$ (Minimum data rate per UE)
- $$ \sum_{k} a_{mk} \leq N_m^{\text{max}} $$ (Max number of UEs served per AP)

which mean:
- Each access point stays within its power limit.
- Each access point doesn't exceed its data link capacity.
- Each user gets at least their minimum service.
- Each access point serves only a limited number of users.

This optimization problem is inherently a **mixed-integer nonconvex** problem, making it extremely challenging to solve quickly in large-scale 6G networks, which is why Learning to Optimize becomes crucial. 

Since this post focuses on the overall concept rather than solving the problem step-by-step, we will skip the detailed mathematical formulation.

### Solving the Optimization Problem

To solve the complex optimization problem of user association and power control, we use a method called **Successive Convex Approximation (SCA)**.

SCA is an advanced technique for solving problems that are too complicated to tackle all at once. Many real-world optimization problems are "non-convex", meaning they have lots of hills and valleys, making it hard to find the very best solution. Instead of trying to solve the messy problem directly, SCA works step-by-step:

- At each step, it **approximates** the difficult problem with a simpler, easier (convex) problem.
- It solves this simpler version to get a better guess.
- Then it updates the problem based on this new guess and repeats the process.

Think of it like climbing a foggy mountain. Instead of trying to find the highest peak right away, you look around your immediate area, climb to the highest nearby point you can see, and then reassess from there. Step by step, you get closer to the top.

When we use SCA, the solution we find satisfies a **necessary condition** for being a global best solution. One type of this condition is called the **Fritz John condition** (named after the mathematician Fritz John).

In simple terms:
- It means the solution we find is at least a "smart" point—not just some random guess.
- It's like reaching a point on the mountain where **you can't immediately find any nearby path that goes uphill**. You might not be at the tallest mountain peak in the entire region, but you are somewhere meaningful and high up.

While SCA doesn't guarantee the absolutely highest peak (global optimum) in all cases, it does ensure that we've reached a point that makes sense mathematically and practically,


## Building the Learning-to-Optimize Dataset

To train a machine learning model to make smart decisions in wireless networks, we first need to build a dataset. 

#### Exploratory Data Analysis (EDA) and Feature Engineering

We simulated different network setups by randomly placing Access Points (APs) and Users (UEs). This helps reflect real-world conditions, where UEs move around and the network changes.

Next, we measured how strong or weak the signals are between each AP and each UE. This is captured by what's called a **large-scale fading coefficient**. It tells us how much the signal weakens as it travels due to distance or things like buildings and walls. Signals are stronger when UEs are close to an AP, and weaker when they're farther away.

This is important because a UE's **large-scale fading coefficient (based on location) directly influences both UE association and power control decisions**. A UE closer to an AP is more likely to be selected by that AP for service because the connection requires less transmission power and results in better signal quality. On the other hand, UEs farther from an AP may require more power to maintain a reliable connection. Since each AP has a limited power budget, serving distant UEs is more costly and might not be efficient. Therefore, the model must learn to balance UE association and power allocation based on spatial relationships between UEs and APs.

We chose to use only this type of signal information (large-scale fading) as input to the model. While other types of signal changes (like quick fluctuations due to movement or interference) exist, they change too fast and are hard to track in real time. Large-scale fading changes slowly and is more reliable for decision-making.

#### What the Dataset Looks Like

For each network setup, the dataset includes:
- **Input**: Large-scale fading coefficients between each AP and each UE.
- **Target output**: The best way to connect UEs to APs and assign power, based on a strong optimization method.

#### Building the Dataset

To figure out what the best decisions look like, we used an optimization method called **Successive Convex Approximation (SCA)**. For each setup, SCA calculated:
- Which AP should serve which UE.
- How much power each AP should use for each UE.

These results became the answers our model should learn from. Once trained, the model can make similar high-quality decisions instantly, without having to run the slow optimization process every time.

This dataset will teaches a model how to make smart network decisions by learning from examples where the best possible choices are already known.

### Model Development

To make fast, smart decisions in a wireless network, we built a machine learning model based on a **Convolutional Neural Network (CNN)**. The goal of the model is to learn from our dataset how to choose the best way to connect users to access points and how to assign transmission power in each scenario.

#### Why We Chose a CNN

We specifically chose a CNN instead of other types of neural networks for several important reasons:

- **Preserving spatial structure**: The problem has an inherent spatial structure (access points and users are laid out in space), and CNNs are excellent at capturing local patterns like strong signals or interference.
- **Efficiency**: CNNs use a small number of parameters compared to fully connected networks. This reduces the risk of overfitting and speeds up training.
- **Scalability**: Because CNNs work locally (only looking at neighboring points at a time), they scale better when the number of APs and UEs grows.
- **Generalization**: CNNs generalize better to unseen network layouts because they focus on local relationships rather than memorizing global structures.
- **Maintaining M x K spatial structure**: CNN operations preserve the dimensions related to APs (M) and UEs (K), ensuring that the output is naturally aligned with the AP-UE relationships.

#### CNN Workflow and Design

For this example, we use normalized power ($$ P_m^{\max} = 1, \forall m $$). 

<img src="/assets/images/CNN_model.png" alt="CNN Model" width="700">

The CNN model architecture is carefully designed based on the following components:

**Input Layer**
- Input: A large-scale fading matrix showing signal strength between APs and UEs.
- Size: M x K (M = number of APs, K = number of UEs).
- Number of training samples: T

**Conv Block 1**
- A 2D convolution layer with 64 3×3 **filters**, giving 64 output learned features (e.g., signal strength patterns, interference cues, proximity effects, context from neighboring APs or UEs, and etc)
- Each filter has weights that are learned during training.
- Purpose: Extract local spatial features in the AP-UE grid. 
- How it interacts with the input: In Conv Block 1, each 3×3 filter slides across the input matrix (which represents AP–UE relationships), examining a small local patch one step at a time. At each position (corresponding to one AP–UE pair), the filter looks at a 3×3 neighborhood, multiplies each input value by its corresponding filter weight, sums them up, and produces one single output feature for that AP–UE pair. Since there are 64 different filters, each AP–UE pair ends up having 64 output features — one feature from each filter — capturing different learned features.
- Why 3×3? A small window captures local dependencies without overwhelming complexity.
- Why 64 output learned features? Enough capacity to learn a wide range of basic features.
- Output size: M x K x 64.
- **Intuition**: The filter acts like a magnifying glass. It scans small local regions of the input. It extracts important local features like "strong connection here," "weak connection there," etc.

**Residual Blocks (ResNet-R)**
- R residual blocks, each containing several convolutional layers.
- Each convolutional layer uses 3×3 filters across 64 channels, maintaining the feature depth.
- Batch Normalization is applied after each convolution to stabilize training by normalizing the feature distributions, speeding up convergence and allowing higher learning rates.
- ReLU activation is applied after Batch Normalization to introduce non-linearity, enabling the network to learn complex, non-linear mappings and preventing vanishing gradients.
- Skip connections are added to each block, allowing the input to be directly added to the output. This helps gradients flow more easily during backpropagation, solving the vanishing gradient problem and enabling much deeper networks to be trained effectively. Skip connections also help the network remember the input, allowing it to preserve important information and only learn the differences (residuals) needed to improve the mapping.
- Purpose: Deepen the network to learn complex hierarchical features, where deeper layers extract more abstract patterns based on local relationships.
- **Intuition**: Normal networks try to learn everything from scratch at every layer. ResNet helps networks learn better by making layers focus only on small improvements from Conv Block 1, not full transformations — while remembering and preserving the good parts of the input through skip connections. Like climbing a mountain. Instead of jumping to random new points, you step slightly up from where you already are. 


**Conv Block 2**
- A 2D convolution layer with 3×3 filters, operating on the 64 feature maps produced by the previous layers.
- Purpose: Refine and unify the deep features extracted by the ResNet blocks.
- Why 3×3? Maintains local spatial relationships while slightly adjusting and smoothing the features before output.
- This layer aggregates the various complex patterns detected by earlier layers into a more consistent, task-friendly representation.
- It prepares a clean and focused feature map that contains distilled information needed for the final predictions made by the output heads.
- Acts as a final adjustment layer to align the network's learned features with the specific needs of power control and user association prediction.


After Conv Block 2, the network splits into two branches:

**Power Control Head**
- Conv layer (3×3, 64 filters) followed by sigmoid activation.
- Outputs normalized power control values (range 0 to 1) for each AP-UE pair.
- Why Conv? Keeps local relationships intact.
- Why Sigmoid? Power values are normalized between 0 and 1.

**User Association Head**
- Conv layer (3×3, 64 filters) followed by sigmoid activation.
- Outputs association probability (range 0 to 1) for each AP-UE pair.
- Why Conv? Maintains the spatial structure.
- Why Sigmoid? Predicts association likelihoods.

Each output head predicts an M x K matrix:
- Rows correspond to Access Points (APs).
- Columns correspond to User Equipments (UEs).

Thus, the model outputs directly reflect the physical structure of the wireless system.

Note that standard practice in CNNs — especially with 3×3 convolutions — involves applying padding of 1 pixel on all sides. This preserves the spatial size of the input, ensuring that the Access Point–User Equipment (AP–UE) relationship grid (M × K) remains consistent throughout the network. Padding is thus implicitly applied in ConvBlock 1, the ResNet blocks, and ConvBlock 2.

#### Why Not Fully Connected Networks?

We considered using fully connected networks but rejected them for several reasons:
- **Too many parameters**: Fully connected layers grow dramatically in size as M and K increase.
- **Loss of spatial structure**: Fully connected layers ignore the natural 2D structure between APs and UEs.
- **Higher overfitting risk**: Fully connected networks tend to memorize training data rather than generalize.



#### Model Training
- dataset
- metric
- matlab

### Model Evaluation

#### Metrics
- Mean Squared Error (MSE) between predicted and true optimization outputs.

#### Results
- Achieved performance very close to traditional optimization (within 97%).
- Reduced run-time by 1000x, enabling real-time deployment.

## Next Steps for Improvement
- Incorporate dynamic network conditions (e.g., moving users) into training data.
- Explore graph neural networks (GNNs) for better scalability to very large networks.
- Add uncertainty quantification to model predictions for reliability.
- Experiment with reinforcement learning to further enhance adaptability.

## Conclusion

Learning to Optimize bridges machine learning and optimization, offering a powerful new way to solve complex business problems faster and smarter.

From network resource allocation to supply chain design, it can change how industries operate.

As problems grow in complexity, Learning to Optimize is a key frontier every data scientist and engineer should watch — and master.

The detailed example in this post is available in my [publication](https://ieeexplore.ieee.org/abstract/document/10387264). 

For further inquiries or collaboration, please contact me at [my email](mailto:tungvutelecom@gmail.com).
