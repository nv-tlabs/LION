# Model Overview

## Description:
LION is a point cloud generation model. This model is for research and development only


### License/Terms of Use: 
https://github.com/nv-tlabs/LION/blob/main/LICENSE.txt 

## References (Leave Blank If None):
https://arxiv.org/abs/2210.06978

## Model Architecture: 
**Architecture Type:** CNN  <br>
**Network Architecture:** PVCNN <br>

## Input:
**Input Type(s):** Noisy point cloud <br>
**Input Format(s):** X,Y,Z <br>
**Input Parameters:** Three-Dimensional (3D) <br>


## Output: 
**Output Type(s):** Point Cloud <br>
**Output Format:** X,Y,Z <br>
**Output Parameters:** 3D <br>


## Software Integration (Required For NVIDIA Commercial Models Only):
**Runtime Engine(s):** 
* PyTorch <br> 

**Supported Hardware Microarchitecture Compatibility:** <br>
* [NVIDIA Ampere] <br>
* [NVIDIA Blackwell] <br>
* [NVIDIA Jetson]  <br>
* [NVIDIA Hopper] <br>
* [NVIDIA Lovelace] <br>
* [NVIDIA Pascal] <br>
* [NVIDIA Turing] <br>
* [NVIDIA Volta] <br>

**[Preferred/Supported] Operating System(s):** <br>
* [Linux] <br>


## Model Version(s): 
V1.0 model describe in the paper <br>

# Training, Testing, and Evaluation Datasets: 

## Training Dataset:

**Link:** ShapeNet
** Data Collection Method by dataset <br>
* Unknown <br>
** Labeling Method by dataset <br>
* Unknown <br>



## Inference:
**Engine:** Pytorch <br>
**Test Hardware [Name the specific test hardware model]:** <br>
* Driver Version: 525.116.03
* NVIDIA RTX A6000 [Ampere architecture] 


## Ethical Considerations (For NVIDIA Models Only):
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  



