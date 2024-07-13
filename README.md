# Protecting Privacy of Data Buyers in Data Markets

This repository contains the code for the paper "Protecting Privacy of Data Buyers in Data Markets." The short version of this paper is accepted in IEEE Internet Computing ([DOI: 10.1109/MIC.2024.3398626](https://doi.org/10.1109/MIC.2024.3398626)
).

## Authors
- Minxing Zhang ([minxing.zhang@duke.edu](mailto:minxing.zhang@duke.edu))
- Jian Pei ([j.pei@duke.edu](mailto:j.pei@duke.edu))

## Abstract
Data markets serve as crucial platforms facilitating data discovery, exchange, sharing, and integration among data users and providers. However, the paramount concern of privacy has predominantly centered on protecting privacy of data owners and third parties, neglecting the challenges associated with protecting the privacy of data buyers. In this article, we address this gap by modeling the intricacies of data buyer privacy protection and investigating the delicate balance between privacy and purchase cost. Through comprehensive experimentation, our results yield valuable insights, shedding light on the efficacy and efficiency of our proposed approaches.

## Running Instructions
To replicate the main results, follow the instructions below:

### Main Results
- **Primary Script**: Run `main.py` to execute the main results, including:
  - PI-uniform attack
  - Efficiency maximization attack
  - Purchased record inference attack
  - Associated defense strategies

### Ablation Studies
- **True Intent Size Influence**:
  - `case_increase_TI_PI.py`: Examines the impact of true intent size for the PI-uniform attack.
  - `case_increase_TI_EM.py`: Examines the impact of true intent size for the efficiency maximization attack.
- **Dimensionality Impact**:
  - `case_reduce_dimensionality_PI.py`: Studies the effects of dimensionality reduction on the PI-uniform attack.
  - `case_reduce_dimensionality_EM.py`: Studies the effects of dimensionality reduction on the efficiency maximization attack.
- **Mixed Dimensions**:
  - `case_increase_TI_mix_age_dimension.py`: Investigates the influence of true intent on the Age dimension for both PI-uniform and efficiency maximization attacks.
  - `case_increase_TI_mix_race_dimension.py`: Investigates the influence of true intent on the Ethnicity dimension for both attack types.
  - `case_increase_TI_mix_hpw_dimension.py`: Investigates the influence of true intent on the Hours Per Week dimension for both attack strategies.
- **Parameter and Threshold Impacts**:
  - `case_weight_parameter.py`: Explores the influence of the parameter alpha used in the expansion method.
  - `case_impact_of_lambda.py`: Explores the influence of the privacy threshold lambda.
