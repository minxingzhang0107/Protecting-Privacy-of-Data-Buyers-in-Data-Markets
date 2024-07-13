# Hyperparameter Selection 

## Data Loading and Iteration Details (Iteration numbers are utilized to track and differentiate various configurations)
- **Iteration 1**: 
  - Data: Real-world Adult Dataset
  - Cost: Unit Cost of 1
  - True Intent Size: 1
- **Iteration 2**: 
  - Data: Real-world Adult Dataset
  - Cost: Unit Cost of 1
  - True Intent Size: 2
- **Iteration 3**: 
  - Data: Synthetic Dataset
  - Cost: Synthetic Gaussian Distribution
  - True Intent Size: 2
- **Iteration 4**: 
  - Data: Synthetic Dataset
  - Cost: Synthetic Gaussian Distribution
  - True Intent Size: 1

## Alpha

### Real-world Adult Dataset
- **True Intent Size of 1**:
  - PI-uniform: 0.5
  - EM-f: 0.5
  - EM-c: 0.5
  - EM-fc: 0.5
- **True Intent Size of 2**:
  - PI-uniform: 1.0
  - EM-f: 0.6
  - EM-c: 0.5
  - EM-fc: 0.6

### Synthetic Dataset
- **True Intent Size of 1**:
  - PI-uniform: 0.8
  - EM-f: 0.4
  - EM-c: 0.8
  - EM-fc: 0.4
- **True Intent Size of 2**:
  - PI-uniform: 0.6
  - EM-f: 0.4
  - EM-c: 0.7
  - EM-fc: 0.5

## Alpha (Dimension Reduction)

### Real-world Data
- **Reduce Age**:
  - PI-uniform: 1
  - EM-fc: 0.6
- **Reduce Ethnicity**:
  - PI-uniform: 1
  - EM-fc: 0.5
- **Reduce Gender**:
  - PI-uniform: 1
  - EM-fc: 0.5
- **Reduce Hours Per Week**:
  - PI-uniform: 0.8
  - EM-fc: 0.6
- **Reduce Income**:
  - PI-uniform: 1
  - EM-fc: 0.4

## Epsilon (Differences in Confidence and Privacy Thresholds) 

### PRI Uniform Attack: G-MCMC
- **Real-world Adult Dataset**:
  - 1 True Intent: 0.15
  - 2 True Intents: 0.01
- **Synthetic set**:
  - 1 True Intent: 0.05
  - 2 True Intents: 0.05
  - 2 True Intents, Unit Cost: 0.05
  - 1 True Intent, Unit Cost: 0.05

### PRI Uniform Attack: MCMC
- **Real-world Adult Dataset**:
  - 1 True Intent: 0.01
  - 2 True Intents: 0.1
- **Synthetic Dataset**:
  - 1 True Intent: 0.0001
  - 2 True Intents: 0.0001
  - 2 True Intents, Unit Cost: 0.1
  - 1 True Intent, Unit Cost: 0.05
