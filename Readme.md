
This repository contains the Pytorch implementation of

A Dynamic Multi-Modal Deep Reinforcement Learning Framework for 3D Bin Packing Problem

Anhao Zhao, Caixia Rong, Tianrui Li,Liangcai Lin

The 3D bin packing problem, a notorious NPhard combinatorial optimization challenge with wide-ranging practical implications, focuses on optimizing spatial allocation within a bin through the arrangement of boxes. Previous studies have shown promise in employing Neural Combinatorial Optimization to tackle such intractable problems. However, due to the inherent diversity in observations and the sparse rewards, current learning-based methods for addressing the 3D bin packing problem have yielded less-than-ideal outcomes. In response to this shortfall, we propose a novel Dynamic Multi-modal deep Reinforcement Learning framework tailored specifically for the 3D Bin Packing Problem, coined as DMRLBPP. This framework stands apart from existing learning-based bin-packing approaches in two pivotal respects. Firstly, in order to capture the range of observations, we introduce an innovative dynamic multi-modal encoder, comprising dynamic boxes state and gradient height map sub-encoders, effectively modeling multi-modal information. Secondly, to mitigate the challenge of sparse rewards, we put forth a novel reward function, offering an efficient and adaptable approach for addressing the 3D bin packing problem. Extensive experimental results validate the superiority of our method over all baseline approaches, across instances of varying scales.

## Usage

### Preparation

1. Install conda
2. Run `conda env create -f environment.yml`

### Train

1. Modify the config file in `config.py` as you need.
2. Run `python main.py`.
