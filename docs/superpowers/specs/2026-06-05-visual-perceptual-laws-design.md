# Visual Perceptual Laws Implementation — Design Doc

## Overview
Implement 4 missing perceptual mechanisms in NotMe's visual pipeline:
- **B**: Brightness/Color Constancy (Divisive Normalization)
- **A**: Gestalt Grouping Layer (Proximity, Continuity, Similarity, Symmetry, Figure-Ground)
- **C**: Visual Saliency Map (Center-surround + Attention Modulation)
- **D**: Visual Predictive Coding (Hierarchical prediction errors V1↔V2↔V4)

Order: B → A → C → D (dependency-driven, validate each stage)

## Module B: Brightness/Color Constancy

**File**: `layer0_visual.py` (modify GaborFilterBank)

**Mechanism**: Divisive normalization (V1 surround suppression)
- R_norm[i,x,y] = R[i,x,y] / (σ² + local_energy[i,x,y])
- local_energy = Gaussian-smoothed |R|² over 3×3 spatial + same-scale neighbor orientations
- σ² = 0.1 (semi-saturation constant)
- For color: cross-channel normalization RG/(1+BY_energy), BY/(1+RG_energy)

**Changes**:
1. New method `_divisive_normalize()` 
2. Precompute surround kernel FFT in `__init__`
3. Insert after Gabor response computation in `encode()`, `encode_v2()`, `encode_v4()`, `encode_color()`

**Acceptance**: Same image × different lighting → cosine > 0.85

## Module A: Gestalt Grouping Layer

**File**: `layer0_gestalt.py` (new)

**Mechanisms**:
1. **Proximity Grouping**: Spatial clustering of Gabor responses within radius r=image_size/16
2. **Continuity (Contour Integration)**: Connect co-linear edge fragments along Gabor orientation axis; chain responses where orientation difference < 30°
3. **Similarity Grouping**: Cross-orientation, same-scale response correlation → texture regions
4. **Symmetry Detection**: Mirror symmetry on V4 global pooling — compare response histogram across vertical/horizontal axes
5. **Figure-Ground Segregation**: Edge density heatmap → threshold → foreground mask. Higher edge density = figure.

**Output**: Grouping feature vector (concatenated to visual sensory before PCA)

**Class**: `GestaltGrouping`
- `group_proximity(responses) → proximity_features`
- `integrate_contours(responses) → contour_features`  
- `group_similarity(responses) → similarity_features`
- `detect_symmetry(v4_features) → symmetry_score`
- `figure_ground_mask(responses) → fg_mask, bg_features`
- `compute_all(image, gabor_bank) → grouping_vector` (main entry)

## Module C: Visual Saliency Map

**File**: `layer0_visual.py` (modify GaborFilterBank, new method)

**Mechanism**: Center-surround difference on Gabor responses (Itti & Koch inspired, but purely signal-processing)
- Saliency[i,x,y] = |R_center[i,x,y] - R_surround[i,x,y]|
- Center: pixel, Surround: Gaussian blurred (σ=image_size/8)
- Cross-scale: fine scale (σ=2) vs coarse scale (σ=8) difference
- Integration: sum across all filters, normalized to [0,1]
- L1 attention_precision modulates saliency gain
- IOR: exponential decay mask on recently attended locations (τ=5 steps)

**New method**: `compute_saliency(image) → (saliency_map, attended_features)`

## Module D: Visual Predictive Coding

**File**: `layer1_free_energy.py` (modify), `layer0_visual.py` (minor)

**Mechanism**: Hierarchical prediction errors between visual areas
- V1 predicts V2 responses; V2 predicts V4 responses
- Prediction error = ||actual - predicted||²
- Precision-weighted by attention (from Module C)
- F_accuracy augmented with visual prediction error component

**Changes**:
1. GaborFilterBank stores last response maps for V1, V2, V4
2. `compute_visual_prediction_error(v1, v2, v4, last_v1, last_v2, last_v4) → F_visual_pred`
3. Integrated into `compute_free_energy()` as additional F_accuracy component

## Core Principles Compliance
- Zero ML training parameters (all fixed mathematical transforms)
- FEP-native: grouping reduces prediction error; saliency = precision modulation
- Hebb-compatible: grouping co-activations → wire together in ClusterNetwork
- No LLM dependency
- Biological homologies documented for each mechanism
