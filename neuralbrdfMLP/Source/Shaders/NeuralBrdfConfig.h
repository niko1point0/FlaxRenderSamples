// Neural BRDF network configuration.
// Shared between HLSL shaders and C++ host code (only #defines / POD here).
//
// Ported from NVIDIA RTXNS "ShaderTraining" sample (NetworkConfig.h). The MLP
// approximates the Disney BRDF used in renderDisney: Disney(NdotL, NdotV, NdotH, LdotH, roughness).

#ifndef NEURAL_BRDF_CONFIG_H
#define NEURAL_BRDF_CONFIG_H

// ----- Network architecture (must match the trained .bin model) -----
#define NB_INPUT_FEATURES      5                                   // NdotL, NdotV, NdotH, LdotH, roughness
#define NB_FREQ_ENCODING       6                                   // each feature expanded to 6 values
#define NB_INPUT_NEURONS       (NB_INPUT_FEATURES * NB_FREQ_ENCODING) // 30
#define NB_HIDDEN_NEURONS      32
#define NB_OUTPUT_NEURONS      4                                   // 4 BRDF values
#define NB_NUM_HIDDEN_LAYERS   3
#define NB_NUM_LAYERS          (NB_NUM_HIDDEN_LAYERS + 1)          // 4 weight matrices

// ----- Training hyper parameters -----
#define NB_LEARNING_RATE       0.001f
#define NB_LOSS_SCALE          128.0f
// Per-output loss weighting (matches RTXNS COMPONENT_WEIGHTS).
#define NB_COMPONENT_WEIGHTS   float4(1.0f, 10.0f, 1.0f, 5.0f)

// Batch configuration. Smaller than RTXNS (1<<16) because the portable HLSL path
// accumulates gradients with atomics rather than hardware cooperative-vector ops.
#define NB_BATCH_SIZE          4096
#define NB_TRAIN_GROUP         64
#define NB_OPTIMIZE_GROUP      64

// ----- Adam -----
#define NB_ADAM_BETA1          0.9f
#define NB_ADAM_BETA2          0.999f
#define NB_ADAM_EPSILON        1e-8f

// ----- Parameter buffer layout (row-major, contiguous fp32 elements) -----
// Layer L has (outputs * inputs) weights followed by (outputs) biases.
//   inputs(L)  = (L == 0) ? NB_INPUT_NEURONS : NB_HIDDEN_NEURONS
//   outputs(L) = (L == NB_NUM_LAYERS-1) ? NB_OUTPUT_NEURONS : NB_HIDDEN_NEURONS
// Offsets below are element indices into the fp32 parameter array.
#define NB_L0_W_OFF   0
#define NB_L0_W_CNT   (NB_HIDDEN_NEURONS * NB_INPUT_NEURONS)        // 32*30 = 960
#define NB_L0_B_OFF   (NB_L0_W_OFF + NB_L0_W_CNT)                   // 960
#define NB_L0_B_CNT   NB_HIDDEN_NEURONS                             // 32

#define NB_L1_W_OFF   (NB_L0_B_OFF + NB_L0_B_CNT)                   // 992
#define NB_L1_W_CNT   (NB_HIDDEN_NEURONS * NB_HIDDEN_NEURONS)       // 1024
#define NB_L1_B_OFF   (NB_L1_W_OFF + NB_L1_W_CNT)                   // 2016
#define NB_L1_B_CNT   NB_HIDDEN_NEURONS                             // 32

#define NB_L2_W_OFF   (NB_L1_B_OFF + NB_L1_B_CNT)                   // 2048
#define NB_L2_W_CNT   (NB_HIDDEN_NEURONS * NB_HIDDEN_NEURONS)       // 1024
#define NB_L2_B_OFF   (NB_L2_W_OFF + NB_L2_W_CNT)                   // 3072
#define NB_L2_B_CNT   NB_HIDDEN_NEURONS                             // 32

#define NB_L3_W_OFF   (NB_L2_B_OFF + NB_L2_B_CNT)                   // 3104
#define NB_L3_W_CNT   (NB_OUTPUT_NEURONS * NB_HIDDEN_NEURONS)       // 128
#define NB_L3_B_OFF   (NB_L3_W_OFF + NB_L3_W_CNT)                   // 3232
#define NB_L3_B_CNT   NB_OUTPUT_NEURONS                             // 4

#define NB_PARAM_COUNT (NB_L3_B_OFF + NB_L3_B_CNT)                  // 3236

#endif // NEURAL_BRDF_CONFIG_H
