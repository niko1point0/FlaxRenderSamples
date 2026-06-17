// Neural BRDF training + optimizer compute shaders.
//
//   CS_Train - one thread per random sample: forward pass, hand-written backprop, atomically
//              accumulates gradients (matches RTXNS computeTraining.slang, but the Slang autodiff
//              backward pass is written out explicitly here since HLSL has no autodiff).
//   CS_Adam  - one thread per parameter: Adam update (matches RTXNS computeOptimizer.slang / Optimizers.slang).
//
// Portable fp32 implementation (runs on stock Flax SM5+). The matrix/vector products and the
// gradient accumulation are the parts that a future SM6.9 + cooperative-vector backend would
// replace with dx::linalg MulAdd / OuterProductAccumulate for hardware acceleration.

#include "./Flax/Common.hlsl"
#include "./neuralbrdf/NeuralBrdfCommon.hlsl"

META_CB_BEGIN(0, TrainData)
uint Seed0;
uint Seed1;
uint Step;          // 1-based Adam step
uint ParamCount;
float LearningRate;
uint BatchSize;
float Pad0;
float Pad1;
META_CB_END

#define NB_LOSS_EPSILON 0.01f

// ---- CS_Train resources ----
Buffer<float> Params           : register(t0); // current weights (read-only during training)
RWByteAddressBuffer Grads      : register(u0); // accumulated gradients (fp32 via atomic add)
RWByteAddressBuffer LossAccum  : register(u1); // single fp32: mean L2-relative error over batch

// Float atomic add via compare-exchange loop (portable; no native fp32 atomics pre-SM6.6).
void AtomicAddFloat(RWByteAddressBuffer buf, uint elemIndex, float value)
{
    uint byteOffset = elemIndex * 4u;
    uint orig = buf.Load(byteOffset);
    uint comp;
    // Bounded retry: caps GPU work so heavy contention can never hang the device (TDR).
    [allow_uav_condition]
    for (uint attempt = 0; attempt < 1024u; ++attempt)
    {
        comp = orig;
        float updated = asfloat(comp) + value;
        buf.InterlockedCompareExchange(byteOffset, comp, asuint(updated), orig);
        if (orig == comp)
            break;
    }
}

META_CS(true, FEATURE_LEVEL_SM5)
[numthreads(NB_TRAIN_GROUP, 1, 1)]
void CS_Train(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= BatchSize)
        return;

    // ---- Randomly generate input parameters (matches computeTraining.slang) ----
    NB_PCG32 rng = NB_PCG_Init(uint2(Seed0 + idx, Seed1));

    // L in the tangent frame, N=(0,0,1); first-quadrant in XZ so NdotL = L.z >= 0.
    float3 L;
    L.y = 0.0f;
    sincos(NB_PCG_Float(rng) * NB_PI * 0.5f, L.z, L.x);

    // V random direction with NdotV = V.z >= 0.
    float sa, ca;
    sincos(-NB_PI + 2.0f * NB_PI * NB_PCG_Float(rng), sa, ca);
    float se, ce;
    sincos(NB_PI * 0.5f * NB_PCG_Float(rng), se, ce);
    float3 V = float3(ce * ca, ce * sa, se);

    float NdotL = L.z;
    float NdotV = V.z;
    float3 H = normalize(L + V);
    float NdotH = H.z;
    float LdotH = dot(L, H);
    float roughness = NB_PCG_Float(rng) * 0.7f + 0.3f;

    float4 target = NB_Disney(NdotL, NdotV, NdotH, LdotH, roughness);

    // ---- Encode inputs ----
    float feats[NB_INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, roughness };
    float enc[NB_INPUT_NEURONS];
    NB_EncodeFrequency(feats, enc);

    // ---- Forward pass (store pre-activations for backprop) ----
    float z0[NB_HIDDEN_NEURONS], a0[NB_HIDDEN_NEURONS];
    for (int o = 0; o < NB_HIDDEN_NEURONS; ++o)
    {
        float s = Params[NB_L0_B_OFF + o];
        for (int i = 0; i < NB_INPUT_NEURONS; ++i)
            s += Params[NB_L0_W_OFF + o * NB_INPUT_NEURONS + i] * enc[i];
        z0[o] = s; a0[o] = max(0.0f, s);
    }

    float z1[NB_HIDDEN_NEURONS], a1[NB_HIDDEN_NEURONS];
    for (int o1 = 0; o1 < NB_HIDDEN_NEURONS; ++o1)
    {
        float s = Params[NB_L1_B_OFF + o1];
        for (int i = 0; i < NB_HIDDEN_NEURONS; ++i)
            s += Params[NB_L1_W_OFF + o1 * NB_HIDDEN_NEURONS + i] * a0[i];
        z1[o1] = s; a1[o1] = max(0.0f, s);
    }

    float z2[NB_HIDDEN_NEURONS], a2[NB_HIDDEN_NEURONS];
    for (int o2 = 0; o2 < NB_HIDDEN_NEURONS; ++o2)
    {
        float s = Params[NB_L2_B_OFF + o2];
        for (int i = 0; i < NB_HIDDEN_NEURONS; ++i)
            s += Params[NB_L2_W_OFF + o2 * NB_HIDDEN_NEURONS + i] * a1[i];
        z2[o2] = s; a2[o2] = max(0.0f, s);
    }

    float z3[NB_OUTPUT_NEURONS];
    float4 outp;
    for (int o3 = 0; o3 < NB_OUTPUT_NEURONS; ++o3)
    {
        float s = Params[NB_L3_B_OFF + o3];
        for (int i = 0; i < NB_HIDDEN_NEURONS; ++i)
            s += Params[NB_L3_W_OFF + o3 * NB_HIDDEN_NEURONS + i] * a2[i];
        z3[o3] = s; outp[o3] = exp(s);
    }

    // ---- Loss gradient (L2 relative, matches Loss.slang L2Relative.deriv) ----
    float sum = NB_LOSS_EPSILON;
    for (int t = 0; t < NB_OUTPUT_NEURONS; ++t)
        sum += target[t] * target[t];
    float invSum = 1.0f / sum;
    float4 scale = (NB_LOSS_SCALE / (BatchSize * 4.0f)) * NB_COMPONENT_WEIGHTS;

    // dL/d(out)
    float gOut[NB_OUTPUT_NEURONS];
    for (int g = 0; g < NB_OUTPUT_NEURONS; ++g)
        gOut[g] = scale[g] * (outp[g] - target[g]) * 2.0f * invSum;

    // ---- Backward pass ----
    // Output layer (exponential activation): dz = dL/dout * d(exp)/dz = gOut * out
    float dz3[NB_OUTPUT_NEURONS];
    for (int b3 = 0; b3 < NB_OUTPUT_NEURONS; ++b3)
        dz3[b3] = gOut[b3] * outp[b3];

    for (int wo3 = 0; wo3 < NB_OUTPUT_NEURONS; ++wo3)
    {
        AtomicAddFloat(Grads, NB_L3_B_OFF + wo3, dz3[wo3]);
        for (int i = 0; i < NB_HIDDEN_NEURONS; ++i)
            AtomicAddFloat(Grads, NB_L3_W_OFF + wo3 * NB_HIDDEN_NEURONS + i, dz3[wo3] * a2[i]);
    }

    // Propagate to layer 2 (ReLU)
    float dz2[NB_HIDDEN_NEURONS];
    for (int p2 = 0; p2 < NB_HIDDEN_NEURONS; ++p2)
    {
        float d = 0.0f;
        for (int o = 0; o < NB_OUTPUT_NEURONS; ++o)
            d += dz3[o] * Params[NB_L3_W_OFF + o * NB_HIDDEN_NEURONS + p2];
        dz2[p2] = (z2[p2] > 0.0f) ? d : 0.0f;
    }
    for (int wo2 = 0; wo2 < NB_HIDDEN_NEURONS; ++wo2)
    {
        AtomicAddFloat(Grads, NB_L2_B_OFF + wo2, dz2[wo2]);
        for (int i = 0; i < NB_HIDDEN_NEURONS; ++i)
            AtomicAddFloat(Grads, NB_L2_W_OFF + wo2 * NB_HIDDEN_NEURONS + i, dz2[wo2] * a1[i]);
    }

    // Propagate to layer 1 (ReLU)
    float dz1[NB_HIDDEN_NEURONS];
    for (int p1 = 0; p1 < NB_HIDDEN_NEURONS; ++p1)
    {
        float d = 0.0f;
        for (int o = 0; o < NB_HIDDEN_NEURONS; ++o)
            d += dz2[o] * Params[NB_L2_W_OFF + o * NB_HIDDEN_NEURONS + p1];
        dz1[p1] = (z1[p1] > 0.0f) ? d : 0.0f;
    }
    for (int wo1 = 0; wo1 < NB_HIDDEN_NEURONS; ++wo1)
    {
        AtomicAddFloat(Grads, NB_L1_B_OFF + wo1, dz1[wo1]);
        for (int i = 0; i < NB_HIDDEN_NEURONS; ++i)
            AtomicAddFloat(Grads, NB_L1_W_OFF + wo1 * NB_HIDDEN_NEURONS + i, dz1[wo1] * a0[i]);
    }

    // Propagate to layer 0 (ReLU), inputs are the encoding (no input gradient needed)
    float dz0[NB_HIDDEN_NEURONS];
    for (int p0 = 0; p0 < NB_HIDDEN_NEURONS; ++p0)
    {
        float d = 0.0f;
        for (int o = 0; o < NB_HIDDEN_NEURONS; ++o)
            d += dz1[o] * Params[NB_L1_W_OFF + o * NB_HIDDEN_NEURONS + p0];
        dz0[p0] = (z0[p0] > 0.0f) ? d : 0.0f;
    }
    for (int wo0 = 0; wo0 < NB_HIDDEN_NEURONS; ++wo0)
    {
        AtomicAddFloat(Grads, NB_L0_B_OFF + wo0, dz0[wo0]);
        for (int i = 0; i < NB_INPUT_NEURONS; ++i)
            AtomicAddFloat(Grads, NB_L0_W_OFF + wo0 * NB_INPUT_NEURONS + i, dz0[wo0] * enc[i]);
    }

    // ---- Accumulate L2 relative error for the UI ----
    float4 diff = outp - target;
    float l2e = sqrt(dot(diff, diff));
    float l2t = sqrt(dot(target, target));
    AtomicAddFloat(LossAccum, 0, (l2e / (l2t + 1e-6f)) / BatchSize);
}

// ---- CS_Adam resources ----
RWBuffer<float> ParamsRW       : register(u0);
RWByteAddressBuffer GradsRW    : register(u1);
RWBuffer<float> Moments1       : register(u2);
RWBuffer<float> Moments2       : register(u3);

META_CS(true, FEATURE_LEVEL_SM5)
[numthreads(NB_OPTIMIZE_GROUP, 1, 1)]
void CS_Adam(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i >= ParamCount)
        return;

    float gradient = asfloat(GradsRW.Load(i * 4u));
    GradsRW.Store(i * 4u, asuint(0.0f)); // reset for next step

    if (!isfinite(gradient))
        return;

    gradient /= NB_LOSS_SCALE;
    if (isnan(gradient) || isinf(gradient))
        gradient = 0.0f;

    float weight = ParamsRW[i];

    float m1 = Moments1[i] * NB_ADAM_BETA1 + gradient * (1.0f - NB_ADAM_BETA1);
    float m2 = Moments2[i] * NB_ADAM_BETA2 + gradient * gradient * (1.0f - NB_ADAM_BETA2);

    float stepF = (float)Step;
    float bc1 = 1.0f - pow(NB_ADAM_BETA1, stepF);
    float bc2 = 1.0f - pow(NB_ADAM_BETA2, stepF);

    float denom = sqrt(m2) * rsqrt(bc2) + NB_ADAM_EPSILON;
    float stepSize = LearningRate / bc1;
    float updated = weight - (m1 / denom) * stepSize;

    Moments1[i] = m1;
    Moments2[i] = m2;
    ParamsRW[i] = updated;
}

// ---------------------------------------------------------------------------
// CS_PackF16: packs the fp32 master weights (Params) into a contiguous fp16 row-major buffer
// (element i at byte 2*i), the source for the GPU cooperative-vector layout conversion on the host.
// Each thread packs two consecutive fp32 elements into one 32-bit store (NB_PARAM_COUNT is even).
// Plain SM5 / cs_6_0 - no cooperative vectors here, just a precision cast.
// ---------------------------------------------------------------------------
RWByteAddressBuffer WeightsF16 : register(u0);

META_CS(true, FEATURE_LEVEL_SM5)
[numthreads(64, 1, 1)]
void CS_PackF16(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint pair = dispatchThreadID.x;
    uint i0 = pair * 2u;
    if (i0 >= NB_PARAM_COUNT)
        return;
    uint lo = f32tof16(Params[i0]);
    uint hi = (i0 + 1u < NB_PARAM_COUNT) ? f32tof16(Params[i0 + 1u]) : 0u;
    WeightsF16.Store(pair * 4u, lo | (hi << 16u));
}
