// Shared neural-BRDF HLSL: Disney BRDF (ground truth), frequency encoding, RNG and activations.
//
// Ported from NVIDIA RTXNS "ShaderTraining": Disney.slang, Utils.slang (EncodeFrequency),
// and Activation.slang (ReLU / Exponential).
//
// Inference itself lives in NeuralBrdf.shader and runs exclusively through cooperative vectors
// (dx::linalg MatVecMulAdd on D3D12, inline SPV_NV_cooperative_vector on Vulkan); the training
// kernels still use explicit fp32 loops for the backward pass.

#ifndef NEURAL_BRDF_COMMON_HLSL
#define NEURAL_BRDF_COMMON_HLSL

// Flax resolves shader includes as "./<ProjectName>/<path under Source/Shaders>".
#include "./neuralbrdf/NeuralBrdfConfig.h"

static const float NB_PI = 3.14159265358979323846f;

// ----------------------------------------------------------------------------
// Disney BRDF (analytic ground truth). Returns 4 channels:
//   x = diffuse (subsurface) term, y = specular D*G, z = Fresnel, w = clearcoat
// ----------------------------------------------------------------------------
float NB_SchlickFresnel(float u)
{
    float m = clamp(1 - u, 0, 1);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

float NB_Gtr1(float NdotH, float a)
{
    if (a >= 1)
        return 1 / NB_PI;
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (NB_PI * log(a2) * t);
}

float NB_Gtr2(float NdotH, float ax)
{
    float a = ax * (1 / ax / ax * (1 - NdotH * NdotH) + NdotH * NdotH);
    return 1 / (NB_PI * a * a);
}

float NB_SmithGGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

float NB_SmithGGXAniso(float NdotV, float ax)
{
    return 1 / (NdotV + sqrt(ax * ax * (1 - NdotV * NdotV) + NdotV * NdotV));
}

float4 NB_Disney(float NdotL, float NdotV, float NdotH, float LdotH, float roughness)
{
    float FL = NB_SchlickFresnel(NdotL), FV = NB_SchlickFresnel(NdotV);
    float Fss90 = LdotH * LdotH * roughness;
    float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.f / (NdotL + NdotV) - .5f) + .5f);

    float ax = max(.001f, roughness * roughness);
    float Ds = NB_Gtr2(NdotH, ax);
    float FH = NB_SchlickFresnel(LdotH);
    float Gs = NB_SmithGGXAniso(NdotL, ax);
    Gs *= NB_SmithGGXAniso(NdotV, ax);

    float Dr = NB_Gtr1(NdotH, .01f);
    float Fr = lerp(.04f, 1.0f, FH);
    float Gr = NB_SmithGGX(NdotL, .25f) * NB_SmithGGX(NdotV, .25f);

    return float4((1 / NB_PI) * ss, Gs * Ds, FH, .25 * Gr * Fr * Dr);
}

// Combine the 4 BRDF channels into a lit color (shared by ground-truth and neural passes).
float3 NB_Shade(float4 brdf, float3 baseColor, float specular, float metallic, float NdotL, float3 lightIntensity)
{
    float3 Cdlin = pow(max(baseColor, 0.0f), 2.2f);
    float3 Cspec0 = lerp(specular * 0.08f.xxx, Cdlin, metallic);
    float3 brdfn = brdf.x * Cdlin * (1 - metallic) + brdf.y * lerp(Cspec0, 1.0f.xxx, brdf.z) + brdf.w;
    return brdfn * NdotL * lightIntensity;
}

// ----------------------------------------------------------------------------
// Frequency encoding (RTXNS Utils.slang EncodeFrequency): expands each input to 6 values
//   sin(pi x), cos(pi x), sin(2pi x), cos(2pi x), sin(4pi x), cos(4pi x)
// ----------------------------------------------------------------------------
void NB_EncodeFrequency(float params[NB_INPUT_FEATURES], out float outEnc[NB_INPUT_NEURONS])
{
    for (int i = 0; i < NB_INPUT_FEATURES; ++i)
    {
        float sn, cn;
        sincos(params[i] * NB_PI, sn, cn);
        int i1 = i * NB_FREQ_ENCODING;
        outEnc[i1 + 0] = sn;
        outEnc[i1 + 1] = cn;
        outEnc[i1 + 2] = 2.0f * outEnc[i1 + 0] * outEnc[i1 + 1];
        outEnc[i1 + 3] = 2.0f * outEnc[i1 + 1] * outEnc[i1 + 1] - 1.0f;
        outEnc[i1 + 4] = 2.0f * outEnc[i1 + 2] * outEnc[i1 + 3];
        outEnc[i1 + 5] = 2.0f * outEnc[i1 + 3] * outEnc[i1 + 3] - 1.0f;
    }
}

// ----------------------------------------------------------------------------
// PCG32 RNG (for generating random training samples on the GPU).
// ----------------------------------------------------------------------------
struct NB_PCG32
{
    uint state;
    uint inc;
};

NB_PCG32 NB_PCG_Init(uint2 seed)
{
    NB_PCG32 rng;
    rng.state = 0u;
    rng.inc = (seed.y << 1u) | 1u;
    // advance once mixing the seed
    rng.state = rng.state * 747796405u + rng.inc;
    rng.state += seed.x;
    rng.state = rng.state * 747796405u + rng.inc;
    return rng;
}

uint NB_PCG_Next(inout NB_PCG32 rng)
{
    uint oldstate = rng.state;
    rng.state = oldstate * 747796405u + rng.inc;
    uint word = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
    return (word >> 22u) ^ word;
}

float NB_PCG_Float(inout NB_PCG32 rng)
{
    return (NB_PCG_Next(rng) >> 8) * (1.0f / 16777216.0f); // [0,1)
}

#endif // NEURAL_BRDF_COMMON_HLSL
