// Neural BRDF draw shader: renders a sphere three ways.
//   PS_Disney     - analytic Disney BRDF (ground truth)
//   PS_InferenceCV - the trained MLP approximation, evaluated with NV cooperative vectors
//   PS_Difference  - |ground truth - neural| (amplified) for visualising approximation error
//
// Inference runs exclusively through cooperative vectors (NVIDIA Neural Shading): dx::linalg
// MatVecMulAdd on D3D12 (DXIL SM6.9), or hand-emitted SPV_NV_cooperative_vector on Vulkan.
// There is no fp32 inference fallback - the sample requires a cooperative-vector-capable backend.
//
// Ported from RTXNS ShaderTraining (renderDisney.slang / renderInference.slang / renderDifference.slang).
//
// coopvec-rev: 6 (bump this to force a shader-cache recompile of the cooperative-vector path)

#include "./Flax/Common.hlsl"
#include "./neuralbrdf/NeuralBrdfCommon.hlsl"

META_CB_BEGIN(0, Data)
float4x4 ViewProject;
float4 CameraPos;
float4 LightDir;        // direction the light travels (as in RTXNS)
float4 LightIntensity;
float4 BaseColor;
float Specular;
float Roughness;
float Metallic;
float DiffScale;        // amplification for the difference view
int4 WeightOffsets;     // coopvec: byte offsets of layer 0..3 MUL_OPTIMAL weight matrices in WeightsOpt
int4 BiasOffsets;       // coopvec: byte offsets of layer 0..3 fp16 bias vectors in WeightsOpt
META_CB_END

struct VS2PS
{
    float4 Position : SV_Position;
    float3 Normal   : NORMAL;
    float3 View     : TEXCOORD0;
};

META_VS(true, FEATURE_LEVEL_SM5)
META_VS_IN_ELEMENT(POSITION, 0, R32G32B32_FLOAT, 0, 0, PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(NORMAL, 0, R32G32B32_FLOAT, 0, 12, PER_VERTEX, 0, true)
VS2PS VS(float3 Position : POSITION0, float3 Normal : NORMAL0)
{
    VS2PS output;
    // Column-vector convention (matrices uploaded with SaschaGpu::StoreVulkan).
    output.Position = mul(ViewProject, float4(Position, 1.0f));
    output.Normal = Normal;
    output.View = CameraPos.xyz - Position;
    return output;
}

// Compute the 4 BRDF angle dot-products shared by all passes.
void GetAngles(VS2PS input, out float NdotL, out float NdotV, out float NdotH, out float LdotH)
{
    float3 view = normalize(input.View);
    float3 norm = normalize(input.Normal);
    float3 h = normalize(-LightDir.xyz + view);
    NdotL = max(0.0f, dot(norm, -LightDir.xyz));
    NdotV = max(0.0f, dot(norm, view));
    NdotH = max(0.0f, dot(norm, h));
    LdotH = max(0.0f, dot(h, -LightDir.xyz));
}

META_PS(true, FEATURE_LEVEL_SM5)
float4 PS_Disney(VS2PS input) : SV_Target0
{
    float NdotL, NdotV, NdotH, LdotH;
    GetAngles(input, NdotL, NdotV, NdotH, LdotH);
    float4 brdf = NB_Disney(NdotL, NdotV, NdotH, LdotH, Roughness);
    float3 color = NB_Shade(brdf, BaseColor.rgb, Specular, Metallic, NdotL, LightIntensity.rgb);
    return float4(color, 1.0f);
}

// ---------------------------------------------------------------------------
// Cooperative-vector neural inference (NVIDIA Neural Shading, Shader Model 6.9 preview).
// Runs the MLP forward pass with cooperative-vector MatVecMulAdd over device-optimal FP16 weights
// produced by GPUContext::ConvertCooperativeVectorMatrices on the host.
//
// Shared by the neural (PS_InferenceCV) and difference (PS_Difference) passes - both are flagged
// META_FLAG(CooperativeVector) so they are routed through DXC (dx::linalg -> DXIL on D3D12, or
// inline SPV_NV_cooperative_vector on Vulkan). Cooperative vectors require a DXC SM6+ backend, so
// the implementation is guarded by __hlsl_dx_compiler; there is no fp32 inference fallback.
// ---------------------------------------------------------------------------
#if (defined(_PS_InferenceCV) || defined(_PS_Difference)) && defined(__hlsl_dx_compiler)
ByteAddressBuffer WeightsOpt : register(t1); // device-optimal fp16 weight matrices + fp16 bias vectors

#if defined(__spirv__)
// -------- Vulkan: real cooperative vectors via inline SPIR-V (SPV_NV_cooperative_vector) --------
// DXC cannot lower dx::linalg to SPIR-V, so the MatVecMulAdd is hand-emitted. See coopvec.h.
#include "./neuralbrdf/spirv/coopvec.h"

float4 NeuralBrdfCV(float NdotL, float NdotV, float NdotH, float LdotH)
{
    float featsF[NB_INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, Roughness };
    float encF[NB_INPUT_NEURONS];
    NB_EncodeFrequency(featsF, encF);

    // Input cooperative vector (30): start from a splat-zero and set each component.
    nbcv::CoopVecHalf30 x = nbcv::Splat<nbcv::CoopVecHalf30>((half)0);
    [unroll] for (uint i = 0; i < NB_INPUT_NEURONS; ++i)
        x = nbcv::Insert<nbcv::CoopVecHalf30>(x, (half)encF[i], i);

    // Weight + bias matrices share the device-optimal buffer; per-layer byte offsets come from the CB.
    nbcv::NB_U32ArrayPtr W = nbcv::ArrayPtr(WeightsOpt, 0);
    nbcv::CoopVecHalf32 zero32 = nbcv::Splat<nbcv::CoopVecHalf32>((half)0);

    // Layer 0: 30 -> 32, ReLU
    nbcv::CoopVecHalf32 a0 = nbcv::MatVecMulAdd<nbcv::CoopVecHalf32, nbcv::CoopVecHalf30>(
        x, NB_CV_FLOAT16,
        W, (uint)WeightOffsets.x, NB_CV_FLOAT16,
        W, (uint)BiasOffsets.x, NB_CV_FLOAT16,
        NB_HIDDEN_NEURONS, NB_INPUT_NEURONS, NB_CV_INFERENCING_OPTIMAL, false);
    a0 = nbcv::Max<nbcv::CoopVecHalf32>(a0, zero32);

    // Layer 1: 32 -> 32, ReLU
    nbcv::CoopVecHalf32 a1 = nbcv::MatVecMulAdd<nbcv::CoopVecHalf32, nbcv::CoopVecHalf32>(
        a0, NB_CV_FLOAT16,
        W, (uint)WeightOffsets.y, NB_CV_FLOAT16,
        W, (uint)BiasOffsets.y, NB_CV_FLOAT16,
        NB_HIDDEN_NEURONS, NB_HIDDEN_NEURONS, NB_CV_INFERENCING_OPTIMAL, false);
    a1 = nbcv::Max<nbcv::CoopVecHalf32>(a1, zero32);

    // Layer 2: 32 -> 32, ReLU
    nbcv::CoopVecHalf32 a2 = nbcv::MatVecMulAdd<nbcv::CoopVecHalf32, nbcv::CoopVecHalf32>(
        a1, NB_CV_FLOAT16,
        W, (uint)WeightOffsets.z, NB_CV_FLOAT16,
        W, (uint)BiasOffsets.z, NB_CV_FLOAT16,
        NB_HIDDEN_NEURONS, NB_HIDDEN_NEURONS, NB_CV_INFERENCING_OPTIMAL, false);
    a2 = nbcv::Max<nbcv::CoopVecHalf32>(a2, zero32);

    // Layer 3: 32 -> 4, exp
    nbcv::CoopVecHalf4 a3 = nbcv::MatVecMulAdd<nbcv::CoopVecHalf4, nbcv::CoopVecHalf32>(
        a2, NB_CV_FLOAT16,
        W, (uint)WeightOffsets.w, NB_CV_FLOAT16,
        W, (uint)BiasOffsets.w, NB_CV_FLOAT16,
        NB_OUTPUT_NEURONS, NB_HIDDEN_NEURONS, NB_CV_INFERENCING_OPTIMAL, false);

    float4 outp;
    [unroll] for (uint o = 0; o < NB_OUTPUT_NEURONS; ++o)
        outp[o] = exp((float)nbcv::Extract<half>(a3, o));
    return outp;
}
#else // __spirv__
// -------- D3D12: cooperative vectors via dx::linalg (lowers to DXIL) --------
#include "./neuralbrdf/dx/linalg.h"
using namespace dx::linalg;

float4 NeuralBrdfCV(float NdotL, float NdotV, float NdotH, float LdotH)
{
    float featsF[NB_INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, Roughness };
    float encF[NB_INPUT_NEURONS];
    NB_EncodeFrequency(featsF, encF);

    vector<half, NB_INPUT_NEURONS> x;
    [unroll] for (int i = 0; i < NB_INPUT_NEURONS; ++i)
        x[i] = (half)encF[i];

    // Layer 0: 30 -> 32, ReLU
    MatrixRef<DATA_TYPE_FLOAT16, NB_HIDDEN_NEURONS, NB_INPUT_NEURONS, MATRIX_LAYOUT_MUL_OPTIMAL> w0 = { WeightsOpt, (uint)WeightOffsets.x, 0 };
    VectorRef<DATA_TYPE_FLOAT16> b0 = { WeightsOpt, (uint)BiasOffsets.x };
    vector<half, NB_HIDDEN_NEURONS> a0 = MulAdd<half>(w0, MakeInterpretedVector<DATA_TYPE_FLOAT16>(x), b0);
    a0 = max(a0, (half)0);

    // Layer 1: 32 -> 32, ReLU
    MatrixRef<DATA_TYPE_FLOAT16, NB_HIDDEN_NEURONS, NB_HIDDEN_NEURONS, MATRIX_LAYOUT_MUL_OPTIMAL> w1 = { WeightsOpt, (uint)WeightOffsets.y, 0 };
    VectorRef<DATA_TYPE_FLOAT16> b1 = { WeightsOpt, (uint)BiasOffsets.y };
    vector<half, NB_HIDDEN_NEURONS> a1 = MulAdd<half>(w1, MakeInterpretedVector<DATA_TYPE_FLOAT16>(a0), b1);
    a1 = max(a1, (half)0);

    // Layer 2: 32 -> 32, ReLU
    MatrixRef<DATA_TYPE_FLOAT16, NB_HIDDEN_NEURONS, NB_HIDDEN_NEURONS, MATRIX_LAYOUT_MUL_OPTIMAL> w2 = { WeightsOpt, (uint)WeightOffsets.z, 0 };
    VectorRef<DATA_TYPE_FLOAT16> b2 = { WeightsOpt, (uint)BiasOffsets.z };
    vector<half, NB_HIDDEN_NEURONS> a2 = MulAdd<half>(w2, MakeInterpretedVector<DATA_TYPE_FLOAT16>(a1), b2);
    a2 = max(a2, (half)0);

    // Layer 3: 32 -> 4, exp
    MatrixRef<DATA_TYPE_FLOAT16, NB_OUTPUT_NEURONS, NB_HIDDEN_NEURONS, MATRIX_LAYOUT_MUL_OPTIMAL> w3 = { WeightsOpt, (uint)WeightOffsets.w, 0 };
    VectorRef<DATA_TYPE_FLOAT16> b3 = { WeightsOpt, (uint)BiasOffsets.w };
    vector<half, NB_OUTPUT_NEURONS> a3 = MulAdd<half>(w3, MakeInterpretedVector<DATA_TYPE_FLOAT16>(a2), b3);

    float4 outp;
    [unroll] for (int o = 0; o < NB_OUTPUT_NEURONS; ++o)
        outp[o] = exp((float)a3[o]);
    return outp;
}
#endif // __spirv__
#endif // (_PS_InferenceCV || _PS_Difference) && __hlsl_dx_compiler

// Middle "neural" view: the trained MLP evaluated entirely with cooperative vectors.
#ifdef _PS_InferenceCV
META_PS(true, FEATURE_LEVEL_SM5)
META_FLAG(CooperativeVector)
float4 PS_InferenceCV(VS2PS input) : SV_Target0
{
    float NdotL, NdotV, NdotH, LdotH;
    GetAngles(input, NdotL, NdotV, NdotH, LdotH);
    float4 brdf = NeuralBrdfCV(NdotL, NdotV, NdotH, LdotH); // cooperative-vector MatVecMulAdd
    float3 color = NB_Shade(brdf, BaseColor.rgb, Specular, Metallic, NdotL, LightIntensity.rgb);
    return float4(color, 1.0f);
}
#endif // _PS_InferenceCV

// Right "difference" view: |analytic ground truth - cooperative-vector neural| (amplified).
#ifdef _PS_Difference
META_PS(true, FEATURE_LEVEL_SM5)
META_FLAG(CooperativeVector)
float4 PS_Difference(VS2PS input) : SV_Target0
{
    float NdotL, NdotV, NdotH, LdotH;
    GetAngles(input, NdotL, NdotV, NdotH, LdotH);
    float4 truth = NB_Disney(NdotL, NdotV, NdotH, LdotH, Roughness);
    float4 pred = NeuralBrdfCV(NdotL, NdotV, NdotH, LdotH);
    float3 cTruth = NB_Shade(truth, BaseColor.rgb, Specular, Metallic, NdotL, LightIntensity.rgb);
    float3 cPred = NB_Shade(pred, BaseColor.rgb, Specular, Metallic, NdotL, LightIntensity.rgb);
    return float4(abs(cTruth - cPred) * DiffScale, 1.0f);
}
#endif // _PS_Difference
