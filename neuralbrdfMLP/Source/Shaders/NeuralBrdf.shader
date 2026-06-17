// Neural BRDF draw shader: renders a sphere three ways.
//   PS_Disney    - analytic Disney BRDF (ground truth)
//   PS_Inference - the trained MLP approximation
//   PS_Difference- |ground truth - neural| (amplified) for visualising training error
//
// Ported from RTXNS ShaderTraining (renderDisney.slang / renderInference.slang / renderDifference.slang).

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
META_CB_END

// Trained MLP parameters (row-major fp32), bound as a typed SRV for inference/difference.
Buffer<float> Weights : register(t0);

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

float4 NeuralBrdf(float NdotL, float NdotV, float NdotH, float LdotH)
{
    float feats[NB_INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, Roughness };
    float enc[NB_INPUT_NEURONS];
    NB_EncodeFrequency(feats, enc);
    return NB_Inference(Weights, enc);
}

META_PS(true, FEATURE_LEVEL_SM5)
float4 PS_Inference(VS2PS input) : SV_Target0
{
    float NdotL, NdotV, NdotH, LdotH;
    GetAngles(input, NdotL, NdotV, NdotH, LdotH);
    float4 brdf = NeuralBrdf(NdotL, NdotV, NdotH, LdotH);
    float3 color = NB_Shade(brdf, BaseColor.rgb, Specular, Metallic, NdotL, LightIntensity.rgb);
    return float4(color, 1.0f);
}

META_PS(true, FEATURE_LEVEL_SM5)
float4 PS_Difference(VS2PS input) : SV_Target0
{
    float NdotL, NdotV, NdotH, LdotH;
    GetAngles(input, NdotL, NdotV, NdotH, LdotH);
    float4 truth = NB_Disney(NdotL, NdotV, NdotH, LdotH, Roughness);
    float4 pred = NeuralBrdf(NdotL, NdotV, NdotH, LdotH);
    float3 cTruth = NB_Shade(truth, BaseColor.rgb, Specular, Metallic, NdotL, LightIntensity.rgb);
    float3 cPred = NB_Shade(pred, BaseColor.rgb, Specular, Metallic, NdotL, LightIntensity.rgb);
    return float4(abs(cTruth - cPred) * DiffScale, 1.0f);
}
