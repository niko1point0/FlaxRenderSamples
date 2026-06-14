// Port of Sascha Willems Vulkan examples/texture shaders.

#include "./Flax/Common.hlsl"

META_CB_BEGIN(0, UBO)
float4x4 ProjectionMatrix;
float4x4 ModelMatrix;
float4 ViewPos;
float LodBias;
META_CB_END

// GLSL locations 0-4: UV, LodBias, Normal, ViewVec, LightVec.
#if VULKAN
struct VS2PS
{
	float4 Position : SV_Position;
	[[vk::location(0)]] float2 UV        : TEXCOORD0;
	[[vk::location(1)]] float  LodBiasV  : TEXCOORD3;
	[[vk::location(2)]] float3 Normal    : NORMAL0;
	[[vk::location(3)]] float3 ViewVec   : TEXCOORD1;
	[[vk::location(4)]] float3 LightVec  : TEXCOORD2;
};
#else
struct VS2PS
{
	float4 Position : SV_Position;
	float2 UV        : TEXCOORD0;
	float  LodBiasV  : TEXCOORD1;
	float3 Normal    : TEXCOORD2;
	float3 ViewVec   : TEXCOORD3;
	float3 LightVec  : TEXCOORD4;
};
#endif

META_VS(true, FEATURE_LEVEL_ES2)
META_VS_IN_ELEMENT(POSITION, 0, R32G32B32_FLOAT, 0, 0, PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(TEXCOORD, 0, R32G32_FLOAT, 0, 12, PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(NORMAL,  0, R32G32B32_FLOAT, 0, 20, PER_VERTEX, 0, true)
VS2PS VS(float3 Position : POSITION0, float2 UV : TEXCOORD0, float3 Normal : NORMAL0)
{
	VS2PS output;
	output.UV = UV;
	output.LodBiasV = LodBias;

	// Sascha texture.vert (HLSL): mul(projection, mul(model, pos)); model = camera.view
	float4 pos = mul(ModelMatrix, float4(Position, 1.0));
	output.Position = mul(ProjectionMatrix, pos);

	float3x3 model3 = (float3x3)ModelMatrix;
	output.Normal = mul(model3, Normal);

	float3 lPos = mul(model3, float3(0.0, 0.0, 0.0));
	output.LightVec = lPos - pos.xyz;
	output.ViewVec = ViewPos.xyz - pos.xyz;
	return output;
}

Texture2D TextureColor : register(t0);

META_PS(true, FEATURE_LEVEL_ES2)
float4 PS(VS2PS input) : SV_Target
{
	float4 color = TextureColor.SampleLevel(SamplerLinearClamp, input.UV, LodBias);

	float3 N = normalize(input.Normal);
	float3 L = normalize(input.LightVec);
	float3 V = normalize(input.ViewVec);
	float3 R = reflect(-L, N);
	float3 diffuse = max(dot(N, L), 0.0) * float3(1.0, 1.0, 1.0);
	float specular = pow(max(dot(R, V), 0.0), 16.0) * color.a;

	return float4(diffuse * color.rgb + specular, 1.0);
}
