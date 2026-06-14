// Port of Sascha Willems Vulkan examples/gears shaders.

#include "./Flax/Common.hlsl"

META_CB_BEGIN(0, UBO)
float4x4 ProjectionMatrix;
float4x4 ViewMatrix;
float4 LightPos;
float4x4 ModelMatrix[3];
uint InstanceBase;
META_CB_END

struct VS2PS
{
	float4 Position : SV_Position;
	float3 Normal   : TEXCOORD0;
	float3 Color    : TEXCOORD1;
	float3 EyePos   : TEXCOORD2;
	float3 LightVec : TEXCOORD3;
};

META_VS(true, FEATURE_LEVEL_ES2)
META_VS_IN_ELEMENT(POSITION, 0, R32G32B32_FLOAT, 0, 0, PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(NORMAL, 0, R32G32B32_FLOAT, 0, 12, PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(COLOR, 0, R32G32B32_FLOAT, 0, 24, PER_VERTEX, 0, true)
VS2PS VS(float3 Position : POSITION0, float3 Normal : NORMAL0, float3 Color : COLOR0, uint instanceId : SV_InstanceID)
{
	VS2PS output;

#if VULKAN
	// SPIRV: SV_InstanceID includes the draw's startInstance
	uint gearIndex = instanceId;
#else
	// DXBC SM5: SV_InstanceID is per-draw only; startInstance is set via InstanceBase
	uint gearIndex = InstanceBase + instanceId;
#endif

	// Row-vector equivalent of gears.vert column math: P * V * M * v  ==  v * M * V * P
	float4x4 modelMatrix = ModelMatrix[gearIndex];
	output.Normal = normalize(mul(Normal, (float3x3)modelMatrix));
	output.Color = Color;

	float4x4 modelView = mul(modelMatrix, ViewMatrix);
	float4 pos = mul(float4(Position, 1.0), modelView);
	output.Position = mul(pos, ProjectionMatrix);

	output.EyePos = mul(pos, modelView).xyz;
	float4 lightPos = mul(float4(LightPos.xyz, 1.0), modelView);
	output.LightVec = normalize(lightPos.xyz - output.EyePos);
	return output;
}

META_PS(true, FEATURE_LEVEL_ES2)
float4 PS(VS2PS input) : SV_Target
{
	float3 Eye = normalize(-input.EyePos);
	float3 Reflected = normalize(reflect(-input.LightVec, input.Normal));

	float4 IAmbient = float4(0.2, 0.2, 0.2, 1.0);
	float4 IDiffuse = float4(0.5, 0.5, 0.5, 0.5) * max(dot(input.Normal, input.LightVec), 0.0);
	float specular = 0.25;
	float4 ISpecular = float4(0.5, 0.5, 0.5, 1.0) * pow(max(dot(Reflected, Eye), 0.0), 0.8) * specular;

	return float4((IAmbient + IDiffuse) * float4(input.Color, 1.0) + ISpecular);
}
