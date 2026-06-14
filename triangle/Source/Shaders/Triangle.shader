// Port of Sascha Willems Vulkan examples/triangle shaders.

#include "./Flax/Common.hlsl"

META_CB_BEGIN(0, UBO)
float4x4 ProjectionMatrix;
float4x4 ModelMatrix;
float4x4 ViewMatrix;
META_CB_END

struct VS2PS
{
	float4 Position : SV_Position;
	float3 Color    : COLOR;
};

META_VS(true, FEATURE_LEVEL_ES2)
META_VS_IN_ELEMENT(POSITION, 0, R32G32B32_FLOAT, 0, 0, PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(COLOR,    0, R32G32B32_FLOAT, 0, 12, PER_VERTEX, 0, true)
VS2PS VS(float3 Position : POSITION0, float3 Color : COLOR0)
{
	VS2PS output;
	// Sascha triangle.vert (HLSL): mul(projection, mul(view, mul(model, pos)))
	output.Position = mul(ProjectionMatrix, mul(ViewMatrix, mul(ModelMatrix, float4(Position, 1.0))));
	output.Color = Color;
	return output;
}

META_PS(true, FEATURE_LEVEL_ES2)
float4 PS(VS2PS input) : SV_Target
{
	return float4(input.Color, 1.0);
}
