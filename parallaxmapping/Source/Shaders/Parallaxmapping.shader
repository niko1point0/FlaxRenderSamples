// Port of Sascha Willems Vulkan examples/parallaxmapping shaders (parallax.vert + parallax.frag).

#include "./Flax/Common.hlsl"

META_CB_BEGIN(0, VertexUBO)
float4x4 ProjectionMatrix;
float4x4 ViewMatrix;
float4x4 ModelMatrix;
float4 LightPos;
float4 CameraPos;
META_CB_END

META_CB_BEGIN(1, FragmentUBO)
float MappingMode;
float HeightScale;
float ParallaxBias;
float NumLayers;
META_CB_END

Texture2D ColorMap : register(t0);
Texture2D NormalHeightMap : register(t1);

struct VS2PS
{
	float4 Position         : SV_Position;
	float2 UV               : TEXCOORD0;
	float3 TangentLightPos  : TEXCOORD1;
	float3 TangentViewPos   : TEXCOORD2;
	float3 TangentFragPos   : TEXCOORD3;
};

META_VS(true, FEATURE_LEVEL_ES2)
META_VS_IN_ELEMENT(POSITION, 0, R32G32B32_FLOAT, 0, 0, PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(TEXCOORD, 0, R32G32_FLOAT, 0, 12, PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(NORMAL,  0, R32G32B32_FLOAT, 0, 20, PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(TANGENT, 0, R32G32B32A32_FLOAT, 0, 32, PER_VERTEX, 0, true)
VS2PS VS(float3 Position : POSITION0, float2 UV : TEXCOORD0, float3 Normal : NORMAL0, float4 Tangent : TANGENT0)
{
	VS2PS output;
	// parallax.vert (Vulkan HLSL): column-vector MVP + StoreVulkan upload
	float4 modelPos = mul(ModelMatrix, float4(Position, 1.0));
	output.Position = mul(ProjectionMatrix, mul(ViewMatrix, modelPos));
	output.UV = UV;

	float3 N = normalize(Normal);
	float3 T = normalize(Tangent.xyz);
	float3 B = normalize(cross(N, T));
	float3x3 TBN = float3x3(T, B, N);

	output.TangentLightPos = mul(TBN, LightPos.xyz);
	output.TangentViewPos = mul(TBN, CameraPos.xyz);
	output.TangentFragPos = mul(TBN, modelPos.xyz);
	return output;
}

float2 ParallaxMapping(float2 uv, float3 viewDir)
{
	float height = 1.0 - NormalHeightMap.SampleLevel(SamplerLinearWrap, uv, 0).a;
	float2 p = viewDir.xy * (height * (HeightScale * 0.5) + ParallaxBias) / viewDir.z;
	return uv - p;
}

float2 SteepParallaxMapping(float2 uv, float3 viewDir)
{
	float layerDepth = 1.0 / NumLayers;
	float currLayerDepth = 0.0;
	float2 deltaUV = viewDir.xy * HeightScale / (viewDir.z * NumLayers);
	float2 currUV = uv;
	float height = 1.0 - NormalHeightMap.SampleLevel(SamplerLinearWrap, currUV, 0).a;
	for (int i = 0; i < (int)NumLayers; i++)
	{
		currLayerDepth += layerDepth;
		currUV -= deltaUV;
		height = 1.0 - NormalHeightMap.SampleLevel(SamplerLinearWrap, currUV, 0).a;
		if (height < currLayerDepth)
			break;
	}
	return currUV;
}

float2 ParallaxOcclusionMapping(float2 uv, float3 viewDir)
{
	float layerDepth = 1.0 / NumLayers;
	float currLayerDepth = 0.0;
	float2 deltaUV = viewDir.xy * HeightScale / (viewDir.z * NumLayers);
	float2 currUV = uv;
	float height = 1.0 - NormalHeightMap.SampleLevel(SamplerLinearWrap, currUV, 0).a;
	for (int i = 0; i < (int)NumLayers; i++)
	{
		currLayerDepth += layerDepth;
		currUV -= deltaUV;
		height = 1.0 - NormalHeightMap.SampleLevel(SamplerLinearWrap, currUV, 0).a;
		if (height < currLayerDepth)
			break;
	}
	float2 prevUV = currUV + deltaUV;
	float nextDepth = height - currLayerDepth;
	float prevDepth = 1.0 - NormalHeightMap.SampleLevel(SamplerLinearWrap, prevUV, 0).a - currLayerDepth + layerDepth;
	return lerp(currUV, prevUV, nextDepth / (nextDepth - prevDepth));
}

META_PS(true, FEATURE_LEVEL_ES2)
float4 PS(VS2PS input) : SV_Target
{
	float3 V = normalize(input.TangentViewPos - input.TangentFragPos);
	float2 uv = input.UV;
	int mode = (int)(MappingMode + 0.5);

	// 0=color only, 1=normal map, 2=parallax, 3=steep, 4=occlusion (Vulkan mappingModes)
	if (mode == 0)
		return ColorMap.Sample(SamplerLinearWrap, input.UV);

	switch (mode)
	{
	case 2:
		uv = ParallaxMapping(input.UV, V);
		break;
	case 3:
		uv = SteepParallaxMapping(input.UV, V);
		break;
	case 4:
		uv = ParallaxOcclusionMapping(input.UV, V);
		break;
	default:
		// mode 1 (normal mapping): lit shading without parallax UV offset
		break;
	}

	if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
		discard;

	float3 N = normalize(NormalHeightMap.SampleLevel(SamplerLinearWrap, uv, 0).rgb * 2.0 - 1.0);
	float3 color = ColorMap.Sample(SamplerLinearWrap, uv).rgb;
	float3 L = normalize(input.TangentLightPos - input.TangentFragPos);
	float3 H = normalize(L + V);

	float3 ambient = 0.2 * color;
	float3 diffuse = max(dot(L, N), 0.0) * color;
	float3 specular = float3(0.15, 0.15, 0.15) * pow(max(dot(N, H), 0.0), 32.0);

	return float4(ambient + diffuse + specular, 1.0);
}
