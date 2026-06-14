// Port of Sascha Willems Vulkan examples/shadowmapping (HLSL shaders).
// Convention matches the confirmed-working parallaxmapping/texture/triangle ports and the
// original Vulkan scene.vert / offscreen.vert: StoreVulkan upload + column-vector mul.
//   clip = mul(Projection, mul(View, mul(Model, pos)))
// Single CB0 holds every uniform (field order MUST match UniformData in the renderer).

#include "./Flax/Common.hlsl"

#define AMBIENT 0.1

META_CB_BEGIN(0, UBO)
float4x4 ProjectionMatrix;
float4x4 ViewMatrix;
float4x4 ModelMatrix;
float4x4 LightSpace;
float4x4 DepthMvp;
float4 LightPos;
float ZNear;
float ZFar;
int EnablePCF;
META_CB_END

// Maps light-clip XY [-1,1] to shadow-map UV [0,1] (row-major literals, used column-vector:
// mul(BiasMat, clip)). The original Vulkan scene.vert uses +0.5 on the Y row because Vulkan's
// framebuffer has a bottom-left/inverted-Y origin. Flax (like D3D) rasterizes into textures with a
// top-left origin (texel Y increases downward, i.e. texelV = 0.5 - 0.5*ndc.y), so the Y row is
// negated here. Without this flip the depth lookup is sampled vertically mirrored and the shadows
// land in the wrong place / appear detached from their casters.
static const float4x4 BiasMat = float4x4(
	0.5,  0.0, 0.0, 0.5,
	0.0, -0.5, 0.0, 0.5,
	0.0,  0.0, 1.0, 0.0,
	0.0,  0.0, 0.0, 1.0);

// The mesh is drawn with its native Flax vertex buffers (MeshBase::Render), so the input layout
// matches Flax's standard model layout (copied from Source/Shaders/Editor/VertexColors.shader):
//   VB0 slot0: Position R32G32B32_FLOAT
//   VB1 slot1: TexCoord0 R16G16_FLOAT, Normal R10G10B10A2_UNORM, Tangent R10G10B10A2_UNORM, TexCoord1 R16G16_FLOAT
//   VB2 slot2: Color R8G8B8A8_UNORM

// --- Offscreen shadow map pass (offscreen.vert) ---
struct OffscreenVS2PS
{
	float4 Position : SV_Position;
};

META_VS(true, FEATURE_LEVEL_ES2)
META_VS_IN_ELEMENT(POSITION, 0, R32G32B32_FLOAT, 0, 0, PER_VERTEX, 0, true)
OffscreenVS2PS VS_Offscreen(float3 Position : POSITION0)
{
	OffscreenVS2PS output;
	float4 worldPos = mul(ModelMatrix, float4(Position, 1.0));
	output.Position = mul(DepthMvp, worldPos);
	return output;
}

META_PS(true, FEATURE_LEVEL_ES2)
void PS_Offscreen(OffscreenVS2PS input)
{
}

// --- Lit scene pass (scene.vert / scene.frag) ---
struct SceneVS2PS
{
	float4 Position    : SV_Position;
	float3 Normal      : NORMAL0;
	float3 Color       : COLOR0;
	float3 ViewVec     : TEXCOORD1;
	float3 LightVec    : TEXCOORD2;
	float4 ShadowCoord : TEXCOORD3;
};

struct SceneInput
{
	float3 Position : POSITION0;
	float2 UV       : TEXCOORD0;
	float4 Normal   : NORMAL0;  // R10G10B10A2_UNORM, normalized range
	float4 Tangent  : TANGENT0; // unused, declared so the slot-1 layout matches Flax
	float2 UV1      : TEXCOORD1;
	float4 Color    : COLOR0;   // R8G8B8A8_UNORM
};

META_VS(true, FEATURE_LEVEL_ES2)
META_VS_IN_ELEMENT(POSITION, 0, R32G32B32_FLOAT,   0, 0,     PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(TEXCOORD, 0, R16G16_FLOAT,      1, 0,     PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(NORMAL,   0, R10G10B10A2_UNORM, 1, ALIGN, PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(TANGENT,  0, R10G10B10A2_UNORM, 1, ALIGN, PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(TEXCOORD, 1, R16G16_FLOAT,      1, ALIGN, PER_VERTEX, 0, true)
META_VS_IN_ELEMENT(COLOR,    0, R8G8B8A8_UNORM,    2, 0,     PER_VERTEX, 0, true)
SceneVS2PS VS_Scene(SceneInput input)
{
	SceneVS2PS output;
	// vkglTF defaults a primitive's vertex color to white when the glTF has no COLOR_0 attribute,
	// which is the case for vulkanscene_shadow/samplescene (POSITION + NORMAL only). The original
	// scene.frag therefore lights the geometry with inColor == white; Flax's imported model likewise
	// has no color vertex buffer (input.Color reads 0), so use white explicitly to reproduce the
	// reference's monochrome lit look instead of rendering black silhouettes.
	output.Color = float3(1.0, 1.0, 1.0);

	// scene.vert: gl_Position = projection * view * model * pos
	float4 worldPos = mul(ModelMatrix, float4(input.Position, 1.0));
	output.Position = mul(ProjectionMatrix, mul(ViewMatrix, worldPos));

	// Flax stores normals packed in [0,1] range; decode to [-1,1] then transform by the model.
	float3 normal = input.Normal.xyz * 2.0 - 1.0;
	output.Normal = mul((float3x3)ModelMatrix, normal);
	output.LightVec = LightPos.xyz - worldPos.xyz;
	output.ViewVec = -worldPos.xyz;
	output.ShadowCoord = mul(BiasMat, mul(LightSpace, worldPos));
	return output;
}

Texture2D ShadowMap : register(t0);

float TextureProj(float4 shadowCoord, float2 offset, float bias)
{
	float shadow = 1.0;
	if (shadowCoord.z > -1.0 && shadowCoord.z < 1.0)
	{
		float dist = ShadowMap.Sample(SamplerLinearClamp, shadowCoord.xy + offset).r;
		// dist is the closest depth seen from the light; the fragment is shadowed when it is
		// farther than that. 'bias' replaces the Vulkan offscreen pipeline's hardware depth bias
		// (vkCmdSetDepthBias), which Flax's pipeline state cannot express, to avoid self-shadow acne.
		if (shadowCoord.w > 0.0 && dist < shadowCoord.z - bias)
			shadow = AMBIENT;
	}
	return shadow;
}

float FilterPCF(float4 sc, float bias)
{
	uint width, height;
	ShadowMap.GetDimensions(width, height);
	float scale = 1.5;
	float dx = scale * 1.0 / float(width);
	float dy = scale * 1.0 / float(height);

	float shadowFactor = 0.0;
	int count = 0;
	int range = 1;

	for (int x = -range; x <= range; x++)
	{
		for (int y = -range; y <= range; y++)
		{
			shadowFactor += TextureProj(sc, float2(dx * x, dy * y), bias);
			count++;
		}
	}
	return shadowFactor / count;
}

META_PS(true, FEATURE_LEVEL_ES2)
float4 PS_Scene(SceneVS2PS input) : SV_Target
{
	float4 sc = input.ShadowCoord / input.ShadowCoord.w;

	float3 N = normalize(input.Normal);
	float3 L = normalize(input.LightVec);
	float3 V = normalize(input.ViewVec);

	// Slope-scaled + constant depth bias emulating the Vulkan offscreen pipeline's depthBiasConstant
	// (1.25) / depthBiasSlope (1.75). Surfaces nearly parallel to the light need a larger bias.
	float NdotL = max(dot(N, L), 0.0);
	float bias = max(0.0025 * (1.0 - NdotL), 0.0008);

	float shadow = (EnablePCF > 0) ? FilterPCF(sc, bias) : TextureProj(sc, float2(0.0, 0.0), bias);

	float3 diffuse = max(dot(N, L), AMBIENT) * input.Color;

	return float4(diffuse * shadow, 1.0);
}

// --- Shadow map debug quad (quad.vert / quad.frag) ---
struct QuadVS2PS
{
	float4 Position : SV_Position;
	float2 UV       : TEXCOORD0;
};

META_VS(true, FEATURE_LEVEL_ES2)
QuadVS2PS VS_Quad(uint VertexIndex : SV_VertexID)
{
	QuadVS2PS output;
	output.UV = float2((VertexIndex << 1) & 2, VertexIndex & 2);
	output.Position = float4(output.UV * 2.0 - 1.0, 0.0, 1.0);
	return output;
}

float LinearizeDepth(float depth)
{
	float n = ZNear;
	float f = ZFar;
	return (2.0 * n) / (f + n - depth * (f - n));
}

META_PS(true, FEATURE_LEVEL_ES2)
float4 PS_Quad(QuadVS2PS input) : SV_Target
{
	float depth = ShadowMap.Sample(SamplerLinearClamp, input.UV).r;
	return float4((1.0 - LinearizeDepth(depth)).xxx, 1.0);
}

// --- Ray-traced lit scene pass (inline ray queries, Shader Model 6.5) ---
// Identical lighting to PS_Scene, except the shadow term is computed with a hardware ray query instead of
// sampling the offscreen shadow map: a shadow ray is launched from the surface toward the light and, if it
// hits any scene geometry before reaching the light, the surface is shadowed. This means the offscreen
// depth pass is not needed when ray tracing is enabled. Ray queries are used (no ray pipelines / shader
// tables). Only compiled on Shader Model 6.5 capable backends (DirectX 12 / Vulkan).
//
// The green background that confirms ray tracing is active is produced by clearing the scene output to
// green on the CPU side (see the renderer) - not here, because this is a shadow trace toward the light and
// a "miss" means "lit", not "background".
#if FEATURE_LEVEL >= FEATURE_LEVEL_SM6

RaytracingAccelerationStructure SceneAS : register(t1);

META_PS(true, FEATURE_LEVEL_SM6)
float4 PS_SceneRT(SceneVS2PS input) : SV_Target
{
	// VS_Scene stores ViewVec = -worldPos and LightVec = LightPos - worldPos (both un-normalized),
	// so the surface world position and the direction/distance to the light are recovered directly.
	float3 worldPos = -input.ViewVec;
	float3 N = normalize(input.Normal);
	float3 toLight = input.LightVec;
	float distToLight = length(toLight);
	float3 L = toLight / max(distToLight, 1e-5);

	float NdotL = dot(N, L);

	// Only light-facing surfaces can be lit, so only those need an occlusion test. Surfaces facing away are
	// already dark from the diffuse term (and tracing them would just self-shadow the whole back side).
	float shadow = 1.0;
	if (NdotL > 0.0)
	{
		// Slope-scaled normal offset: surfaces nearly parallel to the light (small NdotL) need a larger
		// push along the normal to keep the ray from grazing back into its own geometry (self-shadow acne).
		// This mirrors the slope-scaled depth bias used by the raster shadow-map path. Keep the grazing
		// offset bounded: too large detaches shadows (peter-panning) and makes near-grazing back-faces
		// flicker between hit/miss as the light orbits.
		float offset = 0.04 / max(NdotL, 0.15);

		RayDesc ray;
		ray.Origin = worldPos + N * offset;
		ray.Direction = L;
		ray.TMin = 0.02;
		ray.TMax = max(distToLight - 0.1, 0.0);

		#define SHADOW_RAY_FLAGS (RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_NON_OPAQUE)
		RayQuery<SHADOW_RAY_FLAGS> query;
		query.TraceRayInline(SceneAS, SHADOW_RAY_FLAGS, 0xFF, ray);
		query.Proceed();

		if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
			shadow = AMBIENT;
	}

	float3 diffuse = max(NdotL, AMBIENT) * input.Color;
	return float4(diffuse * shadow, 1.0);
}

#endif
