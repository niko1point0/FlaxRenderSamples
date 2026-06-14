// Port of Sascha Willems Vulkan examples/shadowmappingomni shaders (offscreen cube faces + scene + debug).

#include "./Flax/Common.hlsl"

#define EPSILON 0.15
#define SHADOW_OPACITY 0.5

// Flax cbuffer members live at global scope, so every member name must be unique across all the
// constant buffers and is referenced directly (not as Block.Member). Layout/order still matches the
// matching C++ structs (OffscreenUniformData / FaceViewData / SceneUniformData).
META_CB_BEGIN(0, OffscreenUBO)
float4x4 OffscreenProjection;
float4x4 OffscreenModel;
float4 OffscreenLightPos;
META_CB_END

// Vulkan push constants emulated with a per-face constant buffer.
META_CB_BEGIN(1, FaceView)
float4x4 FaceViewMatrix;
META_CB_END

META_CB_BEGIN(2, SceneUBO)
float4x4 SceneProjection;
float4x4 SceneView;
float4x4 SceneModel;
float4 SceneLightPos;
META_CB_END

// --- Offscreen cube face pass (stores distance to light in R32 cubemap) ---
struct OffscreenVS2PS
{
	float4 Position : SV_Position;
	float3 WorldPos : TEXCOORD0;
	float3 LightPos : TEXCOORD1;
};

META_VS(true, FEATURE_LEVEL_ES2)
META_VS_IN_ELEMENT(POSITION, 0, R32G32B32_FLOAT, 0, 0, PER_VERTEX, 0, true)
OffscreenVS2PS VS_Offscreen(float3 Position : POSITION0)
{
	OffscreenVS2PS output;
	// offscreen.vert: gl_Position = projection * faceView * model * pos (column-vector, matches GLSL).
	// OffscreenModel maps the baked vertex into corrected world space; FaceViewMatrix = faceRotation *
	// translate(-lightPos). Distance is then measured in that same corrected world space as the scene.
	float4 worldPos = mul(OffscreenModel, float4(Position, 1.0));
	output.Position = mul(OffscreenProjection, mul(FaceViewMatrix, worldPos));
	output.WorldPos = worldPos.xyz;
	output.LightPos = OffscreenLightPos.xyz;
	return output;
}

META_PS(true, FEATURE_LEVEL_ES2)
float4 PS_Offscreen(OffscreenVS2PS input) : SV_Target
{
	float3 lightVec = input.WorldPos - input.LightPos;
	float dist = length(lightVec);
	return float4(dist, 0.0, 0.0, 1.0);
}

// --- Lit scene with omni shadow cubemap ---
struct SceneVS2PS
{
	float4 Position : SV_Position;
	float3 Normal   : NORMAL0;
	float3 Color    : COLOR0;
	float3 WorldPos : TEXCOORD0;
	float3 LightPos : TEXCOORD1;
};

// Flax lays imported glTF mesh vertices across multiple vertex buffers: position in slot 0, the
// texcoord/normal/tangent block packed in slot 1 (normals are R10G10B10A2_UNORM encoded in [0,1]),
// and an optional color buffer in slot 2. Declaring everything as float3 in slot 0 at offsets 0/12/24
// makes the input assembler use a 36-byte stride over the 12-byte position-only buffer, scrambling
// every vertex past the first so the geometry renders off-screen. This layout must match Flax exactly.
struct SceneInput
{
	float3 Position : POSITION0;
	float2 UV       : TEXCOORD0;
	float4 Normal   : NORMAL0;  // R10G10B10A2_UNORM, [0,1] encoded
	float4 Tangent  : TANGENT0; // unused, declared so the slot-1 layout matches Flax
	float2 UV1      : TEXCOORD1;
	float4 Color    : COLOR0;   // R8G8B8A8_UNORM (shadowscene_fire has no vertex color)
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
	// shadowscene_fire has no COLOR_0 attribute; vkglTF defaults missing vertex color to white, so the
	// scene is lit as monochrome white (matching the original sample's look).
	output.Color = float3(1.0, 1.0, 1.0);

	// Flax stores normals packed in [0,1]; decode to [-1,1].
	float3 normal = input.Normal.xyz * 2.0 - 1.0;

	// scene.vert: gl_Position = projection * view * model * pos (column-vector, matches GLSL). model
	// carries the Flax-import correction, so worldPos is in the corrected space shared with the offscreen
	// pass - the shadow cubemap lookup (worldPos - lightPos) is therefore consistent between both passes.
	float4 worldPos = mul(SceneModel, float4(input.Position, 1.0));
	output.Position = mul(SceneProjection, mul(SceneView, worldPos));

	output.Normal = mul((float3x3)SceneModel, normal);
	output.WorldPos = worldPos.xyz;
	output.LightPos = SceneLightPos.xyz;
	return output;
}

TextureCube ShadowCubeMap : register(t0);

META_PS(true, FEATURE_LEVEL_ES2)
float4 PS_Scene(SceneVS2PS input) : SV_Target
{
	float3 N = normalize(input.Normal);
	float3 L = normalize(input.LightPos - input.WorldPos);

	float4 ambient = float4(0.05, 0.05, 0.05, 1.0);
	float4 diffuse = float4(1.0, 1.0, 1.0, 1.0) * max(dot(N, L), 0.0);
	float4 outColor = float4(ambient + diffuse * float4(input.Color, 1.0));

	float3 lightVec = input.WorldPos - input.LightPos;
	float sampledDist = ShadowCubeMap.Sample(SamplerLinearClamp, lightVec).r;
	float dist = length(lightVec);
	float shadow = (dist <= sampledDist + EPSILON) ? 1.0 : SHADOW_OPACITY;

	outColor.rgb *= shadow;
	return outColor;
}

// --- Cubemap debug visualization (M key) ---
struct CubemapVS2PS
{
	float4 Position : SV_Position;
	float2 UV       : TEXCOORD0;
};

META_VS(true, FEATURE_LEVEL_ES2)
CubemapVS2PS VS_CubemapDisplay(uint VertexIndex : SV_VertexID)
{
	CubemapVS2PS output;
	output.UV = float2((VertexIndex << 1) & 2, VertexIndex & 2);
	output.Position = float4(output.UV * 2.0 - 1.0, 0.0, 1.0);
	return output;
}

META_PS(true, FEATURE_LEVEL_ES2)
float4 PS_CubemapDisplay(CubemapVS2PS input) : SV_Target
{
	float4 outColor = float4(0.05, 0.05, 0.05, 1.0);
	float3 samplePos = float3(0, 0, 0);

	int x = int(floor(input.UV.x / 0.25));
	int y = int(floor(input.UV.y / (1.0 / 3.0)));
	if (y == 1)
	{
		float2 uv = float2(input.UV.x * 4.0, (input.UV.y - 1.0 / 3.0) * 3.0);
		uv = 2.0 * float2(uv.x - float(x) * 1.0, uv.y) - 1.0;
		switch (x)
		{
		case 0: samplePos = float3(-1.0, uv.y, uv.x); break;
		case 1: samplePos = float3(uv.x, uv.y, 1.0); break;
		case 2: samplePos = float3(1.0, uv.y, -uv.x); break;
		case 3: samplePos = float3(-uv.x, uv.y, -1.0); break;
		}
	}
	else if (x == 1)
	{
		float2 uv = float2((input.UV.x - 0.25) * 4.0, (input.UV.y - float(y) / 3.0) * 3.0);
		uv = 2.0 * uv - 1.0;
		if (y == 0) samplePos = float3(uv.x, -1.0, uv.y);
		if (y == 2) samplePos = float3(uv.x, 1.0, -uv.y);
	}

	if (samplePos.x != 0.0 || samplePos.y != 0.0)
	{
		float dist = ShadowCubeMap.Sample(SamplerLinearClamp, samplePos).r * 0.005;
		outColor = float4(dist, dist, dist, 1.0);
	}
	return outColor;
}

// --- Ray-traced omni shadow scene pass (inline ray queries, Shader Model 6.5) ---
// Identical lighting to PS_Scene, except the omni point-light shadow term is computed with a hardware ray
// query instead of sampling the distance cube map: a shadow ray is launched from the surface toward the
// point light and, if it hits scene geometry before reaching the light, the surface is shadowed. This means
// the 6-face offscreen cube pass is not needed when ray tracing is enabled. Ray queries are used (no ray
// pipelines / shader tables), so this only compiles on Shader Model 6.5 capable backends (DirectX 12 / Vulkan).
//
// The green background that confirms ray tracing is active is produced by clearing the scene output to green
// on the CPU side (see the renderer) - a ray "miss" here means "lit", not "background".
#if FEATURE_LEVEL >= FEATURE_LEVEL_SM6

RaytracingAccelerationStructure SceneAS : register(t1);

META_PS(true, FEATURE_LEVEL_SM6)
float4 PS_SceneRT(SceneVS2PS input) : SV_Target
{
	// WorldPos / LightPos arrive in the same corrected world space the TLAS instance is built in
	// (model scale 0.01,-0.01,-0.01), so the ray origin/direction match the acceleration structure.
	float3 N = normalize(input.Normal);
	float3 toLight = input.LightPos - input.WorldPos;
	float distToLight = length(toLight);
	float3 L = toLight / max(distToLight, 1e-5);

	float NdotL = max(dot(N, L), 0.0);

	// Trace a shadow ray from this pixel straight toward the point light. The ray starts just off the surface
	// and ends at the light, so a committed hit means another piece of scene geometry is between the pixel and
	// the light - i.e. the pixel is occluded and must be shadowed. A small fixed normal offset keeps the ray
	// from grazing back into its own triangle (self-shadow acne). These parameters and RAY_FLAG_NONE match the
	// query that was verified to hit the acceleration structure correctly across the whole scene.
	float shadow = 1.0;
	{
		RayDesc ray;
		ray.Origin = input.WorldPos + N * 0.05; // start at the lit pixel, nudged along the normal
		ray.Direction = L;                      // aim at the light
		ray.TMin = 0.001;
		ray.TMax = distToLight;                 // stop at the light

		RayQuery<RAY_FLAG_NONE> query;
		query.TraceRayInline(SceneAS, RAY_FLAG_NONE, 0xFF, ray);
		query.Proceed();

		// Hit something before reaching the light => occluded. Dim by SHADOW_OPACITY, exactly like the cube-map
		// path in PS_Scene, so the ray-traced shadows have the same darkness as the non-RT shadows.
		if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
			shadow = SHADOW_OPACITY;
	}

	// Identical shading to PS_Scene: ambient + N.L diffuse, then the whole color is scaled by the shadow term.
	float4 ambient = float4(0.05, 0.05, 0.05, 1.0);
	float4 diffuse = float4(1.0, 1.0, 1.0, 1.0) * NdotL;
	float4 outColor = float4(ambient + diffuse * float4(input.Color, 1.0));
	outColor.rgb *= shadow;
	return outColor;
}

#endif
