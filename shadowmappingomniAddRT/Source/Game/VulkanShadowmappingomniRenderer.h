// Port of Sascha Willems Vulkan examples/shadowmappingomni to Flax.

#pragma once

#include "Engine/Scripting/Script.h"
#include "SaschaCamera.h"
#include "SaschaModelDraw.h"

class GPUConstantBuffer;
class GPUPipelineState;
class GPUTexture;
class GPUTextureView;
class GPUBuffer;
class GPUAccelerationStructure;
class RenderTask;
class GPUContext;

/// <summary>
/// Omni-directional point light shadows via dynamic cube shadow map (6 face renders).
/// </summary>
API_CLASS(Namespace="") class VulkanShadowmappingomniRenderer : public Script
{
    API_AUTO_SERIALIZATION();
    DECLARE_SCRIPTING_TYPE(VulkanShadowmappingomniRenderer);

    static constexpr int32 ShadowDim = 1024;

    SaschaCamera _camera;
    SaschaModelDraw _scene;

    GPUTexture* _shadowCubeMap = nullptr;
    GPUTexture* _offscreenDepth = nullptr;
    GPUTextureView* _offscreenDepthView = nullptr;

    GPUPipelineState* _offscreenPipeline = nullptr;
    GPUPipelineState* _scenePipeline = nullptr;
    GPUPipelineState* _cubemapDisplayPipeline = nullptr;
    GPUPipelineState* _rayTracingPipeline = nullptr;

    GPUConstantBuffer* _offscreenCb = nullptr;
    GPUConstantBuffer* _faceViewCb = nullptr;
    GPUConstantBuffer* _sceneCb = nullptr;

    // Hardware ray tracing (inline ray queries). Only used on DirectX 12 / Vulkan when the device supports it.
    // The scene meshes are merged into one combined position/index buffer so the whole room is one BLAS.
    GPUAccelerationStructure* _accelerationStructure = nullptr;
    GPUBuffer* _rtVertexBuffer = nullptr;
    GPUBuffer* _rtIndexBuffer = nullptr;
    int32 _rtVertexCount = 0;
    int32 _rtIndexCount = 0;
    bool _rayTracingSupported = false;
    bool _rayTracingEnabled = false;

    API_FIELD() bool _displayCubeMap = false;

    float _zNear = 0.1f;
    float _zFar = 1024.0f;
    float _timer = 0.0f;
    Float3 _lightPos = Float3(0.0f, -2.5f, 0.0f);

    void CreateShadowTargets();
    void DestroyShadowTargets();
    void UpdateLight();
    void UpdateOffscreenUniforms(GPUContext* context);
    void UpdateSceneUniforms(GPUContext* context, float aspect);
    void RenderShadowCubeFaces(GPUContext* context);
    void RenderScenePass(GPUContext* context, GPUTextureView* depth, GPUTextureView* output);
    void RenderCubemapDebugPass(GPUContext* context, GPUTextureView* depth, GPUTextureView* output);

    void BuildRayTracingGeometry();
    void EnsureAccelerationStructure(GPUContext* context);
    void RenderRayTracePass(GPUContext* context, GPUTextureView* depth, GPUTextureView* output);

    void OnMainRender(RenderTask* task, GPUContext* context);

    void OnStart() override;
    void OnUpdate() override;
    void OnDestroy() override;
};
