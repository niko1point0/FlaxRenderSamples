// Port of Sascha Willems Vulkan examples/shadowmapping to Flax.
// Kept structurally close to shadowmapping.cpp: updateLight + updateUniformBuffers, an offscreen
// depth pass and a lit scene pass, both drawing the scene mesh-by-mesh (like vkglTF primitives).

#pragma once

#include "Engine/Core/Math/Viewport.h"
#include "Engine/Scripting/Script.h"
#include "SaschaCamera.h"

class GPUConstantBuffer;
class GPUPipelineState;
class GPUTexture;
class GPUTextureView;
class Model;
class RenderTask;
class GPUContext;

/// <summary>
/// Directional shadow mapping: offscreen depth pass + forward lit scene with optional PCF.
/// </summary>
API_CLASS(Namespace="") class VulkanShadowmappingRenderer : public Script
{
    API_AUTO_SERIALIZATION();
    DECLARE_SCRIPTING_TYPE(VulkanShadowmappingRenderer);

    static constexpr int32 SceneCount = 2;
    static constexpr int32 ShadowDim = 2048;

    SaschaCamera _camera;
    Model* _sceneModels[SceneCount] = {};

    GPUTexture* _shadowMap = nullptr;
    GPUTextureView* _shadowMapView = nullptr;

    GPUPipelineState* _offscreenPipeline = nullptr;
    GPUPipelineState* _scenePipeline = nullptr;
    GPUPipelineState* _debugPipeline = nullptr;

    GPUConstantBuffer* _constantBuffer = nullptr;

    API_FIELD() int32 _sceneIndex = 0;
    API_FIELD() bool _displayShadowMap = false;
    API_FIELD() bool _filterPCF = true;

    // shadowmapping.cpp: zNear = 1.0f, zFar = 96.0f, lightFOV = 45.0f.
    float _zNear = 1.0f;
    float _zFar = 96.0f;
    float _lightFov = 45.0f;
    float _timer = 0.0f;
    glm::vec3 _lightPos = glm::vec3(0.0f);

    void DrawScene(GPUContext* context) const;
    void CreateShadowMap();
    void DestroyShadowMap();
    void UpdateLight();
    void UpdateUniformBuffers(GPUContext* context, float aspect);
    void RenderShadowPass(GPUContext* context);
    void RenderScenePass(GPUContext* context, const Viewport& viewport);
    void RenderDebugPass(GPUContext* context, const Viewport& viewport);

    void OnMainRender(RenderTask* task, GPUContext* context);

    void OnStart() override;
    void OnUpdate() override;
    void OnDestroy() override;
};
