// Port of Sascha Willems Vulkan examples/parallaxmapping to Flax.

#pragma once

#include "Engine/Core/Math/Viewport.h"
#include "Engine/Scripting/Script.h"
#include "SaschaCamera.h"

class GPUBuffer;
class GPUConstantBuffer;
class GPUPipelineState;
class GPUShader;
class GPUVertexLayout;
class Model;
class RenderTask;
class GPUContext;
class Texture;

/// <summary>
/// Parallax mapping on a subdivided plane with color + normal/height maps.
/// Keys: M = cycle mode, +/- = height scale, PgUp/PgDn = layer count.
/// </summary>
API_CLASS(Namespace="") class VulkanParallaxmappingRenderer : public Script
{
    API_AUTO_SERIALIZATION();
    DECLARE_SCRIPTING_TYPE(VulkanParallaxmappingRenderer);

    Texture* _colorMap = nullptr;
    Texture* _normalHeightMap = nullptr;
    GPUBuffer* _vertexBuffer = nullptr;
    GPUBuffer* _indexBuffer = nullptr;
    GPUPipelineState* _pipeline = nullptr;
    GPUConstantBuffer* _vertexCb = nullptr;
    GPUConstantBuffer* _fragmentCb = nullptr;
    GPUVertexLayout* _vertexLayout = nullptr;
    SaschaCamera _camera;

    float _heightScale = 0.1f;
    float _parallaxBias = -0.02f;
    float _numLayers = 48.0f;
    int32 _mappingMode = 4;
    int32 _indexCount = 0;
    // VulkanExampleBase timerSpeed 0.25 * parallaxmapping timerSpeed *= 0.5 => 0.125
    float _lightTimer = 0.0f;
    static constexpr float LightTimerSpeed = 0.125f;

    void BuildMesh();
    void DrawPlane(GPUContext* context, const Viewport& viewport, float aspect);
    void OnMainRender(RenderTask* task, GPUContext* context);
    void UpdateUniforms(GPUContext* context, float aspect);
    void HandleInput();

    void OnStart() override;
    void OnUpdate() override;
    void OnDestroy() override;
};
