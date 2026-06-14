// Port of Sascha Willems Vulkan examples/triangle to Flax.

#pragma once

#include "Engine/Scripting/Script.h"
#include "SaschaCamera.h"

class GPUBuffer;
class GPUConstantBuffer;
class GPUPipelineState;
class RenderTask;
class GPUContext;

/// <summary>
/// Renders the Vulkan triangle example: indexed colored triangle with MVP uniforms.
/// </summary>
API_CLASS(Namespace="") class VulkanTriangleRenderer : public Script
{
    API_AUTO_SERIALIZATION();
    DECLARE_SCRIPTING_TYPE(VulkanTriangleRenderer);

    GPUBuffer* _vertexBuffer = nullptr;
    GPUBuffer* _indexBuffer = nullptr;
    GPUPipelineState* _pipeline = nullptr;
    GPUConstantBuffer* _constantBuffer = nullptr;
    SaschaCamera _camera;

    void OnMainRender(RenderTask* task, GPUContext* context);
    void UpdateUniforms(GPUContext* context, float aspect);

    void OnStart() override;
    void OnUpdate() override;
    void OnDestroy() override;
};
