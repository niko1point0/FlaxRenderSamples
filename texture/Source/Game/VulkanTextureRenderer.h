// Port of Sascha Willems Vulkan examples/texture to Flax.

#pragma once

#include "Engine/Scripting/Script.h"
#include "SaschaCamera.h"

class GPUBuffer;
class GPUConstantBuffer;
class GPUPipelineState;
class RenderTask;
class GPUContext;
class Texture;

/// <summary>
/// Renders the Vulkan texture example: textured quad with simple lighting.
/// </summary>
API_CLASS(Namespace="") class VulkanTextureRenderer : public Script
{
    API_AUTO_SERIALIZATION();
    DECLARE_SCRIPTING_TYPE(VulkanTextureRenderer);

    GPUBuffer* _vertexBuffer = nullptr;
    GPUBuffer* _indexBuffer = nullptr;
    GPUPipelineState* _pipeline = nullptr;
    GPUConstantBuffer* _constantBuffer = nullptr;
    Texture* _texture = nullptr;
    uint32 _indexCount = 0;
    SaschaCamera _camera;

    void OnMainRender(RenderTask* task, GPUContext* context);
    void UpdateUniforms(GPUContext* context, float aspect);

    void OnStart() override;
    void OnUpdate() override;
    void OnDestroy() override;
};
