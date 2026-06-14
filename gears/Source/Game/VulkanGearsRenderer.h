// Port of Sascha Willems Vulkan examples/gears to Flax.

#pragma once

#include "Engine/Core/Types/BaseTypes.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Scripting/Script.h"
#include "SaschaCamera.h"

class GPUBuffer;
class GPUConstantBuffer;
class GPUPipelineState;
class RenderTask;
class GPUContext;

/// <summary>
/// Renders the Vulkan gears example: three procedurally generated animated gears with Phong lighting.
/// </summary>
API_CLASS(Namespace="") class VulkanGearsRenderer : public Script
{
    API_AUTO_SERIALIZATION();
    DECLARE_SCRIPTING_TYPE(VulkanGearsRenderer);

    static constexpr int32 NumGears = 3;

    struct GearDrawData
    {
        Float3 Pos;
        float RotSpeed = 0.0f;
        float RotOffset = 0.0f;
        uint32 IndexCount = 0;
        uint32 IndexStart = 0;
    };

    GPUBuffer* _vertexBuffer = nullptr;
    GPUBuffer* _indexBuffer = nullptr;
    GPUPipelineState* _pipeline = nullptr;
    GPUConstantBuffer* _constantBuffer = nullptr;
    SaschaCamera _camera;
    GearDrawData _gears[NumGears];
    int32 _gearCount = 0;

    void PrepareGears();
    void UpdateUniforms(GPUContext* context, float aspect, uint32 instanceBase);
    void OnMainRender(RenderTask* task, GPUContext* context);

    void OnStart() override;
    void OnUpdate() override;
    void OnDestroy() override;
};
