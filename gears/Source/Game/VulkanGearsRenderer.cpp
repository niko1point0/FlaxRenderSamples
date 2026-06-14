// Port of Sascha Willems Vulkan examples/gears to Flax.

#include "VulkanGearsRenderer.h"
#include "SaschaGpu.h"
#include "Engine/Content/Assets/Shader.h"
#include "Engine/Content/Content.h"
#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Math/Color.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Engine/Globals.h"
#include "Engine/Engine/Time.h"
#include "Engine/Graphics/GPUBuffer.h"
#include "Engine/Graphics/GPUContext.h"
#include "Engine/Graphics/GPUDevice.h"
#include "Engine/Graphics/GPUPipelineState.h"
#include "Engine/Graphics/GPUResource.h"
#include "Engine/Graphics/RenderBuffers.h"
#include "Engine/Graphics/RenderTask.h"
#include "Engine/Graphics/Shaders/GPUShader.h"
#include "Engine/Graphics/Shaders/GPUVertexLayout.h"
#include "Engine/Profiler/Profiler.h"

namespace
{
    struct GearDefinition
    {
        float InnerRadius;
        float OuterRadius;
        float Width;
        int32 NumTeeth;
        float ToothDepth;
        Float3 Color;
        Float3 Pos;
        float RotSpeed;
        float RotOffset;
    };

    class Gear
    {
    public:
        struct Vertex
        {
            Float3 Position;
            Float3 Normal;
            Float3 Color;
        };

        Float3 Color;
        Float3 Pos;
        float RotSpeed = 0.0f;
        float RotOffset = 0.0f;
        uint32 IndexCount = 0;
        uint32 IndexStart = 0;

        void Generate(GearDefinition& gearDefinition, Array<Vertex>& vertexBuffer, Array<uint32>& indexBuffer)
        {
            Color = gearDefinition.Color;
            Pos = gearDefinition.Pos;
            RotOffset = gearDefinition.RotOffset;
            RotSpeed = gearDefinition.RotSpeed;

            IndexStart = (uint32)indexBuffer.Count();

            const float r0 = gearDefinition.InnerRadius;
            const float r1 = gearDefinition.OuterRadius - gearDefinition.ToothDepth * 0.5f;
            const float r2 = gearDefinition.OuterRadius + gearDefinition.ToothDepth * 0.5f;
            const float da = TWO_PI / (float)gearDefinition.NumTeeth / 4.0f;

            auto addFace = [&indexBuffer](int32 a, int32 b, int32 c)
            {
                indexBuffer.Add((uint32)a);
                indexBuffer.Add((uint32)b);
                indexBuffer.Add((uint32)c);
            };

            auto addVertex = [this, &vertexBuffer](float x, float y, float z, const Float3& normal) -> int32
            {
                Vertex v;
                v.Position = Float3(x, y, z);
                v.Normal = normal;
                v.Color = Color;
                vertexBuffer.Add(v);
                return vertexBuffer.Count() - 1;
            };

            for (int32 i = 0; i < gearDefinition.NumTeeth; i++)
            {
                const float ta = (float)i * TWO_PI / (float)gearDefinition.NumTeeth;

                const float cosTa = Math::Cos(ta);
                const float cosTa1da = Math::Cos(ta + da);
                const float cosTa2da = Math::Cos(ta + 2.0f * da);
                const float cosTa3da = Math::Cos(ta + 3.0f * da);
                const float cosTa4da = Math::Cos(ta + 4.0f * da);
                const float sinTa = Math::Sin(ta);
                const float sinTa1da = Math::Sin(ta + da);
                const float sinTa2da = Math::Sin(ta + 2.0f * da);
                const float sinTa3da = Math::Sin(ta + 3.0f * da);
                const float sinTa4da = Math::Sin(ta + 4.0f * da);

                float u1 = r2 * cosTa1da - r1 * cosTa;
                float v1 = r2 * sinTa1da - r1 * sinTa;
                float len = Math::Sqrt(u1 * u1 + v1 * v1);
                u1 /= len;
                v1 /= len;
                const float u2 = r1 * cosTa3da - r2 * cosTa2da;
                const float v2 = r1 * sinTa3da - r2 * sinTa2da;

                Float3 normal;
                int32 ix0, ix1, ix2, ix3, ix4, ix5;

                normal = Float3(0.0f, 0.0f, 1.0f);
                ix0 = addVertex(r0 * cosTa, r0 * sinTa, gearDefinition.Width * 0.5f, normal);
                ix1 = addVertex(r1 * cosTa, r1 * sinTa, gearDefinition.Width * 0.5f, normal);
                ix2 = addVertex(r0 * cosTa, r0 * sinTa, gearDefinition.Width * 0.5f, normal);
                ix3 = addVertex(r1 * cosTa3da, r1 * sinTa3da, gearDefinition.Width * 0.5f, normal);
                ix4 = addVertex(r0 * cosTa4da, r0 * sinTa4da, gearDefinition.Width * 0.5f, normal);
                ix5 = addVertex(r1 * cosTa4da, r1 * sinTa4da, gearDefinition.Width * 0.5f, normal);
                addFace(ix0, ix1, ix2);
                addFace(ix1, ix3, ix2);
                addFace(ix2, ix3, ix4);
                addFace(ix3, ix5, ix4);

                normal = Float3(0.0f, 0.0f, 1.0f);
                ix0 = addVertex(r1 * cosTa, r1 * sinTa, gearDefinition.Width * 0.5f, normal);
                ix1 = addVertex(r2 * cosTa1da, r2 * sinTa1da, gearDefinition.Width * 0.5f, normal);
                ix2 = addVertex(r1 * cosTa3da, r1 * sinTa3da, gearDefinition.Width * 0.5f, normal);
                ix3 = addVertex(r2 * cosTa2da, r2 * sinTa2da, gearDefinition.Width * 0.5f, normal);
                addFace(ix0, ix1, ix2);
                addFace(ix1, ix3, ix2);

                normal = Float3(0.0f, 0.0f, -1.0f);
                ix0 = addVertex(r1 * cosTa, r1 * sinTa, -gearDefinition.Width * 0.5f, normal);
                ix1 = addVertex(r0 * cosTa, r0 * sinTa, -gearDefinition.Width * 0.5f, normal);
                ix2 = addVertex(r1 * cosTa3da, r1 * sinTa3da, -gearDefinition.Width * 0.5f, normal);
                ix3 = addVertex(r0 * cosTa, r0 * sinTa, -gearDefinition.Width * 0.5f, normal);
                ix4 = addVertex(r1 * cosTa4da, r1 * sinTa4da, -gearDefinition.Width * 0.5f, normal);
                ix5 = addVertex(r0 * cosTa4da, r0 * sinTa4da, -gearDefinition.Width * 0.5f, normal);
                addFace(ix0, ix1, ix2);
                addFace(ix1, ix3, ix2);
                addFace(ix2, ix3, ix4);
                addFace(ix3, ix5, ix4);

                normal = Float3(0.0f, 0.0f, -1.0f);
                ix0 = addVertex(r1 * cosTa3da, r1 * sinTa3da, -gearDefinition.Width * 0.5f, normal);
                ix1 = addVertex(r2 * cosTa2da, r2 * sinTa2da, -gearDefinition.Width * 0.5f, normal);
                ix2 = addVertex(r1 * cosTa, r1 * sinTa, -gearDefinition.Width * 0.5f, normal);
                ix3 = addVertex(r2 * cosTa1da, r2 * sinTa1da, -gearDefinition.Width * 0.5f, normal);
                addFace(ix0, ix1, ix2);
                addFace(ix1, ix3, ix2);

                normal = Float3(v1, -u1, 0.0f);
                ix0 = addVertex(r1 * cosTa, r1 * sinTa, gearDefinition.Width * 0.5f, normal);
                ix1 = addVertex(r1 * cosTa, r1 * sinTa, -gearDefinition.Width * 0.5f, normal);
                ix2 = addVertex(r2 * cosTa1da, r2 * sinTa1da, gearDefinition.Width * 0.5f, normal);
                ix3 = addVertex(r2 * cosTa1da, r2 * sinTa1da, -gearDefinition.Width * 0.5f, normal);
                addFace(ix0, ix1, ix2);
                addFace(ix1, ix3, ix2);

                normal = Float3(cosTa, sinTa, 0.0f);
                ix0 = addVertex(r2 * cosTa1da, r2 * sinTa1da, gearDefinition.Width * 0.5f, normal);
                ix1 = addVertex(r2 * cosTa1da, r2 * sinTa1da, -gearDefinition.Width * 0.5f, normal);
                ix2 = addVertex(r2 * cosTa2da, r2 * sinTa2da, gearDefinition.Width * 0.5f, normal);
                ix3 = addVertex(r2 * cosTa2da, r2 * sinTa2da, -gearDefinition.Width * 0.5f, normal);
                addFace(ix0, ix1, ix2);
                addFace(ix1, ix3, ix2);

                normal = Float3(v2, -u2, 0.0f);
                ix0 = addVertex(r2 * cosTa2da, r2 * sinTa2da, gearDefinition.Width * 0.5f, normal);
                ix1 = addVertex(r2 * cosTa2da, r2 * sinTa2da, -gearDefinition.Width * 0.5f, normal);
                ix2 = addVertex(r1 * cosTa3da, r1 * sinTa3da, gearDefinition.Width * 0.5f, normal);
                ix3 = addVertex(r1 * cosTa3da, r1 * sinTa3da, -gearDefinition.Width * 0.5f, normal);
                addFace(ix0, ix1, ix2);
                addFace(ix1, ix3, ix2);

                normal = Float3(cosTa, sinTa, 0.0f);
                ix0 = addVertex(r1 * cosTa3da, r1 * sinTa3da, gearDefinition.Width * 0.5f, normal);
                ix1 = addVertex(r1 * cosTa3da, r1 * sinTa3da, -gearDefinition.Width * 0.5f, normal);
                ix2 = addVertex(r1 * cosTa4da, r1 * sinTa4da, gearDefinition.Width * 0.5f, normal);
                ix3 = addVertex(r1 * cosTa4da, r1 * sinTa4da, -gearDefinition.Width * 0.5f, normal);
                addFace(ix0, ix1, ix2);
                addFace(ix1, ix3, ix2);

                ix0 = addVertex(r0 * cosTa, r0 * sinTa, -gearDefinition.Width * 0.5f, Float3(-cosTa, -sinTa, 0.0f));
                ix1 = addVertex(r0 * cosTa, r0 * sinTa, gearDefinition.Width * 0.5f, Float3(-cosTa, -sinTa, 0.0f));
                ix2 = addVertex(r0 * cosTa4da, r0 * sinTa4da, -gearDefinition.Width * 0.5f, Float3(-cosTa4da, -sinTa4da, 0.0f));
                ix3 = addVertex(r0 * cosTa4da, r0 * sinTa4da, gearDefinition.Width * 0.5f, Float3(-cosTa4da, -sinTa4da, 0.0f));
                addFace(ix0, ix1, ix2);
                addFace(ix1, ix3, ix2);
            }

            IndexCount = (uint32)indexBuffer.Count() - IndexStart;
        }
    };

    static VertexElement g_LayoutElements[3] =
    {
        { VertexElement::Types::Position, 0, 0, 0, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::Normal, 0, 12, 0, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::Color, 0, 24, 0, PixelFormat::R32G32B32_Float },
    };

    GPU_CB_STRUCT(UniformData {
        Matrix ProjectionMatrix;
        Matrix ViewMatrix;
        Float4 LightPos;
        Matrix ModelMatrix[3];
        uint32 InstanceBase;
    });

    Span<GPUBuffer*> g_BindVB;
    static GPUVertexLayout* g_VertexLayout;

    void BindAndClearOutput(SceneRenderTask* sceneTask, GPUContext* context, GPUTextureView* output, const Color& clearColor)
    {
        GPUTextureView* depth = (sceneTask->Buffers && sceneTask->Buffers->DepthBuffer)
            ? sceneTask->Buffers->DepthBuffer->View()
            : nullptr;

        if (depth)
        {
            context->SetRenderTarget(depth, output);
            context->Clear(output, clearColor);
            context->ClearDepth(depth);
        }
        else
        {
            context->Clear(output, clearColor);
            context->SetRenderTarget(output);
        }
    }

}

VulkanGearsRenderer::VulkanGearsRenderer(const SpawnParams& params)
    : Script(params)
{
    _tickUpdate = true;
}

void VulkanGearsRenderer::PrepareGears()
{
    GearDefinition gearDefinitions[NumGears];

    gearDefinitions[0].InnerRadius = 1.0f;
    gearDefinitions[0].OuterRadius = 4.0f;
    gearDefinitions[0].Width = 1.0f;
    gearDefinitions[0].NumTeeth = 20;
    gearDefinitions[0].ToothDepth = 0.7f;
    gearDefinitions[0].Color = Float3(1.0f, 0.0f, 0.0f);
    gearDefinitions[0].Pos = Float3(-3.0f, 0.0f, 0.0f);
    gearDefinitions[0].RotSpeed = 1.0f;
    gearDefinitions[0].RotOffset = 0.0f;

    gearDefinitions[1].InnerRadius = 0.5f;
    gearDefinitions[1].OuterRadius = 2.0f;
    gearDefinitions[1].Width = 2.0f;
    gearDefinitions[1].NumTeeth = 10;
    gearDefinitions[1].ToothDepth = 0.7f;
    gearDefinitions[1].Color = Float3(0.0f, 1.0f, 0.2f);
    gearDefinitions[1].Pos = Float3(3.1f, 0.0f, 0.0f);
    gearDefinitions[1].RotSpeed = -2.0f;
    gearDefinitions[1].RotOffset = -9.0f;

    gearDefinitions[2].InnerRadius = 1.3f;
    gearDefinitions[2].OuterRadius = 2.0f;
    gearDefinitions[2].Width = 0.5f;
    gearDefinitions[2].NumTeeth = 10;
    gearDefinitions[2].ToothDepth = 0.7f;
    gearDefinitions[2].Color = Float3(0.0f, 0.0f, 1.0f);
    gearDefinitions[2].Pos = Float3(-3.1f, -6.2f, 0.0f);
    gearDefinitions[2].RotSpeed = -2.0f;
    gearDefinitions[2].RotOffset = -30.0f;

    Array<Gear::Vertex> vertices;
    Array<uint32> indices;
    Gear gears[NumGears];

    for (int32 i = 0; i < NumGears; i++)
        gears[i].Generate(gearDefinitions[i], vertices, indices);

    _gearCount = NumGears;
    for (int32 i = 0; i < NumGears; i++)
    {
        _gears[i].Pos = gears[i].Pos;
        _gears[i].RotSpeed = gears[i].RotSpeed;
        _gears[i].RotOffset = gears[i].RotOffset;
        _gears[i].IndexCount = gears[i].IndexCount;
        _gears[i].IndexStart = gears[i].IndexStart;
    }

    static GPUVertexLayout::Elements layoutList(g_LayoutElements, 3);
    g_VertexLayout = GPUVertexLayout::Get(layoutList, true);

    _vertexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("GearsVB"));
    _vertexBuffer->Init(GPUBufferDescription::Vertex(g_VertexLayout, sizeof(Gear::Vertex), vertices.Count(), vertices.Get()));
    g_BindVB = Span<GPUBuffer*>(&_vertexBuffer, 1);

    _indexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("GearsIB"));
    _indexBuffer->Init(GPUBufferDescription::Index(sizeof(uint32), indices.Count(), indices.Get()));
}

void VulkanGearsRenderer::OnStart()
{
    PrepareGears();

    Shader* shader = Content::Load<Shader>(Globals::ProjectContentFolder / TEXT("Shaders/Gears.flax"));
    GPUShader* gpuShader = shader->GetShader();
    _constantBuffer = gpuShader->GetCB(0);

    _pipeline = GPUDevice::Instance->CreatePipelineState();
    GPUPipelineState::Description psDesc = GPUPipelineState::Description::Default;
    psDesc.CullMode = CullMode::Normal;
    psDesc.VS = gpuShader->GetVS("VS");
    psDesc.PS = gpuShader->GetPS("PS");
    _pipeline->Init(psDesc);

    _camera.setPosition(glm::vec3(0.0f, 2.5f, -16.0f));
    _camera.setRotation(glm::vec3(0.0f));
    _camera.setPerspective(60.0f, 1.0f, 0.001f, 256.0f);
    _camera.rotationSpeed *= 0.25f;

    MainRenderTask::Instance->IsCustomRendering = true;
    MainRenderTask::Instance->Render.Bind<VulkanGearsRenderer, &VulkanGearsRenderer::OnMainRender>(this);
}

void VulkanGearsRenderer::OnUpdate()
{
    _camera.HandleMouseInput();
}

void VulkanGearsRenderer::UpdateUniforms(GPUContext* context, float aspect, uint32 instanceBase)
{
    // gears.cpp: degree = timer * 360 (timerSpeed *= 0.25 in constructor)
    const float degree = Time::GetGameTime() * 0.25f * 360.0f;

    _camera.syncProjection(aspect);
    const glm::mat4& projection = _camera.matrices.perspective;
    const glm::mat4& view = _camera.matrices.view;

    UniformData data;
    SaschaGpu::StoreRowVector(data.ProjectionMatrix, projection);
    SaschaGpu::StoreRowVector(data.ViewMatrix, view);
    data.LightPos = Float4(0.0f, 0.0f, 2.5f, 1.0f);

    for (int32 i = 0; i < _gearCount; i++)
    {
        // Vulkan: translate then rotate on the same matrix chain
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, SaschaGlm::ToGlm(_gears[i].Pos));
        model = glm::rotate(model, glm::radians(_gears[i].RotSpeed * degree + _gears[i].RotOffset), glm::vec3(0.0f, 0.0f, 1.0f));
        SaschaGpu::StoreRowVector(data.ModelMatrix[i], model);
    }

    data.InstanceBase = instanceBase;
    context->UpdateCB(_constantBuffer, &data);
}

void VulkanGearsRenderer::OnDestroy()
{
    MainRenderTask::Instance->Render.Unbind<VulkanGearsRenderer, &VulkanGearsRenderer::OnMainRender>(this);
    MainRenderTask::Instance->IsCustomRendering = false;

    SAFE_DELETE_GPU_RESOURCE(_pipeline);
    SAFE_DELETE_GPU_RESOURCE(_vertexBuffer);
    SAFE_DELETE_GPU_RESOURCE(_indexBuffer);
    _constantBuffer = nullptr;
}

void VulkanGearsRenderer::OnMainRender(RenderTask* task, GPUContext* context)
{
    SceneRenderTask* sceneTask = static_cast<SceneRenderTask*>(task);
    GPUTextureView* output = sceneTask->GetOutputView();
    Viewport viewport = sceneTask->GetViewport();
    const float aspect = viewport.GetAspectRatio() > 0.0f ? viewport.GetAspectRatio() : 1.0f;

    BindAndClearOutput(sceneTask, context, output, Color(0.025f, 0.025f, 0.025f, 1.0f));
    context->SetViewportAndScissors(viewport);
    context->BindVB(g_BindVB, nullptr, g_VertexLayout);
    context->BindIB(_indexBuffer);
    context->SetState(_pipeline);

    context->BindCB(0, _constantBuffer);

    {
        PROFILE_GPU("Gears");
        for (int32 j = 0; j < _gearCount; j++)
        {
            UpdateUniforms(context, aspect, (uint32)j);
            const GearDrawData& gear = _gears[j];
            switch (j)
            {
            case 0:
            {
                PROFILE_GPU("Red Gear");
                context->DrawIndexedInstanced(gear.IndexCount, 1, j, 0, (int32)gear.IndexStart);
                break;
            }
            case 1:
            {
                PROFILE_GPU("Green Gear");
                context->DrawIndexedInstanced(gear.IndexCount, 1, j, 0, (int32)gear.IndexStart);
                break;
            }
            case 2:
            {
                PROFILE_GPU("Blue Gear");
                context->DrawIndexedInstanced(gear.IndexCount, 1, j, 0, (int32)gear.IndexStart);
                break;
            }
            default:
                context->DrawIndexedInstanced(gear.IndexCount, 1, j, 0, (int32)gear.IndexStart);
                break;
            }
        }
    }
}
