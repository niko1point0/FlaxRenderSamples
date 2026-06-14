// Port of Sascha Willems Vulkan examples/triangle to Flax.

#include "VulkanTriangleRenderer.h"
#include "SaschaGpu.h"
#include "Engine/Content/Assets/Shader.h"
#include "Engine/Content/Content.h"
#include "Engine/Core/Math/Color.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Engine/Globals.h"
#include "Engine/Graphics/GPUBuffer.h"
#include "Engine/Graphics/GPUContext.h"
#include "Engine/Graphics/GPUDevice.h"
#include "Engine/Graphics/GPUPipelineState.h"
#include "Engine/Graphics/GPUResource.h"
#include "Engine/Graphics/RenderBuffers.h"
#include "Engine/Graphics/RenderTask.h"
#include "Engine/Graphics/Shaders/GPUShader.h"
#include "Engine/Graphics/Shaders/GPUVertexLayout.h"

namespace
{
    struct TriangleVertex
    {
        Float3 Position;
        Float3 Color;
    };

    // Same vertices as Vulkan/examples/triangle/triangle.cpp
    static const TriangleVertex g_Vertices[3] =
    {
        { Float3(1.0f, 1.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f) },
        { Float3(-1.0f, 1.0f, 0.0f), Float3(0.0f, 1.0f, 0.0f) },
        { Float3(0.0f, -1.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f) },
    };

    static const uint32 g_Indices[3] = { 0, 1, 2 };

    static VertexElement g_LayoutElements[2] =
    {
        { VertexElement::Types::Position, 0, 0, 0, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::Color, 0, 12, 0, PixelFormat::R32G32B32_Float },
    };

    GPU_CB_STRUCT(UniformData {
        Matrix ProjectionMatrix;
        Matrix ModelMatrix;
        Matrix ViewMatrix;
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

VulkanTriangleRenderer::VulkanTriangleRenderer(const SpawnParams& params)
    : Script(params)
{
    _tickUpdate = true;
}

void VulkanTriangleRenderer::OnStart()
{
    Shader* shader = Content::Load<Shader>(Globals::ProjectContentFolder / TEXT("Shaders/Triangle.flax"));
    GPUShader* gpuShader = shader->GetShader();

    static GPUVertexLayout::Elements layoutList(g_LayoutElements, 2);
    g_VertexLayout = GPUVertexLayout::Get(layoutList, true);

    _vertexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("TriangleVB"));
    _vertexBuffer->Init(GPUBufferDescription::Vertex(g_VertexLayout, sizeof(TriangleVertex), 3, g_Vertices));
    g_BindVB = Span<GPUBuffer*>(&_vertexBuffer, 1);

    _indexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("TriangleIB"));
    _indexBuffer->Init(GPUBufferDescription::Index(sizeof(uint32), 3, g_Indices));

    _constantBuffer = gpuShader->GetCB(0);

    _pipeline = GPUDevice::Instance->CreatePipelineState();
    GPUPipelineState::Description psDesc = GPUPipelineState::Description::Default;
    psDesc.CullMode = CullMode::TwoSided;
    psDesc.VS = gpuShader->GetVS("VS");
    psDesc.PS = gpuShader->GetPS("PS");
    _pipeline->Init(psDesc);

    _camera.setPosition(glm::vec3(0.0f, 0.0f, -2.5f));
    _camera.setRotation(glm::vec3(0.0f));
    _camera.setPerspective(60.0f, 1.0f, 1.0f, 256.0f);

    MainRenderTask::Instance->IsCustomRendering = true;
    MainRenderTask::Instance->Render.Bind<VulkanTriangleRenderer, &VulkanTriangleRenderer::OnMainRender>(this);
}

void VulkanTriangleRenderer::OnUpdate()
{
    _camera.HandleMouseInput();
}

void VulkanTriangleRenderer::OnDestroy()
{
    MainRenderTask::Instance->Render.Unbind<VulkanTriangleRenderer, &VulkanTriangleRenderer::OnMainRender>(this);
    MainRenderTask::Instance->IsCustomRendering = false;

    SAFE_DELETE_GPU_RESOURCE(_pipeline);
    SAFE_DELETE_GPU_RESOURCE(_vertexBuffer);
    SAFE_DELETE_GPU_RESOURCE(_indexBuffer);
    _constantBuffer = nullptr;
}

void VulkanTriangleRenderer::UpdateUniforms(GPUContext* context, float aspect)
{
    // triangle.cpp: shaderData.projectionMatrix = camera.matrices.perspective; etc.
    _camera.syncProjection(aspect);
    const glm::mat4& projection = _camera.matrices.perspective;
    const glm::mat4& view = _camera.matrices.view;
    const glm::mat4 model = glm::mat4(1.0f);

    UniformData data;
    SaschaGpu::StoreVulkan(data.ProjectionMatrix, projection);
    SaschaGpu::StoreVulkan(data.ModelMatrix, model);
    SaschaGpu::StoreVulkan(data.ViewMatrix, view);
    context->UpdateCB(_constantBuffer, &data);
}

void VulkanTriangleRenderer::OnMainRender(RenderTask* task, GPUContext* context)
{
    SceneRenderTask* sceneTask = static_cast<SceneRenderTask*>(task);
    GPUTextureView* output = sceneTask->GetOutputView();
    // Match Vulkan vkCmdSetViewport / vkCmdSetScissor (0,0,width,height) and scene depth buffer size.
    Viewport viewport = sceneTask->GetViewport();
    const float aspect = viewport.GetAspectRatio() > 0.0f ? viewport.GetAspectRatio() : 1.0f;

    UpdateUniforms(context, aspect);

    BindAndClearOutput(sceneTask, context, output, Color(0.0f, 0.0f, 0.2f, 1.0f));
    context->SetViewportAndScissors(viewport);
    context->BindVB(g_BindVB, nullptr, g_VertexLayout);
    context->BindIB(_indexBuffer);
    context->BindCB(0, _constantBuffer);
    context->SetState(_pipeline);
    context->DrawIndexed(3);
}
