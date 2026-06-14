// Port of Sascha Willems Vulkan examples/texture to Flax.

#include "VulkanTextureRenderer.h"
#include "SaschaGpu.h"
#include "Engine/Content/Assets/Shader.h"
#include "Engine/Content/Assets/Texture.h"
#include "Engine/Content/Content.h"
#include "Engine/Core/Math/Color.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Math/Matrix.h"
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
    struct QuadVertex
    {
        Float3 Position;
        Float2 UV;
        Float3 Normal;
    };

    // Same quad as Vulkan/examples/texture/texture.cpp
    static const QuadVertex g_Vertices[4] =
    {
        { Float3(1.0f, 1.0f, 0.0f), Float2(1.0f, 1.0f), Float3(0.0f, 0.0f, 1.0f) },
        { Float3(-1.0f, 1.0f, 0.0f), Float2(0.0f, 1.0f), Float3(0.0f, 0.0f, 1.0f) },
        { Float3(-1.0f, -1.0f, 0.0f), Float2(0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f) },
        { Float3(1.0f, -1.0f, 0.0f), Float2(1.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f) },
    };

    static const uint32 g_Indices[6] = { 0, 1, 2, 2, 3, 0 };

    static VertexElement g_LayoutElements[3] =
    {
        { VertexElement::Types::Position, 0, 0, 0, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::TexCoord0, 0, 12, 0, PixelFormat::R32G32_Float },
        { VertexElement::Types::Normal, 0, 20, 0, PixelFormat::R32G32B32_Float },
    };

    GPU_CB_STRUCT(UniformData {
        Matrix ProjectionMatrix;
        Matrix ModelMatrix;
        Float4 ViewPos;
        float LodBias;
        float Padding[3];
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

    static Texture* LoadMetalPlateTexture()
    {
        return Content::Load<Texture>(Globals::ProjectContentFolder / TEXT("Textures/metalplate01_rgba.flax"));
    }
}

VulkanTextureRenderer::VulkanTextureRenderer(const SpawnParams& params)
    : Script(params)
{
    _tickUpdate = true;
}

void VulkanTextureRenderer::OnStart()
{
    _texture = LoadMetalPlateTexture();

    Shader* shader = Content::Load<Shader>(Globals::ProjectContentFolder / TEXT("Shaders/Texture.flax"));
    GPUShader* gpuShader = shader->GetShader();

    static GPUVertexLayout::Elements layoutList(g_LayoutElements, 3);
    g_VertexLayout = GPUVertexLayout::Get(layoutList, true);

    _vertexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("TextureVB"));
    _vertexBuffer->Init(GPUBufferDescription::Vertex(g_VertexLayout, sizeof(QuadVertex), 4, g_Vertices));
    g_BindVB = Span<GPUBuffer*>(&_vertexBuffer, 1);

    _indexCount = 6;
    _indexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("TextureIB"));
    _indexBuffer->Init(GPUBufferDescription::Index(sizeof(uint32), _indexCount, g_Indices));

    _constantBuffer = gpuShader->GetCB(0);

    _pipeline = GPUDevice::Instance->CreatePipelineState();
    GPUPipelineState::Description psDesc = GPUPipelineState::Description::Default;
    psDesc.CullMode = CullMode::TwoSided;
    psDesc.VS = gpuShader->GetVS("VS");
    psDesc.PS = gpuShader->GetPS("PS");
    _pipeline->Init(psDesc);

    _camera.setPosition(glm::vec3(0.0f, 0.0f, -2.5f));
    _camera.setRotation(glm::vec3(0.0f, 15.0f, 0.0f));

    _camera.setPerspective(60.0f, 1.0f, 0.1f, 256.0f);
    MainRenderTask::Instance->IsCustomRendering = true;
    MainRenderTask::Instance->Render.Bind<VulkanTextureRenderer, &VulkanTextureRenderer::OnMainRender>(this);
}

void VulkanTextureRenderer::OnUpdate()
{
    _camera.HandleMouseInput();
}

void VulkanTextureRenderer::OnDestroy()
{
    MainRenderTask::Instance->Render.Unbind<VulkanTextureRenderer, &VulkanTextureRenderer::OnMainRender>(this);
    MainRenderTask::Instance->IsCustomRendering = false;

    SAFE_DELETE_GPU_RESOURCE(_pipeline);
    SAFE_DELETE_GPU_RESOURCE(_vertexBuffer);
    SAFE_DELETE_GPU_RESOURCE(_indexBuffer);
    _constantBuffer = nullptr;
    _texture = nullptr;
}

void VulkanTextureRenderer::UpdateUniforms(GPUContext* context, float aspect)
{
    // Sascha texture.cpp + texture.vert: gl_Position = P * model * pos, model = camera.view
    _camera.syncProjection(aspect);

    const glm::mat4& projection = _camera.matrices.perspective;
    const glm::mat4 model = _camera.matrices.view;

    UniformData data;
    SaschaGpu::StoreVulkan(data.ProjectionMatrix, projection);
    SaschaGpu::StoreVulkan(data.ModelMatrix, model);
    data.ViewPos = _camera.GetViewPos();
    data.LodBias = 0.0f;
    data.Padding[0] = data.Padding[1] = data.Padding[2] = 0.0f;
    context->UpdateCB(_constantBuffer, &data);
}

void VulkanTextureRenderer::OnMainRender(RenderTask* task, GPUContext* context)
{
    GPUTexture* gpuTexture = _texture ? _texture->GetTexture() : nullptr;
    if (gpuTexture == nullptr)
        return;

    SceneRenderTask* sceneTask = static_cast<SceneRenderTask*>(task);
    GPUTextureView* output = sceneTask->GetOutputView();
    Viewport viewport = sceneTask->GetViewport();
    const float aspect = viewport.GetAspectRatio() > 0.0f ? viewport.GetAspectRatio() : 1.0f;

    UpdateUniforms(context, aspect);

    // vulkanexamplebase.h defaultClearColor (texture.cpp uses defaultClearColor, not triangle's blue)
    BindAndClearOutput(sceneTask, context, output, Color(0.025f, 0.025f, 0.025f, 1.0f));
    context->SetViewportAndScissors(viewport);
    context->BindVB(g_BindVB, nullptr, g_VertexLayout);
    context->BindIB(_indexBuffer);
    context->BindCB(0, _constantBuffer);
    context->BindSR(0, gpuTexture);
    context->SetState(_pipeline);
    context->DrawIndexed(_indexCount);
}
