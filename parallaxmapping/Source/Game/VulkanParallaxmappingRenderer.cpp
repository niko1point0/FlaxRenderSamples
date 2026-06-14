// Port of Sascha Willems Vulkan examples/parallaxmapping to Flax.

#include "VulkanParallaxmappingRenderer.h"
#include "SaschaGpu.h"
#include "Engine/Content/Assets/Shader.h"
#include "Engine/Content/Assets/Texture.h"
#include "Engine/Content/Content.h"
#include "Engine/Core/Math/Color.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Core/Types/Span.h"
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
#include "Engine/Input/Enums.h"
#include "Engine/Input/Input.h"

namespace
{
    struct PlaneVertex
    {
        Float3 Position;
        Float2 UV;
        Float3 Normal;
        Float4 Tangent;
    };

    // Vulkan plane.gltf + FileLoadingFlags::FlipY (parallaxmapping.cpp) flips normal.y
    static const PlaneVertex g_PlaneVertices[4] =
    {
        { Float3(5.0f, 0.0f, 5.0f), Float2(1.0f, 1.0f), Float3(0.0f, -1.0f, 0.0f), Float4(1.0f, 0.0f, 0.0f, 1.0f) },
        { Float3(5.0f, 0.0f, -5.0f), Float2(1.0f, 0.0f), Float3(0.0f, -1.0f, 0.0f), Float4(1.0f, 0.0f, 0.0f, 1.0f) },
        { Float3(-5.0f, 0.0f, -5.0f), Float2(0.0f, 0.0f), Float3(0.0f, -1.0f, 0.0f), Float4(1.0f, 0.0f, 0.0f, 1.0f) },
        { Float3(-5.0f, 0.0f, 5.0f), Float2(0.0f, 1.0f), Float3(0.0f, -1.0f, 0.0f), Float4(1.0f, 0.0f, 0.0f, 1.0f) },
    };

    static const uint32 g_PlaneIndices[6] = { 0, 1, 2, 0, 2, 3 };

    static VertexElement g_LayoutElements[4] =
    {
        { VertexElement::Types::Position, 0, 0, 0, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::TexCoord0, 0, 12, 0, PixelFormat::R32G32_Float },
        { VertexElement::Types::Normal, 0, 20, 0, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::Tangent, 0, 32, 0, PixelFormat::R32G32B32A32_Float },
    };

    GPU_CB_STRUCT(VertexUniformData {
        Matrix ProjectionMatrix;
        Matrix ViewMatrix;
        Matrix ModelMatrix;
        Float4 LightPos;
        Float4 CameraPos;
    });

    GPU_CB_STRUCT(FragmentUniformData {
        float MappingMode;
        float HeightScale;
        float ParallaxBias;
        float NumLayers;
    });

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

VulkanParallaxmappingRenderer::VulkanParallaxmappingRenderer(const SpawnParams& params)
    : Script(params)
{
    _tickUpdate = true;
}

void VulkanParallaxmappingRenderer::OnStart()
{
    _camera.type = SaschaCamera::Type::FirstPerson;
    _camera.setPosition(glm::vec3(0.0f, 1.25f, -1.5f));
    _camera.setRotation(glm::vec3(-45.0f, 0.0f, 0.0f));

    _camera.setPerspective(60.0f, 1.0f, 0.1f, 256.0f);
    _colorMap = Content::Load<Texture>(Globals::ProjectContentFolder / TEXT("Textures/rocks_color_rgba.flax"));
    _normalHeightMap = Content::Load<Texture>(Globals::ProjectContentFolder / TEXT("Textures/rocks_normal_height_rgba.flax"));

    Shader* shader = Content::Load<Shader>(Globals::ProjectContentFolder / TEXT("Shaders/Parallaxmapping.flax"));
    GPUShader* gpuShader = shader->GetShader();
    _vertexCb = gpuShader->GetCB(0);
    _fragmentCb = gpuShader->GetCB(1);

    static GPUVertexLayout::Elements layoutList(g_LayoutElements, 4);
    _vertexLayout = GPUVertexLayout::Get(layoutList, true);

    _pipeline = GPUDevice::Instance->CreatePipelineState();
    GPUPipelineState::Description psDesc = GPUPipelineState::Description::Default;
    psDesc.CullMode = CullMode::TwoSided;
    psDesc.VS = gpuShader->GetVS("VS");
    psDesc.PS = gpuShader->GetPS("PS");
    _pipeline->Init(psDesc);

    BuildMesh();

    MainRenderTask::Instance->IsCustomRendering = true;
    MainRenderTask::Instance->Render.Bind<VulkanParallaxmappingRenderer, &VulkanParallaxmappingRenderer::OnMainRender>(this);
}

void VulkanParallaxmappingRenderer::BuildMesh()
{
    if (_vertexBuffer == nullptr)
        _vertexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("ParallaxmappingVB"));
    if (_indexBuffer == nullptr)
        _indexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("ParallaxmappingIB"));

    _vertexBuffer->Init(GPUBufferDescription::Vertex(_vertexLayout, sizeof(PlaneVertex), 4, g_PlaneVertices));
    _indexBuffer->Init(GPUBufferDescription::Index(sizeof(uint32), 6, g_PlaneIndices));
    _indexCount = 6;
}

void VulkanParallaxmappingRenderer::HandleInput()
{
    if (Input::GetKeyDown(KeyboardKeys::M))
        _mappingMode = (_mappingMode + 1) % 5;

    if (Input::GetKey(KeyboardKeys::NumpadAdd))
        _heightScale = Math::Min(1.0f, _heightScale + 0.005f);
    if (Input::GetKey(KeyboardKeys::NumpadSubtract) || Input::GetKey(KeyboardKeys::Minus))
        _heightScale = Math::Max(0.0f, _heightScale - 0.005f);

    if (Input::GetKey(KeyboardKeys::PageUp))
        _numLayers = Math::Min(128.0f, _numLayers + 1.0f);
    if (Input::GetKey(KeyboardKeys::PageDown))
        _numLayers = Math::Max(1.0f, _numLayers - 1.0f);
}

void VulkanParallaxmappingRenderer::OnUpdate()
{
    _camera.HandleMouseInput();
    HandleInput();
    _lightTimer += Time::GetDeltaTime() * LightTimerSpeed;
    if (_lightTimer > 1.0f)
        _lightTimer -= 1.0f;
}

void VulkanParallaxmappingRenderer::OnDestroy()
{
    MainRenderTask::Instance->Render.Unbind<VulkanParallaxmappingRenderer, &VulkanParallaxmappingRenderer::OnMainRender>(this);
    MainRenderTask::Instance->IsCustomRendering = false;

    SAFE_DELETE_GPU_RESOURCE(_pipeline);
    SAFE_DELETE_GPU_RESOURCE(_vertexBuffer);
    SAFE_DELETE_GPU_RESOURCE(_indexBuffer);
    _vertexCb = nullptr;
    _fragmentCb = nullptr;
    _colorMap = nullptr;
    _normalHeightMap = nullptr;
    _vertexLayout = nullptr;
}

void VulkanParallaxmappingRenderer::UpdateUniforms(GPUContext* context, float aspect)
{
    _camera.syncProjection(aspect);

    const glm::mat4& projection = _camera.matrices.perspective;
    const glm::mat4& view = _camera.matrices.view;
    const glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(0.2f));

    VertexUniformData vertexData;
    SaschaGpu::StoreVulkan(vertexData.ProjectionMatrix, projection);
    SaschaGpu::StoreVulkan(vertexData.ViewMatrix, view);
    SaschaGpu::StoreVulkan(vertexData.ModelMatrix, model);
    // parallaxmapping.cpp: lightPos default (0, -2, 0, 1); x/z animated via sin/cos(radians(timer * 360)) * 1.5
    const float lightAngle = _lightTimer * 360.0f * DegreesToRadians;
    vertexData.LightPos = Float4(
        Math::Sin(lightAngle) * 1.5f,
        -2.0f,
        Math::Cos(lightAngle) * 1.5f,
        1.0f);
    // parallaxmapping.cpp: cameraPos = vec4(camera.position, -1) * -1 (negates xyz, w -> 1)
    vertexData.CameraPos = SaschaGlm::ToFloat4(glm::vec4(_camera.position, -1.0f) * -1.0f);
    context->UpdateCB(_vertexCb, &vertexData);
    context->BindCB(0, _vertexCb);

    FragmentUniformData fragmentData;
    fragmentData.MappingMode = (float)_mappingMode;
    fragmentData.HeightScale = _heightScale;
    fragmentData.ParallaxBias = _parallaxBias;
    fragmentData.NumLayers = _numLayers;
    context->UpdateCB(_fragmentCb, &fragmentData);
    context->BindCB(1, _fragmentCb);

    if (_colorMap)
        context->BindSR(0, _colorMap->GetTexture());
    if (_normalHeightMap)
        context->BindSR(1, _normalHeightMap->GetTexture());
}

void VulkanParallaxmappingRenderer::DrawPlane(GPUContext* context, const Viewport& viewport, float aspect)
{
    if (_vertexBuffer == nullptr || _indexBuffer == nullptr || _indexCount <= 0)
        return;

    UpdateUniforms(context, aspect);
    context->SetViewportAndScissors(viewport);
    context->SetState(_pipeline);

    Span<GPUBuffer*> bindVB(&_vertexBuffer, 1);
    context->BindVB(bindVB, nullptr, _vertexLayout);
    context->BindIB(_indexBuffer);
    context->DrawIndexed(_indexCount);
}

void VulkanParallaxmappingRenderer::OnMainRender(RenderTask* task, GPUContext* context)
{
    SceneRenderTask* sceneTask = static_cast<SceneRenderTask*>(task);
    GPUTextureView* output = sceneTask->GetOutputView();
    Viewport viewport = sceneTask->GetViewport();
    const float aspect = viewport.GetAspectRatio() > 0.0f ? viewport.GetAspectRatio() : 1.0f;

    BindAndClearOutput(sceneTask, context, output, Color(0.025f, 0.025f, 0.025f, 1.0f));
    DrawPlane(context, viewport, aspect);
}
