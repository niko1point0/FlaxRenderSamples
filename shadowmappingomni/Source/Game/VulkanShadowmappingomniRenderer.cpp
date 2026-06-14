// Port of Sascha Willems Vulkan examples/shadowmappingomni to Flax.

#include "VulkanShadowmappingomniRenderer.h"
#include "SaschaGpu.h"
#include "Engine/Content/Assets/Shader.h"
#include "Engine/Content/Content.h"
#include "Engine/Core/Math/Color.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Engine/Globals.h"
#include "Engine/Engine/Time.h"
#include "Engine/Graphics/GPUContext.h"
#include "Engine/Graphics/GPUDevice.h"
#include "Engine/Graphics/GPUPipelineState.h"
#include "Engine/Graphics/GPUResource.h"
#include "Engine/Graphics/RenderBuffers.h"
#include "Engine/Graphics/RenderTask.h"
#include "Engine/Graphics/Shaders/GPUShader.h"
#include "Engine/Graphics/Textures/GPUTexture.h"
#include "Engine/Graphics/Textures/GPUTextureDescription.h"
#include "Engine/Input/Enums.h"
#include "Engine/Input/Input.h"

namespace
{
    GPU_CB_STRUCT(OffscreenUniformData {
        Matrix ProjectionMatrix;
        Matrix ModelMatrix;
        Float4 LightPos;
    });

    GPU_CB_STRUCT(FaceViewData {
        Matrix ViewMatrix;
    });

    GPU_CB_STRUCT(SceneUniformData {
        Matrix ProjectionMatrix;
        Matrix ViewMatrix;
        Matrix ModelMatrix;
        Float4 LightPos;
    });

    GPUPipelineState* CreatePipeline(GPUShader* shader, const char* vs, const char* ps, CullMode cull, bool depthWrite = true)
    {
        GPUPipelineState* pipeline = GPUDevice::Instance->CreatePipelineState();
        GPUPipelineState::Description desc = GPUPipelineState::Description::Default;
        desc.CullMode = cull;
        desc.DepthWriteEnable = depthWrite;
        // Standard-Z port (glm/Vulkan 0..1 depth, cleared to 1.0). Pin the comparison to Less so the
        // sample behaves identically whether or not the engine is built with reversed-Z (Description::
        // Default.DepthFunc resolves to Greater under REVERSE_Z, which would invert depth sorting).
        desc.DepthFunc = ComparisonFunc::Less;
        desc.VS = shader->GetVS(vs);
        if (ps)
            desc.PS = shader->GetPS(ps);
        pipeline->Init(desc);
        return pipeline;
    }

    GPUPipelineState* CreateFullscreenPipeline(GPUShader* shader)
    {
        GPUPipelineState* pipeline = GPUDevice::Instance->CreatePipelineState();
        GPUPipelineState::Description desc = GPUPipelineState::Description::DefaultFullscreenTriangle;
        desc.CullMode = CullMode::Inverted;
        desc.DepthEnable = false;
        desc.DepthWriteEnable = false;
        desc.VS = shader->GetVS("VS_CubemapDisplay");
        desc.PS = shader->GetPS("PS_CubemapDisplay");
        pipeline->Init(desc);
        return pipeline;
    }

    glm::mat4 BuildCubeFaceView(int32 faceIndex)
    {
        glm::mat4 view = glm::mat4(1.0f);
        glm::mat4 rot, tmp;
        switch (faceIndex)
        {
        case 0: // POSITIVE_X
            rot = glm::rotate(glm::mat4(1.0f), 90.0f * DegreesToRadians, glm::vec3(0.0f, 1.0f, 0.0f));
            view = rot;
            rot = glm::rotate(glm::mat4(1.0f), 180.0f * DegreesToRadians, glm::vec3(1.0f, 0.0f, 0.0f));
            tmp = view * rot;
            view = tmp;
            break;
        case 1: // NEGATIVE_X
            rot = glm::rotate(glm::mat4(1.0f), -90.0f * DegreesToRadians, glm::vec3(0.0f, 1.0f, 0.0f));
            view = rot;
            rot = glm::rotate(glm::mat4(1.0f), 180.0f * DegreesToRadians, glm::vec3(1.0f, 0.0f, 0.0f));
            tmp = view * rot;
            view = tmp;
            break;
        case 2: // POSITIVE_Y
            view = glm::rotate(glm::mat4(1.0f), -90.0f * DegreesToRadians, glm::vec3(1.0f, 0.0f, 0.0f));
            break;
        case 3: // NEGATIVE_Y
            view = glm::rotate(glm::mat4(1.0f), 90.0f * DegreesToRadians, glm::vec3(1.0f, 0.0f, 0.0f));
            break;
        case 4: // POSITIVE_Z
            view = glm::rotate(glm::mat4(1.0f), 180.0f * DegreesToRadians, glm::vec3(1.0f, 0.0f, 0.0f));
            break;
        case 5: // NEGATIVE_Z
            view = glm::rotate(glm::mat4(1.0f), 180.0f * DegreesToRadians, glm::vec3(0.0f, 0.0f, 1.0f));
            break;
        default:
            break;
        }
        return view;
    }
}

VulkanShadowmappingomniRenderer::VulkanShadowmappingomniRenderer(const SpawnParams& params)
    : Script(params)
{
    _tickUpdate = true;
}

void VulkanShadowmappingomniRenderer::OnStart()
{
    _camera.setPosition(glm::vec3(0.0f, 0.5f, -15.0f));
    _camera.setRotation(glm::vec3(-20.5f, -673.0f, 0.0f));
    _camera.setPerspective(45.0f, 1.0f, _zNear, _zFar);
    _camera.rotationSpeed *= 0.5f;

    _scene.Load(TEXT("Models/shadowscene_fire.flax"));

    Shader* shader = Content::Load<Shader>(Globals::ProjectContentFolder / TEXT("Shaders/Shadowmappingomni.flax"));
    GPUShader* gpuShader = shader->GetShader();

    _offscreenCb = gpuShader->GetCB(0);
    _faceViewCb = gpuShader->GetCB(1);
    _sceneCb = gpuShader->GetCB(2);

    _offscreenPipeline = CreatePipeline(gpuShader, "VS_Offscreen", "PS_Offscreen", CullMode::TwoSided);
    _scenePipeline = CreatePipeline(gpuShader, "VS_Scene", "PS_Scene", CullMode::Normal);
    _cubemapDisplayPipeline = CreateFullscreenPipeline(gpuShader);

    CreateShadowTargets();

    MainRenderTask::Instance->IsCustomRendering = true;
    MainRenderTask::Instance->Render.Bind<VulkanShadowmappingomniRenderer, &VulkanShadowmappingomniRenderer::OnMainRender>(this);
}

void VulkanShadowmappingomniRenderer::OnUpdate()
{
    _camera.HandleMouseInput();
    _timer += Time::GetDeltaTime() * 0.5f;

    if (Input::GetKeyDown(KeyboardKeys::M))
        _displayCubeMap = !_displayCubeMap;
}

void VulkanShadowmappingomniRenderer::OnDestroy()
{
    MainRenderTask::Instance->Render.Unbind<VulkanShadowmappingomniRenderer, &VulkanShadowmappingomniRenderer::OnMainRender>(this);
    MainRenderTask::Instance->IsCustomRendering = false;

    DestroyShadowTargets();
    SAFE_DELETE_GPU_RESOURCE(_offscreenPipeline);
    SAFE_DELETE_GPU_RESOURCE(_scenePipeline);
    SAFE_DELETE_GPU_RESOURCE(_cubemapDisplayPipeline);
    _offscreenCb = nullptr;
    _faceViewCb = nullptr;
    _sceneCb = nullptr;
}

void VulkanShadowmappingomniRenderer::CreateShadowTargets()
{
    _shadowCubeMap = GPUDevice::Instance->CreateTexture(TEXT("ShadowmappingomniCubeShadow"));
    auto cubeDesc = GPUTextureDescription::NewCube(ShadowDim, PixelFormat::R32_Float,
        GPUTextureFlags::ShaderResource | GPUTextureFlags::RenderTarget);
    _shadowCubeMap->Init(cubeDesc);

    _offscreenDepth = GPUDevice::Instance->CreateTexture(TEXT("ShadowmappingomniOffscreenDepth"));
    auto depthDesc = GPUTextureDescription::New2D(ShadowDim, ShadowDim, PixelFormat::D32_Float, GPUTextureFlags::DepthStencil);
    _offscreenDepth->Init(depthDesc);
    _offscreenDepthView = _offscreenDepth->View();
}

void VulkanShadowmappingomniRenderer::DestroyShadowTargets()
{
    _offscreenDepthView = nullptr;
    SAFE_DELETE_GPU_RESOURCE(_offscreenDepth);
    SAFE_DELETE_GPU_RESOURCE(_shadowCubeMap);
}

void VulkanShadowmappingomniRenderer::UpdateLight()
{
    const float timer = _timer;
    const float spin = timer * 360.0f * DegreesToRadians;
    _lightPos.X = Math::Sin(spin) * 0.15f;
    _lightPos.Y = -2.5f;
    _lightPos.Z = Math::Cos(spin) * 0.15f;
}

void VulkanShadowmappingomniRenderer::UpdateOffscreenUniforms(GPUContext* context)
{
    // shadowmappingomni.cpp updateUniformBuffers(): offscreen projection is a 90deg cube-face frustum.
    // The per-face rotation and the -lightPos translate are supplied separately via FaceView. The model
    // matrix carries the Flax-import correction (Flax bakes the glTF at x100 cm with Y/Z negated, so this
    // is the inverse of that import transform) - identical to the working shadowmapping port - which puts
    // the geometry back into the original Vulkan-authored space the camera/light are defined in.
    const glm::mat4 projection = SaschaCamera::Perspective(90.0f, 1.0f, _zNear, _zFar);
    const glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(0.01f, -0.01f, -0.01f));

    OffscreenUniformData data;
    SaschaGpu::Store(data.ProjectionMatrix, projection);
    SaschaGpu::Store(data.ModelMatrix, model);
    data.LightPos = Float4(_lightPos, 1.0f);
    context->UpdateCB(_offscreenCb, &data);
}

void VulkanShadowmappingomniRenderer::UpdateSceneUniforms(GPUContext* context, float aspect)
{
    _camera.syncProjection(aspect);

    const glm::mat4& projection = _camera.matrices.perspective;
    const glm::mat4 view = _camera.matrices.view;
    // Vulkan uses model = identity; Flax bakes the glTF at x100 cm with Y/Z negated, so undo that import
    // transform here (matching the working shadowmapping port) to render in the original-authored space.
    const glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(0.01f, -0.01f, -0.01f));

    SceneUniformData data;
    SaschaGpu::Store(data.ProjectionMatrix, projection);
    SaschaGpu::Store(data.ViewMatrix, view);
    SaschaGpu::Store(data.ModelMatrix, model);
    data.LightPos = Float4(_lightPos, 1.0f);
    context->UpdateCB(_sceneCb, &data);
}

void VulkanShadowmappingomniRenderer::RenderShadowCubeFaces(GPUContext* context)
{
    Viewport shadowVp;
    shadowVp.X = shadowVp.Y = 0;
    shadowVp.Width = (float)ShadowDim;
    shadowVp.Height = (float)ShadowDim;
    shadowVp.MinDepth = 0.0f;
    shadowVp.MaxDepth = 1.0f;

    // The cube map is left bound as a shader resource (SR slot 0) by the previous frame's scene/debug
    // pass. Re-binding its faces as render targets while it is still an SRV trips a read/write hazard,
    // so detach it first - matching shadowmapping.cpp's RenderShadowPass.
    context->UnBindSR(0);

    context->BindCB(0, _offscreenCb);
    context->SetState(_offscreenPipeline);
    context->SetViewportAndScissors(shadowVp);

    for (int32 face = 0; face < 6; face++)
    {
        GPUTextureView* faceView = _shadowCubeMap->View(face);
        context->SetRenderTarget(_offscreenDepthView, faceView);
        context->Clear(faceView, Color::Black);
        context->ClearDepth(_offscreenDepthView, 1.0f);

        FaceViewData faceData;
        // offscreen.vert: gl_Position = projection * faceRotation * translate(-lightPos) * model * pos.
        // model already maps the vertex into corrected world space, so the light translate must be applied
        // there too (after the model) - fold it into the per-face view as faceRotation * translate(-lightPos).
        const glm::mat4 lightTranslate = glm::translate(glm::mat4(1.0f), glm::vec3(-_lightPos.X, -_lightPos.Y, -_lightPos.Z));
        const glm::mat4 faceViewMatrix = BuildCubeFaceView(face) * lightTranslate;
        SaschaGpu::Store(faceData.ViewMatrix, faceViewMatrix);
        context->UpdateCB(_faceViewCb, &faceData);
        context->BindCB(1, _faceViewCb);

        _scene.Draw(context);
    }
}

void VulkanShadowmappingomniRenderer::RenderScenePass(GPUContext* context, GPUTextureView* depth, GPUTextureView* output)
{
    if (depth)
    {
        context->SetRenderTarget(depth, output);
        context->Clear(output, Color(0.025f, 0.025f, 0.025f, 1.0f));
        context->ClearDepth(depth, 1.0f);
    }
    else
    {
        context->SetRenderTarget(output);
        context->Clear(output, Color(0.025f, 0.025f, 0.025f, 1.0f));
    }

    context->BindCB(2, _sceneCb);
    if (_shadowCubeMap)
        // GPUTexture::View() returns the first face's 2D view; the shader samples a TextureCube, so bind
        // the whole-cube view (ViewArray) instead - otherwise the cubemap lookup reads a single 2D face.
        context->BindSR(0, _shadowCubeMap->ViewArray());
    context->SetState(_scenePipeline);
    _scene.Draw(context);
}

void VulkanShadowmappingomniRenderer::RenderCubemapDebugPass(GPUContext* context, GPUTextureView* depth, GPUTextureView* output)
{
    if (depth)
    {
        context->SetRenderTarget(depth, output);
        context->Clear(output, Color(0.025f, 0.025f, 0.025f, 1.0f));
        context->ClearDepth(depth, 1.0f);
    }
    else
    {
        context->SetRenderTarget(output);
        context->Clear(output, Color(0.025f, 0.025f, 0.025f, 1.0f));
    }

    context->BindCB(2, _sceneCb);
    if (_shadowCubeMap)
        context->BindSR(0, _shadowCubeMap->ViewArray());
    context->SetState(_cubemapDisplayPipeline);
    context->DrawInstanced(3, 1);
}

void VulkanShadowmappingomniRenderer::OnMainRender(RenderTask* task, GPUContext* context)
{
    SceneRenderTask* sceneTask = static_cast<SceneRenderTask*>(task);
    GPUTextureView* output = sceneTask->GetOutputView();
    Viewport viewport = sceneTask->GetViewport();
    const float aspect = viewport.GetAspectRatio() > 0.0f ? viewport.GetAspectRatio() : 1.0f;

    UpdateLight();
    UpdateOffscreenUniforms(context);
    UpdateSceneUniforms(context, aspect);

    RenderShadowCubeFaces(context);

    // The DX11/DX12 contexts flush shader resources before render targets, so the cube map faces would
    // still be bound as color targets when the scene pass binds the cube map as an SRV - the runtime
    // then force-nulls the SRV and the shadows read as empty. ResetRenderTarget() detaches the output
    // merger first, mirroring the Vulkan sample's subpass dependency that transitions the cube map to
    // SHADER_READ_ONLY between the offscreen and scene passes.
    context->ResetRenderTarget();
    context->SetViewportAndScissors(viewport);

    GPUTextureView* depth = (sceneTask->Buffers && sceneTask->Buffers->DepthBuffer)
        ? sceneTask->Buffers->DepthBuffer->View()
        : nullptr;

    if (_displayCubeMap)
        RenderCubemapDebugPass(context, depth, output);
    else
        RenderScenePass(context, depth, output);
}
