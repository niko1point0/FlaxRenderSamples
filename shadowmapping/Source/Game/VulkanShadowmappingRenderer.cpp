// Port of Sascha Willems Vulkan examples/shadowmapping to Flax.
//
// Mapping to shadowmapping.cpp:
//   loadAssets()           -> OnStart() (Content::Load of the two scene models)
//   updateLight()          -> UpdateLight()
//   updateUniformBuffers() -> UpdateUniformBuffers()
//   buildCommandBuffer()   -> OnMainRender() (offscreen depth pass, then scene/debug pass)
//   scenes[i].draw(cmd)    -> DrawScene() (one MeshBase::Render per mesh = vkglTF primitive draw)
//
// Geometry is left exactly as imported: DrawScene binds each mesh's native Flax vertex/index
// buffers (positions in VB0, packed UV/normal/tangent in VB1, color in VB2) via MeshBase::Render,
// and the shader declares that same layout (see Shadowmapping.shader META_VS_IN_ELEMENT, copied
// from the engine's VertexColors.shader). No vertices are modified on the CPU.
//
// Matrix math matches the working parallaxmapping/texture ports: SaschaGpu::StoreVulkan upload +
// column-vector multiply in the shader, with SaschaCamera + PerspectiveForFlax. The model matrix
// is scale(1,-1,-1): Flax imports glTF with aiProcess_ConvertToLeftHanded (negates Z) whereas the
// Vulkan sample loads it with FlipY (negates Y); negating Y and Z makes the transformed geometry
// match what the Vulkan sample feeds to the identical camera/projection.

#include "VulkanShadowmappingRenderer.h"
#include "SaschaGpu.h"
#include "SaschaGlm.h"
#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Core/Math/Vector4.h"
#include "Engine/Content/Assets/Model.h"
#include "Engine/Content/Assets/Shader.h"
#include "Engine/Content/Content.h"
#include "Engine/Core/Math/Color.h"
#include "Engine/Core/Math/Viewport.h"
#include "Engine/Engine/Globals.h"
#include "Engine/Engine/Time.h"
#include "Engine/Graphics/GPUContext.h"
#include "Engine/Graphics/GPUDevice.h"
#include "Engine/Graphics/GPUPipelineState.h"
#include "Engine/Graphics/Models/MeshBase.h"
#include "Engine/Graphics/RenderBuffers.h"
#include "Engine/Graphics/RenderTask.h"
#include "Engine/Graphics/Shaders/GPUShader.h"
#include "Engine/Graphics/Textures/GPUTexture.h"
#include "Engine/Graphics/Textures/GPUTextureDescription.h"
#include "Engine/Input/Enums.h"
#include "Engine/Input/Input.h"
#include "Engine/Platform/Platform.h"
#include "Engine/Streaming/Streaming.h"
#include "Engine/Threading/Task.h"

namespace
{
    void WaitForModelMeshes(Model* model)
    {
        if (model == nullptr)
            return;

        if (model->WaitForLoaded())
            return;

        const double deadline = Platform::GetTimeSeconds() + 30.0;
        while (!model->CanBeRendered())
        {
            if (Platform::GetTimeSeconds() > deadline)
                break;

            if (!model->CanBeUpdated())
            {
                Platform::Yield();
                continue;
            }

            const int32 targetResidency = model->GetLODsCount();
            const int32 currentResidency = model->GetCurrentResidency();
            if (currentResidency >= targetResidency)
                break;

            Task* task = model->CreateStreamingTask(currentResidency + 1);
            if (task == nullptr)
            {
                model->RequestStreamingUpdate();
                Streaming::RequestStreamingUpdate();
                Platform::Yield();
                continue;
            }

            task->Start();
            task->Wait(30000.0);
        }
    }

    // struct UniformDataScene / UniformDataOffscreen from shadowmapping.cpp, packed into one CB.
    GPU_CB_STRUCT(UniformData {
        Matrix ProjectionMatrix; // uniformDataScene.projection
        Matrix ViewMatrix;       // uniformDataScene.view
        Matrix ModelMatrix;      // uniformDataScene.model
        Matrix LightSpace;       // uniformDataScene.depthBiasMVP
        Matrix DepthMvp;         // uniformDataOffscreen.depthMVP
        Float4 LightPos;         // uniformDataScene.lightPos
        float ZNear;
        float ZFar;
        int32 EnablePCF;
    });

    GPUPipelineState* CreatePipeline(GPUShader* shader, const char* vs, const char* ps, CullMode cull)
    {
    GPUPipelineState* pipeline = GPUDevice::Instance->CreatePipelineState();
    GPUPipelineState::Description desc = GPUPipelineState::Description::Default;
    desc.CullMode = cull;
    desc.DepthWriteEnable = true;
    // This is a standard-Z port (glm/Vulkan 0..1 depth, cleared to 1.0). Pin the comparison to Less so
    // the project behaves identically whether or not the engine is built with reversed-Z: Flax v1.13
    // makes Description::Default.DepthFunc resolve to Greater under REVERSE_Z, which would invert depth
    // sorting against this sample's standard-Z geometry.
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
        desc.VS = shader->GetVS("VS_Quad");
        desc.PS = shader->GetPS("PS_Quad");
        pipeline->Init(desc);
        return pipeline;
    }

    void BindAndClearOutput(SceneRenderTask* sceneTask, GPUContext* context, GPUTextureView* output, const Color& clearColor)
    {
        GPUTextureView* depth = (sceneTask->Buffers && sceneTask->Buffers->DepthBuffer)
            ? sceneTask->Buffers->DepthBuffer->View()
            : nullptr;

        if (depth)
        {
            context->SetRenderTarget(depth, output);
            context->Clear(output, clearColor);
            // Standard-Z: clear to the far value 1.0 explicitly. The engine default is
            // GPU_DEPTH_RANGE_MAX, which becomes 0.0 under REVERSE_Z and would break this sample.
            context->ClearDepth(depth, 1.0f);
        }
        else
        {
            context->Clear(output, clearColor);
            context->SetRenderTarget(output);
        }
    }
}

VulkanShadowmappingRenderer::VulkanShadowmappingRenderer(const SpawnParams& params)
    : Script(params)
{
    _tickUpdate = true;
}

void VulkanShadowmappingRenderer::DrawScene(GPUContext* context) const
{
    // vkglTF::Model::draw: bind the model geometry and issue one indexed draw per primitive.
    Model* model = _sceneModels[_sceneIndex];
    if (model == nullptr || !model->CanBeRendered())
        return;

    Array<MeshBase*> meshes;
    model->GetMeshes(meshes, 0);
    for (int32 i = 0; i < meshes.Count(); i++)
    {
        MeshBase* mesh = meshes[i];
        if (mesh && mesh->IsInitialized())
            mesh->Render(context);
    }
}

void VulkanShadowmappingRenderer::OnStart()
{
    // shadowmapping.cpp constructor camera setup.
    _camera.setPosition(glm::vec3(0.0f, 0.0f, -12.5f));
    _camera.setRotation(glm::vec3(-25.0f, -390.0f, 0.0f));
    _camera.setPerspective(60.0f, 1.0f, _zNear, _zFar);
    _camera.rotationSpeed *= 0.5f;

    // loadAssets(): scenes[0] = vulkanscene_shadow, scenes[1] = samplescene.
    _sceneModels[0] = Content::Load<Model>(Globals::ProjectContentFolder / TEXT("Models/vulkanscene_shadow.flax"));
    WaitForModelMeshes(_sceneModels[0]);

    _sceneModels[1] = Content::Load<Model>(Globals::ProjectContentFolder / TEXT("Models/samplescene.flax"));
    WaitForModelMeshes(_sceneModels[1]);

    Shader* shader = Content::Load<Shader>(Globals::ProjectContentFolder / TEXT("Shaders/Shadowmapping.flax"));
    if (shader == nullptr)
        return;

    GPUShader* gpuShader = shader->GetShader();
    if (gpuShader == nullptr)
        return;

    _constantBuffer = gpuShader->GetCB(0);
    if (_constantBuffer == nullptr)
        return;

    // pipelines.offscreen / pipelines.sceneShadow(PCF) / pipelines.debug.
    // Offscreen disables culling so all faces contribute to the shadow map (matches Vulkan).
    _offscreenPipeline = CreatePipeline(gpuShader, "VS_Offscreen", "PS_Offscreen", CullMode::TwoSided);
    _scenePipeline = CreatePipeline(gpuShader, "VS_Scene", "PS_Scene", CullMode::Normal);
    _debugPipeline = CreateFullscreenPipeline(gpuShader);

    CreateShadowMap();

    MainRenderTask::Instance->IsCustomRendering = true;
    MainRenderTask::Instance->Render.Bind<VulkanShadowmappingRenderer, &VulkanShadowmappingRenderer::OnMainRender>(this);
}

void VulkanShadowmappingRenderer::OnUpdate()
{
    _camera.HandleMouseInput();
    _timer += Time::GetDeltaTime() * 0.5f; // timerSpeed *= 0.5f

    if (Input::GetKeyDown(KeyboardKeys::Alpha1))
        _sceneIndex = 0;
    if (Input::GetKeyDown(KeyboardKeys::Alpha2))
        _sceneIndex = 1;
    if (Input::GetKeyDown(KeyboardKeys::M))
        _displayShadowMap = !_displayShadowMap;
    if (Input::GetKeyDown(KeyboardKeys::P))
        _filterPCF = !_filterPCF;
}

void VulkanShadowmappingRenderer::OnDestroy()
{
    MainRenderTask::Instance->Render.Unbind<VulkanShadowmappingRenderer, &VulkanShadowmappingRenderer::OnMainRender>(this);
    MainRenderTask::Instance->IsCustomRendering = false;

    DestroyShadowMap();
    SAFE_DELETE_GPU_RESOURCE(_offscreenPipeline);
    SAFE_DELETE_GPU_RESOURCE(_scenePipeline);
    SAFE_DELETE_GPU_RESOURCE(_debugPipeline);
    _constantBuffer = nullptr;

    for (int32 i = 0; i < SceneCount; i++)
        _sceneModels[i] = nullptr;
}

void VulkanShadowmappingRenderer::CreateShadowMap()
{
    _shadowMap = GPUDevice::Instance->CreateTexture(TEXT("ShadowmappingShadowMap"));
    auto shadowDesc = GPUTextureDescription::New2D(ShadowDim, ShadowDim, PixelFormat::D32_Float,
        GPUTextureFlags::ShaderResource | GPUTextureFlags::DepthStencil);
    _shadowMap->Init(shadowDesc);
    _shadowMapView = _shadowMap->View();
}

void VulkanShadowmappingRenderer::DestroyShadowMap()
{
    _shadowMapView = nullptr;
    SAFE_DELETE_GPU_RESOURCE(_shadowMap);
}

void VulkanShadowmappingRenderer::UpdateLight()
{
    // shadowmapping.cpp updateLight()
    _lightPos.x = glm::cos(glm::radians(_timer * 360.0f)) * 40.0f;
    _lightPos.y = -50.0f + glm::sin(glm::radians(_timer * 360.0f)) * 20.0f;
    _lightPos.z = 25.0f + glm::sin(glm::radians(_timer * 360.0f)) * 5.0f;
}

void VulkanShadowmappingRenderer::UpdateUniformBuffers(GPUContext* context, float aspect)
{
    if (_constantBuffer == nullptr)
        return;

    _camera.syncProjection(aspect);

    // Offscreen: matrix from the light's point of view (shadowmapping.cpp updateUniformBuffers).
    const glm::mat4 depthProjectionMatrix = glm::perspective(glm::radians(_lightFov), 1.0f, _zNear, _zFar);
    const glm::mat4 depthViewMatrix = glm::lookAt(_lightPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 depthModelMatrix = glm::mat4(1.0f);
    const glm::mat4 depthMVP = depthProjectionMatrix * depthViewMatrix * depthModelMatrix;

    // Scene uniforms. Vulkan uses model = identity (its glTF is FlipY-loaded); Flax imports with a
    // Z negation instead, so the model matrix negates Y and Z to reproduce the same geometry.
    const glm::mat4& projection = _camera.matrices.perspective;
    const glm::mat4& view = _camera.matrices.view;
    // Flax imports the glTF in centimeters (x100) and bakes the per-node x3 scale, so the model
    // spans ~+/-750 vs Sascha's ~+/-7.5. Scale by 0.01 to match the Vulkan sample's units (the
    // camera at z=-12.5 and light near/far 1..96 are authored for that scale). Y and Z are negated
    // because Flax imports with aiProcess_ConvertToLeftHanded (negated Z) while Vulkan loads FlipY.
    const glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(0.01f, -0.01f, -0.01f));

    UniformData data;
    SaschaGpu::StoreVulkan(data.ProjectionMatrix, projection);
    SaschaGpu::StoreVulkan(data.ViewMatrix, view);
    SaschaGpu::StoreVulkan(data.ModelMatrix, model);
    SaschaGpu::StoreVulkan(data.LightSpace, depthMVP); // depthBiasMVP
    SaschaGpu::StoreVulkan(data.DepthMvp, depthMVP);
    data.LightPos = Float4(_lightPos.x, _lightPos.y, _lightPos.z, 1.0f);
    data.ZNear = _zNear;
    data.ZFar = _zFar;
    data.EnablePCF = _filterPCF ? 1 : 0;
    context->UpdateCB(_constantBuffer, &data);
}

void VulkanShadowmappingRenderer::RenderShadowPass(GPUContext* context)
{
    if (_constantBuffer == nullptr || _offscreenPipeline == nullptr)
        return;

    Viewport shadowVp;
    shadowVp.X = shadowVp.Y = 0;
    shadowVp.Width = (float)ShadowDim;
    shadowVp.Height = (float)ShadowDim;
    shadowVp.MinDepth = 0.0f;
    shadowVp.MaxDepth = 1.0f;

    // Unbind the shadow map as a shader resource (the previous frame's scene/debug pass left it in
    // SR slot 0). Re-binding it as the depth-stencil target while it is still an SRV would otherwise
    // trip a read/write hazard on the next draw.
    context->UnBindSR(0);

    context->BindCB(0, _constantBuffer);
    context->SetState(_offscreenPipeline);
    context->SetRenderTarget(_shadowMapView, nullptr);
    // Standard-Z: clear the shadow map to the far value 1.0 explicitly (engine default flips to 0.0
    // under REVERSE_Z). The PCF lookup in the shader compares against this standard-Z depth.
    context->ClearDepth(_shadowMapView, 1.0f);
    context->SetViewportAndScissors(shadowVp);
    DrawScene(context);
}

void VulkanShadowmappingRenderer::RenderScenePass(GPUContext* context, const Viewport& viewport)
{
    if (_constantBuffer == nullptr || _scenePipeline == nullptr)
        return;

    context->SetViewportAndScissors(viewport);
    context->BindCB(0, _constantBuffer);
    if (_shadowMap)
        context->BindSR(0, _shadowMap);
    context->SetState(_scenePipeline);
    DrawScene(context);
}

void VulkanShadowmappingRenderer::RenderDebugPass(GPUContext* context, const Viewport& viewport)
{
    if (_constantBuffer == nullptr || _debugPipeline == nullptr)
        return;

    context->SetViewportAndScissors(viewport);
    context->BindCB(0, _constantBuffer);
    if (_shadowMap)
        context->BindSR(0, _shadowMap);
    context->SetState(_debugPipeline);
    context->DrawInstanced(3, 1);
}

void VulkanShadowmappingRenderer::OnMainRender(RenderTask* task, GPUContext* context)
{
    if (_constantBuffer == nullptr)
        return;

    SceneRenderTask* sceneTask = static_cast<SceneRenderTask*>(task);
    GPUTextureView* output = sceneTask->GetOutputView();
    Viewport viewport = sceneTask->GetViewport();
    const float aspect = viewport.GetAspectRatio() > 0.0f ? viewport.GetAspectRatio() : 1.0f;

    UpdateLight();
    UpdateUniformBuffers(context, aspect);

    // First render pass: generate shadow map (offscreen depth only).
    RenderShadowPass(context);

    // The DX11/DX12 contexts flush shader resources BEFORE render targets in onDrawCall, so the
    // shadow map would still be bound as the depth-stencil output when the scene pass binds it as an
    // SRV - the runtime then force-nulls the SRV and shadows read as empty (scene renders dark/wrong).
    // ResetRenderTarget() flushes the output-merger immediately, detaching the shadow map as a target
    // before it is sampled. This mirrors the Vulkan sample's subpass dependency that transitions the
    // depth attachment to SHADER_READ_ONLY between the offscreen and scene passes.
    context->ResetRenderTarget();

    // Second pass: scene as viewed by the camera, sampling the shadow map.
    // Matches the Vulkan sample's defaultClearColor (vulkanexamplebase.h): { 0.025, 0.025, 0.025, 1 }.
    BindAndClearOutput(sceneTask, context, output, Color(0.025f, 0.025f, 0.025f, 1.0f));

    if (_displayShadowMap)
        RenderDebugPass(context, viewport);
    else
        RenderScenePass(context, viewport);
}
