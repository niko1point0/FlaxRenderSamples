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
#include "Engine/Core/Log.h"
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
#include "Engine/Graphics/GPUAccelerationStructure.h"
#include "Engine/Graphics/GPUBuffer.h"
#include "Engine/Graphics/GPUContext.h"
#include "Engine/Graphics/GPUDevice.h"
#include "Engine/Graphics/GPULimits.h"
#include "Engine/Graphics/GPUPipelineState.h"
#include "Engine/Graphics/Enums.h"
#include "Engine/Graphics/Models/MeshBase.h"
#include "Engine/Graphics/Models/Mesh.h"
#include "Engine/Graphics/RenderBuffers.h"
#include "Engine/Graphics/RenderTask.h"
#include "Engine/Graphics/Shaders/GPUShader.h"
#include "Engine/Graphics/Shaders/GPUVertexLayout.h"
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

    GPUPipelineState* CreateRayTracingPipeline(GPUShader* shader)
    {
        // Ray-traced scene pass: same geometry/state as the legacy scene pipeline, but PS_SceneRT computes
        // shadows with an inline ray query. PS_SceneRT only exists on Shader Model 6.5 backends (DX12 / Vulkan).
        auto vs = shader->GetVS("VS_Scene");
        auto ps = shader->GetPS("PS_SceneRT");
        if (vs == nullptr || ps == nullptr)
            return nullptr;
        GPUPipelineState* pipeline = GPUDevice::Instance->CreatePipelineState();
        GPUPipelineState::Description desc = GPUPipelineState::Description::Default;
        desc.CullMode = CullMode::Normal;
        desc.DepthWriteEnable = true;
        desc.DepthFunc = ComparisonFunc::Less; // standard-Z, matches CreatePipeline()
        desc.VS = vs;
        desc.PS = ps;
        if (pipeline->Init(desc))
        {
            SAFE_DELETE_GPU_RESOURCE(pipeline);
            return nullptr;
        }
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

    // Ray tracing is only available on DirectX 12 and Vulkan (never DirectX 11) and only when the GPU supports it.
    const RendererType renderer = GPUDevice::Instance->GetRendererType();
    const bool backendSupportsRayTracing = renderer == RendererType::DirectX12 || renderer == RendererType::Vulkan;
    _rayTracingSupported = backendSupportsRayTracing && GPUDevice::Instance->Limits.HasRayTracing;
    if (_rayTracingSupported)
    {
        _rayTracingPipeline = CreateRayTracingPipeline(gpuShader);
        _accelerationStructure = GPUDevice::Instance->CreateAccelerationStructure(TEXT("ShadowmappingRT"));
        if (_rayTracingPipeline == nullptr || _accelerationStructure == nullptr)
        {
            // Ray tracing shader/structure unavailable - keep it disabled
            _rayTracingSupported = false;
        }
        else
        {
            LOG(Info, "Ray tracing is supported. Press 'R' to toggle ray-traced shadows (the background turns green while ray tracing is active).");
        }
    }

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

    // 'R' toggles hardware ray tracing (only meaningful on DirectX 12 / Vulkan with a ray tracing capable GPU).
    if (Input::GetKeyDown(KeyboardKeys::R) && _rayTracingSupported)
    {
        _rayTracingEnabled = !_rayTracingEnabled;
        LOG(Info, "Ray tracing {0}.", _rayTracingEnabled ? TEXT("enabled") : TEXT("disabled"));
    }
}

void VulkanShadowmappingRenderer::OnDestroy()
{
    MainRenderTask::Instance->Render.Unbind<VulkanShadowmappingRenderer, &VulkanShadowmappingRenderer::OnMainRender>(this);
    MainRenderTask::Instance->IsCustomRendering = false;

    DestroyShadowMap();
    SAFE_DELETE_GPU_RESOURCE(_offscreenPipeline);
    SAFE_DELETE_GPU_RESOURCE(_scenePipeline);
    SAFE_DELETE_GPU_RESOURCE(_debugPipeline);
    SAFE_DELETE_GPU_RESOURCE(_rayTracingPipeline);
    if (_accelerationStructure != nullptr)
    {
        _accelerationStructure->DeleteObjectNow();
        _accelerationStructure = nullptr;
    }
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

void VulkanShadowmappingRenderer::EnsureAccelerationStructure(GPUContext* context)
{
    if (!_rayTracingSupported || _accelerationStructure == nullptr)
        return;
    // Build (or rebuild when the active scene changes) the acceleration structure from the scene mesh geometry
    if (_accelerationStructureScene == _sceneIndex)
        return;

    Model* model = _sceneModels[_sceneIndex];
    if (model == nullptr || !model->CanBeRendered())
        return;
    Array<MeshBase*> meshes;
    model->GetMeshes(meshes, 0);
    if (meshes.Count() == 0)
        return;
    MeshBase* mesh = meshes[0];
    if (mesh == nullptr || !mesh->IsInitialized() || mesh->GetIndexBuffer() == nullptr || mesh->GetVertexBuffer(0) == nullptr)
        return;

    GPUBuffer* vb0 = mesh->GetVertexBuffer(0);
    GPUBuffer* ib = mesh->GetIndexBuffer();

    // Read the actual position element layout. Flax 1.13 stores positions compressed (e.g. R16G16B16A16_Float,
    // 8-byte stride) rather than tightly-packed Float3, so the BLAS must use the real stride, byte offset and
    // pixel format - otherwise it reads scrambled (but in-bounds) triangles and the ray queries return garbage.
    VertexElement posElement;
    posElement.Format = PixelFormat::R32G32B32_Float;
    posElement.Offset = 0;
    if (GPUVertexLayout* layout = vb0->GetVertexLayout())
        posElement = layout->FindElement(VertexElement::Types::Position);

    GPUAccelerationStructureGeometry geometry;
    geometry.VertexBuffer = vb0;
    geometry.VertexBufferOffset = posElement.Offset;
    geometry.VertexStride = vb0->GetStride();
    geometry.VertexFormat = posElement.Format;
    geometry.VertexCount = (uint32)mesh->GetVertexCount();
    geometry.IndexBuffer = ib;
    geometry.IndexCount = (uint32)mesh->GetTriangleCount() * 3;
    // Same model transform used by the raster scene pass (Flax import scale + Y/Z negation).
    geometry.Transform = Matrix::Scaling(0.01f, -0.01f, -0.01f);

    LOG(Info, "[RT] Building AS: vertices={0}, vb0 stride={1}, pos offset={2}, pos format={3}, triangles={4}, ib stride={5}",
        geometry.VertexCount, vb0->GetStride(), (uint32)posElement.Offset, (int32)posElement.Format, (uint32)mesh->GetTriangleCount(), ib->GetStride());

    _accelerationStructure->SetGeometry(geometry);
    context->BuildAccelerationStructure(_accelerationStructure);
    _accelerationStructureScene = _sceneIndex;
}

void VulkanShadowmappingRenderer::RenderRayTracePass(GPUContext* context, const Viewport& viewport)
{
    if (_constantBuffer == nullptr || _rayTracingPipeline == nullptr || _accelerationStructure == nullptr || !_accelerationStructure->IsBuilt())
        return;

    // Ray-traced scene draw: same geometry as the legacy scene pass, but the shader resolves shadows via a
    // ray query against the TLAS instead of sampling the offscreen shadow map (no shadow pass needed).
    context->UnBindSR(0); // PS_SceneRT does not use the shadow map
    context->SetViewportAndScissors(viewport);
    context->BindCB(0, _constantBuffer);
    context->BindSR(1, _accelerationStructure->GetView()); // RaytracingAccelerationStructure SceneAS : register(t1)
    context->SetState(_rayTracingPipeline);
    DrawScene(context);
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

    // Ray tracing is an additional, opt-in feature (toggled with 'R', DirectX 12 / Vulkan only). When it is
    // active the shadows are resolved with ray queries in the scene pass, so the offscreen shadow-map pass is
    // skipped entirely - the legacy depth/shadow-map path below is preserved for everything else.
    bool rayTracingActive = false;
    if (_rayTracingEnabled && _accelerationStructure != nullptr)
    {
        EnsureAccelerationStructure(context);
        rayTracingActive = _accelerationStructure->IsBuilt();
    }

    if (!rayTracingActive)
    {
        // First render pass: generate shadow map (offscreen depth only).
        RenderShadowPass(context);

        // The DX11/DX12 contexts flush shader resources BEFORE render targets in onDrawCall, so the
        // shadow map would still be bound as the depth-stencil output when the scene pass binds it as an
        // SRV - the runtime then force-nulls the SRV and shadows read as empty (scene renders dark/wrong).
        // ResetRenderTarget() flushes the output-merger immediately, detaching the shadow map as a target
        // before it is sampled. This mirrors the Vulkan sample's subpass dependency that transitions the
        // depth attachment to SHADER_READ_ONLY between the offscreen and scene passes.
        context->ResetRenderTarget();
    }

    // Scene as viewed by the camera. Legacy clear matches the Vulkan sample's defaultClearColor
    // { 0.025, 0.025, 0.025, 1 }; when ray tracing is active the background is cleared to green so it is
    // immediately obvious that the ray-traced path (not the shadow-map path) is running.
    const Color clearColor = rayTracingActive
        ? Color(0.0f, 1.0f, 0.0f, 1.0f)
        : Color(0.025f, 0.025f, 0.025f, 1.0f);
    BindAndClearOutput(sceneTask, context, output, clearColor);

    if (rayTracingActive)
        RenderRayTracePass(context, viewport);
    else if (_displayShadowMap)
        RenderDebugPass(context, viewport);
    else
        RenderScenePass(context, viewport);
}
