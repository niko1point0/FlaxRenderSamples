// Port of Sascha Willems Vulkan examples/shadowmappingomni to Flax.

#include "VulkanShadowmappingomniRenderer.h"
#include "SaschaGpu.h"
#include "Engine/Content/Assets/Model.h"
#include "Engine/Content/Assets/Shader.h"
#include "Engine/Content/Content.h"
#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Log.h"
#include "Engine/Core/Math/Color.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Core/Types/Span.h"
#include "Engine/Engine/Globals.h"
#include "Engine/Engine/Time.h"
#include "Engine/Graphics/Enums.h"
#include "Engine/Graphics/GPUAccelerationStructure.h"
#include "Engine/Graphics/GPUBuffer.h"
#include "Engine/Graphics/GPUBufferDescription.h"
#include "Engine/Graphics/GPUContext.h"
#include "Engine/Graphics/GPUDevice.h"
#include "Engine/Graphics/GPULimits.h"
#include "Engine/Graphics/GPUPipelineState.h"
#include "Engine/Graphics/GPUResource.h"
#include "Engine/Graphics/Models/MeshAccessor.h"
#include "Engine/Graphics/Models/MeshBase.h"
#include "Engine/Graphics/Models/Types.h"
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

    GPUPipelineState* CreateRayTracingPipeline(GPUShader* shader)
    {
        // Ray-traced scene pass: same geometry/state as the legacy scene pipeline, but PS_SceneRT resolves the
        // omni shadow with an inline ray query. PS_SceneRT only exists on Shader Model 6.5 backends (DX12 / Vulkan),
        // so a null VS/PS here means ray tracing stays unavailable.
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

    // Ray tracing is only available on DirectX 12 and Vulkan (never DirectX 11) and only when the GPU supports it.
    const RendererType renderer = GPUDevice::Instance->GetRendererType();
    const bool backendSupportsRayTracing = renderer == RendererType::DirectX12 || renderer == RendererType::Vulkan;
    _rayTracingSupported = backendSupportsRayTracing && GPUDevice::Instance->Limits.HasRayTracing;
    if (_rayTracingSupported)
    {
        _rayTracingPipeline = CreateRayTracingPipeline(gpuShader);
        _accelerationStructure = GPUDevice::Instance->CreateAccelerationStructure(TEXT("ShadowmappingomniRT"));
        if (_rayTracingPipeline == nullptr || _accelerationStructure == nullptr)
        {
            _rayTracingSupported = false;
        }
        else
        {
            // The merged-geometry buffers are built lazily on the first ray-traced frame (during rendering),
            // not here: at OnStart the device is not rendering and the meshes may not be fully streamed in, so
            // uploading now would go through an async copy on a separate context and the acceleration-structure
            // build could read an empty vertex buffer.
            LOG(Info, "Ray tracing is supported. Press 'R' to toggle ray-traced omni shadows (the background turns green while ray tracing is active).");
        }
    }

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

    // 'R' toggles hardware ray tracing (only meaningful on DirectX 12 / Vulkan with a ray tracing capable GPU).
    if (Input::GetKeyDown(KeyboardKeys::R) && _rayTracingSupported)
    {
        _rayTracingEnabled = !_rayTracingEnabled;
        LOG(Info, "Ray tracing {0}.", _rayTracingEnabled ? TEXT("enabled") : TEXT("disabled"));
    }
}

void VulkanShadowmappingomniRenderer::OnDestroy()
{
    MainRenderTask::Instance->Render.Unbind<VulkanShadowmappingomniRenderer, &VulkanShadowmappingomniRenderer::OnMainRender>(this);
    MainRenderTask::Instance->IsCustomRendering = false;

    DestroyShadowTargets();
    SAFE_DELETE_GPU_RESOURCE(_offscreenPipeline);
    SAFE_DELETE_GPU_RESOURCE(_scenePipeline);
    SAFE_DELETE_GPU_RESOURCE(_cubemapDisplayPipeline);
    SAFE_DELETE_GPU_RESOURCE(_rayTracingPipeline);
    SAFE_DELETE_GPU_RESOURCE(_rtVertexBuffer);
    SAFE_DELETE_GPU_RESOURCE(_rtIndexBuffer);
    if (_accelerationStructure != nullptr)
    {
        _accelerationStructure->DeleteObjectNow();
        _accelerationStructure = nullptr;
    }
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

void VulkanShadowmappingomniRenderer::BuildRayTracingGeometry()
{
    // Merge every LOD0 mesh of the room into a single combined position/index buffer so the whole scene is
    // covered by one bottom-level acceleration structure (the engine's AS exposes a single geometry/instance).
    // Each Flax mesh owns its own vertex/index buffers, so they are read on the CPU (MeshAccessor decodes the
    // packed vertex layout to Float3), concatenated, and the indices are rebased onto the merged vertex array.
    Model* model = _scene.ModelAsset;
    if (model == nullptr || !model->CanBeRendered())
        return;

    Array<MeshBase*> meshes;
    model->GetMeshes(meshes, 0);
    if (meshes.Count() == 0)
        return;

    Array<Float3> positions;
    Array<uint32> indices;
    for (int32 m = 0; m < meshes.Count(); m++)
    {
        MeshBase* mesh = meshes[m];
        if (mesh == nullptr || !mesh->IsInitialized())
            continue;

        MeshAccessor accessor;
        MeshBufferType bufferTypes[2] = { MeshBufferType::Index, MeshBufferType::Vertex0 };
        if (accessor.LoadMesh(mesh, false, Span<MeshBufferType>(bufferTypes, 2)))
            continue;

        MeshAccessor::Stream positionStream = accessor.Position();
        MeshAccessor::Stream indexStream = accessor.Index();
        if (!positionStream.IsValid() || !indexStream.IsValid())
            continue;

        const int32 baseVertex = positions.Count();
        const int32 vertexCount = positionStream.GetCount();
        for (int32 i = 0; i < vertexCount; i++)
            positions.Add(positionStream.GetFloat3(i));

        const int32 indexCount = indexStream.GetCount();
        for (int32 i = 0; i < indexCount; i++)
            indices.Add((uint32)(baseVertex + indexStream.GetInt(i)));
    }

    if (positions.Count() == 0 || indices.Count() == 0)
        return;

    _rtVertexCount = positions.Count();
    _rtIndexCount = indices.Count();

    _rtVertexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("ShadowmappingomniRT.VB"));
    _rtVertexBuffer->Init(GPUBufferDescription::Buffer(_rtVertexCount * sizeof(Float3), GPUBufferFlags::VertexBuffer, PixelFormat::Unknown, positions.Get(), sizeof(Float3)));

    _rtIndexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("ShadowmappingomniRT.IB"));
    _rtIndexBuffer->Init(GPUBufferDescription::Buffer(_rtIndexCount * sizeof(uint32), GPUBufferFlags::IndexBuffer, PixelFormat::R32_UInt, indices.Get(), sizeof(uint32)));

    LOG(Info, "[RT] Merged {0} meshes for omni shadows: vertices={1}, triangles={2}", meshes.Count(), _rtVertexCount, _rtIndexCount / 3);
}

void VulkanShadowmappingomniRenderer::EnsureAccelerationStructure(GPUContext* context)
{
    if (!_rayTracingSupported || _accelerationStructure == nullptr)
        return;
    if (_accelerationStructure->IsBuilt())
        return;

    // Build the merged geometry now (first ray-traced frame). We are inside OnMainRender, so the device is
    // rendering on the main thread: GPUBuffer::Init uploads the vertex/index data immediately on this same
    // context, guaranteeing it is resident before BuildAccelerationStructure copies it below.
    if (_rtVertexBuffer == nullptr || _rtIndexBuffer == nullptr)
        BuildRayTracingGeometry();
    if (_rtVertexBuffer == nullptr || _rtIndexBuffer == nullptr)
        return;

    GPUAccelerationStructureGeometry geometry;
    geometry.VertexBuffer = _rtVertexBuffer;
    geometry.VertexBufferOffset = 0;
    geometry.VertexStride = sizeof(Float3);
    geometry.VertexFormat = PixelFormat::R32G32B32_Float;
    geometry.VertexCount = (uint32)_rtVertexCount;
    geometry.IndexBuffer = _rtIndexBuffer;
    geometry.IndexCount = (uint32)_rtIndexCount;
    // Same model transform the raster scene pass uses (Flax import scale + Y/Z negation) so ray origins in the
    // pixel shader (which work in this corrected world space) match the acceleration structure geometry.
    geometry.Transform = Matrix::Scaling(0.01f, -0.01f, -0.01f);

    _accelerationStructure->SetGeometry(geometry);
    context->BuildAccelerationStructure(_accelerationStructure);
}

void VulkanShadowmappingomniRenderer::RenderRayTracePass(GPUContext* context, GPUTextureView* depth, GPUTextureView* output)
{
    // Ray tracing is active: clear the background to green so it is immediately obvious the ray-traced path
    // (not the cube-shadow path) is running, then draw the scene with PS_SceneRT, which resolves the omni
    // shadow via a ray query against the TLAS instead of sampling the distance cube map.
    const Color clearColor(0.0f, 1.0f, 0.0f, 1.0f);
    if (depth)
    {
        context->SetRenderTarget(depth, output);
        context->Clear(output, clearColor);
        context->ClearDepth(depth, 1.0f);
    }
    else
    {
        context->SetRenderTarget(output);
        context->Clear(output, clearColor);
    }

    context->UnBindSR(0); // PS_SceneRT does not use the shadow cube map
    context->BindCB(2, _sceneCb);
    context->BindSR(1, _accelerationStructure->GetView()); // RaytracingAccelerationStructure SceneAS : register(t1)
    context->SetState(_rayTracingPipeline);
    _scene.Draw(context);
}

void VulkanShadowmappingomniRenderer::OnMainRender(RenderTask* task, GPUContext* context)
{
    SceneRenderTask* sceneTask = static_cast<SceneRenderTask*>(task);
    GPUTextureView* output = sceneTask->GetOutputView();
    Viewport viewport = sceneTask->GetViewport();
    const float aspect = viewport.GetAspectRatio() > 0.0f ? viewport.GetAspectRatio() : 1.0f;

    UpdateLight();
    UpdateSceneUniforms(context, aspect);

    // Ray tracing is an additional, opt-in feature (toggled with 'R', DirectX 12 / Vulkan only). When it is
    // active the omni shadow is resolved with ray queries in the scene pass, so the 6-face offscreen cube pass
    // is skipped entirely - the legacy cube-shadow path below is preserved for everything else.
    bool rayTracingActive = false;
    if (_rayTracingEnabled && _accelerationStructure != nullptr)
    {
        EnsureAccelerationStructure(context);
        rayTracingActive = _accelerationStructure->IsBuilt();
    }

    if (!rayTracingActive)
    {
        UpdateOffscreenUniforms(context);
        RenderShadowCubeFaces(context);

        // The DX11/DX12 contexts flush shader resources before render targets, so the cube map faces would
        // still be bound as color targets when the scene pass binds the cube map as an SRV - the runtime
        // then force-nulls the SRV and the shadows read as empty. ResetRenderTarget() detaches the output
        // merger first, mirroring the Vulkan sample's subpass dependency that transitions the cube map to
        // SHADER_READ_ONLY between the offscreen and scene passes.
        context->ResetRenderTarget();
    }

    context->SetViewportAndScissors(viewport);

    GPUTextureView* depth = (sceneTask->Buffers && sceneTask->Buffers->DepthBuffer)
        ? sceneTask->Buffers->DepthBuffer->View()
        : nullptr;

    if (rayTracingActive)
        RenderRayTracePass(context, depth, output);
    else if (_displayCubeMap)
        RenderCubemapDebugPass(context, depth, output);
    else
        RenderScenePass(context, depth, output);
}
