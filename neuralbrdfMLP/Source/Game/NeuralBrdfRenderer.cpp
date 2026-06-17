// Neural BRDF sample renderer implementation. See NeuralBrdfRenderer.h.

#include "NeuralBrdfRenderer.h"
#include "SaschaGpu.h"
#include "../Shaders/NeuralBrdfConfig.h"
#include "Engine/Content/Content.h"
#include "Engine/Content/Assets/Shader.h"
#include "Engine/Core/Log.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Core/Math/Color.h"
#include "Engine/Core/Collections/Array.h"
#include "Engine/Engine/Globals.h"
#include "Engine/Engine/Engine.h"
#include "Engine/Platform/Platform.h"
#include "Engine/Graphics/Config.h"
#include "Engine/Graphics/GPUBuffer.h"
#include "Engine/Graphics/GPUBufferDescription.h"
#include "Engine/Graphics/GPUContext.h"
#include "Engine/Graphics/GPUDevice.h"
#include "Engine/Graphics/GPUPipelineState.h"
#include "Engine/Graphics/RenderTask.h"
#include "Engine/Graphics/RenderBuffers.h"
#include "Engine/Graphics/Shaders/GPUShader.h"
#include "Engine/Graphics/Shaders/GPUVertexLayout.h"
#include "Engine/Graphics/Shaders/VertexElement.h"
#include "Engine/Graphics/Textures/GPUTexture.h"
#include "Engine/Graphics/Textures/GPUTextureDescription.h"
#include "Engine/Profiler/ProfilerGPU.h"

NeuralBrdfRenderer* NeuralBrdfRenderer::_instance = nullptr;

namespace
{
    struct NBVertex
    {
        Float3 Position;
        Float3 Normal;
    };

    GPU_CB_STRUCT(DrawData {
        Matrix ViewProject;
        Float4 CameraPos;
        Float4 LightDir;
        Float4 LightIntensity;
        Float4 BaseColor;
        float Specular;
        float Roughness;
        float Metallic;
        float DiffScale;
    });

    GPU_CB_STRUCT(TrainData {
        uint32 Seed0;
        uint32 Seed1;
        uint32 Step;
        uint32 ParamCount;
        float LearningRate;
        uint32 BatchSize;
        float Pad0;
        float Pad1;
    });
}

NeuralBrdfRenderer::NeuralBrdfRenderer(const SpawnParams& params)
    : Script(params)
{
    _tickUpdate = true;
}

NeuralBrdfRenderer* NeuralBrdfRenderer::GetInstance()
{
    return _instance;
}

void NeuralBrdfRenderer::BuildSphere()
{
    // UV sphere, radius 1. Positions on the unit sphere double as normals.
    const int32 stacks = 64, slices = 64;
    Array<NBVertex> vertices;
    Array<uint32> indices;
    for (int32 i = 0; i <= stacks; ++i)
    {
        const float v = (float)i / stacks;
        const float phi = v * PI; // 0..pi
        const float sinPhi = Math::Sin(phi), cosPhi = Math::Cos(phi);
        for (int32 j = 0; j <= slices; ++j)
        {
            const float u = (float)j / slices;
            const float theta = u * 2.0f * PI;
            const float sinTheta = Math::Sin(theta), cosTheta = Math::Cos(theta);
            Float3 n(sinPhi * cosTheta, cosPhi, sinPhi * sinTheta);
            NBVertex vert;
            vert.Position = n;
            vert.Normal = n;
            vertices.Add(vert);
        }
    }
    const int32 ringVerts = slices + 1;
    for (int32 i = 0; i < stacks; ++i)
    {
        for (int32 j = 0; j < slices; ++j)
        {
            const uint32 a = i * ringVerts + j;
            const uint32 b = (i + 1) * ringVerts + j;
            indices.Add(a); indices.Add(b); indices.Add(a + 1);
            indices.Add(a + 1); indices.Add(b); indices.Add(b + 1);
        }
    }
    _indexCount = indices.Count();

    static VertexElement elements[2] =
    {
        { VertexElement::Types::Position, 0, 0, 0, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::Normal, 0, 12, 0, PixelFormat::R32G32B32_Float },
    };
    GPUVertexLayout::Elements elementList(elements, 2);
    _layout = GPUVertexLayout::Get(elementList, true);

    _vb = GPUDevice::Instance->CreateBuffer(TEXT("NeuralBrdfVB"));
    _vb->Init(GPUBufferDescription::Vertex(_layout, sizeof(NBVertex), vertices.Count(), vertices.Get()));
    _ib = GPUDevice::Instance->CreateBuffer(TEXT("NeuralBrdfIB"));
    _ib->Init(GPUBufferDescription::Index(sizeof(uint32), indices.Count(), indices.Get()));
}

void NeuralBrdfRenderer::OnStart()
{
    _instance = this;

    // Load compiled shaders (imported by NeuralBrdfEditorPlugin.cs).
    _drawShaderAsset = Content::Load<Shader>(Globals::ProjectContentFolder / TEXT("Shaders/NeuralBrdf.flax"));
    _trainShaderAsset = Content::Load<Shader>(Globals::ProjectContentFolder / TEXT("Shaders/NeuralBrdfTraining.flax"));
    if (!_drawShaderAsset.Get() || !_trainShaderAsset.Get())
    {
        LOG(Error, "NeuralBrdf: failed to load shaders. Open the editor once so they get imported to Content/Shaders.");
        return;
    }
    GPUShader* drawShader = _drawShaderAsset.Get()->GetShader();
    GPUShader* trainShader = _trainShaderAsset.Get()->GetShader();

    BuildSphere();

    // Neural buffers.
    _params = GPUDevice::Instance->CreateBuffer(TEXT("NeuralBrdfParams"));
    _params->Init(GPUBufferDescription::Typed(_net.Params, NB_PARAM_COUNT, PixelFormat::R32_Float, true));
    _grads = GPUDevice::Instance->CreateBuffer(TEXT("NeuralBrdfGrads"));
    _grads->Init(GPUBufferDescription::Raw(NB_PARAM_COUNT * sizeof(float), GPUBufferFlags::UnorderedAccess));
    _moments1 = GPUDevice::Instance->CreateBuffer(TEXT("NeuralBrdfM1"));
    _moments1->Init(GPUBufferDescription::Typed(NB_PARAM_COUNT, PixelFormat::R32_Float, true));
    _moments2 = GPUDevice::Instance->CreateBuffer(TEXT("NeuralBrdfM2"));
    _moments2->Init(GPUBufferDescription::Typed(NB_PARAM_COUNT, PixelFormat::R32_Float, true));
    _lossAccum = GPUDevice::Instance->CreateBuffer(TEXT("NeuralBrdfLoss"));
    _lossAccum->Init(GPUBufferDescription::Raw(sizeof(float), GPUBufferFlags::UnorderedAccess));
    _lossStaging = GPUDevice::Instance->CreateBuffer(TEXT("NeuralBrdfLossStaging"));
    _lossStaging->Init(_lossAccum->GetDescription().ToStagingReadback());
    _paramsStaging = GPUDevice::Instance->CreateBuffer(TEXT("NeuralBrdfParamsStaging"));
    _paramsStaging->Init(_params->GetDescription().ToStagingReadback());

    // Pipelines.
    GPUPipelineState::Description desc = GPUPipelineState::Description::Default;
    desc.CullMode = CullMode::TwoSided;
    desc.DepthEnable = true;
    desc.DepthWriteEnable = true;
    desc.DepthFunc = ComparisonFunc::Less;
    desc.VS = drawShader->GetVS("VS");
    desc.PS = drawShader->GetPS("PS_Disney");
    _psDisney = GPUDevice::Instance->CreatePipelineState();
    _psDisney->Init(desc);
    desc.PS = drawShader->GetPS("PS_Inference");
    _psInference = GPUDevice::Instance->CreatePipelineState();
    _psInference->Init(desc);
    desc.PS = drawShader->GetPS("PS_Difference");
    _psDifference = GPUDevice::Instance->CreatePipelineState();
    _psDifference->Init(desc);

    _csTrain = trainShader->GetCS("CS_Train");
    _csAdam = trainShader->GetCS("CS_Adam");
    _drawCB = drawShader->GetCB(0);
    _trainCB = trainShader->GetCB(0);

    _camera.setPosition(glm::vec3(0.0f, 0.0f, -2.5f));
    _camera.setRotation(glm::vec3(0.0f));
    _camera.setPerspective(60.0f, 1.0f, 0.1f, 256.0f);

    MainRenderTask::Instance->IsCustomRendering = true;
    MainRenderTask::Instance->Render.Bind<NeuralBrdfRenderer, &NeuralBrdfRenderer::OnMainRender>(this);
}

void NeuralBrdfRenderer::OnUpdate()
{
    // Only drive the camera while the game viewport is focused. Otherwise dragging the
    // editor sliders (a left-mouse drag) would be read as a camera rotation, spinning the
    // spheres and "eating" the slider input.
    if (Engine::HasGameViewportFocus())
        _camera.HandleMouseInput();
}

void NeuralBrdfRenderer::OnDestroy()
{
    if (_instance == this)
        _instance = nullptr;
    MainRenderTask::Instance->Render.Unbind<NeuralBrdfRenderer, &NeuralBrdfRenderer::OnMainRender>(this);
    MainRenderTask::Instance->IsCustomRendering = false;

    SAFE_DELETE_GPU_RESOURCE(_psDisney);
    SAFE_DELETE_GPU_RESOURCE(_psInference);
    SAFE_DELETE_GPU_RESOURCE(_psDifference);
    SAFE_DELETE_GPU_RESOURCE(_vb);
    SAFE_DELETE_GPU_RESOURCE(_ib);
    SAFE_DELETE_GPU_RESOURCE(_depthBuffer);
    SAFE_DELETE_GPU_RESOURCE(_params);
    SAFE_DELETE_GPU_RESOURCE(_grads);
    SAFE_DELETE_GPU_RESOURCE(_moments1);
    SAFE_DELETE_GPU_RESOURCE(_moments2);
    SAFE_DELETE_GPU_RESOURCE(_lossAccum);
    SAFE_DELETE_GPU_RESOURCE(_lossStaging);
    SAFE_DELETE_GPU_RESOURCE(_paramsStaging);
    _drawCB = nullptr;
    _trainCB = nullptr;
}

void NeuralBrdfRenderer::SetTrainingEnabled(bool enabled)
{
    _training = enabled;
}

void NeuralBrdfRenderer::ResetTraining()
{
    _net.Initialise();
    _resetPending = true;
}

void NeuralBrdfRenderer::SaveModel(const String& path)
{
    _savePath = path;
    _savePending = true;
    _saveCopyCountdown = -1;
}

void NeuralBrdfRenderer::LoadModel(const String& path)
{
    _loadPath = path;
    _loadPending = true;
}

void NeuralBrdfRenderer::UploadParams(GPUContext* context)
{
    context->UpdateBuffer(_params, _net.Params, sizeof(_net.Params));
}

void NeuralBrdfRenderer::ClearTrainingState(GPUContext* context)
{
    context->ClearUA(_grads, Float4::Zero);
    context->ClearUA(_moments1, Float4::Zero);
    context->ClearUA(_moments2, Float4::Zero);
    context->ClearUA(_lossAccum, Float4::Zero);
    _step = 0;
    _epochs = 0;
    _loss = 0.0f;
    _lossStagingValid = false;
}

void NeuralBrdfRenderer::ReadLoss()
{
    if (!_lossStagingValid)
        return;
    void* mapped = _lossStaging->Map(GPUResourceMapMode::Read);
    if (mapped)
    {
        _loss = *(const float*)mapped;
        _lossStaging->Unmap();
    }
}

void NeuralBrdfRenderer::RunTraining(GPUContext* context)
{
    TrainData data;
    _step++;
    data.Seed0 = _step * 9781u + 1u;
    data.Seed1 = 0x9E3779B9u ^ (_step * 2654435761u);
    data.Step = _step;
    data.ParamCount = (uint32)NB_PARAM_COUNT;
    data.LearningRate = NB_LEARNING_RATE;
    data.BatchSize = (uint32)NB_BATCH_SIZE;
    data.Pad0 = 0; data.Pad1 = 0;
    context->UpdateCB(_trainCB, &data);

    // Fresh gradient accumulation + loss for this step.
    context->ClearUA(_grads, Float4::Zero);
    context->ClearUA(_lossAccum, Float4::Zero);

    // Training pass: read weights (SRV), accumulate gradients (UAV) + loss (UAV).
    context->BindCB(0, _trainCB);
    context->BindSR(0, _params->View());
    context->BindUA(0, _grads->View());
    context->BindUA(1, _lossAccum->View());
    context->Dispatch(_csTrain, Math::DivideAndRoundUp((uint32)NB_BATCH_SIZE, (uint32)NB_TRAIN_GROUP), 1, 1);
    context->ResetUA();
    context->ResetSR();

    // Optimizer pass: Adam update of the weights.
    context->BindCB(0, _trainCB);
    context->BindUA(0, _params->View());
    context->BindUA(1, _grads->View());
    context->BindUA(2, _moments1->View());
    context->BindUA(3, _moments2->View());
    context->Dispatch(_csAdam, Math::DivideAndRoundUp((uint32)NB_PARAM_COUNT, (uint32)NB_OPTIMIZE_GROUP), 1, 1);
    context->ResetUA();

    _epochs++;

    // Deferred loss readback (1 frame latency).
    ReadLoss();
    context->CopyResource(_lossStaging, _lossAccum);
    _lossStagingValid = true;
}

void NeuralBrdfRenderer::ServiceSave(GPUContext* context)
{
    if (!_savePending)
        return;
    if (_saveCopyCountdown < 0)
    {
        context->CopyResource(_paramsStaging, _params);
        _saveCopyCountdown = 2; // let the GPU finish the copy
        return;
    }
    if (_saveCopyCountdown > 0)
    {
        _saveCopyCountdown--;
        return;
    }
    // Countdown elapsed: read back and write the file.
    void* mapped = _paramsStaging->Map(GPUResourceMapMode::Read);
    if (mapped)
    {
        Platform::MemoryCopy(_net.Params, mapped, sizeof(_net.Params));
        _paramsStaging->Unmap();
        _net.Save(_savePath);
        LOG(Info, "NeuralBrdf: saved trained model to '{0}'.", _savePath);
    }
    _savePending = false;
    _saveCopyCountdown = -1;
}

void NeuralBrdfRenderer::OnMainRender(RenderTask* task, GPUContext* context)
{
    if (!_params)
        return;
    SceneRenderTask* sceneTask = static_cast<SceneRenderTask*>(task);
    GPUTextureView* output = sceneTask->GetOutputView();
    // Use the OUTPUT viewport (post-upsampling), not GetViewport() (render resolution before
    // upsampling). We render into GetOutputView(), so the depth buffer and sub-viewports must
    // match its size; otherwise any editor resolution-scale/upsampling makes the depth buffer a
    // different size than the color target and the GPU clips rendering into screen-aligned bars.
    Viewport viewport = sceneTask->GetOutputViewport();

    // Apply pending model changes (load/reset/initial upload) on the render thread.
    if (_loadPending)
    {
        if (_net.Load(_loadPath))
        {
            UploadParams(context);
            ClearTrainingState(context);
            LOG(Info, "NeuralBrdf: loaded model '{0}'.", _loadPath);
        }
        _loadPending = false;
    }
    if (_resetPending)
    {
        UploadParams(context);
        ClearTrainingState(context);
        _resetPending = false;
    }
    if (_uploadPending)
    {
        UploadParams(context);
        ClearTrainingState(context);
        _uploadPending = false;
    }

    if (_training)
        RunTraining(context);

    ServiceSave(context);

    // ---- Draw the three views ----
    const float h = viewport.Height;
    const float w = viewport.Width;
    const float third = w / 3.0f;
    const float aspect = (third > 0.0f && h > 0.0f) ? (third / h) : 1.0f;

    _camera.syncProjection(aspect);
    glm::mat4 viewProject = _camera.matrices.perspective * _camera.matrices.view;

    DrawData data;
    SaschaGpu::StoreVulkan(data.ViewProject, viewProject);
    data.CameraPos = Float4(0.0f, 0.0f, 2.5f, 0.0f);
    Float3 ld = _lightDir.GetNormalized();
    data.LightDir = Float4(ld, 0.0f);
    data.LightIntensity = Float4(_lightIntensity);
    data.BaseColor = _baseColor;
    data.Specular = _specular;
    data.Roughness = _roughness;
    data.Metallic = _metallic;
    data.DiffScale = 4.0f;
    context->UpdateCB(_drawCB, &data);

    // A sphere needs depth testing so the nearest surface wins (otherwise back faces
    // overwrite front faces depending on draw order, which breaks as the camera rotates).
    // Use our OWN depth buffer rather than the scene's: the editor's depth buffer already
    // contains the grid and actor icons, which would reject our sphere fragments (black bars).
    const int32 vw = (int32)viewport.Width;
    const int32 vh = (int32)viewport.Height;
    if (vw > 0 && vh > 0)
    {
        if (!_depthBuffer)
            _depthBuffer = GPUDevice::Instance->CreateTexture(TEXT("NeuralBrdfDepth"));
        if (_depthBuffer->Width() != vw || _depthBuffer->Height() != vh)
            _depthBuffer->Init(GPUTextureDescription::New2D(vw, vh, PixelFormat::D32_Float, GPUTextureFlags::DepthStencil));
    }
    GPUTextureView* depth = (_depthBuffer && _depthBuffer->IsAllocated()) ? _depthBuffer->View() : nullptr;
    const Color clearColor = Color::Black;
    if (depth)
    {
        context->SetRenderTarget(depth, output);
        context->Clear(output, clearColor);
        // Standard-Z convention (GLM_FORCE_DEPTH_ZERO_TO_ONE: near->0, far->1) with DepthFunc=Less.
        // Must clear to far=1.0 explicitly; the engine default is reversed-Z far (0.0), which would reject everything.
        context->ClearDepth(depth, 1.0f);
    }
    else
    {
        context->Clear(output, clearColor);
        context->SetRenderTarget(output);
    }

    GPUPipelineState* passes[3] = { _psDisney, _psInference, _psDifference };
    // GPU Profiler region names for each viewport pass (visible in the Flax GPU Profiler).
    const Char* passNames[3] =
    {
        TEXT("Disney (Ground Truth)"),
        TEXT("Neural Inference (MLP)"),
        TEXT("Difference"),
    };
    Span<GPUBuffer*> vb(&_vb, 1);
    for (int32 i = 0; i < 3; ++i)
    {
        ScopeProfileBlockGPU profile(passNames[i]);
        Viewport sub(third * i, 0.0f, third, h);
        context->SetViewportAndScissors(sub);
        context->BindCB(0, _drawCB);
        context->BindSR(0, _params->View());
        context->BindVB(vb, nullptr, _layout);
        context->BindIB(_ib);
        context->SetState(passes[i]);
        context->DrawIndexed(_indexCount);
    }
    context->ResetSR();
}
