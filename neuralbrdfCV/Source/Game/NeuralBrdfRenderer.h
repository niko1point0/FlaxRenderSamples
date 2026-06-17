// Neural BRDF sample renderer.
//
// Trains a small MLP to approximate the Disney BRDF entirely on the GPU and renders a
// sphere three ways side-by-side: analytic ground truth | trained neural | difference.
// Mirrors NVIDIA RTXNS "ShaderTraining" but uses portable HLSL compute (no cooperative
// vectors / Slang autodiff), so it runs on stock Flax (D3D12 and Vulkan).
//
// Exposes Train / Reset / Save / Load and material controls to the editor (see
// NeuralBrdfEditorPlugin.cs) via a static Instance.

#pragma once

#include "Engine/Scripting/Script.h"
#include "Engine/Content/AssetReference.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Core/Math/Vector4.h"
#include "NeuralBrdfNetwork.h"
#include "SaschaCamera.h"
#include "../Shaders/NeuralBrdfConfig.h"

class GPUBuffer;
class GPUTexture;
class GPUConstantBuffer;
class GPUPipelineState;
class GPUShaderProgramCS;
class GPUVertexLayout;
class RenderTask;
class GPUContext;

API_CLASS(Namespace="") class NeuralBrdfRenderer : public Script
{
    API_AUTO_SERIALIZATION();
    DECLARE_SCRIPTING_TYPE(NeuralBrdfRenderer);

public:
    // Singleton accessor so the editor window can drive the live renderer.
    API_PROPERTY() static NeuralBrdfRenderer* GetInstance();

    // ----- Buttons (ported from RTXNS UIWidget) -----
    API_FUNCTION() void SetTrainingEnabled(bool enabled);
    API_PROPERTY() bool GetTrainingEnabled() const { return _training; }
    API_FUNCTION() void ResetTraining();
    API_FUNCTION() void SaveModel(const String& path);
    API_FUNCTION() void LoadModel(const String& path);

    // ----- Live stats for the UI -----
    API_PROPERTY() int32 GetEpochs() const { return _epochs; }
    API_PROPERTY() float GetLoss() const { return _loss; }

    // ----- Material / light controls (ported from RTXNS sliders) -----
    API_PROPERTY() void SetLightIntensity(float v) { _lightIntensity = v; }
    API_PROPERTY() float GetLightIntensity() const { return _lightIntensity; }
    API_PROPERTY() void SetSpecular(float v) { _specular = v; }
    API_PROPERTY() float GetSpecular() const { return _specular; }
    API_PROPERTY() void SetRoughness(float v) { _roughness = v; }
    API_PROPERTY() float GetRoughness() const { return _roughness; }
    API_PROPERTY() void SetMetallic(float v) { _metallic = v; }
    API_PROPERTY() float GetMetallic() const { return _metallic; }

private:
    void OnStart() override;
    void OnUpdate() override;
    void OnDestroy() override;
    void OnMainRender(RenderTask* task, GPUContext* context);

    void BuildSphere();
    void UploadParams(GPUContext* context);
    void ClearTrainingState(GPUContext* context);
    void RunTraining(GPUContext* context);
    void ReadLoss();
    void ServiceSave(GPUContext* context);
    bool InitCoopVec(class GPUShader* drawShader, class GPUShader* trainShader);
    void RefreshCoopVecWeights(GPUContext* context);

    // Assets / pipeline
    AssetReference<class Shader> _drawShaderAsset;
    AssetReference<class Shader> _trainShaderAsset;
    GPUPipelineState* _psDisney = nullptr;
    GPUPipelineState* _psDifference = nullptr;
    GPUShaderProgramCS* _csTrain = nullptr;
    GPUShaderProgramCS* _csAdam = nullptr;
    GPUConstantBuffer* _drawCB = nullptr;
    GPUConstantBuffer* _trainCB = nullptr;

    // Geometry
    GPUBuffer* _vb = nullptr;
    GPUBuffer* _ib = nullptr;
    GPUVertexLayout* _layout = nullptr;
    int32 _indexCount = 0;

    // Private depth buffer (avoids sharing the editor's depth which contains grid/icons).
    GPUTexture* _depthBuffer = nullptr;

    // Neural buffers
    GPUBuffer* _params = nullptr;      // typed RW fp32, the trained weights
    GPUBuffer* _grads = nullptr;       // raw RW fp32, accumulated gradients
    GPUBuffer* _moments1 = nullptr;    // typed RW fp32 (Adam)
    GPUBuffer* _moments2 = nullptr;    // typed RW fp32 (Adam)
    GPUBuffer* _lossAccum = nullptr;   // raw RW fp32, mean batch error
    GPUBuffer* _lossStaging = nullptr; // staging readback of loss
    GPUBuffer* _paramsStaging = nullptr; // staging readback for SaveModel

    // Cooperative-vector (NVIDIA Neural Shading) inference path - the only inference path. The neural
    // and difference views run cooperative-vector MatVecMulAdd over device-optimal FP16 weights. The
    // fp32 _params buffer remains the trained master (drives training and Save/Load); it is packed to
    // fp16 and converted to the optimal layout each frame the weights may have changed. Requires a
    // cooperative-vector-capable backend (no fp32 inference fallback).
    bool _coopVec = false;
    GPUBuffer* _weightsF16 = nullptr;        // raw fp16 row-major mirror of _params (pack target)
    GPUBuffer* _weightsOpt = nullptr;        // raw device-optimal fp16 weight matrices + fp16 bias vectors
    GPUPipelineState* _psInferenceCV = nullptr;
    GPUShaderProgramCS* _csPackF16 = nullptr;
    uint32 _wOptOff[NB_NUM_LAYERS] = {};     // byte offset of each layer's MUL_OPTIMAL weight matrix in _weightsOpt
    uint32 _bOptOff[NB_NUM_LAYERS] = {};     // byte offset of each layer's fp16 bias vector in _weightsOpt
    uint32 _weightsOptSize = 0;

    NeuralBrdfNetwork _net;

    // State
    bool _training = false;
    bool _uploadPending = true;
    bool _resetPending = false;
    bool _loadPending = false;
    String _loadPath;
    bool _savePending = false;
    String _savePath;
    int32 _saveCopyCountdown = -1;
    bool _lossStagingValid = false;
    int32 _epochs = 0;
    uint32 _step = 0;
    float _loss = 0.0f;

    // Material / light
    float _lightIntensity = 3.0f;
    float _specular = 0.5f;
    float _roughness = 0.5f;
    float _metallic = 0.0f;
    Float4 _baseColor = Float4(0.82f, 0.67f, 0.16f, 1.0f);
    Float3 _lightDir = Float3(-0.761f, -0.467f, -0.450f);

    SaschaCamera _camera;

    static NeuralBrdfRenderer* _instance;
};
