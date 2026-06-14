#pragma once

#include "SaschaGlm.h"
#include "SaschaGpu.h"
#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Core/Math/Quaternion.h"
#include "Engine/Core/Math/Vector2.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Core/Math/Vector4.h"
#include "Engine/Core/Types/BaseTypes.h"
#include "Engine/Core/Types/String.h"
#include "Engine/Graphics/GPUBuffer.h"
#include "Engine/Graphics/GPUContext.h"
#include "Engine/Graphics/Shaders/GPUShader.h"
#include "Engine/Graphics/Shaders/GPUVertexLayout.h"
#include "Engine/Graphics/Textures/GPUTexture.h"

class GPUConstantBuffer;
class GPUPipelineState;

/// <summary>Animated skinned glTF model loader/renderer (Vulkan gltfskinning port).</summary>
class SaschaSkinnedGltf
{
public:
    struct Vertex
    {
        Float3 Position;
        Float3 Normal;
        Float2 UV;
        Float3 Color;
        Float4 JointIndices;
        Float4 JointWeights;
    };

    struct Primitive
    {
        uint32 FirstIndex = 0;
        uint32 IndexCount = 0;
        int32 MaterialIndex = 0;
    };

    struct Mesh
    {
        Array<Primitive> Primitives;
    };

    struct Node;

    struct Material
    {
        Float4 BaseColorFactor = Float4(1.0f);
        int32 BaseColorTextureIndex = -1;
    };

    struct Image
    {
        Array<byte> Pixels;
        int32 Width = 0;
        int32 Height = 0;
        GPUTexture* Texture = nullptr;
    };

    struct Texture
    {
        int32 ImageIndex = 0;
    };

    struct Skin
    {
        Array<glm::mat4> InverseBindMatrices;
        Array<Node*> Joints;
        GPUBuffer* JointBuffer = nullptr;
    };

    struct AnimationSampler
    {
        String Interpolation;
        Array<float> Inputs;
        Array<Float4> Outputs;
    };

    struct AnimationChannel
    {
        String Path;
        Node* TargetNode = nullptr;
        uint32 SamplerIndex = 0;
    };

    struct Animation
    {
        String Name;
        Array<AnimationSampler> Samplers;
        Array<AnimationChannel> Channels;
        float Start = FLT_MAX;
        float End = FLT_MIN;
        float CurrentTime = 0.0f;
    };

    struct Node
    {
        Node* Parent = nullptr;
        uint32 Index = 0;
        Array<Node*> Children;
        Mesh MeshData;
        glm::vec3 Translation = glm::vec3(0.0f);
        glm::vec3 Scale = glm::vec3(1.0f);
        glm::quat Rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        int32 SkinIndex = -1;
        glm::mat4 Matrix = glm::mat4(1.0f);

        glm::mat4 GetLocalMatrix() const;
    };

    GPUVertexLayout* VertexLayout = nullptr;
    GPUBuffer* VertexBuffer = nullptr;
    GPUBuffer* IndexBuffer = nullptr;
    uint32 IndexCount = 0;

    Array<Image> Images;
    Array<Texture> Textures;
    Array<Material> Materials;
    Array<Node*> Nodes;
    Array<Skin> Skins;
    Array<Animation> Animations;

    uint32 ActiveAnimation = 0;

    bool Load(const Char* gltfPath);
    void UploadGpuResources(GPUContext* context);
    void UpdateAnimation(float deltaTime);
    void Draw(GPUContext* context, GPUConstantBuffer* sceneCb, GPUConstantBuffer* nodeCb) const;
    void ReleaseGpuResources();

private:
    bool _gpuReady = false;

    Node* FindNode(Node* parent, uint32 index);
    Node* NodeFromIndex(uint32 index);
    glm::mat4 GetNodeMatrix(Node* node) const;
    void UpdateJoints(Node* node);
    void DrawNode(GPUContext* context, GPUConstantBuffer* nodeCb, Node* node) const;
    static glm::mat4 LoadGltfMatrix(const float* src);
};
