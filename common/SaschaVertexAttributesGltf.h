#pragma once

#include "SaschaGlm.h"
#include "Engine/Content/Assets/Texture.h"
#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Core/Math/Vector2.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Core/Math/Vector4.h"
#include "Engine/Core/Types/BaseTypes.h"
#include "Engine/Core/Types/String.h"
#include "Engine/Graphics/GPUBuffer.h"
#include "Engine/Graphics/GPUContext.h"
#include "Engine/Graphics/Shaders/GPUShader.h"
#include "Engine/Graphics/Shaders/GPUVertexLayout.h"

class GPUConstantBuffer;

/// <summary>glTF scene with interleaved and separate vertex attribute buffers (Vulkan vertexattributes port).</summary>
class SaschaVertexAttributesGltf
{
public:
    struct Vertex
    {
        Float3 Position;
        Float3 Normal;
        Float2 UV;
        Float4 Tangent;
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
        int32 NormalTextureIndex = -1;
        String AlphaMode = TEXT("OPAQUE");
        float AlphaCutoff = 0.5f;
    };

    struct Image
    {
        String Uri;
        Texture* TextureAsset = nullptr;
    };

    struct GltfTextureSlot
    {
        int32 ImageIndex = 0;
    };

    struct Node
    {
        Node* Parent = nullptr;
        Array<Node*> Children;
        Mesh MeshData;
        glm::mat4 LocalMatrix = glm::mat4(1.0f);
    };

    GPUVertexLayout* InterleavedLayout = nullptr;
    GPUVertexLayout* SeparateLayout = nullptr;

    GPUBuffer* InterleavedVertexBuffer = nullptr;
    GPUBuffer* SeparatePosBuffer = nullptr;
    GPUBuffer* SeparateNormalBuffer = nullptr;
    GPUBuffer* SeparateUvBuffer = nullptr;
    GPUBuffer* SeparateTangentBuffer = nullptr;
    GPUBuffer* IndexBuffer = nullptr;

    Array<Image> Images;
    Array<GltfTextureSlot> Textures;
    Array<Material> Materials;
    Array<Node*> Nodes;

    bool Load(const Char* gltfPath);
    void ResolveTextures();
    void ReleaseGpuResources();

    void Draw(GPUContext* context, GPUConstantBuffer* sceneCb, GPUConstantBuffer* pushCb, bool separateAttributes) const;

private:
    void DrawNode(GPUContext* context, GPUConstantBuffer* pushCb, Node* node) const;
    glm::mat4 GetWorldMatrix(Node* node) const;
    static String GetTextureContentPath(const String& uri);
};
