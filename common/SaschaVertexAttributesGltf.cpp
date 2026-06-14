#include "SaschaGpu.h"
#include "SaschaGpu.h"
#include "SaschaVertexAttributesGltf.h"

#include <cstring>
#include "Engine/Content/Content.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Math/Quaternion.h"
#include "Engine/Core/Memory/Memory.h"
#include "Engine/Engine/Globals.h"
#include "Engine/Graphics/GPUDevice.h"
#include "Engine/Graphics/GPUResource.h"
#include "Engine/Platform/File.h"
#include "Engine/Platform/FileSystem.h"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include "tiny_gltf.h"

namespace
{
    static VertexElement g_InterleavedLayoutElements[4] =
    {
        { VertexElement::Types::Position, 0, 0, 0, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::Normal, 0, 12, 0, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::TexCoord0, 0, 24, 0, PixelFormat::R32G32_Float },
        { VertexElement::Types::TexCoord1, 0, 32, 0, PixelFormat::R32G32B32A32_Float },
    };

    static VertexElement g_SeparateLayoutElements[4] =
    {
        { VertexElement::Types::Position, 0, 0, 0, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::Normal, 0, 0, 1, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::TexCoord0, 0, 0, 2, PixelFormat::R32G32_Float },
        { VertexElement::Types::TexCoord1, 0, 0, 3, PixelFormat::R32G32B32A32_Float },
    };

    GPU_CB_STRUCT(PushData {
        Matrix NodeMatrix;
        uint32 AlphaMask;
        float AlphaMaskCutoff;
        float Padding;
    });

    glm::mat4 LoadGltfMatrix(const float* src)
    {
        glm::mat4 m(1.0f);
        memcpy(&m, src, sizeof(glm::mat4));
        return m;
    }

    glm::mat4 BuildNodeMatrix(const tinygltf::Node& inputNode)
    {
        if (inputNode.matrix.size() == 16)
            return LoadGltfMatrix((const float*)inputNode.matrix.data());

        glm::mat4 result = glm::mat4(1.0f);
        if (inputNode.translation.size() == 3)
        {
            result = glm::translate(result, glm::vec3(
                (float)inputNode.translation[0],
                (float)inputNode.translation[1],
                (float)inputNode.translation[2]));
        }
        if (inputNode.rotation.size() == 4)
        {
            const glm::quat rot(
                (float)inputNode.rotation[3],
                (float)inputNode.rotation[0],
                (float)inputNode.rotation[1],
                (float)inputNode.rotation[2]);
            result = result * glm::mat4(rot);
        }
        if (inputNode.scale.size() == 3)
        {
            result = glm::scale(result, glm::vec3(
                (float)inputNode.scale[0],
                (float)inputNode.scale[1],
                (float)inputNode.scale[2]));
        }
        return result;
    }
}

String SaschaVertexAttributesGltf::GetTextureContentPath(const String& uri)
{
    int32 slash = uri.FindLast(TEXT('/'));
    int32 backslash = uri.FindLast(TEXT('\\'));
    int32 split = Math::Max(slash, backslash);
    String name = split >= 0 ? uri.Substring(split + 1) : uri;
    const int32 dot = name.FindLast(TEXT('.'));
    if (dot >= 0)
        name = name.Substring(0, dot);
    return Globals::ProjectContentFolder / (TEXT("Textures/") + name + TEXT(".flax"));
}

bool SaschaVertexAttributesGltf::Load(const Char* gltfPath)
{
    if (!FileSystem::FileExists(gltfPath))
        return false;

    tinygltf::Model input;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    const std::string path((const char*)gltfPath);
    if (!loader.LoadASCIIFromFile(&input, &err, &warn, path))
        return false;

    Images.Resize(input.images.size());
    for (size_t i = 0; i < input.images.size(); i++)
        Images[i].Uri = input.images[i].uri.c_str();

    Textures.Resize(input.textures.size());
    for (size_t i = 0; i < input.textures.size(); i++)
        Textures[i].ImageIndex = input.textures[i].source;

    Materials.Resize(input.materials.size());
    for (size_t i = 0; i < input.materials.size(); i++)
    {
        const tinygltf::Material& src = input.materials[i];
        Material& dst = Materials[i];
        auto factorIt = src.values.find("baseColorFactor");
        if (factorIt != src.values.end())
        {
            const auto& c = factorIt->second.ColorFactor();
            dst.BaseColorFactor = Float4((float)c[0], (float)c[1], (float)c[2], (float)c[3]);
        }
        auto colorTexIt = src.values.find("baseColorTexture");
        if (colorTexIt != src.values.end())
            dst.BaseColorTextureIndex = colorTexIt->second.TextureIndex();
        auto normalTexIt = src.additionalValues.find("normalTexture");
        if (normalTexIt != src.additionalValues.end())
            dst.NormalTextureIndex = normalTexIt->second.TextureIndex();
        dst.AlphaMode = src.alphaMode.c_str();
        dst.AlphaCutoff = (float)src.alphaCutoff;
    }

    Array<uint32> indexBuffer;
    Array<Vertex> interleavedVertices;
    Array<Float3> separatePos;
    Array<Float3> separateNormal;
    Array<Float2> separateUv;
    Array<Float4> separateTangent;

    auto loadNode = [&](const tinygltf::Node& inputNode, Node* parent, auto&& loadNodeRef) -> void
    {
        Node* node = New<Node>();
        node->Parent = parent;
        node->LocalMatrix = BuildNodeMatrix(inputNode);

        for (int childIndex : inputNode.children)
            loadNodeRef(input.nodes[childIndex], node, loadNodeRef);

        if (inputNode.mesh > -1)
        {
            const tinygltf::Mesh& mesh = input.meshes[inputNode.mesh];
            for (const tinygltf::Primitive& primitive : mesh.primitives)
            {
                const uint32 firstIndex = (uint32)indexBuffer.Count();
                const uint32 vertexStart = (uint32)interleavedVertices.Count();

                const float* positionBuffer = nullptr;
                const float* normalsBuffer = nullptr;
                const float* texCoordsBuffer = nullptr;
                const float* tangentsBuffer = nullptr;
                size_t vertexCount = 0;

                auto posIt = primitive.attributes.find("POSITION");
                if (posIt != primitive.attributes.end())
                {
                    const tinygltf::Accessor& accessor = input.accessors[posIt->second];
                    const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
                    positionBuffer = reinterpret_cast<const float*>(&input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
                    vertexCount = accessor.count;
                }

                auto normIt = primitive.attributes.find("NORMAL");
                if (normIt != primitive.attributes.end())
                {
                    const tinygltf::Accessor& accessor = input.accessors[normIt->second];
                    const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
                    normalsBuffer = reinterpret_cast<const float*>(&input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
                }

                auto uvIt = primitive.attributes.find("TEXCOORD_0");
                if (uvIt != primitive.attributes.end())
                {
                    const tinygltf::Accessor& accessor = input.accessors[uvIt->second];
                    const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
                    texCoordsBuffer = reinterpret_cast<const float*>(&input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
                }

                auto tangentIt = primitive.attributes.find("TANGENT");
                if (tangentIt != primitive.attributes.end())
                {
                    const tinygltf::Accessor& accessor = input.accessors[tangentIt->second];
                    const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
                    tangentsBuffer = reinterpret_cast<const float*>(&input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
                }

                for (size_t v = 0; v < vertexCount; v++)
                {
                    Vertex vert;
                    vert.Position = Float3(positionBuffer[v * 3 + 0], positionBuffer[v * 3 + 1], positionBuffer[v * 3 + 2]);
                    if (normalsBuffer)
                        vert.Normal = Float3(normalsBuffer[v * 3 + 0], normalsBuffer[v * 3 + 1], normalsBuffer[v * 3 + 2]).GetNormalized();
                    else
                        vert.Normal = Float3::Up;
                    if (texCoordsBuffer)
                        vert.UV = Float2(texCoordsBuffer[v * 2 + 0], texCoordsBuffer[v * 2 + 1]);
                    if (tangentsBuffer)
                        vert.Tangent = Float4(tangentsBuffer[v * 4 + 0], tangentsBuffer[v * 4 + 1], tangentsBuffer[v * 4 + 2], tangentsBuffer[v * 4 + 3]);
                    interleavedVertices.Add(vert);

                    separatePos.Add(vert.Position);
                    separateNormal.Add(vert.Normal);
                    separateUv.Add(vert.UV);
                    separateTangent.Add(vert.Tangent);
                }

                const tinygltf::Accessor& accessor = input.accessors[primitive.indices];
                const tinygltf::BufferView& bufferView = input.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& buffer = input.buffers[bufferView.buffer];
                const uint32 primitiveIndexCount = (uint32)accessor.count;

                switch (accessor.componentType)
                {
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
                {
                    const uint32_t* buf = reinterpret_cast<const uint32_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                    for (size_t index = 0; index < accessor.count; index++)
                        indexBuffer.Add(buf[index] + vertexStart);
                    break;
                }
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
                {
                    const uint16_t* buf = reinterpret_cast<const uint16_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                    for (size_t index = 0; index < accessor.count; index++)
                        indexBuffer.Add(buf[index] + vertexStart);
                    break;
                }
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
                {
                    const uint8_t* buf = reinterpret_cast<const uint8_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                    for (size_t index = 0; index < accessor.count; index++)
                        indexBuffer.Add(buf[index] + vertexStart);
                    break;
                }
                default:
                    break;
                }

                Primitive prim;
                prim.FirstIndex = firstIndex;
                prim.IndexCount = primitiveIndexCount;
                prim.MaterialIndex = primitive.material;
                node->MeshData.Primitives.Add(prim);
            }
        }

        if (parent)
            parent->Children.Add(node);
        else
            Nodes.Add(node);
    };

    if (!input.scenes.empty())
    {
        const int sceneIndex = input.defaultScene > -1 ? input.defaultScene : 0;
        const tinygltf::Scene& scene = input.scenes[sceneIndex];
        for (int nodeIndex : scene.nodes)
            loadNode(input.nodes[nodeIndex], nullptr, loadNode);
    }

    if (!interleavedVertices.HasItems() || !indexBuffer.HasItems())
        return false;

    static GPUVertexLayout::Elements interleavedList(g_InterleavedLayoutElements, 4);
    static GPUVertexLayout::Elements separateList(g_SeparateLayoutElements, 4);
    InterleavedLayout = GPUVertexLayout::Get(interleavedList, true);
    SeparateLayout = GPUVertexLayout::Get(separateList, true);

    InterleavedVertexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("VertexAttrInterleavedVB"));
    InterleavedVertexBuffer->Init(GPUBufferDescription::Vertex(InterleavedLayout, sizeof(Vertex), interleavedVertices.Count(), interleavedVertices.Get()));

    SeparatePosBuffer = GPUDevice::Instance->CreateBuffer(TEXT("VertexAttrPosVB"));
    SeparatePosBuffer->Init(GPUBufferDescription::Vertex(SeparateLayout, sizeof(Float3), separatePos.Count(), separatePos.Get()));

    SeparateNormalBuffer = GPUDevice::Instance->CreateBuffer(TEXT("VertexAttrNormalVB"));
    SeparateNormalBuffer->Init(GPUBufferDescription::Vertex(SeparateLayout, sizeof(Float3), separateNormal.Count(), separateNormal.Get()));

    SeparateUvBuffer = GPUDevice::Instance->CreateBuffer(TEXT("VertexAttrUvVB"));
    SeparateUvBuffer->Init(GPUBufferDescription::Vertex(SeparateLayout, sizeof(Float2), separateUv.Count(), separateUv.Get()));

    SeparateTangentBuffer = GPUDevice::Instance->CreateBuffer(TEXT("VertexAttrTangentVB"));
    SeparateTangentBuffer->Init(GPUBufferDescription::Vertex(SeparateLayout, sizeof(Float4), separateTangent.Count(), separateTangent.Get()));

    IndexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("VertexAttrIB"));
    IndexBuffer->Init(GPUBufferDescription::Index(sizeof(uint32), indexBuffer.Count(), indexBuffer.Get()));

    return true;
}

void SaschaVertexAttributesGltf::ResolveTextures()
{
    for (Image& image : Images)
    {
        if (image.Uri.IsEmpty())
            continue;
        image.TextureAsset = Content::Load<Texture>(GetTextureContentPath(image.Uri));
    }
}

void SaschaVertexAttributesGltf::ReleaseGpuResources()
{
    SAFE_DELETE_GPU_RESOURCE(InterleavedVertexBuffer);
    SAFE_DELETE_GPU_RESOURCE(SeparatePosBuffer);
    SAFE_DELETE_GPU_RESOURCE(SeparateNormalBuffer);
    SAFE_DELETE_GPU_RESOURCE(SeparateUvBuffer);
    SAFE_DELETE_GPU_RESOURCE(SeparateTangentBuffer);
    SAFE_DELETE_GPU_RESOURCE(IndexBuffer);
}

glm::mat4 SaschaVertexAttributesGltf::GetWorldMatrix(Node* node) const
{
    glm::mat4 nodeMatrix = node->LocalMatrix;
    Node* parent = node->Parent;
    while (parent)
    {
        nodeMatrix = parent->LocalMatrix * nodeMatrix;
        parent = parent->Parent;
    }
    return nodeMatrix;
}

void SaschaVertexAttributesGltf::DrawNode(GPUContext* context, GPUConstantBuffer* pushCb, Node* node) const
{
    if (node->MeshData.Primitives.HasItems())
    {
        glm::mat4 worldMatrix = GetWorldMatrix(node);
        Matrix gpuMatrix;
        SaschaGpu::StoreTransposed(gpuMatrix, worldMatrix);

        for (const Primitive& primitive : node->MeshData.Primitives)
        {
            if (primitive.IndexCount == 0)
                continue;

            PushData push;
            push.NodeMatrix = gpuMatrix;
            push.AlphaMask = 0;
            push.AlphaMaskCutoff = 0.5f;

            GPUTexture* colorMap = GPUDevice::Instance->GetDefaultWhiteTexture();
            GPUTexture* normalMap = GPUDevice::Instance->GetDefaultNormalMap();
            if (!normalMap)
                normalMap = GPUDevice::Instance->GetDefaultWhiteTexture();
            if (primitive.MaterialIndex >= 0 && primitive.MaterialIndex < Materials.Count())
            {
                const Material& material = Materials[primitive.MaterialIndex];
                push.AlphaMask = material.AlphaMode == TEXT("MASK") ? 1u : 0u;
                push.AlphaMaskCutoff = material.AlphaCutoff;

                if (material.BaseColorTextureIndex >= 0 && material.BaseColorTextureIndex < Textures.Count())
                {
                    const int32 imageIndex = Textures[material.BaseColorTextureIndex].ImageIndex;
                    if (imageIndex >= 0 && imageIndex < Images.Count() && Images[imageIndex].TextureAsset)
                        colorMap = Images[imageIndex].TextureAsset->GetTexture();
                }
                if (material.NormalTextureIndex >= 0 && material.NormalTextureIndex < Textures.Count())
                {
                    const int32 imageIndex = Textures[material.NormalTextureIndex].ImageIndex;
                    if (imageIndex >= 0 && imageIndex < Images.Count() && Images[imageIndex].TextureAsset)
                        normalMap = Images[imageIndex].TextureAsset->GetTexture();
                }
            }

            context->UpdateCB(pushCb, &push);
            context->BindCB(1, pushCb);
            context->BindSR(0, colorMap);
            context->BindSR(1, normalMap);
            context->DrawIndexed(primitive.IndexCount, primitive.FirstIndex, 0);
        }
    }

    for (Node* child : node->Children)
        DrawNode(context, pushCb, child);
}

void SaschaVertexAttributesGltf::Draw(GPUContext* context, GPUConstantBuffer* sceneCb, GPUConstantBuffer* pushCb, bool separateAttributes) const
{
    if (!IndexBuffer)
        return;

    context->BindCB(0, sceneCb);
    context->BindIB(IndexBuffer);

    if (separateAttributes)
    {
        GPUBuffer* vbs[4] = { SeparatePosBuffer, SeparateNormalBuffer, SeparateUvBuffer, SeparateTangentBuffer };
        context->BindVB(Span<GPUBuffer*>(vbs, 4), nullptr, SeparateLayout);
    }
    else
    {
        GPUBuffer* vb = InterleavedVertexBuffer;
        context->BindVB(Span<GPUBuffer*>(&vb, 1), nullptr, InterleavedLayout);
    }

    for (Node* node : Nodes)
        DrawNode(context, pushCb, node);
}
