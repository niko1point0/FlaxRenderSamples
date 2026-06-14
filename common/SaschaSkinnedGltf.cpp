#include "SaschaGpu.h"
#include "SaschaSkinnedGltf.h"

#include <cstring>
#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Memory/Memory.h"
#include "Engine/Graphics/Enums.h"
#include "Engine/Graphics/GPUDevice.h"
#include "Engine/Graphics/GPUResource.h"
#include "Engine/Graphics/Textures/GPUTextureDescription.h"
#include "Engine/Platform/File.h"
#include "Engine/Platform/FileSystem.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

namespace
{
    static VertexElement g_LayoutElements[6] =
    {
        { VertexElement::Types::Position, 0, 0, 0, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::Normal, 0, 12, 0, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::TexCoord0, 0, 24, 0, PixelFormat::R32G32_Float },
        { VertexElement::Types::Color, 0, 32, 0, PixelFormat::R32G32B32_Float },
        { VertexElement::Types::TexCoord1, 0, 44, 0, PixelFormat::R32G32B32A32_Float },
        { VertexElement::Types::TexCoord2, 0, 60, 0, PixelFormat::R32G32B32A32_Float },
    };

    static GPUVertexLayout* g_VertexLayout = nullptr;
}

glm::mat4 SaschaSkinnedGltf::LoadGltfMatrix(const float* src)
{
    glm::mat4 m(1.0f);
    memcpy(&m, src, sizeof(glm::mat4));
    return m;
}

glm::mat4 SaschaSkinnedGltf::Node::GetLocalMatrix() const
{
    return glm::translate(glm::mat4(1.0f), Translation) * glm::mat4(Rotation) * glm::scale(glm::mat4(1.0f), Scale) * Matrix;
}

SaschaSkinnedGltf::Node* SaschaSkinnedGltf::FindNode(Node* parent, uint32 index)
{
    if (parent->Index == index)
        return parent;
    for (Node* child : parent->Children)
    {
        Node* found = FindNode(child));
        if (found)
            return found;
    }
    return nullptr;
}

SaschaSkinnedGltf::Node* SaschaSkinnedGltf::NodeFromIndex(uint32 index)
{
    for (Node* node : Nodes)
    {
        Node* found = FindNode(node, index);
        if (found)
            return found;
    }
    return nullptr;
}

bool SaschaSkinnedGltf::Load(const Char* gltfPath)
{
    if (!FileSystem::FileExists(gltfPath))
        return false;

    Array<byte> fileBytes;
    if (!File::ReadAllBytes(gltfPath, fileBytes))
        return false;

    tinygltf::Model input;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    if (!loader.LoadASCIIFromString(&input, &err, &warn, (const char*)fileBytes.Get(), (uint32)fileBytes.Count(), ""))
        return false;

    Images.Resize(input.images.size());
    for (size_t i = 0; i < input.images.size(); i++)
    {
        const tinygltf::Image& gltfImage = input.images[i];
        Image& image = Images[i];
        image.Width = gltfImage.width;
        image.Height = gltfImage.height;
        if (gltfImage.component == 3)
        {
            const size_t pixelCount = (size_t)gltfImage.width * (size_t)gltfImage.height;
            image.Pixels.Resize(pixelCount * 4);
            for (size_t p = 0; p < pixelCount; p++)
            {
                image.Pixels[p * 4 + 0] = gltfImage.image[p * 3 + 0];
                image.Pixels[p * 4 + 1] = gltfImage.image[p * 3 + 1];
                image.Pixels[p * 4 + 2] = gltfImage.image[p * 3 + 2];
                image.Pixels[p * 4 + 3] = 255;
            }
        }
        else
        {
            image.Pixels.Set((const byte*)gltfImage.image.data(), gltfImage.image.size());
        }
    }

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
        auto texIt = src.values.find("baseColorTexture");
        if (texIt != src.values.end())
            dst.BaseColorTextureIndex = texIt->second.TextureIndex();
    }

    Array<uint32> indexBuffer;
    Array<Vertex> vertexBuffer;

    auto loadNode = [&](const tinygltf::Node& inputNode, Node* parent, uint32 nodeIndex, auto&& loadNodeRef) -> void
    {
        Node* node = New<Node>();
        node->Parent = parent;
        node->Index = nodeIndex;
        node->SkinIndex = inputNode.skin;

        if (inputNode.translation.size() == 3)
            node->Translation = glm::vec3((float)inputNode.translation[0], (float)inputNode.translation[1], (float)inputNode.translation[2]);
        if (inputNode.rotation.size() == 4)
            node->Rotation = glm::quat((float)inputNode.rotation[3], (float)inputNode.rotation[0], (float)inputNode.rotation[1], (float)inputNode.rotation[2]);
        if (inputNode.scale.size() == 3)
            node->Scale = glm::vec3((float)inputNode.scale[0], (float)inputNode.scale[1], (float)inputNode.scale[2]);
        if (inputNode.matrix.size() == 16)
            node->Matrix = LoadGltfMatrix((const float*)inputNode.matrix.data());

        for (int childIndex : inputNode.children)
            loadNodeRef(input.nodes[childIndex], node, (uint32)childIndex, loadNodeRef);

        if (inputNode.mesh > -1)
        {
            const tinygltf::Mesh& mesh = input.meshes[inputNode.mesh];
            for (const tinygltf::Primitive& primitive : mesh.primitives)
            {
                const uint32 firstIndex = (uint32)indexBuffer.Count();
                const uint32 vertexStart = (uint32)vertexBuffer.Count();
                uint32 primitiveIndexCount = 0;

                const float* positionBuffer = nullptr;
                const float* normalsBuffer = nullptr;
                const float* texCoordsBuffer = nullptr;
                const uint16_t* jointIndicesBuffer = nullptr;
                const float* jointWeightsBuffer = nullptr;
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

                auto jointsIt = primitive.attributes.find("JOINTS_0");
                if (jointsIt != primitive.attributes.end())
                {
                    const tinygltf::Accessor& accessor = input.accessors[jointsIt->second];
                    const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
                    jointIndicesBuffer = reinterpret_cast<const uint16_t*>(&input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
                }

                auto weightsIt = primitive.attributes.find("WEIGHTS_0");
                if (weightsIt != primitive.attributes.end())
                {
                    const tinygltf::Accessor& accessor = input.accessors[weightsIt->second];
                    const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
                    jointWeightsBuffer = reinterpret_cast<const float*>(&input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
                }

                const bool hasSkin = jointIndicesBuffer && jointWeightsBuffer;
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
                    vert.Color = Float3::One;
                    if (hasSkin)
                    {
                        vert.JointIndices = Float4((float)jointIndicesBuffer[v * 4 + 0], (float)jointIndicesBuffer[v * 4 + 1], (float)jointIndicesBuffer[v * 4 + 2], (float)jointIndicesBuffer[v * 4 + 3]);
                        vert.JointWeights = Float4(jointWeightsBuffer[v * 4 + 0], jointWeightsBuffer[v * 4 + 1], jointWeightsBuffer[v * 4 + 2], jointWeightsBuffer[v * 4 + 3]);
                    }
                    vertexBuffer.Add(vert);
                }

                const tinygltf::Accessor& accessor = input.accessors[primitive.indices];
                const tinygltf::BufferView& bufferView = input.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& buffer = input.buffers[bufferView.buffer];
                primitiveIndexCount = (uint32)accessor.count;

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
            loadNode(input.nodes[nodeIndex], nullptr, (uint32)nodeIndex, loadNode);
    }

    Skins.Resize(input.skins.size());
    for (size_t i = 0; i < input.skins.size(); i++)
    {
        const tinygltf::Skin& gltfSkin = input.skins[i];
        Skin& skin = Skins[i];
        for (int jointIndex : gltfSkin.joints)
        {
            Node* jointNode = NodeFromIndex((uint32)jointIndex);
            if (jointNode)
                skin.Joints.Add(jointNode);
        }

        if (gltfSkin.inverseBindMatrices > -1)
        {
            const tinygltf::Accessor& accessor = input.accessors[gltfSkin.inverseBindMatrices];
            const tinygltf::BufferView& bufferView = input.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = input.buffers[bufferView.buffer];
            skin.InverseBindMatrices.Resize(accessor.count);
            const float* src = reinterpret_cast<const float*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
            for (size_t j = 0; j < accessor.count; j++)
                skin.InverseBindMatrices[j] = LoadGltfMatrix(src + j * 16);
        }
    }

    Animations.Resize(input.animations.size());
    for (size_t i = 0; i < input.animations.size(); i++)
    {
        const tinygltf::Animation& gltfAnim = input.animations[i];
        Animation& anim = Animations[i];
        anim.Name = gltfAnim.name.c_str();

        anim.Samplers.Resize(gltfAnim.samplers.size());
        for (size_t j = 0; j < gltfAnim.samplers.size(); j++)
        {
            const tinygltf::AnimationSampler& gltfSampler = gltfAnim.samplers[j];
            AnimationSampler& sampler = anim.Samplers[j];
            sampler.Interpolation = gltfSampler.interpolation.c_str();

            const tinygltf::Accessor& inAccessor = input.accessors[gltfSampler.input];
            const tinygltf::BufferView& inView = input.bufferViews[inAccessor.bufferView];
            const tinygltf::Buffer& inBuffer = input.buffers[inView.buffer];
            const float* inBuf = reinterpret_cast<const float*>(&inBuffer.data[inAccessor.byteOffset + inView.byteOffset]);
            sampler.Inputs.Resize(inAccessor.count);
            for (size_t k = 0; k < inAccessor.count; k++)
            {
                sampler.Inputs[k] = inBuf[k];
                anim.Start = Math::Min(anim.Start, inBuf[k]);
                anim.End = Math::Max(anim.End, inBuf[k]);
            }

            const tinygltf::Accessor& outAccessor = input.accessors[gltfSampler.output];
            const tinygltf::BufferView& outView = input.bufferViews[outAccessor.bufferView];
            const tinygltf::Buffer& outBuffer = input.buffers[outView.buffer];
            const void* outPtr = &outBuffer.data[outAccessor.byteOffset + outView.byteOffset];
            sampler.Outputs.Resize(outAccessor.count);
            if (outAccessor.type == TINYGLTF_TYPE_VEC3)
            {
                const Float3* buf = static_cast<const Float3*>(outPtr);
                for (size_t k = 0; k < outAccessor.count; k++)
                    sampler.Outputs[k] = Float4(buf[k], 0.0f);
            }
            else if (outAccessor.type == TINYGLTF_TYPE_VEC4)
            {
                const Float4* buf = static_cast<const Float4*>(outPtr);
                for (size_t k = 0; k < outAccessor.count; k++)
                    sampler.Outputs[k] = buf[k];
            }
        }

        anim.Channels.Resize(gltfAnim.channels.size());
        for (size_t j = 0; j < gltfAnim.channels.size(); j++)
        {
            const tinygltf::AnimationChannel& gltfChannel = gltfAnim.channels[j];
            AnimationChannel& channel = anim.Channels[j];
            channel.Path = gltfChannel.target_path.c_str();
            channel.SamplerIndex = (uint32)gltfChannel.sampler;
            channel.TargetNode = NodeFromIndex((uint32)gltfChannel.target_node);
        }
    }

    static GPUVertexLayout::Elements layoutList(g_LayoutElements, 6);
    VertexLayout = GPUVertexLayout::Get(layoutList, true);

    VertexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("SkinnedGltfVB"));
    VertexBuffer->Init(GPUBufferDescription::Vertex(VertexLayout, sizeof(Vertex), vertexBuffer.Count(), vertexBuffer.Get()));

    IndexCount = (uint32)indexBuffer.Count();
    IndexBuffer = GPUDevice::Instance->CreateBuffer(TEXT("SkinnedGltfIB"));
    IndexBuffer->Init(GPUBufferDescription::Index(sizeof(uint32), indexBuffer.Count(), indexBuffer.Get()));

    for (Node* node : Nodes)
        UpdateJoints(node);

    return vertexBuffer.HasItems() && indexBuffer.HasItems();
}

void SaschaSkinnedGltf::UploadGpuResources(GPUContext* context)
{
    if (_gpuReady)
        return;

    for (Image& image : Images)
    {
        if (!image.Pixels.HasItems())
            continue;
        image.Texture = GPUDevice::Instance->CreateTexture(TEXT("SkinnedGltfTex"));
        image.Texture->Init(GPUTextureDescription::New2D(
            image.Width,
            image.Height,
            PixelFormat::R8G8B8A8_UNorm,
            GPUTextureFlags::ShaderResource));
        const uint32 rowPitch = (uint32)image.Width * 4;
        const uint32 slicePitch = rowPitch * (uint32)image.Height;
        context->UpdateTexture(image.Texture, 0, 0, image.Pixels.Get(), rowPitch, slicePitch);
        image.Pixels.Clear();
    }

    for (Skin& skin : Skins)
    {
        if (!skin.Joints.HasItems())
            continue;
        skin.JointBuffer = GPUDevice::Instance->CreateBuffer(TEXT("SkinnedGltfJoints"));
        skin.JointBuffer->Init(GPUBufferDescription::Buffer(
            skin.Joints.Count() * sizeof(Matrix),
            GPUBufferFlags::Structured | GPUBufferFlags::ShaderResource,
            PixelFormat::Unknown,
            nullptr,
            sizeof(Matrix)));
    }

    _gpuReady = true;
}

void SaschaSkinnedGltf::ReleaseGpuResources()
{
    for (Image& image : Images)
        SAFE_DELETE_GPU_RESOURCE(image.Texture);
    SAFE_DELETE_GPU_RESOURCE(VertexBuffer);
    SAFE_DELETE_GPU_RESOURCE(IndexBuffer);
    for (Skin& skin : Skins)
        SAFE_DELETE_GPU_RESOURCE(skin.JointBuffer);
    _gpuReady = false;
}

glm::mat4 SaschaSkinnedGltf::GetNodeMatrix(Node* node) const
{
    glm::mat4 nodeMatrix = node->GetLocalMatrix();
    Node* parent = node->Parent;
    while (parent)
    {
        nodeMatrix = parent->GetLocalMatrix() * nodeMatrix;
        parent = parent->Parent;
    }
    return nodeMatrix;
}

void SaschaSkinnedGltf::UpdateJoints(Node* node)
{
    if (node->SkinIndex > -1 && node->SkinIndex < Skins.Count())
    {
        Skin& skin = Skins[node->SkinIndex];
        const glm::mat4 inverseTransform = glm::inverse(GetNodeMatrix(node));

        const int32 jointCount = skin.Joints.Count();
        Array<Matrix> jointMatrices;
        jointMatrices.Resize(jointCount);
        for (int32 i = 0; i < jointCount; i++)
        {
            const glm::mat4 jointMatrix = inverseTransform * GetNodeMatrix(skin.Joints[i]) * skin.InverseBindMatrices[i];
            SaschaGpu::StoreTransposed(jointMatrices[i], jointMatrix);
        }

        if (skin.JointBuffer)
            GPUDevice::Instance->GetMainContext()->UpdateBuffer(skin.JointBuffer, jointMatrices.Get(), jointMatrices.Count() * sizeof(Matrix));
    }

    for (Node* child : node->Children)
        UpdateJoints(child);
}

void SaschaSkinnedGltf::UpdateAnimation(float deltaTime)
{
    if (Animations.Count() == 0 || ActiveAnimation >= (uint32)Animations.Count())
        return;

    Animation& animation = Animations[ActiveAnimation];
    animation.CurrentTime += deltaTime;
    if (animation.End > animation.Start && animation.CurrentTime > animation.End)
        animation.CurrentTime -= (animation.End - animation.Start);

    for (AnimationChannel& channel : animation.Channels)
    {
        if (!channel.TargetNode)
            continue;
        AnimationSampler& sampler = animation.Samplers[channel.SamplerIndex];
        if (sampler.Interpolation != String("LINEAR"))
            continue;

        for (int32 i = 0; i < sampler.Inputs.Count() - 1; i++)
        {
            if (animation.CurrentTime < sampler.Inputs[i] || animation.CurrentTime > sampler.Inputs[i + 1])
                continue;

            const float range = sampler.Inputs[i + 1] - sampler.Inputs[i];
            const float a = range > 0.0f ? (animation.CurrentTime - sampler.Inputs[i]) / range : 0.0f;
            if (channel.Path == String("translation"))
            {
                const Float4& o0 = sampler.Outputs[i];
                const Float4& o1 = sampler.Outputs[i + 1];
                channel.TargetNode->Translation = glm::mix(glm::vec3(o0.X, o0.Y, o0.Z), glm::vec3(o1.X, o1.Y, o1.Z), a);
            }
            else if (channel.Path == String("rotation"))
            {
                const Float4& o0 = sampler.Outputs[i];
                const Float4& o1 = sampler.Outputs[i + 1];
                const glm::quat q1(o0.W, o0.X, o0.Y, o0.Z);
                const glm::quat q2(o1.W, o1.X, o1.Y, o1.Z);
                channel.TargetNode->Rotation = glm::normalize(glm::slerp(q1, q2, a));
            }
            else if (channel.Path == String("scale"))
            {
                const Float4& o0 = sampler.Outputs[i];
                const Float4& o1 = sampler.Outputs[i + 1];
                channel.TargetNode->Scale = glm::mix(glm::vec3(o0.X, o0.Y, o0.Z), glm::vec3(o1.X, o1.Y, o1.Z), a);
            }
            break;
        }
    }

    for (Node* node : Nodes)
        UpdateJoints(node);
}

void SaschaSkinnedGltf::DrawNode(GPUContext* context, GPUConstantBuffer* nodeCb, Node* node) const
{
    if (node->MeshData.Primitives.HasItems())
    {
        glm::mat4 nodeMatrix = node->Matrix;
        Node* parent = node->Parent;
        while (parent)
        {
            nodeMatrix = parent->Matrix * nodeMatrix;
            parent = parent->Parent;
        }

        Matrix gpuNodeMatrix;
        SaschaGpu::StoreTransposed(gpuNodeMatrix, nodeMatrix);
        context->UpdateCB(nodeCb, &gpuNodeMatrix);

        if (node->SkinIndex > -1 && node->SkinIndex < Skins.Count())
        {
            const Skin& skin = Skins[node->SkinIndex];
            if (skin.JointBuffer)
                context->BindSR(1, skin.JointBuffer->View());
        }

        for (const Primitive& primitive : node->MeshData.Primitives)
        {
            if (primitive.IndexCount == 0)
                continue;

            GPUTexture* colorMap = nullptr;
            if (primitive.MaterialIndex >= 0 && primitive.MaterialIndex < Materials.Count())
            {
                const Material& material = Materials[primitive.MaterialIndex];
                if (material.BaseColorTextureIndex >= 0 && material.BaseColorTextureIndex < (int32)Textures.Count())
                {
                    const int32 imageIndex = Textures[material.BaseColorTextureIndex].ImageIndex;
                    if (imageIndex >= 0 && imageIndex < Images.Count())
                        colorMap = Images[imageIndex].Texture;
                }
            }
            if (colorMap)
                context->BindSR(0, colorMap->View());

            context->DrawIndexed(primitive.IndexCount, primitive.FirstIndex, 0);
        }
    }

    for (Node* child : node->Children)
        DrawNode(context, nodeCb, child);
}

void SaschaSkinnedGltf::Draw(GPUContext* context, GPUConstantBuffer* sceneCb, GPUConstantBuffer* nodeCb) const
{
    if (!VertexBuffer || !IndexBuffer)
        return;

    GPUBuffer* vb = VertexBuffer;
    Span<GPUBuffer*> bindVb(&vb, 1);
    context->BindVB(bindVb, nullptr, VertexLayout);
    context->BindIB(IndexBuffer);
    context->BindCB(0, sceneCb);
    context->BindCB(1, nodeCb);

    for (Node* node : Nodes)
        DrawNode(context, nodeCb, node);
}
