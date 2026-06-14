#pragma once

#include "Engine/Content/Assets/Model.h"
#include "Engine/Graphics/GPUContext.h"
#include "Engine/Graphics/Shaders/GPUVertexLayout.h"

/// <summary>Draws all meshes from a loaded Flax Model (LOD0) with the current GPU pipeline state.</summary>
class SaschaModelDraw
{
public:
    Model* ModelAsset = nullptr;
    GPUVertexLayout* VertexLayout = nullptr;

    bool Load(const Char* contentPath);
    void Draw(GPUContext* context) const;
};
