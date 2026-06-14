#include "SaschaModelDraw.h"
#include "SaschaContent.h"
#include "Engine/Content/Content.h"
#include "Engine/Graphics/GPUBuffer.h"
#include "Engine/Graphics/Models/MeshBase.h"

bool SaschaModelDraw::Load(const Char* contentPath)
{
    ModelAsset = Content::Load<Model>(SaschaContent::FlaxAssetPath(contentPath));
    if (ModelAsset == nullptr || ModelAsset->GetLODsCount() == 0)
        return false;

    const MeshBase* mesh = ModelAsset->GetMesh(0, 0);
    if (mesh == nullptr)
        return false;

    GPUBuffer* vb = mesh->GetVertexBuffer(0);
    if (vb == nullptr)
        return false;

    VertexLayout = vb->GetVertexLayout();
    return VertexLayout != nullptr;
}

void SaschaModelDraw::Draw(GPUContext* context) const
{
    if (ModelAsset)
        ModelAsset->Render(context, 0);
}
