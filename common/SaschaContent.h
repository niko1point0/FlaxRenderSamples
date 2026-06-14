#pragma once

#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Types/String.h"
#include "Engine/Engine/Globals.h"

namespace SaschaContent
{
    /// Appends ".flax" when @param relativePath has no file extension (Flax content asset id).
    inline String FlaxAssetPath(const Char* relativePath)
    {
        String path(relativePath);
        const int32 slash = Math::Max(path.FindLast('/'), path.FindLast('\\'));
        const String fileName = slash >= 0 ? path.Substring(slash + 1) : path;
        if (fileName.Find('.') < 0)
            path += TEXT(".flax");
        return Globals::ProjectContentFolder / path;
    }
}
