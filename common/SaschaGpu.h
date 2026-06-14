// GPU matrix upload: CPU math is GLM (Vulkan); upload convention depends on the Flax shader.

#pragma once

#include "Engine/Core/Math/Matrix.h"
#include "SaschaGlm.h"

#include <cstring>

namespace SaschaGpu
{
    /// Copy glm matrix into Flax row-major Matrix (mathematical element row r, col c).
    inline void GlmToFlaxMatrix(Matrix& dst, const glm::mat4& src)
    {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                dst.Values[r][c] = src[c][r];
    }

    /// <summary>
    /// Vulkan memcpy upload. Use with column-vector shaders: mul(Matrix, float4).
    /// glm column-major memory maps to HLSL column-major float4x4 when stored in Flax Matrix.
    /// </summary>
    inline void StoreVulkan(Matrix& dst, const glm::mat4& src)
    {
        std::memcpy(&dst, &src, sizeof(Matrix));
    }

    /// <summary>
    /// Flax row-vector upload: mul(float4, Matrix). Stores glm matrix rows in Flax row-major layout.
    /// </summary>
    inline void StoreRowVector(Matrix& dst, const glm::mat4& src)
    {
        GlmToFlaxMatrix(dst, src);
    }

    inline void Store(Matrix& dst, const glm::mat4& src) { StoreVulkan(dst, src); }
    inline void StoreTransposed(Matrix& dst, const glm::mat4& src) { StoreRowVector(dst, src); }
}
