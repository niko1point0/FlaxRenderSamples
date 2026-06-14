// Port of Sascha Willems Vulkan/base/frustum.hpp.

#pragma once

#include "SaschaGlm.h"

#include <array>
#include <cmath>

class SaschaFrustum
{
public:
    enum class Side
    {
        Left = 0,
        Right = 1,
        Top = 2,
        Bottom = 3,
        Back = 4,
        Front = 5,
    };

    std::array<glm::vec4, 6> planes;

    void update(const glm::mat4& matrix)
    {
        planes[(size_t)Side::Left].x = matrix[0].w + matrix[0].x;
        planes[(size_t)Side::Left].y = matrix[1].w + matrix[1].x;
        planes[(size_t)Side::Left].z = matrix[2].w + matrix[2].x;
        planes[(size_t)Side::Left].w = matrix[3].w + matrix[3].x;

        planes[(size_t)Side::Right].x = matrix[0].w - matrix[0].x;
        planes[(size_t)Side::Right].y = matrix[1].w - matrix[1].x;
        planes[(size_t)Side::Right].z = matrix[2].w - matrix[2].x;
        planes[(size_t)Side::Right].w = matrix[3].w - matrix[3].x;

        planes[(size_t)Side::Top].x = matrix[0].w - matrix[0].y;
        planes[(size_t)Side::Top].y = matrix[1].w - matrix[1].y;
        planes[(size_t)Side::Top].z = matrix[2].w - matrix[2].y;
        planes[(size_t)Side::Top].w = matrix[3].w - matrix[3].y;

        planes[(size_t)Side::Bottom].x = matrix[0].w + matrix[0].y;
        planes[(size_t)Side::Bottom].y = matrix[1].w + matrix[1].y;
        planes[(size_t)Side::Bottom].z = matrix[2].w + matrix[2].z;
        planes[(size_t)Side::Bottom].w = matrix[3].w + matrix[3].y;

        planes[(size_t)Side::Back].x = matrix[0].w + matrix[0].z;
        planes[(size_t)Side::Back].y = matrix[1].w + matrix[1].z;
        planes[(size_t)Side::Back].z = matrix[2].w + matrix[2].z;
        planes[(size_t)Side::Back].w = matrix[3].w + matrix[3].z;

        planes[(size_t)Side::Front].x = matrix[0].w - matrix[0].z;
        planes[(size_t)Side::Front].y = matrix[1].w - matrix[1].z;
        planes[(size_t)Side::Front].z = matrix[2].w - matrix[2].z;
        planes[(size_t)Side::Front].w = matrix[3].w - matrix[3].z;

        for (glm::vec4& plane : planes)
        {
            const float length = std::sqrt(plane.x * plane.x + plane.y * plane.y + plane.z * plane.z);
            if (length > 1e-6f)
                plane /= length;
        }
    }

    bool checkSphere(const glm::vec3& pos, float radius) const
    {
        for (const glm::vec4& plane : planes)
        {
            if (plane.x * pos.x + plane.y * pos.y + plane.z * pos.z + plane.w <= -radius)
                return false;
        }
        return true;
    }
};
