// GLM configuration matching Sascha Willems Vulkan examples (base/camera.hpp, etc.).

#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "Engine/Core/Math/Quaternion.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Core/Math/Vector4.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace SaschaGlm
{
    /// <summary>
    /// glm::perspective matches Vulkan/OpenGL clip space (Y up); Flax uses D3D (Y down).
    /// Negate the Y scale so Sascha samples look correct when rendered in Flax.
    /// </summary>
    inline glm::mat4 PerspectiveForFlax(float fovDegrees, float aspect, float zNear, float zFar)
    {
        glm::mat4 projection = glm::perspective(glm::radians(fovDegrees), aspect, zNear, zFar);
        projection[1][1] *= -1.0f;
        return projection;
    }

    inline void FlipProjectionYForFlax(glm::mat4& projection)
    {
        projection[1][1] *= -1.0f;
    }

    inline glm::vec3 ToGlm(const Float3& v) { return glm::vec3(v.X, v.Y, v.Z); }
    inline glm::vec4 ToGlm(const Float4& v) { return glm::vec4(v.X, v.Y, v.Z, v.W); }
    inline Float3 ToFloat3(const glm::vec3& v) { return Float3(v.x, v.y, v.z); }
    inline Float4 ToFloat4(const glm::vec4& v) { return Float4(v.x, v.y, v.z, v.w); }

    inline glm::vec3 TransformPoint(const glm::mat4& m, const glm::vec3& p)
    {
        return glm::vec3(m * glm::vec4(p, 1.0f));
    }

    inline glm::vec3 TransformPoint(const glm::mat4& m, const Float3& p)
    {
        return TransformPoint(m, ToGlm(p));
    }

    inline glm::vec3 TransformVector(const glm::mat4& m, const glm::vec3& v)
    {
        return glm::vec3(m * glm::vec4(v, 0.0f));
    }

    inline glm::vec4 TransformPoint4(const glm::mat4& m, const glm::vec4& p)
    {
        return m * p;
    }

    inline glm::quat ToGlm(const Quaternion& q)
    {
        return glm::quat(q.W, q.X, q.Y, q.Z);
    }

    /// <summary>View matrix with translation removed (skybox rendering, matches Vulkan examples).</summary>
    /// Vulkan skybox pattern: glm::mat4(glm::mat3(camera.matrices.view))
    inline glm::mat4 ViewRotationOnly(const glm::mat4& view)
    {
        return glm::mat4(glm::mat3(view));
    }

    inline glm::mat4 ViewWithoutTranslation(const glm::mat4& view)
    {
        return ViewRotationOnly(view);
    }

    inline glm::vec3 Translation(const glm::mat4& m)
    {
        return glm::vec3(m[3]);
    }
}
