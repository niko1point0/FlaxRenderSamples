// Port of Sascha Willems Vulkan/base/camera.hpp (lookat camera).

#pragma once

#include "SaschaGlm.h"

/// <summary>
/// Sascha Willems lookat camera. Matrix math is GLM-only; matches Vulkan camera.hpp.
/// </summary>
class SaschaCamera
{
public:
    enum class Type
    {
        Lookat,
        FirstPerson,
    };

    Type type = Type::Lookat;

    glm::vec3 rotation = glm::vec3(0.0f);
    glm::vec3 position = glm::vec3(0.0f, 0.0f, -2.5f);
    glm::vec4 viewPos = glm::vec4(0.0f);
    float rotationSpeed = 1.0f;
    float movementSpeed = 1.0f;
    bool updated = true;
    bool flipY = false;

    struct
    {
        glm::mat4 perspective = glm::mat4(1.0f);
        glm::mat4 view = glm::mat4(1.0f);
    } matrices;

    SaschaCamera();

    void setPerspective(float fovDegrees, float aspect, float zNear, float zFar);
    void updateAspectRatio(float aspect);
    void setPosition(const glm::vec3& pos);
    void setRotation(const glm::vec3& rot);
    void rotate(const glm::vec3& delta);
    void translate(const glm::vec3& delta);

    /// <summary>Left drag = rotate, right = zoom, middle = pan, wheel = zoom (Flax input).</summary>
    void HandleMouseInput();

    /// <summary>One-shot perspective without mutating camera clip planes.</summary>
    static glm::mat4 Perspective(float fovDegrees, float aspect, float zNear, float zFar);

    /// <summary>Match Vulkan camera.matrices.perspective each frame after setPerspective in OnStart.</summary>
    void syncProjection(float aspect) { updateAspectRatio(aspect); }

    /// <summary>Matches Vulkan camera.viewPos / camPos upload (position with x,z negated).</summary>
    Float4 GetViewPos() const { return SaschaGlm::ToFloat4(viewPos); }

private:
    float _fov = 60.0f;
    float _zNear = 0.1f;
    float _zFar = 256.0f;

    void updateViewMatrix();
};
