// Port of Sascha Willems Vulkan/base/camera.hpp + vulkanexamplebase mouse controls.

#include "SaschaCamera.h"
#include "Engine/Core/Math/Vector2.h"
#include "Engine/Input/Enums.h"
#include "Engine/Input/Input.h"

SaschaCamera::SaschaCamera()
{
    updateViewMatrix();
}

void SaschaCamera::setPerspective(float fovDegrees, float aspect, float zNear, float zFar)
{
    const glm::mat4 currentMatrix = matrices.perspective;
    _fov = fovDegrees;
    _zNear = zNear;
    _zFar = zFar;
    matrices.perspective = SaschaGlm::PerspectiveForFlax(fovDegrees, aspect, zNear, zFar);
    if (matrices.perspective != currentMatrix)
        updated = true;
}

void SaschaCamera::updateAspectRatio(float aspect)
{
    const glm::mat4 currentMatrix = matrices.perspective;
    matrices.perspective = SaschaGlm::PerspectiveForFlax(_fov, aspect, _zNear, _zFar);
    if (matrices.perspective != currentMatrix)
        updated = true;
}

void SaschaCamera::setPosition(const glm::vec3& pos)
{
    position = pos;
    updateViewMatrix();
}

void SaschaCamera::setRotation(const glm::vec3& rot)
{
    rotation = rot;
    updateViewMatrix();
}

void SaschaCamera::rotate(const glm::vec3& delta)
{
    rotation += delta;
    updateViewMatrix();
}

void SaschaCamera::translate(const glm::vec3& delta)
{
    position += delta;
    updateViewMatrix();
}

glm::mat4 SaschaCamera::Perspective(float fovDegrees, float aspect, float zNear, float zFar)
{
    return SaschaGlm::PerspectiveForFlax(fovDegrees, aspect, zNear, zFar);
}

void SaschaCamera::updateViewMatrix()
{
    const glm::mat4 currentMatrix = matrices.view;

    glm::mat4 rotM = glm::mat4(1.0f);
    rotM = glm::rotate(rotM, glm::radians(rotation.x * (flipY ? -1.0f : 1.0f)), glm::vec3(1.0f, 0.0f, 0.0f));
    rotM = glm::rotate(rotM, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    rotM = glm::rotate(rotM, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

    glm::vec3 translation = position;
    if (flipY)
        translation.y *= -1.0f;
    const glm::mat4 transM = glm::translate(glm::mat4(1.0f), translation);

    if (type == Type::FirstPerson)
        matrices.view = rotM * transM;
    else
        matrices.view = transM * rotM;

    viewPos = glm::vec4(position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

    if (matrices.view != currentMatrix)
        updated = true;
}

void SaschaCamera::HandleMouseInput()
{
    // Sascha: dx = prev.x - x; Flax delta is current - prev.
    const Float2 delta = Input::GetMousePositionDelta();
    const float dx = -delta.X;
    const float dy = -delta.Y;

    if (Input::GetMouseButton(MouseButton::Left))
        rotate(glm::vec3(dy * rotationSpeed, -dx * rotationSpeed, 0.0f));
    if (Input::GetMouseButton(MouseButton::Right))
        translate(glm::vec3(0.0f, 0.0f, dy * 0.005f));
    if (Input::GetMouseButton(MouseButton::Middle))
        translate(glm::vec3(-dx * 0.005f, -dy * 0.005f, 0.0f));

    const float wheel = Input::GetMouseScrollDelta();
    if (wheel != 0.0f)
        translate(glm::vec3(0.0f, 0.0f, wheel * 0.005f));
}
