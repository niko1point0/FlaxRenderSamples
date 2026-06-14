// Copyright (c) Flax Engine. Sample: exit on Escape (C++).

#pragma once

#include "Engine/Scripting/Script.h"

/// <summary>
/// Requests application exit when the Escape key is released.
/// </summary>
API_CLASS(Namespace="") class ExitOnEsc : public Script
{
    API_AUTO_SERIALIZATION();
    DECLARE_SCRIPTING_TYPE(ExitOnEsc);

    // [Script]
    API_FUNCTION() void OnUpdate() override;
};
