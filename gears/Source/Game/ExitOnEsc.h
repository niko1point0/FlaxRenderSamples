// Copyright (c) Flax Engine. Sample: exit on Escape (C++).

#pragma once

#include "Engine/Scripting/Script.h"

API_CLASS(Namespace="") class ExitOnEsc : public Script
{
    API_AUTO_SERIALIZATION();
    DECLARE_SCRIPTING_TYPE(ExitOnEsc);

    API_FUNCTION() void OnUpdate() override;
};
