// Copyright (c) Flax Engine. Sample: exit on Escape (C++).

#include "ExitOnEsc.h"
#include "Engine/Engine/Engine.h"
#include "Engine/Input/Input.h"

ExitOnEsc::ExitOnEsc(const SpawnParams& params)
    : Script(params)
{
    _tickUpdate = true;
}

void ExitOnEsc::OnUpdate()
{
    if (Input::GetKeyUp(KeyboardKeys::Escape))
        Engine::RequestExit();
}
