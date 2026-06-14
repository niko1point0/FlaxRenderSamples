# Sascha Willems' Vulkan Samples — Ported to Flax

> **Credits to [Sascha Willems](https://github.com/SaschaWillems/vulkan) for writing the original Vulkan samples.**
> This repository ports a selection of them to the [Flax Engine](https://flaxengine.com/) while keeping the original rendering logic (camera math, GLSL-equivalent shaders, glTF scenes) intact.

A collection of classic real-time rendering samples — a spinning triangle, animated gears, omni/directional shadow mapping, and hardware ray-traced shadows — rebuilt as standalone Flax projects.

## Objective

- **Learn the Flax graphics abstraction layer** by porting well-understood samples to it.
- **Build tech demos that auto-port across backends** — the same code targets DirectX 11, DirectX 12, Vulkan, and WebGPU.
- **Feed the lessons back into the engine** — use the experience gained here to potentially extend or modify the full Flax renderer.

## Sample Compatibility

Every sample targets **Flax 1.13**. Some need a small engine patch applied on top of 1.13. The patches ship inside **`FlaxPatches.zip`** in this repository — see [Applying a Patch](#applying-a-patch).

| Sample | Patch required | What it shows |
|--------|----------------|---------------|
| `triangle` | — | Minimal "hello triangle" |
| `gears` | — | Animated mechanical gears |
| `shadowmapping` | — | Directional/spot shadow maps |
| `shadowmappingomni` | — | Omni-directional (point light) cube shadow maps |
| `texture` | `KTX.diff` | KTX texture loading |
| `parallaxmapping` | `KTX.diff` | Parallax-occlusion mapping |
| `shadowmappingAddRT` | `HWRT_D3D12_VK.diff` | Hardware ray-traced shadows (directional) |
| `shadowmappingomniAddRT` | `HWRT_D3D12_VK.diff` | Hardware ray-traced omni shadows |

- **No patch** → runs on a stock Flax 1.13 build.
- **`KTX.diff`** → adds KTX texture support: `texture`, `parallaxmapping`.
- **`HWRT_D3D12_VK.diff`** → adds hardware ray tracing on DirectX 12 and Vulkan: `shadowmappingAddRT`, `shadowmappingomniAddRT`. Running these on **Vulkan** also needs a SPIR-V-enabled DXC — see [Ray Tracing on Vulkan: SPIR-V Shader Compiler](#ray-tracing-on-vulkan-spir-v-shader-compiler).

## Requirements

- Windows x64
- Flax Engine **1.13** (built from source — see below)
- For the ray tracing samples: a GPU + driver supporting DirectX 12 or Vulkan hardware ray tracing
- For the ray tracing samples on **Vulkan**: a SPIR-V-enabled DirectX Shader Compiler (see [Ray Tracing on Vulkan: SPIR-V Shader Compiler](#ray-tracing-on-vulkan-spir-v-shader-compiler))

## Getting Flax 1.13

Clone the engine and **check out the `1.13` branch** (the `-b 1.13` flag is required to get the right branch):

```bash
git clone -b 1.13 https://github.com/FlaxEngine/FlaxEngine.git
```

## Applying a Patch

Some samples require an engine patch. The patches are distributed as **`FlaxPatches.zip`** in this repository, which contains `KTX.diff` and `HWRT_D3D12_VK.diff`.

> **Why a zip?** A `.diff` is byte-sensitive — Git apply fails if even the line endings change. When raw `.diff` files are committed to GitHub, Git's automatic line-ending normalization (and, for some files, LFS handling) rewrites them on checkout, so the copy you clone no longer matches the original and `git apply` rejects it. Packing them inside `FlaxPatches.zip` keeps the bytes exactly as authored, so the extracted files apply cleanly. **Unzip `FlaxPatches.zip` first — don't try to use any loose `.diff` from the repo tree.**

> **Important:** these `.diff` files patch the **[FlaxEngine](https://github.com/FlaxEngine/FlaxEngine) repository (branch `1.13`)** — *not* this samples repository. Apply them inside your cloned engine source tree, then rebuild the engine.

Extract `FlaxPatches.zip`, copy the `.diff` you need into the root of your Flax 1.13 engine clone (or reference it by full path), apply it there, then rebuild the engine:

```bash
# 1. Unzip FlaxPatches.zip to get KTX.diff and HWRT_D3D12_VK.diff.
# 2. Run these from inside your FlaxEngine clone (the 1.13 branch), NOT from the samples repo.

# KTX texture support — needed for: texture, parallaxmapping
git apply KTX.diff

# Hardware ray tracing (DirectX 12 / Vulkan) — needed for: shadowmappingAddRT, shadowmappingomniAddRT
git apply HWRT_D3D12_VK.diff
```

> Apply only the patch(es) for the sample(s) you intend to run. The patches are independent.

## Ray Tracing on Vulkan: SPIR-V Shader Compiler

> Only needed for the ray tracing samples (`shadowmappingAddRT`, `shadowmappingomniAddRT`) **when running on Vulkan**. DirectX 12 works without this step.

The ray tracing samples compile their inline ray-query shaders with the DirectX Shader Compiler (DXC). On Vulkan these shaders must be emitted as **SPIR-V**, but the `dxcompiler.dll` that ships with the Windows SDK is built **without** SPIR-V code generation. Using it produces:

```
Failed to compile 'Shadowmapping'. SPIR-V CodeGen not available. Please recompile with -DENABLE_SPIRV_CODEGEN=ON.
```

To fix this, swap in a SPIR-V-enabled build of DXC:

1. Download the latest release from the official [DirectX Shader Compiler releases](https://github.com/microsoft/DirectXShaderCompiler/releases).
2. From the downloaded package, open **`bin/x64`**.
3. Copy **`dxcompiler.dll`** and **`dxil.dll`** into your engine's third-party binaries folder:

```
FlaxEngine\Source\Platforms\Windows\Binaries\ThirdParty\x64
```

Overwrite the existing files, then rebuild/relaunch the editor. DirectX 12 ray tracing is unaffected by this swap.

## Running a Sample

1. Build the **Flax Editor** from your 1.13 engine source.
2. Open the sample as a project in the Flax Editor (open its `*.flaxproj`).
3. In the **Content Browser**, search for **`Main`** and open the **`Main`** scene.
4. Press **Play** in the editor.

You can also **cook** the project to produce a standalone build.

## Debugging the Ray Tracing Samples

The ray tracing demos (`shadowmappingAddRT`, `shadowmappingomniAddRT`) can only be GPU-debugged with **NVIDIA Nsight Graphics**. They will **not** work in RenderDoc.
