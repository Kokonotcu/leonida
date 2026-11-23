# Ahead-of-time GPU code: no shader or pipeline compiles while the game runs

**Status: local work branch (`leonida/work`). Steps 1 and 2 are implemented (step 2 behind `LEONIDA_SHADER_OBJECTS=1`,
graphics and compute); step 3 is the plan.**

How to use it today:

```powershell
gta6rt.exe --prebuild-shaders            # once per driver: every translated shader -> shader_binaries\
$env:LEONIDA_SHADER_OBJECTS = "1"        # the run then creates its shaders from those binaries
gta6rt.exe --run game_scarlett_master.exe --game-dir <package>
gta6rt.exe --prebuild-pipelines          # without shader objects: compile what the last run recorded
```

`make_game.py`'s pipelines stage runs both prebuilds. Checked on an RTX 3080 Laptop (NVIDIA 617.42) by
`shader_object_draw_test`: the shader-object draw writes the pipeline draw's pixels, and after `--prebuild-shaders`
a new device makes its shaders from the stored binaries (0 compiled from SPIR-V) and draws the same pixels.

The game's CPU code needs no translation: the Xbox Series executable is x86-64 and `gta6rt` maps and runs it as is.
Its GPU programs are RDNA2 machine code; the build translates all of them to SPIR-V once (`make_game.py`, shaders
stage). What is still left to run time is the driver's half: turning SPIR-V plus fixed-function state into the
GPU's own machine code, once per pipeline. On a cold cache that is a compile on the draw that first needs it, which
the player sees as a hitch. The aim of this page is that the game run compiles nothing.

## 1. Record every pipeline, prebuild it before the next run (implemented)

- `runtime/src/pso_recipe.h` holds a graphics pipeline's whole Vulkan description as plain data (a *recipe*):
  the stages by translator index name plus a hash of the exact SPIR-V words, vertex input, input assembly,
  rasterization (with the conservative chain), multisample, depth-stencil, colour blend, the dynamic-rendering
  formats and the pipeline layout (one push-constant range, set 0 when bindless).
- `GraphicsCreateInfo` turns a recipe into a `VkGraphicsPipelineCreateInfo`. The runtime builds its pipelines
  through it and the prebuild replays through it, so the two create infos cannot differ: a pipeline cache is keyed
  by the whole create info, and one differing field is a miss.
- The runtime appends each new recipe to `pso_recipes.jsonl` beside `gta6rt.exe` (first line: the set-0 layout,
  so a file from another device or another bindless sizing is started again rather than replayed wrongly).
- `gta6rt --prebuild-pipelines` creates the Vulkan device exactly as a game run does (same extensions, features
  and robustness, all of which change the compiled code), builds every recorded pipeline into the device's
  pipeline cache and saves it. `make_game.py`'s pipelines stage runs it when recipes exist.

What it does not cover: a pipeline the game has never created on this machine. The first run still compiles
those; every later run does not.

## 2. Compile every shader once at build time (plan: shader objects)

`VK_EXT_shader_object` compiles each stage on its own, with all fixed-function state set by commands at draw
time, and `vkGetShaderBinaryDataEXT` returns the driver's compiled binary. So the build can compile every
translated vertex, mesh, geometry and pixel shader into the driver's machine code, store the binaries, and the
runtime creates its shaders from them (`VK_SHADER_CODE_TYPE_BINARY_EXT`): no compile and no link at run time,
even on the first run. A binary carries the driver's `shaderBinaryUUID` and `shaderBinaryVersion`; a driver
update makes the runtime fall back to SPIR-V for that run and the next build regenerates them.

Requirements on the runtime side: every state the recipe holds must be set per draw (depth/stencil, raster,
blend per attachment and colour write mask through `VK_EXT_extended_dynamic_state3`, vertex input through
`VK_EXT_vertex_input_dynamic_state`), `vkCmdBindShadersEXT` instead of `vkCmdBindPipeline`. The recipe already
holds exactly that state, so the replay can emit it from the recipe. Behind `LEONIDA_SHADER_OBJECTS=1` until
it is checked against the pipeline path draw for draw.

Measured support (RTX 3080 Laptop, NVIDIA 617.42, Windows): `shaderObject = true`, `shaderBinaryVersion = 1`,
`VK_EXT_extended_dynamic_state3`, `VK_EXT_vertex_input_dynamic_state`, `VK_EXT_mesh_shader`, and
`graphicsPipelineLibrary` with `graphicsPipelineLibraryFastLinking = true` (the fallback in step 3).

## 3. Fallback for devices without shader objects: pipeline libraries

`VK_EXT_graphics_pipeline_library` splits a pipeline into vertex input, pre-rasterization shaders, fragment
shader and fragment output. With the state those parts take made dynamic, the pre-rasterization and fragment
shader libraries depend only on the shader and the layout, so the build compiles one per shader into the
pipeline cache and the runtime fast-links (no optimisation pass) at draw time.

## 4. The game's own pipeline list

The package's PSO cache (`docs/pso_cache.md`) lists 25,471 graphics and 2,693 compute pipelines with their stage
and input-layout records. Once its blend and render-target fields are decoded it can produce recipes for step 1
before any run. Steps 2 and 3 do not need it.
