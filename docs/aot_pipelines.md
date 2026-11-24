# Ahead-of-time GPU code: no shader or pipeline compiles while the game runs

**Status: local work branch (`leonida/work`). Steps 1, 2 and 3 are implemented (step 2 behind
`LEONIDA_SHADER_OBJECTS=1`, graphics and compute; step 3, graphics, on by default where shader objects are not in use).**

How to use it today:

```powershell
gta6rt.exe --prebuild-shaders            # once per driver: every translated shader -> shader_binaries\ (shader
                                         # objects), or into the pipeline cache as pipeline-library parts (without)
$env:LEONIDA_SHADER_OBJECTS = "1"        # the run then creates its shaders from those binaries
gta6rt.exe --run game_scarlett_master.exe --game-dir <package>
gta6rt.exe --prebuild-pipelines          # whole pipelines: compile what the last run recorded
```

Switches: `LEONIDA_SHADER_OBJECTS=1` (shader objects, off by default; `=0` also keeps `--prebuild-shaders` off them),
`LEONIDA_GPL=0/1` (graphics PSOs linked from pipeline libraries, step 3: on by default where shader objects are not in
use and the device has `graphicsPipelineLibraryFastLinking`; `0` builds whole pipelines, `1` links even without fast
linking), `LEONIDA_SHADER_BINARIES=<dir>` (where the binaries are, default `shader_binaries` beside `gta6rt.exe`),
`LEONIDA_PSO_RECORD=0` (record no recipes), `LEONIDA_PSO_RECIPES=<file>` (record elsewhere than `pso_recipes.jsonl`),
and `LEONIDA_DRAW_SCOPES=0` (one rendering scope per draw again instead of one for consecutive draws into the same
attachments; vulkan_replay.h emit).

`make_game.py`'s pipelines stage runs both prebuilds. Checked on an RTX 3080 Laptop (NVIDIA 617.42) by
`shader_object_draw_test`: the shader-object draw writes the pipeline draw's pixels, and after `--prebuild-shaders`
a new device makes its shaders from the stored binaries (0 compiled from SPIR-V) and draws the same pixels. And by
`gpl_draw_test` for step 3 (below).

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

## 3. Fallback for devices without shader objects: pipeline libraries (implemented)

`VK_EXT_graphics_pipeline_library` splits a pipeline into vertex input, pre-rasterization shaders, fragment
shader and fragment output. With the state those parts take made dynamic, the pre-rasterization and fragment
shader libraries depend only on the shader and the layout, so the build compiles one per shader into the
pipeline cache and the runtime fast-links (no optimisation pass) when the PSO is created. For AMD's and Intel's
Windows drivers, which have no `VK_EXT_shader_object`.

- `runtime/src/pso_library.h` makes each part's create info from the PSO's recipe (`LibraryCreateInfo`, the
  counterpart of step 1's `GraphicsCreateInfo`, so the runtime and the prebuild ask for the very same parts):
  - vertex input: the recipe's bindings, attributes, divisors and topology class. Made per PSO (no shader: cheap).
  - pre-rasterization shaders: the vertex shader (and a geometry shader), keyed by the shaders' SPIR-V hash, flags
    and subgroup request, the depth clamp (`DepthClipEnable = FALSE`), the fill mode and conservative rasterization,
    which need `VK_EXT_extended_dynamic_state3` to be dynamic. Shared by every PSO with that key.
  - fragment shader: the pixel shader, keyed the same way; the multisample state only for a shader that runs per
    sample (`Sample` inputs, `SampleId`, `SamplePosition`, or sample shading), which the spec requires
    (VUID-VkGraphicsPipelineCreateInfo-Input-06632; any other fragment part must not carry it, since it would have
    to equal the output part's, -06635). A PSO without a pixel shader shares one part with no stage.
  - fragment output: blend state, multisample state and attachment formats. Made per PSO, and per unbound-slot
    variant.
- Dynamic in these pipelines, on top of step 1's list, and set at every draw from the recipe
  (`GraphicsPipeline::set_state`, `LibraryState`): cull mode, front face, depth bias enable and values, rasterizer
  discard, line width, depth test, write, compare and bounds, stencil test, ops and masks. All core Vulkan 1.3 (or
  `VK_EXT_extended_dynamic_state(2)` on 1.2), so no new feature is needed beyond `graphicsPipelineLibrary`.
- Every part and every PSO uses one layout shape: a push-constant range over every graphics stage the device has
  (as the shader objects', step 2) and set 0 whenever the device has it, because the two shader parts of a link
  must have identically defined layouts (VUID-VkGraphicsPipelineCreateInfo-pLibraries-06613).
- The PSO links its four parts without `LINK_TIME_OPTIMIZATION` first (`LinkCreateInfo`, `Link::fast`): that is
  cheap, but a fast-linked pipeline can cost GPU time. So its first draw (`prepare_draw`, `GraphicsPipeline::Upgrade`)
  queues the same parts' optimised link (`Link::optimized`; the parts are made with
  `RETAIN_LINK_TIME_OPTIMIZATION_INFO`) on the device's one background worker, below normal priority. The link goes
  through the pipeline cache, and `VulkanContext::replace_graphics_pipeline` swaps it in under `resource_mutex_` and
  `tracker_mutex_`, which together cover every list from choosing its pipeline to its submit: each list either
  recorded the fast pipeline and is submitted (the fast one goes to the deferred deletion queue, which waits for that
  work) or prepares after the swap and draws the optimised one. A PSO released while its link runs waits for it; one
  whose link has not started cancels it. The next run's PSO asks the cache for its optimised link first
  (`Link::cached`: `VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT`, with `pipelineCreationCacheControl`)
  and takes it at creation, with no fast link and no background work; only when the cache does not have it
  (`VK_PIPELINE_COMPILE_REQUIRED`) does it fast-link. Variants for unbound attachment slots stay fast-linked.
  `LEONIDA_GPL_CACHED_LINK=0` skips that question (always fast-link, then optimise in the background); the test uses
  it because NVIDIA's own disk cache answers it too once the same shaders have run on the machine before.
- `gta6rt --prebuild-shaders` on a device without shader objects (or with `LEONIDA_SHADER_OBJECTS=0`) makes every
  translated vertex shader's pre-rasterization part (depth clipped and depth clamped) and every pixel shader's
  fragment part through the device's pipeline cache and saves the cache (`prebuild_pipeline_libraries`). What it
  cannot know ahead: a geometry shader's part (it pairs with a vertex shader), shaders the runtime edits per PSO
  (the f16 VS/PS link, a D24 depth-bias pixel shader), wireframe and conservative parts, and per-sample pixel
  shaders at other sample counts. Those compile on first use and are cached for the next run.
- The recipe recorder (step 1) still records every PSO, whole-pipeline recipe and all, so `--prebuild-pipelines`
  keeps working with `LEONIDA_GPL=0`. Mesh-shader PSOs and shader objects keep their own paths.

Checked on the RTX 3080 Laptop (NVIDIA 617.42, `graphicsPipelineLibraryFastLinking = true`) by `gpl_draw_test`: three
PSOs that differ in exactly the state the libraries make dynamic (depth test and write, depth bias, cull mode) and in
blending draw the same pixels linked as whole (`LEONIDA_GPL=0`); after `--prebuild-shaders` on an empty cache
directory, a new device's PSOs make 2 of 2 shared shader parts from the pipeline cache (0 compiled) and link 3 PSOs;
their first draws get 3 optimised links made in the background and swapped in, which draw the same pixels; the next
device's 3 PSOs take their optimised links from the pipeline cache at creation (0 fast links, 0 background links) and
draw the same pixels; 0 validation errors with `LEONIDA_VK_VALIDATION=1`. Not measured on AMD or Intel hardware here.

## 4. The game's own pipeline list

The package's PSO cache (`docs/pso_cache.md`) lists 25,471 graphics and 2,693 compute pipelines with their stage
and input-layout records. Once its blend and render-target fields are decoded it can produce recipes for step 1
before any run. Steps 2 and 3 do not need it.
