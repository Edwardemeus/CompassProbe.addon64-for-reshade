// compass_probe.cpp  -- Roblox Studio compass add-on (final version)
// (The file name is kept so the existing GitHub build keeps working.)
//
// What it does
//   * Watches which buffer Studio binds to vertex-shader constant-buffer slots
//     2 and 0 (this is where Studio keeps its per-frame camera block).
//   * At the first draw of every frame it copies that range back from the GPU
//     into a small staging buffer, and reads it a few frames later (so Studio
//     is never stalled).
//   * It finds the camera "Right / Up / Back / Position" rows by their shape
//     (not by a fixed offset), computes the compass heading and hands it to the
//     ReShade effect Compass.fx as the uniforms CompassHeading / CompassValid.
//
// Heading convention: 0 deg = world -Z ("north"), 90 deg = world +X ("east").
//
// Build: x64 DLL, C++17 -> compass_probe.addon64 (see build.yml, needs user32.lib)

#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define IMGUI_DISABLE_INCLUDE_IMCONFIG_H
#include <imgui.h>      // must come BEFORE reshade.hpp
#include <reshade.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>

using namespace reshade::api;

extern "C" __declspec(dllexport) const char *NAME = "Roblox Studio compass";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "Reads the camera heading from Studio's constant buffers and feeds it to Compass.fx.";

// ---------------------------------------------------------------------------
// Settings. If a Studio update moves the camera block, these are the numbers
// to change (the probe report told us: VS slot 2, 256 B, then VS slot 0, 1024 B).
// ---------------------------------------------------------------------------
static const uint32_t CAM_SLOTS[2]   = { 2, 0 };       // vertex-shader constant-buffer slots to try, in order
static const uint32_t NSLOT          = 2;
static const uint32_t RING           = 6;              // staging ring: frames in flight
static const uint64_t CELL           = 2048;           // bytes copied per slot per frame (max)
static const uint64_t STAGING_SIZE   = uint64_t(RING) * NSLOT * CELL;
static const uint64_t MIN_RANGE      = 192;            // ignore bound ranges smaller than this
static const uint64_t READ_AGE       = 3;              // read data that is this many frames old
static const uint64_t STALE_FRAMES   = 240;            // no valid data for this long -> CompassValid = 0
static const char    *EFFECT_FILE    = "Compass.fx";

// ---------------------------------------------------------------------------
// Debug values for the overlay
// ---------------------------------------------------------------------------
static std::atomic<float>    g_dbg_heading{ 0.f };
static std::atomic<float>    g_dbg_right[3];
static std::atomic<int>      g_dbg_slot{ -1 };
static std::atomic<uint64_t> g_dbg_updates{ 0 };
static std::atomic<uint64_t> g_dbg_copies{ 0 };

// ---------------------------------------------------------------------------
// Parsing: find  [Right][Up][Back][Position]  as four consecutive float4 rows.
//   Right and Back: unit-length xyz, w = 0, perpendicular to each other.
//   Position row: w = 1.
// ---------------------------------------------------------------------------
static bool unit_axis(const float *v)
{
    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2]) || !std::isfinite(v[3]))
        return false;
    const float len = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return fabsf(len - 1.0f) < 0.01f && fabsf(v[3]) < 0.01f;
}

static bool parse_camera(const float *f, size_t nfloats, float right[3])
{
    const size_t rows = nfloats / 4;
    for (size_t r = 0; r + 3 < rows; ++r)
    {
        const float *R = f + r * 4;     // Right
        const float *U = R + 4;         // Up (only checked for being finite)
        const float *B = R + 8;         // Back (= -Look)
        const float *P = R + 12;        // Position, w = 1

        if (!unit_axis(R) || !unit_axis(B))
            continue;
        if (!std::isfinite(U[0]) || !std::isfinite(U[1]) || !std::isfinite(U[2]))
            continue;
        if (fabsf(R[0] * B[0] + R[1] * B[1] + R[2] * B[2]) > 0.01f)
            continue;
        if (!std::isfinite(P[3]) || fabsf(P[3] - 1.0f) > 0.001f)
            continue;

        right[0] = R[0];
        right[1] = R[1];
        right[2] = R[2];
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Per-device state
// ---------------------------------------------------------------------------
struct BoundCB { uint64_t buffer = 0, offset = 0, size = 0; };

struct Cell {
    bool has = false;
    uint32_t count = 0;
    uint32_t which[NSLOT] = {};
    uint32_t size[NSLOT] = {};
};

struct DevState {
    resource staging{};
    bool failed = false;
    uint64_t frame = 0;
    uint64_t queued_for = UINT64_MAX;
    BoundCB vs[NSLOT];
    Cell cells[RING];
    float heading = 0.f;
    uint64_t last_ok = 0;
    bool ever_ok = false;
};

static std::unordered_map<device *, DevState> g_dev;
static std::mutex g_mutex;

static bool ensure_staging(device *dev, DevState &ds)
{
    if (ds.staging.handle != 0)
        return true;
    if (ds.failed)
        return false;
    const resource_desc d(STAGING_SIZE, memory_heap::gpu_to_cpu, resource_usage::copy_dest);
    if (!dev->create_resource(d, nullptr, resource_usage::copy_dest, &ds.staging))
    {
        ds.staging = {};
        ds.failed = true;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Track what is bound to the camera constant-buffer slots
// ---------------------------------------------------------------------------
static void on_push_descriptors(command_list *cmd, shader_stage stages, pipeline_layout,
                                uint32_t, const descriptor_table_update &update)
{
    if (update.type != descriptor_type::constant_buffer || update.descriptors == nullptr)
        return;
    if ((static_cast<uint32_t>(stages) & static_cast<uint32_t>(shader_stage::vertex)) == 0)
        return;

    bool wanted = false;
    for (uint32_t i = 0; i < update.count; ++i)
        for (uint32_t k = 0; k < NSLOT; ++k)
            if (update.binding + i == CAM_SLOTS[k])
                wanted = true;
    if (!wanted)
        return;

    device *dev = cmd->get_device();
    const buffer_range *r = static_cast<const buffer_range *>(update.descriptors);

    std::lock_guard<std::mutex> lock(g_mutex);
    DevState &ds = g_dev[dev];

    for (uint32_t i = 0; i < update.count; ++i)
    {
        for (uint32_t k = 0; k < NSLOT; ++k)
        {
            if (update.binding + i != CAM_SLOTS[k])
                continue;

            BoundCB b;
            b.buffer = r[i].buffer.handle;
            b.offset = r[i].offset;
            b.size = r[i].size;
            if (b.buffer != 0 && (b.size == UINT64_MAX || b.size == 0))
            {
                const uint64_t total = dev->get_resource_desc(r[i].buffer).buffer.size;
                b.size = total > b.offset ? total - b.offset : 0;
            }
            ds.vs[k] = b;
        }
    }
}

// ---------------------------------------------------------------------------
// First draw of every frame: queue a GPU copy of the bound camera range(s)
// ---------------------------------------------------------------------------
static void queue_copies(command_list *cmd)
{
    device *dev = cmd->get_device();

    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_dev.find(dev);
    if (it == g_dev.end())
        return;
    DevState &ds = it->second;
    if (ds.queued_for == ds.frame)
        return;                                   // already queued this frame

    const uint64_t cell_index = ds.frame % RING;
    Cell fresh;

    for (uint32_t k = 0; k < NSLOT; ++k)
    {
        const BoundCB &b = ds.vs[k];
        if (b.buffer == 0 || b.size < MIN_RANGE)
            continue;
        if (!ensure_staging(dev, ds))
            return;

        const uint64_t sz = std::min<uint64_t>(b.size, CELL);
        cmd->copy_buffer_region(resource{ b.buffer }, b.offset, ds.staging,
                                (cell_index * NSLOT + k) * CELL, sz);
        fresh.which[fresh.count] = k;
        fresh.size[fresh.count] = static_cast<uint32_t>(sz);
        fresh.count++;
    }

    if (fresh.count == 0)
        return;

    fresh.has = true;
    ds.cells[cell_index] = fresh;
    ds.queued_for = ds.frame;
    g_dbg_copies.fetch_add(1, std::memory_order_relaxed);
}

static bool on_draw(command_list *cmd, uint32_t, uint32_t, uint32_t, uint32_t)
{
    queue_copies(cmd);
    return false;
}

static bool on_draw_indexed(command_list *cmd, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)
{
    queue_copies(cmd);
    return false;
}

// ---------------------------------------------------------------------------
// Read an older frame's copy back and update the heading (caller holds g_mutex)
// ---------------------------------------------------------------------------
static void read_oldest(device *dev, DevState &ds)
{
    if (ds.frame < READ_AGE || ds.staging.handle == 0)
        return;

    const uint64_t cell_index = (ds.frame - READ_AGE) % RING;
    Cell &c = ds.cells[cell_index];
    if (!c.has)
        return;

    void *ptr = nullptr;
    if (dev->map_buffer_region(ds.staging, 0, STAGING_SIZE, map_access::read_only, &ptr) && ptr != nullptr)
    {
        const uint8_t *base = static_cast<const uint8_t *>(ptr);
        for (uint32_t i = 0; i < c.count; ++i)
        {
            const float *fl = reinterpret_cast<const float *>(
                base + (cell_index * NSLOT + c.which[i]) * CELL);
            float right[3];
            if (!parse_camera(fl, c.size[i] / 4, right))
                continue;

            float deg = atan2f(right[2], right[0]) * (180.0f / 3.14159265f);
            if (deg < 0.f)
                deg += 360.f;

            ds.heading = deg;
            ds.last_ok = ds.frame;
            ds.ever_ok = true;

            g_dbg_heading = deg;
            g_dbg_right[0] = right[0];
            g_dbg_right[1] = right[1];
            g_dbg_right[2] = right[2];
            g_dbg_slot = static_cast<int>(CAM_SLOTS[c.which[i]]);
            g_dbg_updates.fetch_add(1, std::memory_order_relaxed);
            break;
        }
        dev->unmap_buffer_region(ds.staging);
    }
    c.has = false;
}

// ---------------------------------------------------------------------------
// Feed the effect, once per frame per window, right before effects are rendered
// ---------------------------------------------------------------------------
static void set_float(effect_runtime *rt, const char *name, float value)
{
    const effect_uniform_variable v = rt->find_uniform_variable(EFFECT_FILE, name);
    if (v.handle != 0)
        rt->set_uniform_value_float(v, &value, 1);
}

static void on_begin_effects(effect_runtime *rt, command_list *, resource_view, resource_view)
{
    device *dev = rt->get_device();
    float heading = 0.f;
    bool valid = false;

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_dev.find(dev);
        if (it != g_dev.end())
        {
            DevState &ds = it->second;
            read_oldest(dev, ds);
            ds.frame++;
            heading = ds.heading;
            valid = ds.ever_ok && (ds.frame - ds.last_ok) <= STALE_FRAMES;
        }
    }

    set_float(rt, "CompassHeading", heading);
    set_float(rt, "CompassValid", valid ? 1.0f : 0.0f);
}

// ---------------------------------------------------------------------------
// Overlay (status only)
// ---------------------------------------------------------------------------
static void draw_overlay(effect_runtime *)
{
    ImGui::TextUnformatted("Roblox Studio compass");
    ImGui::Text("Heading: %.1f deg   (0 = world -Z, 90 = world +X)", g_dbg_heading.load());
    ImGui::Text("Camera Right vector: %.3f %.3f %.3f",
                g_dbg_right[0].load(), g_dbg_right[1].load(), g_dbg_right[2].load());
    ImGui::Text("Found in vertex constant-buffer slot: %d   updates: %llu   copies queued: %llu",
                g_dbg_slot.load(), (unsigned long long)g_dbg_updates.load(),
                (unsigned long long)g_dbg_copies.load());
    ImGui::TextUnformatted("If 'updates' stays 0, Studio's camera block moved: run the probe again.");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID)
{
    switch (reason)
    {
    case DLL_PROCESS_ATTACH:
        if (!reshade::register_addon(hModule))
            return FALSE;
        reshade::register_event<reshade::addon_event::push_descriptors>(on_push_descriptors);
        reshade::register_event<reshade::addon_event::draw>(on_draw);
        reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
        reshade::register_event<reshade::addon_event::reshade_begin_effects>(on_begin_effects);
        reshade::register_overlay(nullptr, draw_overlay);
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_addon(hModule);
        break;
    }
    return TRUE;
}
