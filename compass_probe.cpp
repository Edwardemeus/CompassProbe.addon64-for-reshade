// compass_probe.cpp
// ReShade add-on, step 1: find where Roblox Studio keeps the camera in its
// D3D11 constant buffers.
//
// Build: x64 DLL, C++17, rename the output to compass_probe.addon64
//        (easiest: the GitHub Actions workflow build.yml that ships with this file)
// Needs: ReShade repo headers (include/) and its bundled ImGui (deps/imgui).
// NOTE: not compiled by the author. The event signatures were checked against
// the ReShade docs (map_buffer_region is called AFTER the Map call,
// update_buffer_region BEFORE the update), but if the compiler complains,
// compare the callbacks below with include/reshade_events.hpp.

#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define IMGUI_DISABLE_INCLUDE_IMCONFIG_H
#include <imgui.h>      // must come BEFORE reshade.hpp (ReShade routes ImGui calls to its own instance)
#include <reshade.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace reshade::api;

extern "C" __declspec(dllexport) const char *NAME = "Roblox camera probe";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "Finds the camera position inside Studio's constant buffers.";

// ---------------------------------------------------------------------------
// Settings (editable in the overlay, so no recompiling per camera position)
// ---------------------------------------------------------------------------
static struct {
    float pos[3] = { 1.222f, 6.617f, 20.091f }; // paste from the Lua output
    float tol = 0.01f;
    bool  cb_only = true;
} S;

// ---------------------------------------------------------------------------
// Hit bookkeeping
// ---------------------------------------------------------------------------
struct Hit {
    uint64_t count = 0;
    float live[3] = {};
    uint64_t snap_start = 0;      // byte offset (in the buffer) of snap[0]
    std::vector<float> snap;      // up to 64 floats = 16 float4 rows around the hit
};

// key: buffer size in bytes, byte offset of the hit, sign mask vs. the Lua values
using Key = std::tuple<uint32_t, uint64_t, int>;
static std::map<Key, Hit> g_hits;
static std::mutex g_hits_mutex;
static std::atomic<uint64_t> g_writes{ 0 };

static std::string sign_label(int m)
{
    char b[16];
    snprintf(b, sizeof b, "(%c,%c,%c)", (m & 1) ? '-' : '+', (m & 2) ? '-' : '+', (m & 4) ? '-' : '+');
    return b;
}

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------
static void scan(device *dev, resource res, uint64_t base, const void *data, uint64_t size)
{
    // Constant buffers are <= 64 KiB; the cap also keeps reads of
    // write-combined mapped memory cheap.
    if (data == nullptr || size < 12 || size > 65536)
        return;

    const resource_desc desc = dev->get_resource_desc(res);
    if (desc.type != resource_type::buffer)
        return;
    if (S.cb_only && (desc.usage & resource_usage::constant_buffer) == 0)
        return;

    g_writes.fetch_add(1, std::memory_order_relaxed);

    const float *f = static_cast<const float *>(data);
    const size_t n = static_cast<size_t>(size / 4);
    const float ax = fabsf(S.pos[0]), ay = fabsf(S.pos[1]), az = fabsf(S.pos[2]);
    const float t = S.tol;

    for (size_t i = 0; i + 3 <= n; ++i)
    {
        // match |x|,|y|,|z| first, then record which signs were flipped
        if (fabsf(fabsf(f[i])     - ax) > t) continue;
        if (fabsf(fabsf(f[i + 1]) - ay) > t) continue;
        if (fabsf(fabsf(f[i + 2]) - az) > t) continue;

        const int mask = (((f[i]     < 0) != (S.pos[0] < 0)) ? 1 : 0)
                       | (((f[i + 1] < 0) != (S.pos[1] < 0)) ? 2 : 0)
                       | (((f[i + 2] < 0) != (S.pos[2] < 0)) ? 4 : 0);

        const uint64_t off = base + static_cast<uint64_t>(i) * 4;

        std::lock_guard<std::mutex> lock(g_hits_mutex);
        Hit &h = g_hits[Key(static_cast<uint32_t>(desc.buffer.size), off, mask)];
        h.count++;
        h.live[0] = f[i];
        h.live[1] = f[i + 1];
        h.live[2] = f[i + 2];

        // snapshot: 16 float4 rows (one HLSL register each), starting up to
        // 4 rows before the hit
        size_t start = (i / 4) * 4;
        start = start >= 16 ? start - 16 : 0;
        const size_t end = std::min(n, start + 64);
        h.snap.assign(f + start, f + end);
        h.snap_start = base + static_cast<uint64_t>(start) * 4;
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
struct MapInfo { void *ptr; uint64_t offset; uint64_t size; };
static std::unordered_map<uint64_t, MapInfo> g_maps;
static std::mutex g_maps_mutex;

// UpdateSubresource-style uploads: data pointer is available immediately.
static bool on_update(device *dev, const void *data, resource dest, uint64_t offset, uint64_t size)
{
    // ReShade 6.7+ may pass UINT64_MAX meaning "from offset to the end of the buffer"
    if (size == UINT64_MAX)
        size = dev->get_resource_desc(dest).buffer.size - offset;
    scan(dev, dest, offset, data, size);
    return false; // let the original call go through
}

// Map/Unmap-style uploads: remember the pointer, read it when the app unmaps.
static void on_map(device *, resource res, uint64_t offset, uint64_t size,
                   map_access access, void **data)
{
    if (access == map_access::read_only || data == nullptr || *data == nullptr)
        return;
    std::lock_guard<std::mutex> lock(g_maps_mutex);
    g_maps[res.handle] = { *data, offset, size };
}

static void on_unmap(device *dev, resource res)
{
    MapInfo mi;
    {
        std::lock_guard<std::mutex> lock(g_maps_mutex);
        auto it = g_maps.find(res.handle);
        if (it == g_maps.end())
            return;
        mi = it->second;
        g_maps.erase(it);
    }
    uint64_t size = mi.size;
    if (size == UINT64_MAX) // whole buffer was mapped
        size = dev->get_resource_desc(res).buffer.size - mi.offset;
    scan(dev, res, mi.offset, mi.ptr, size);
}

// ---------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------
static std::string report()
{
    std::string s;
    char line[256];
    snprintf(line, sizeof line, "pos %.3f %.3f %.3f  tol %.4f  cb_only %d  writes %llu\n",
             S.pos[0], S.pos[1], S.pos[2], S.tol, S.cb_only ? 1 : 0,
             (unsigned long long)g_writes.load());
    s += line;
    std::lock_guard<std::mutex> lock(g_hits_mutex);
    if (g_hits.empty())
        s += "no hits\n";
    for (const auto &kv : g_hits)
    {
        const Hit &h = kv.second;
        snprintf(line, sizeof line,
                 "buffer %u bytes, offset %llu, signs %s, hits %llu, live %.3f %.3f %.3f\n",
                 std::get<0>(kv.first), (unsigned long long)std::get<1>(kv.first),
                 sign_label(std::get<2>(kv.first)).c_str(), (unsigned long long)h.count,
                 h.live[0], h.live[1], h.live[2]);
        s += line;
        for (size_t r = 0; r + 4 <= h.snap.size(); r += 4)
        {
            snprintf(line, sizeof line, "  +%llu: %10.5f %10.5f %10.5f %10.5f\n",
                     (unsigned long long)(h.snap_start + r * 4),
                     h.snap[r], h.snap[r + 1], h.snap[r + 2], h.snap[r + 3]);
            s += line;
        }
    }
    return s;
}

static std::string g_saved_path;

// Writes the report to compass_probe_report.txt in the same folder as the exe.
static void save_report()
{
    char path[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    std::string p(path);
    p = p.substr(0, p.find_last_of("\\/") + 1) + "compass_probe_report.txt";

    const std::string text = report();
    FILE *fp = fopen(p.c_str(), "wb");
    if (fp != nullptr)
    {
        fwrite(text.data(), 1, text.size(), fp);
        fclose(fp);
        g_saved_path = p;
    }
    else
        g_saved_path = "FAILED to write " + p;
}

static void draw_overlay(effect_runtime *)
{
    ImGui::TextUnformatted("1) Run the Lua command in Studio  2) type the position below  3) keep the camera still");
    ImGui::InputFloat3("Camera position", S.pos, "%.3f");
    ImGui::SliderFloat("Tolerance", &S.tol, 0.001f, 0.1f, "%.3f");
    ImGui::Checkbox("Constant buffers only", &S.cb_only);
    ImGui::Text("Buffer writes scanned: %llu  (stays 0 = events not firing)",
                (unsigned long long)g_writes.load());

    if (ImGui::Button("Clear hits"))
    {
        std::lock_guard<std::mutex> lock(g_hits_mutex);
        g_hits.clear();
    }
    ImGui::SameLine();
    if (ImGui::Button("Save report to file"))
        save_report();
    ImGui::SameLine();
    if (ImGui::Button("Copy to clipboard"))
        ImGui::SetClipboardText(report().c_str());
    if (!g_saved_path.empty())
        ImGui::TextWrapped("Report saved to: %s", g_saved_path.c_str());

    std::lock_guard<std::mutex> lock(g_hits_mutex);
    ImGui::Text("Hits: %zu", g_hits.size());
    for (const auto &kv : g_hits)
    {
        const Hit &h = kv.second;
        char label[192];
        snprintf(label, sizeof label,
                 "buffer %u B  offset %llu  signs %s  live %.3f %.3f %.3f###%u_%llu_%d",
                 std::get<0>(kv.first), (unsigned long long)std::get<1>(kv.first),
                 sign_label(std::get<2>(kv.first)).c_str(),
                 h.live[0], h.live[1], h.live[2],
                 std::get<0>(kv.first), (unsigned long long)std::get<1>(kv.first),
                 std::get<2>(kv.first));
        if (ImGui::TreeNode(label))
        {
            for (size_t r = 0; r + 4 <= h.snap.size(); r += 4)
                ImGui::Text("+%4llu: %10.5f %10.5f %10.5f %10.5f",
                            (unsigned long long)(h.snap_start + r * 4),
                            h.snap[r], h.snap[r + 1], h.snap[r + 2], h.snap[r + 3]);
            ImGui::TreePop();
        }
    }
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
        reshade::register_event<reshade::addon_event::update_buffer_region>(on_update);
        reshade::register_event<reshade::addon_event::map_buffer_region>(on_map);
        reshade::register_event<reshade::addon_event::unmap_buffer_region>(on_unmap);
        reshade::register_overlay(nullptr, draw_overlay);
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_addon(hModule);
        break;
    }
    return TRUE;
}
