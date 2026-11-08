// render_diag.cpp -- ReShade add-on: what is this game's renderer actually doing?
//
// It counts, per graphics device:
//   * frames presented and the back-buffer size,
//   * every kind of draw / dispatch / indirect call and executed command list,
//   * how many draws go into which render target (size, format, depth buffer),
//   * which kinds of descriptors (constant buffers, textures, ...) get bound.
// Every 10 seconds it writes render_diag.txt next to the game's .exe, covering
// the last 10 seconds, then starts counting again.
//
// Build: x64 DLL, C++17 -> render_diag.addon64 (see build-diag.yml, needs user32.lib)

#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <reshade.hpp>

#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>

using namespace reshade::api;

extern "C" __declspec(dllexport) const char *NAME = "Render diagnostic";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "Counts draw calls, render targets and descriptor bindings; writes render_diag.txt every 10 s.";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
using RTKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;   // width, height, colour format, depth format (0 = none)

struct CmdState {
    RTKey key{ 0, 0, 0, 0 };
};

struct DevStats {
    uint32_t api = 0;
    uint32_t bb_w = 0, bb_h = 0, bb_fmt = 0;
    uint64_t presents = 0;
    uint64_t draws = 0, draws_indexed = 0, indirect = 0, dispatches = 0, secondary = 0;
    std::map<RTKey, uint64_t> rt_draws;
    std::map<std::pair<uint32_t, uint32_t>, uint64_t> push;   // (stage bits, descriptor type) -> descriptors bound
    std::set<command_list *> lists;                           // command lists that issued draws
};

static std::map<device *, DevStats> g_stats;
static std::unordered_map<command_list *, CmdState> g_cmd;
static std::mutex g_mutex;

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
static void on_bind_rt(command_list *cmd, uint32_t count, const resource_view *rtvs, resource_view dsv)
{
    device *dev = cmd->get_device();
    uint32_t w = 0, h = 0, fmt = 0, dfmt = 0;

    if (count > 0 && rtvs != nullptr && rtvs[0].handle != 0)
    {
        const resource res = dev->get_resource_from_view(rtvs[0]);
        if (res.handle != 0)
        {
            const resource_desc d = dev->get_resource_desc(res);
            if (d.type != resource_type::buffer)
            {
                w = d.texture.width;
                h = d.texture.height;
                fmt = static_cast<uint32_t>(d.texture.format);
            }
        }
    }
    if (dsv.handle != 0)
    {
        const resource res = dev->get_resource_from_view(dsv);
        if (res.handle != 0)
        {
            const resource_desc d = dev->get_resource_desc(res);
            if (d.type != resource_type::buffer)
            {
                dfmt = static_cast<uint32_t>(d.texture.format);
                if (w == 0)
                {
                    w = d.texture.width;
                    h = d.texture.height;
                }
            }
        }
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    g_cmd[cmd].key = RTKey(w, h, fmt, dfmt);
}

// kind: 0 = draw, 1 = draw_indexed, 2 = indirect, 3 = dispatch
static void count_call(command_list *cmd, int kind)
{
    device *dev = cmd->get_device();
    std::lock_guard<std::mutex> lock(g_mutex);
    DevStats &s = g_stats[dev];

    switch (kind)
    {
    case 0: s.draws++; break;
    case 1: s.draws_indexed++; break;
    case 2: s.indirect++; break;
    default: s.dispatches++; break;
    }
    s.lists.insert(cmd);

    if (kind != 3)
    {
        auto it = g_cmd.find(cmd);
        const RTKey k = (it != g_cmd.end()) ? it->second.key : RTKey(0, 0, 0, 0);
        s.rt_draws[k]++;
    }
}

static bool on_draw(command_list *cmd, uint32_t, uint32_t, uint32_t, uint32_t)
{
    count_call(cmd, 0);
    return false;
}

static bool on_draw_indexed(command_list *cmd, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)
{
    count_call(cmd, 1);
    return false;
}

static bool on_indirect(command_list *cmd, indirect_command, resource, uint64_t, uint32_t, uint32_t)
{
    count_call(cmd, 2);
    return false;
}

static bool on_dispatch(command_list *cmd, uint32_t, uint32_t, uint32_t)
{
    count_call(cmd, 3);
    return false;
}

static void on_secondary(command_list *cmd, command_list *)
{
    device *dev = cmd->get_device();
    std::lock_guard<std::mutex> lock(g_mutex);
    g_stats[dev].secondary++;
}

static void on_push(command_list *cmd, shader_stage stages, pipeline_layout, uint32_t,
                    const descriptor_table_update &update)
{
    device *dev = cmd->get_device();
    std::lock_guard<std::mutex> lock(g_mutex);
    g_stats[dev].push[{ static_cast<uint32_t>(stages), static_cast<uint32_t>(update.type) }] += update.count;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------
static void write_report(double seconds)
{
    std::string s;
    char line[640];

    std::lock_guard<std::mutex> lock(g_mutex);

    snprintf(line, sizeof line,
             "Render diagnostic: counts over the last %.1f s\n"
             "descriptor_type numbers: sampler=%u shader_resource_view=%u constant_buffer=%u\n"
             "shader_stage bits: vertex=%u hull=%u domain=%u geometry=%u pixel=%u compute=%u\n",
             seconds,
             static_cast<uint32_t>(descriptor_type::sampler),
             static_cast<uint32_t>(descriptor_type::shader_resource_view),
             static_cast<uint32_t>(descriptor_type::constant_buffer),
             static_cast<uint32_t>(shader_stage::vertex), static_cast<uint32_t>(shader_stage::hull),
             static_cast<uint32_t>(shader_stage::domain), static_cast<uint32_t>(shader_stage::geometry),
             static_cast<uint32_t>(shader_stage::pixel), static_cast<uint32_t>(shader_stage::compute));
    s += line;

    for (auto &kv : g_stats)
    {
        DevStats &d = kv.second;
        const uint64_t total = d.draws + d.draws_indexed + d.indirect + d.dispatches;
        if (d.presents == 0 && total == 0)
            continue;

        const double frames = d.presents > 0 ? static_cast<double>(d.presents) : 1.0;

        snprintf(line, sizeof line,
                 "\n== device %p, graphics api 0x%x ==\n"
                 "frames presented through ReShade: %llu (%.1f per second), back buffer %ux%u, format %u\n"
                 "draw %llu, draw_indexed %llu, indirect %llu, dispatch %llu, secondary command lists executed %llu\n"
                 "distinct command lists that issued draws: %zu\n"
                 "draw calls per frame: %.1f\n",
                 (void *)kv.first, d.api, (unsigned long long)d.presents, d.presents / (seconds > 0 ? seconds : 1.0),
                 d.bb_w, d.bb_h, d.bb_fmt,
                 (unsigned long long)d.draws, (unsigned long long)d.draws_indexed,
                 (unsigned long long)d.indirect, (unsigned long long)d.dispatches,
                 (unsigned long long)d.secondary, d.lists.size(),
                 (d.draws + d.draws_indexed + d.indirect) / frames);
        s += line;

        s += "draws by render target (width, height, colour format, depth format; depth 0 = no depth buffer):\n";
        for (const auto &r : d.rt_draws)
        {
            snprintf(line, sizeof line, "  %ux%u  colour fmt %u  depth fmt %u : %llu draws (%.1f per frame)\n",
                     std::get<0>(r.first), std::get<1>(r.first), std::get<2>(r.first), std::get<3>(r.first),
                     (unsigned long long)r.second, r.second / frames);
            s += line;
        }

        s += "descriptors bound (shader stage bits, descriptor type): count\n";
        for (const auto &p : d.push)
        {
            snprintf(line, sizeof line, "  stage %u, type %u : %llu\n",
                     p.first.first, p.first.second, (unsigned long long)p.second);
            s += line;
        }

        // start a fresh window
        d.presents = 0;
        d.draws = d.draws_indexed = d.indirect = d.dispatches = d.secondary = 0;
        d.rt_draws.clear();
        d.push.clear();
        d.lists.clear();
    }

    s += "\nHow to read this: the 3D scene is the render target with a non-zero depth format and the\n"
         "highest draw count. A game scene normally has hundreds of draws per frame there.\n";

    char path[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    std::string p(path);
    p = p.substr(0, p.find_last_of("\\/") + 1) + "render_diag.txt";

    FILE *fp = fopen(p.c_str(), "wb");
    if (fp != nullptr)
    {
        fwrite(s.data(), 1, s.size(), fp);
        fclose(fp);
    }
}

static void on_present(effect_runtime *rt)
{
    device *dev = rt->get_device();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        DevStats &s = g_stats[dev];
        s.presents++;
        s.api = static_cast<uint32_t>(dev->get_api());

        const resource bb = rt->get_current_back_buffer();
        if (bb.handle != 0)
        {
            const resource_desc d = dev->get_resource_desc(bb);
            if (d.type != resource_type::buffer)
            {
                s.bb_w = d.texture.width;
                s.bb_h = d.texture.height;
                s.bb_fmt = static_cast<uint32_t>(d.texture.format);
            }
        }
    }

    static ULONGLONG last = 0;
    const ULONGLONG now = GetTickCount64();
    if (last == 0)
        last = now;
    if (now - last >= 10000)
    {
        write_report((now - last) / 1000.0);
        last = now;
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
        reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind_rt);
        reshade::register_event<reshade::addon_event::push_descriptors>(on_push);
        reshade::register_event<reshade::addon_event::draw>(on_draw);
        reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
        reshade::register_event<reshade::addon_event::draw_or_dispatch_indirect>(on_indirect);
        reshade::register_event<reshade::addon_event::dispatch>(on_dispatch);
        reshade::register_event<reshade::addon_event::execute_secondary_command_list>(on_secondary);
        reshade::register_event<reshade::addon_event::reshade_present>(on_present);
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_addon(hModule);
        break;
    }
    return TRUE;
}
