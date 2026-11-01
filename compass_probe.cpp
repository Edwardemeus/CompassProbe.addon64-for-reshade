// compass_probe.cpp  (v4)
// ReShade add-on: find where Roblox Studio keeps the camera in its D3D11
// buffers.
//
// How it works
//   * It watches every buffer upload.
//   * Hold the camera still in state A and press Ctrl+Shift+F7 (2.5 s capture),
//     put the camera in state B and press Ctrl+Shift+F8, then press
//     Ctrl+Shift+F9: a report is written next to the Studio exe listing the
//     values whose range in state A does not overlap their range in state B,
//     plus a census of every buffer that was written.
//   * Beeps confirm each step, so the ReShade overlay is not required.
//
// v4 changes: range-based comparison (tolerates jitter), tracks all buffer
// types up to 64 KB (not only constant buffers), census + skip counters.
//
// Build: x64 DLL, C++17 -> compass_probe.addon64 (see build.yml)

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
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace reshade::api;

extern "C" __declspec(dllexport) const char *NAME = "Roblox camera probe";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "Finds the camera data inside Studio's buffers (A/B diff).";

// ---------------------------------------------------------------------------
// Hotkeys: hold Ctrl+Shift, then press the F-key
// ---------------------------------------------------------------------------
static const int       KEY_CAPTURE_A = VK_F7;
static const int       KEY_CAPTURE_B = VK_F8;
static const int       KEY_SAVE_DIFF = VK_F9;
static const ULONGLONG CAPTURE_MS    = 2500;     // length of one capture
static const uint64_t  SCAN_CAP      = 65536;    // writes bigger than this are skipped

// ---------------------------------------------------------------------------
// Settings (editable in the overlay if you can reach it)
// ---------------------------------------------------------------------------
static struct {
    float pos[3]  = { -1.080f, 6.822f, 25.367f };  // camera position in state A
    float posB[3] = { -3.080f, 6.822f, 23.367f };  // camera position in state B
    float tol = 0.01f;
    bool  cb_only = false;         // false = look at every buffer type
    bool  group_by_size = false;   // merge all buffers of the same size
} S;

static std::atomic<uint64_t> g_writes{ 0 };        // writes that were scanned
static std::atomic<uint64_t> g_seen{ 0 };          // all buffer writes seen
static std::atomic<uint64_t> g_skip_big{ 0 };      // skipped: bigger than SCAN_CAP
static std::atomic<uint64_t> g_skip_notcb{ 0 };    // skipped: not a constant buffer (cb_only)
static std::atomic<uint64_t> g_max_size{ 0 };      // largest write size seen
static std::string g_status = "Idle. Hotkeys: Ctrl+Shift+F7 = capture A, F8 = capture B, F9 = save diff.";

// ---------------------------------------------------------------------------
// Part 1: position search
// ---------------------------------------------------------------------------
struct Hit {
    uint64_t count = 0;
    float live[3] = {};
    uint64_t snap_start = 0;
    std::vector<float> snap;
};
using Key = std::tuple<uint32_t, uint64_t, int>;   // buffer size, byte offset, sign mask
static std::map<Key, Hit> g_hits;
static std::mutex g_hits_mutex;

static std::string sign_label(int m)
{
    char b[16];
    snprintf(b, sizeof b, "(%c,%c,%c)", (m & 1) ? '-' : '+', (m & 2) ? '-' : '+', (m & 4) ? '-' : '+');
    return b;
}

static void search_position(const resource_desc &desc, uint64_t base, const float *f, size_t n)
{
    const float ax = fabsf(S.pos[0]), ay = fabsf(S.pos[1]), az = fabsf(S.pos[2]);
    const float t = S.tol;

    for (size_t i = 0; i + 3 <= n; ++i)
    {
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

        size_t start = (i / 4) * 4;
        start = start >= 16 ? start - 16 : 0;
        const size_t end = std::min(n, start + 64);
        h.snap.assign(f + start, f + end);
        h.snap_start = base + static_cast<uint64_t>(start) * 4;
    }
}

// ---------------------------------------------------------------------------
// Part 2: A/B capture and diff
// ---------------------------------------------------------------------------
struct Loc {                       // one float inside one buffer: range seen during a capture
    float mn = 0.f, mx = 0.f;
    double sum = 0.0;
    uint32_t n = 0;
    float mean() const { return n ? static_cast<float>(sum / n) : 0.f; }
    float width() const { return mx - mn; }
};
struct BufTrack {
    uint64_t size = 0;
    uint64_t writes = 0, via_map = 0, via_update = 0;
    std::string info;              // usage flags + heap
    std::vector<Loc> locs;         // indexed by (byte offset / 4)
};

static std::unordered_map<uint64_t, BufTrack> g_cap[2];   // 0 = state A, 1 = state B
static std::atomic<int> g_capturing{ -1 };                 // -1 = off, 0 = A, 1 = B
static ULONGLONG g_cap_start = 0;
static std::mutex g_cap_mutex;                             // also guards g_status

static std::string describe(const resource_desc &d)
{
    std::string s;
    auto add = [&](resource_usage f, const char *name) {
        if ((d.usage & f) == 0)
            return;
        if (!s.empty()) s += "|";
        s += name;
    };
    add(resource_usage::constant_buffer, "CB");
    add(resource_usage::shader_resource, "SRV");
    add(resource_usage::unordered_access, "UAV");
    add(resource_usage::vertex_buffer, "VB");
    add(resource_usage::index_buffer, "IB");
    char b[32];
    snprintf(b, sizeof b, " heap=%d", static_cast<int>(d.heap));
    s += b;
    return s;
}

static void track(uint64_t id, const resource_desc &desc, uint64_t base,
                  const float *f, size_t n, int path)
{
    std::lock_guard<std::mutex> lock(g_cap_mutex);
    const int which = g_capturing.load();
    if (which < 0)
        return;

    if (GetTickCount64() - g_cap_start > CAPTURE_MS)
    {
        g_capturing = -1;
        g_status = std::string("Capture ") + (which == 0 ? "A" : "B") + " finished ("
                 + std::to_string(g_cap[which].size()) + " buffers tracked).";
        MessageBeep(MB_ICONEXCLAMATION);
        return;
    }

    BufTrack &bt = g_cap[which][id];
    if (bt.info.empty())
    {
        bt.size = desc.buffer.size;
        bt.info = describe(desc);
    }
    bt.writes++;
    if (path == 0) bt.via_update++; else bt.via_map++;

    const size_t first = static_cast<size_t>(base / 4);
    if (bt.locs.size() < first + n)
        bt.locs.resize(first + n);

    for (size_t i = 0; i < n; ++i)
    {
        const float v = f[i];
        if (!std::isfinite(v))
            continue;
        Loc &L = bt.locs[first + i];
        if (L.n == 0) { L.mn = L.mx = v; }
        else { if (v < L.mn) L.mn = v; if (v > L.mx) L.mx = v; }
        L.sum += v;
        L.n++;
    }
}

static void start_capture(int which)
{
    std::lock_guard<std::mutex> lock(g_cap_mutex);
    g_cap[which].clear();
    g_cap_start = GetTickCount64();
    g_capturing = which;
    g_status = std::string("Capturing state ") + (which == 0 ? "A" : "B")
             + " ... keep the camera still until the second beep.";
    MessageBeep(MB_OK);
}

static std::string fmt_loc(const Loc &l)
{
    char b[32];
    if (l.n == 0)
        return "      --  ";
    snprintf(b, sizeof b, "%9.4f%c", l.mean(), l.width() > 1e-5f ? '~' : ' ');   // ~ = value jittered
    return b;
}

static bool near_f(float a, float b) { return fabsf(a - b) <= S.tol; }

// A value counts as "changed" when its range in A and its range in B do not
// overlap, and the gap between them is much bigger than the wobble inside each.
static bool changed_loc(const Loc &a, const Loc &b)
{
    if (a.n == 0 || b.n == 0)
        return false;
    const float gap = std::max(a.mn - b.mx, b.mn - a.mx);
    if (gap <= 1e-4f)
        return false;
    return (a.width() + b.width()) < 0.1f * gap;
}

static std::string build_diff(std::string &summary)
{
    struct Entry {
        uint64_t id;
        const BufTrack *a;
        const BufTrack *b;
        std::vector<size_t> changed;
    };
    std::vector<Entry> entries;
    std::string s;
    char line[640];
    int pos_tags = 0;

    std::lock_guard<std::mutex> lock(g_cap_mutex);

    for (const auto &kv : g_cap[0])
    {
        auto itB = g_cap[1].find(kv.first);
        if (itB == g_cap[1].end())
            continue;
        Entry e{ kv.first, &kv.second, &itB->second, {} };
        const size_t n = std::min(e.a->locs.size(), e.b->locs.size());
        for (size_t i = 0; i < n; ++i)
            if (changed_loc(e.a->locs[i], e.b->locs[i]))
                e.changed.push_back(i);
        if (!e.changed.empty())
            entries.push_back(std::move(e));
    }

    std::sort(entries.begin(), entries.end(),
              [](const Entry &x, const Entry &y) { return x.changed.size() > y.changed.size(); });

    snprintf(line, sizeof line,
             "A: %zu buffers tracked, B: %zu buffers tracked, %zu buffers have values whose range "
             "differs between A and B.\n"
             "Position A = %.3f %.3f %.3f   Position B = %.3f %.3f %.3f   tol %.3f\n"
             "Buffer writes seen: %llu, scanned: %llu, skipped as bigger than %llu B: %llu "
             "(largest write: %llu B), skipped as non-constant-buffer: %llu (cb_only=%d)\n"
             "Each row is one float4 register. '~' after a value = it jittered during the capture. "
             "Tags <-POS.x/y/z matched the camera position.\n",
             g_cap[0].size(), g_cap[1].size(), entries.size(),
             S.pos[0], S.pos[1], S.pos[2], S.posB[0], S.posB[1], S.posB[2], S.tol,
             (unsigned long long)g_seen.load(), (unsigned long long)g_writes.load(),
             (unsigned long long)SCAN_CAP, (unsigned long long)g_skip_big.load(),
             (unsigned long long)g_max_size.load(), (unsigned long long)g_skip_notcb.load(),
             S.cb_only ? 1 : 0);
    s += line;

    std::set<uint64_t> changed_ids;
    for (size_t k = 0; k < entries.size() && k < 12; ++k)
    {
        const Entry &e = entries[k];
        changed_ids.insert(e.id);
        snprintf(line, sizeof line, "\n== buffer %s %llu (size %llu bytes, %s): %zu changed values ==\n",
                 S.group_by_size ? "size-group" : "handle",
                 (unsigned long long)e.id, (unsigned long long)e.a->size, e.a->info.c_str(),
                 e.changed.size());
        s += line;

        std::set<size_t> regs;
        for (size_t i : e.changed)
            regs.insert(i / 4);

        size_t shown = 0;
        for (size_t r : regs)
        {
            if (shown++ >= 40) { s += "  ...\n"; break; }

            std::string rowA, rowB, tags;
            for (size_t c = 0; c < 4; ++c)
            {
                const size_t idx = r * 4 + c;
                Loc la, lb;
                if (idx < e.a->locs.size()) la = e.a->locs[idx];
                if (idx < e.b->locs.size()) lb = e.b->locs[idx];
                rowA += fmt_loc(la) + " ";
                rowB += fmt_loc(lb) + " ";

                if (la.n > 0 && lb.n > 0)
                {
                    static const char axes[3] = { 'x', 'y', 'z' };
                    for (int ax = 0; ax < 3; ++ax)
                    {
                        if (near_f(la.mean(), S.pos[ax]) && near_f(lb.mean(), S.posB[ax]))
                        { tags += " <-POS."; tags += axes[ax]; ++pos_tags; }
                        else if (near_f(la.mean(), -S.pos[ax]) && near_f(lb.mean(), -S.posB[ax]))
                        { tags += " <-(-POS)."; tags += axes[ax]; ++pos_tags; }
                    }
                }
            }
            snprintf(line, sizeof line, "  +%4zu B | A: %s| B: %s|%s\n",
                     r * 16, rowA.c_str(), rowB.c_str(), tags.c_str());
            s += line;
        }
    }

    // ---- census of every tracked buffer ----
    struct Row { uint64_t id; const BufTrack *a; const BufTrack *b; };
    std::vector<Row> rows;
    for (const auto &kv : g_cap[0])
    {
        auto itB = g_cap[1].find(kv.first);
        rows.push_back({ kv.first, &kv.second, itB == g_cap[1].end() ? nullptr : &itB->second });
    }
    std::sort(rows.begin(), rows.end(),
              [](const Row &x, const Row &y) { return x.a->writes > y.a->writes; });

    s += "\n== census: every buffer written during capture A (sorted by writes) ==\n";
    size_t listed = 0;
    for (const Row &r : rows)
    {
        if (listed++ >= 60) { s += "  ...\n"; break; }
        snprintf(line, sizeof line,
                 "  id %-12llu %7llu B  %-28s writes A=%llu (update %llu / map %llu)  B=%llu%s\n",
                 (unsigned long long)r.id, (unsigned long long)r.a->size, r.a->info.c_str(),
                 (unsigned long long)r.a->writes, (unsigned long long)r.a->via_update,
                 (unsigned long long)r.a->via_map,
                 (unsigned long long)(r.b ? r.b->writes : 0),
                 changed_ids.count(r.id) ? "   <-- has changed values" : "");
        s += line;
    }

    summary = std::to_string(entries.size()) + " buffers changed, "
            + std::to_string(pos_tags) + " position matches.";
    return s;
}

// ---------------------------------------------------------------------------
// File output (next to the exe)
// ---------------------------------------------------------------------------
static std::string write_next_to_exe(const char *name, const std::string &text)
{
    char path[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    std::string p(path);
    p = p.substr(0, p.find_last_of("\\/") + 1) + name;

    FILE *fp = fopen(p.c_str(), "wb");
    if (fp == nullptr)
        return "FAILED to write " + p;
    fwrite(text.data(), 1, text.size(), fp);
    fclose(fp);
    return p;
}

static void save_diff()
{
    std::string summary;
    const std::string text = build_diff(summary);
    const std::string where = write_next_to_exe("compass_probe_diff.txt", text);
    {
        std::lock_guard<std::mutex> lock(g_cap_mutex);
        g_status = "Diff saved: " + summary + " -> " + where;
    }
    MessageBeep(MB_ICONASTERISK);
}

static std::string hits_report()
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

// ---------------------------------------------------------------------------
// Buffer write scanning (path: 0 = update_buffer_region, 1 = map/unmap)
// ---------------------------------------------------------------------------
static void scan(device *dev, resource res, uint64_t base, const void *data, uint64_t size, int path)
{
    if (data == nullptr || size < 12)
        return;

    g_seen.fetch_add(1, std::memory_order_relaxed);
    uint64_t prev = g_max_size.load(std::memory_order_relaxed);
    while (size > prev && !g_max_size.compare_exchange_weak(prev, size)) {}

    if (size > SCAN_CAP)
    {
        g_skip_big.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    const resource_desc desc = dev->get_resource_desc(res);
    if (desc.type != resource_type::buffer)
        return;
    if (S.cb_only && (desc.usage & resource_usage::constant_buffer) == 0)
    {
        g_skip_notcb.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    g_writes.fetch_add(1, std::memory_order_relaxed);

    const float *f = static_cast<const float *>(data);
    const size_t n = static_cast<size_t>(size / 4);

    if (g_capturing.load() >= 0)
        track(S.group_by_size ? desc.buffer.size : res.handle, desc, base, f, n, path);

    search_position(desc, base, f, n);
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
struct MapInfo { void *ptr; uint64_t offset; uint64_t size; };
static std::unordered_map<uint64_t, MapInfo> g_maps;
static std::mutex g_maps_mutex;

static bool on_update(device *dev, const void *data, resource dest, uint64_t offset, uint64_t size)
{
    if (size == UINT64_MAX)   // ReShade 6.7+: "from offset to end of buffer"
        size = dev->get_resource_desc(dest).buffer.size - offset;
    scan(dev, dest, offset, data, size, 0);
    return false;
}

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
    if (size == UINT64_MAX)
        size = dev->get_resource_desc(res).buffer.size - mi.offset;
    scan(dev, res, mi.offset, mi.ptr, size, 1);
}

static bool key_down(int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; }

// Called once per presented frame (for every window ReShade is attached to).
static void on_present(effect_runtime *)
{
    static bool prev_a = false, prev_b = false, prev_s = false;

    const bool mod = key_down(VK_CONTROL) && key_down(VK_SHIFT);
    const bool a = mod && key_down(KEY_CAPTURE_A);
    const bool b = mod && key_down(KEY_CAPTURE_B);
    const bool s = mod && key_down(KEY_SAVE_DIFF);

    if (a && !prev_a) start_capture(0);
    if (b && !prev_b) start_capture(1);
    if (s && !prev_s) save_diff();

    prev_a = a; prev_b = b; prev_s = s;
}

// ---------------------------------------------------------------------------
// Overlay (optional)
// ---------------------------------------------------------------------------
static std::string g_saved_path;

static void save_hits()
{
    g_saved_path = write_next_to_exe("compass_probe_report.txt", hits_report());
}

static void draw_overlay(effect_runtime *)
{
    ImGui::TextUnformatted("Hotkeys (hold Ctrl+Shift): F7 = capture A, F8 = capture B, F9 = save diff");
    {
        std::lock_guard<std::mutex> lock(g_cap_mutex);
        ImGui::TextWrapped("%s", g_status.c_str());
    }
    if (ImGui::Button("Capture A")) start_capture(0);
    ImGui::SameLine();
    if (ImGui::Button("Capture B")) start_capture(1);
    ImGui::SameLine();
    if (ImGui::Button("Save diff")) save_diff();

    ImGui::Separator();
    ImGui::InputFloat3("Camera position A", S.pos, "%.3f");
    ImGui::InputFloat3("Camera position B", S.posB, "%.3f");
    ImGui::SliderFloat("Tolerance", &S.tol, 0.001f, 0.1f, "%.3f");
    ImGui::Checkbox("Constant buffers only", &S.cb_only);
    ImGui::Checkbox("Group buffers by size", &S.group_by_size);
    ImGui::Text("Writes seen: %llu  scanned: %llu  too big: %llu  largest: %llu B",
                (unsigned long long)g_seen.load(), (unsigned long long)g_writes.load(),
                (unsigned long long)g_skip_big.load(), (unsigned long long)g_max_size.load());

    if (ImGui::Button("Clear position hits"))
    {
        std::lock_guard<std::mutex> lock(g_hits_mutex);
        g_hits.clear();
    }
    ImGui::SameLine();
    if (ImGui::Button("Save position hits to file"))
        save_hits();
    if (!g_saved_path.empty())
        ImGui::TextWrapped("Saved to: %s", g_saved_path.c_str());

    std::lock_guard<std::mutex> lock(g_hits_mutex);
    ImGui::Text("Position hits: %zu", g_hits.size());
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
        reshade::register_event<reshade::addon_event::reshade_present>(on_present);
        reshade::register_overlay(nullptr, draw_overlay);
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_addon(hModule);
        break;
    }
    return TRUE;
}
