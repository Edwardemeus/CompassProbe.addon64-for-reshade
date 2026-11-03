// compass_probe.cpp  (v5)
// ReShade add-on: find where Roblox Studio keeps the camera in its D3D11
// constant buffers.
//
// How it works
//   * Hold the camera still in state A and press Ctrl+Shift+F7 (2.5 s capture),
//     put the camera in state B and press Ctrl+Shift+F8, then press
//     Ctrl+Shift+F9: a report is written next to the Studio exe.
//   * During a capture the add-on
//       1. records which buffer/offset/size is bound to every constant-buffer
//          slot at every draw call (binding census),
//       2. copies each distinct bound range (<= 16 KB) back from the GPU and
//          compares state A with state B (readback diff),
//       3. keeps the older CPU-side "what was written" diff and a census of the
//          big (> 64 KB) buffers.
//   * Beeps confirm each step, so the ReShade overlay is not required.
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
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace reshade::api;

extern "C" __declspec(dllexport) const char *NAME = "Roblox camera probe v5";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "v5: finds the camera data inside Studio's constant buffers (bound-buffer readback A/B diff).";

// ---------------------------------------------------------------------------
// Hotkeys: hold Ctrl+Shift, then press the F-key
// ---------------------------------------------------------------------------
static const int       KEY_CAPTURE_A = VK_F7;
static const int       KEY_CAPTURE_B = VK_F8;
static const int       KEY_SAVE_DIFF = VK_F9;
static const ULONGLONG CAPTURE_MS     = 2500;            // length of one capture
static const uint64_t  SCAN_CAP       = 65536;           // CPU-side writes bigger than this are skipped
static const uint64_t  STAGING_SIZE   = 1u << 20;        // 1 MB readback staging buffer per device
static const size_t    MAX_READS      = 48;              // max ranges read back per frame
static const uint64_t  MAX_READ_RANGE = 16384;           // bound ranges bigger than this are not read back

// ---------------------------------------------------------------------------
// Settings (editable in the overlay if you can reach it)
// ---------------------------------------------------------------------------
static struct {
    float pos[3]  = { -1.080f, 6.822f, 25.367f };  // camera position in state A
    float posB[3] = { -3.080f, 6.822f, 23.367f };  // camera position in state B
    float tol = 0.01f;
    bool  cb_only = false;         // false = look at every buffer type (CPU-side writes)
    bool  group_by_size = false;   // merge all buffers of the same size (CPU-side writes)
} S;

static std::atomic<uint64_t> g_writes{ 0 };        // CPU-side writes that were scanned
static std::atomic<uint64_t> g_seen{ 0 };          // all buffer writes seen
static std::atomic<uint64_t> g_skip_big{ 0 };      // skipped: bigger than SCAN_CAP
static std::atomic<uint64_t> g_skip_notcb{ 0 };    // skipped: not a constant buffer (cb_only)
static std::atomic<uint64_t> g_max_size{ 0 };      // largest write size seen
static std::string g_status = "Idle. Hotkeys: Ctrl+Shift+F7 = capture A, F8 = capture B, F9 = save diff.";

// ---------------------------------------------------------------------------
// Part 1: position search (CPU-side writes)
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
// Shared tracking structures
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
    uint64_t writes = 0, via_map = 0, via_update = 0;   // for readback classes: writes = number of reads
    std::string info;
    std::vector<Loc> locs;         // indexed by (byte offset / 4)
};

static void add_value(Loc &L, float v)
{
    if (!std::isfinite(v))
        return;
    if (L.n == 0) { L.mn = L.mx = v; }
    else { if (v < L.mn) L.mn = v; if (v > L.mx) L.mx = v; }
    L.sum += v;
    L.n++;
}

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

static const char *stage_name(uint32_t si)
{
    static const char *n[6] = { "VS", "HS", "DS", "GS", "PS", "CS" };
    return si < 6 ? n[si] : "??";
}

// Everything below that is capture-related is guarded by g_cap_mutex.
static std::unordered_map<uint64_t, BufTrack> g_cap[2];   // CPU-side writes: 0 = state A, 1 = state B
static std::unordered_map<uint64_t, BufTrack> g_rb[2];    // GPU readback of bound constant buffers
static std::atomic<int> g_capturing{ -1 };                 // -1 = off, 0 = A, 1 = B
static ULONGLONG g_cap_start = 0;
static std::mutex g_cap_mutex;                             // also guards g_status, g_bound, g_dev, g_buf_info

// ---- binding census ----
using BindKey = std::tuple<uint32_t, uint32_t, uint64_t, uint64_t, uint64_t>;   // stage, slot, buffer, offset, size
static std::map<BindKey, uint64_t> g_bind_census[2];       // draws per binding
struct BufMeta { uint64_t size = 0; std::string info; };
static std::unordered_map<uint64_t, BufMeta> g_buf_info;   // buffer handle -> size + usage

struct BoundCB { uint64_t buffer = 0, offset = 0, size = 0; };
struct BoundTable { BoundCB cb[6][16]; };
static std::unordered_map<device *, BoundTable> g_bound;

// ---- readback state, one per device ----
struct Pending { uint32_t stage; uint32_t slot; uint64_t size; uint64_t off; };
struct DevState {
    resource staging{};
    bool staging_failed = false;
    uint64_t used = 0;
    int pending_which = -1;
    std::vector<Pending> pending;
    std::set<BindKey> seen;                                // ranges already queued this frame
};
static std::unordered_map<device *, DevState> g_dev;

// ---- big-buffer census (always on, guarded by g_big_mutex) ----
struct BigInfo {
    uint64_t size = 0;
    std::string info;
    uint64_t writes = 0, via_map = 0, via_update = 0;
};
static std::unordered_map<uint64_t, BigInfo> g_big;
static std::mutex g_big_mutex;

static void note_big(device *dev, resource res, int path)
{
    std::lock_guard<std::mutex> lock(g_big_mutex);
    BigInfo &b = g_big[res.handle];
    if (b.info.empty())
    {
        const resource_desc d = dev->get_resource_desc(res);
        b.size = d.buffer.size;
        b.info = describe(d);
    }
    b.writes++;
    if (path == 0) b.via_update++; else b.via_map++;
}

// Returns the active capture (0/1) or -1; ends the capture when its time is up.
// Caller must hold g_cap_mutex.
static int capture_which_locked()
{
    const int which = g_capturing.load();
    if (which < 0)
        return -1;
    if (GetTickCount64() - g_cap_start > CAPTURE_MS)
    {
        g_capturing = -1;
        g_status = std::string("Capture ") + (which == 0 ? "A" : "B") + " finished ("
                 + std::to_string(g_cap[which].size()) + " written buffers, "
                 + std::to_string(g_rb[which].size()) + " read-back classes).";
        MessageBeep(MB_ICONEXCLAMATION);
        return -1;
    }
    return which;
}

// CPU-side write tracking (path: 0 = update_buffer_region, 1 = map/unmap)
static void track(uint64_t id, const resource_desc &desc, uint64_t base,
                  const float *f, size_t n, int path)
{
    std::lock_guard<std::mutex> lock(g_cap_mutex);
    const int which = capture_which_locked();
    if (which < 0)
        return;

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
        add_value(bt.locs[first + i], f[i]);
}

static void start_capture(int which)
{
    std::lock_guard<std::mutex> lock(g_cap_mutex);
    g_cap[which].clear();
    g_rb[which].clear();
    g_bind_census[which].clear();
    g_cap_start = GetTickCount64();
    g_capturing = which;
    g_status = std::string("Capturing state ") + (which == 0 ? "A" : "B")
             + " ... keep the camera still until the second beep.";
    MessageBeep(MB_OK);
}

// ---------------------------------------------------------------------------
// Binding tracking + draw hook + readback
// ---------------------------------------------------------------------------
static void on_push_descriptors(command_list *cmd, shader_stage stages, pipeline_layout,
                                uint32_t, const descriptor_table_update &update)
{
    if (update.type != descriptor_type::constant_buffer || update.descriptors == nullptr)
        return;

    device *dev = cmd->get_device();
    const buffer_range *r = static_cast<const buffer_range *>(update.descriptors);
    const uint32_t bits = static_cast<uint32_t>(stages);

    std::lock_guard<std::mutex> lock(g_cap_mutex);
    BoundTable &t = g_bound[dev];

    for (uint32_t i = 0; i < update.count; ++i)
    {
        const uint32_t slot = update.binding + i;
        if (slot >= 16)
            break;

        BoundCB b;
        b.buffer = r[i].buffer.handle;
        b.offset = r[i].offset;
        b.size = r[i].size;

        if (b.buffer != 0)
        {
            auto it = g_buf_info.find(b.buffer);
            if (it == g_buf_info.end())
            {
                const resource_desc d = dev->get_resource_desc(r[i].buffer);
                BufMeta m;
                m.size = d.buffer.size;
                m.info = describe(d);
                it = g_buf_info.emplace(b.buffer, m).first;
            }
            const uint64_t total = it->second.size;
            if (b.size == UINT64_MAX || b.size == 0 || b.offset + b.size > total)
                b.size = total > b.offset ? total - b.offset : 0;
        }

        for (uint32_t si = 0; si < 6; ++si)
            if (bits & (1u << si))
                t.cb[si][slot] = b;
    }
}

static void on_draw_common(command_list *cmd)
{
    if (g_capturing.load() < 0)
        return;

    device *dev = cmd->get_device();
    std::lock_guard<std::mutex> lock(g_cap_mutex);
    const int which = capture_which_locked();
    if (which < 0)
        return;

    auto bt = g_bound.find(dev);
    if (bt == g_bound.end())
        return;
    DevState &ds = g_dev[dev];

    for (uint32_t si = 0; si < 6; ++si)
    {
        for (uint32_t slot = 0; slot < 16; ++slot)
        {
            const BoundCB &b = bt->second.cb[si][slot];
            if (b.buffer == 0 || b.size == 0)
                continue;

            const BindKey bk(si, slot, b.buffer, b.offset, b.size);
            g_bind_census[which][bk]++;

            // queue a GPU readback of this range (once per frame)
            if (b.size < 16 || b.size > MAX_READ_RANGE || ds.pending.size() >= MAX_READS)
                continue;
            const uint64_t aligned = (b.size + 15) & ~uint64_t(15);
            if (ds.used + aligned > STAGING_SIZE)
                continue;
            if (!ds.seen.insert(bk).second)
                continue;

            if (ds.staging.handle == 0 && !ds.staging_failed)
            {
                const resource_desc d(STAGING_SIZE, memory_heap::gpu_to_cpu, resource_usage::copy_dest);
                if (!dev->create_resource(d, nullptr, resource_usage::copy_dest, &ds.staging))
                {
                    ds.staging = {};
                    ds.staging_failed = true;
                }
            }
            if (ds.staging.handle == 0)
                continue;

            cmd->copy_buffer_region(resource{ b.buffer }, b.offset, ds.staging, ds.used, b.size);
            ds.pending.push_back({ si, slot, b.size, ds.used });
            ds.pending_which = which;
            ds.used += aligned;
        }
    }
}

static bool on_draw(command_list *cmd, uint32_t, uint32_t, uint32_t, uint32_t)
{
    on_draw_common(cmd);
    return false;
}

static bool on_draw_indexed(command_list *cmd, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)
{
    on_draw_common(cmd);
    return false;
}

// Called at present time: read the copied ranges back and fold them into the
// per-class value ranges (class = stage + slot + range size).
static void process_readbacks(device *dev)
{
    std::lock_guard<std::mutex> lock(g_cap_mutex);
    auto it = g_dev.find(dev);
    if (it == g_dev.end())
        return;
    DevState &ds = it->second;
    ds.seen.clear();

    if (ds.pending.empty() || ds.staging.handle == 0 || ds.pending_which < 0)
    {
        ds.pending.clear();
        ds.used = 0;
        return;
    }

    void *ptr = nullptr;
    if (dev->map_buffer_region(ds.staging, 0, ds.used, map_access::read_only, &ptr) && ptr != nullptr)
    {
        const uint8_t *base = static_cast<const uint8_t *>(ptr);
        const int which = ds.pending_which;
        for (const Pending &p : ds.pending)
        {
            const uint64_t cls = (uint64_t(p.stage) << 56) | (uint64_t(p.slot) << 48) | p.size;
            BufTrack &bt = g_rb[which][cls];
            if (bt.info.empty())
            {
                char b[64];
                snprintf(b, sizeof b, "%s slot %u, %llu B", stage_name(p.stage), p.slot,
                         (unsigned long long)p.size);
                bt.info = b;
                bt.size = p.size;
            }
            bt.writes++;

            const float *f = reinterpret_cast<const float *>(base + p.off);
            const size_t n = static_cast<size_t>(p.size / 4);
            if (bt.locs.size() < n)
                bt.locs.resize(n);
            for (size_t i = 0; i < n; ++i)
                add_value(bt.locs[i], f[i]);
        }
        dev->unmap_buffer_region(ds.staging);
    }
    ds.pending.clear();
    ds.used = 0;
}

// ---------------------------------------------------------------------------
// Diff + report
// ---------------------------------------------------------------------------
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

static std::string diff_section(const std::unordered_map<uint64_t, BufTrack> &MA,
                                const std::unordered_map<uint64_t, BufTrack> &MB,
                                const char *label, int &pos_tags,
                                std::set<uint64_t> &changed_ids, size_t &n_changed)
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

    for (const auto &kv : MA)
    {
        auto itB = MB.find(kv.first);
        if (itB == MB.end())
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
    n_changed = entries.size();

    snprintf(line, sizeof line, "%zu of %zu %s (present in A) have values whose range differs between A and B.\n",
             entries.size(), MA.size(), label);
    s += line;

    for (size_t k = 0; k < entries.size() && k < 12; ++k)
    {
        const Entry &e = entries[k];
        changed_ids.insert(e.id);
        snprintf(line, sizeof line, "\n== %s (size %llu bytes): %zu changed values ==\n",
                 e.a->info.c_str(), (unsigned long long)e.a->size, e.changed.size());
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
    return s;
}

static std::string class_census(const std::unordered_map<uint64_t, BufTrack> &MA,
                                const std::unordered_map<uint64_t, BufTrack> &MB,
                                const std::set<uint64_t> &changed_ids, const char *what)
{
    struct Row { uint64_t id; const BufTrack *a; const BufTrack *b; };
    std::vector<Row> rows;
    for (const auto &kv : MA)
    {
        auto itB = MB.find(kv.first);
        rows.push_back({ kv.first, &kv.second, itB == MB.end() ? nullptr : &itB->second });
    }
    std::sort(rows.begin(), rows.end(),
              [](const Row &x, const Row &y) { return x.a->writes > y.a->writes; });

    std::string s;
    char line[320];
    snprintf(line, sizeof line, "\n== census: %s (sorted by count) ==\n", what);
    s += line;
    size_t listed = 0;
    for (const Row &r : rows)
    {
        if (listed++ >= 60) { s += "  ...\n"; break; }
        snprintf(line, sizeof line, "  %-34s %7llu B  count A=%llu (update %llu / map %llu)  B=%llu%s\n",
                 r.a->info.c_str(), (unsigned long long)r.a->size,
                 (unsigned long long)r.a->writes, (unsigned long long)r.a->via_update,
                 (unsigned long long)r.a->via_map,
                 (unsigned long long)(r.b ? r.b->writes : 0),
                 changed_ids.count(r.id) ? "   <-- has changed values" : "");
        s += line;
    }
    return s;
}

static std::string bind_census(int which)
{
    struct Agg {
        uint64_t draws = 0;
        size_t distinct = 0;
        uint64_t min_size = UINT64_MAX, max_size = 0;
        BindKey top{};
        uint64_t top_draws = 0;
    };
    std::map<std::pair<uint32_t, uint32_t>, Agg> agg;
    for (const auto &kv : g_bind_census[which])
    {
        const BindKey &k = kv.first;
        Agg &a = agg[{ std::get<0>(k), std::get<1>(k) }];
        a.draws += kv.second;
        a.distinct++;
        a.min_size = std::min(a.min_size, std::get<4>(k));
        a.max_size = std::max(a.max_size, std::get<4>(k));
        if (kv.second > a.top_draws) { a.top_draws = kv.second; a.top = k; }
    }

    std::string s;
    char line[512];
    s += "\n== bound constant buffers: draws per (stage, slot) in this state ==\n";
    for (const auto &kv : agg)
    {
        const Agg &a = kv.second;
        const uint64_t h = std::get<2>(a.top);
        auto mi = g_buf_info.find(h);
        snprintf(line, sizeof line,
                 "  %s slot %2u: %6llu draws, %5zu distinct bindings, sizes %llu..%llu B | busiest: buffer %llu "
                 "(%s, %llu B total) offset %llu size %llu, %llu draws\n",
                 stage_name(kv.first.first), kv.first.second, (unsigned long long)a.draws, a.distinct,
                 (unsigned long long)a.min_size, (unsigned long long)a.max_size,
                 (unsigned long long)h, mi != g_buf_info.end() ? mi->second.info.c_str() : "?",
                 (unsigned long long)(mi != g_buf_info.end() ? mi->second.size : 0),
                 (unsigned long long)std::get<3>(a.top), (unsigned long long)std::get<4>(a.top),
                 (unsigned long long)a.top_draws);
        s += line;
    }
    if (agg.empty())
        s += "  (no constant-buffer bindings were seen: push_descriptors did not fire)\n";
    return s;
}

static std::string big_census()
{
    std::lock_guard<std::mutex> lock(g_big_mutex);
    std::vector<std::pair<uint64_t, const BigInfo *>> rows;
    for (const auto &kv : g_big)
        rows.push_back({ kv.first, &kv.second });
    std::sort(rows.begin(), rows.end(),
              [](const auto &x, const auto &y) { return x.second->writes > y.second->writes; });

    std::string s = "\n== big buffers: writes larger than 64 KB (counted since Studio started) ==\n";
    char line[320];
    size_t listed = 0;
    for (const auto &r : rows)
    {
        if (listed++ >= 15) { s += "  ...\n"; break; }
        snprintf(line, sizeof line, "  buffer %llu: %8llu B  %-24s writes %llu (update %llu / map %llu)\n",
                 (unsigned long long)r.first, (unsigned long long)r.second->size, r.second->info.c_str(),
                 (unsigned long long)r.second->writes, (unsigned long long)r.second->via_update,
                 (unsigned long long)r.second->via_map);
        s += line;
    }
    if (rows.empty())
        s += "  (none)\n";
    return s;
}

static std::string build_diff(std::string &summary)
{
    std::string s;
    char line[640];
    int pos_tags = 0;
    size_t n_rb = 0, n_wr = 0;
    std::set<uint64_t> rb_changed, wr_changed;

    std::lock_guard<std::mutex> lock(g_cap_mutex);

    snprintf(line, sizeof line,
             "Position A = %.3f %.3f %.3f   Position B = %.3f %.3f %.3f   tol %.3f\n"
             "CPU-side writes seen: %llu, scanned: %llu, skipped as bigger than %llu B: %llu "
             "(largest write: %llu B), skipped as non-constant-buffer: %llu (cb_only=%d)\n"
             "'~' after a value = it jittered during the capture. Tags <-POS.x/y/z matched the camera position.\n"
             "Each row is one float4 register.\n",
             S.pos[0], S.pos[1], S.pos[2], S.posB[0], S.posB[1], S.posB[2], S.tol,
             (unsigned long long)g_seen.load(), (unsigned long long)g_writes.load(),
             (unsigned long long)SCAN_CAP, (unsigned long long)g_skip_big.load(),
             (unsigned long long)g_max_size.load(), (unsigned long long)g_skip_notcb.load(),
             S.cb_only ? 1 : 0);
    s += line;

    s += "\n######## 1. READBACK DIFF: bound constant buffers read back from the GPU ########\n";
    s += diff_section(g_rb[0], g_rb[1], "read-back classes", pos_tags, rb_changed, n_rb);
    s += class_census(g_rb[0], g_rb[1], rb_changed, "read-back classes (stage, slot, size) and how often they were read");
    s += bind_census(0);

    s += "\n######## 2. WRITE DIFF: CPU-side buffer writes (as in v4) ########\n";
    s += diff_section(g_cap[0], g_cap[1], "written buffers", pos_tags, wr_changed, n_wr);
    s += class_census(g_cap[0], g_cap[1], wr_changed, "written buffers");

    s += "\n######## 3. BIG BUFFERS ########";
    s += big_census();

    summary = std::to_string(n_rb) + " read-back classes and " + std::to_string(n_wr)
            + " written buffers changed, " + std::to_string(pos_tags) + " position matches.";
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
    const std::string where = write_next_to_exe("compass_probe_v5_diff.txt", text);
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
// CPU-side buffer write scanning (path: 0 = update_buffer_region, 1 = map/unmap)
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
        note_big(dev, res, path);
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
static void on_present(effect_runtime *rt)
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

    process_readbacks(rt->get_device());
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
        reshade::register_event<reshade::addon_event::push_descriptors>(on_push_descriptors);
        reshade::register_event<reshade::addon_event::draw>(on_draw);
        reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
        reshade::register_event<reshade::addon_event::reshade_present>(on_present);
        reshade::register_overlay(nullptr, draw_overlay);
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_addon(hModule);
        break;
    }
    return TRUE;
}
