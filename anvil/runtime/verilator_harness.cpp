// Verilator harness for the `gpu` top: runs one Anvil Job and writes per-launch cycles, counters and final memory.
//
// It replays sim/test/helpers/{runner,setup,memory}.py at the cycle level under cocotb's timing on iverilog:
// at each rising edge the testbench reads the DUT's pre-edge outputs, and its writes land after the edge (cocotb
// queues them to the ReadWrite phase), so the DUT samples them at the next edge. Per launch (setup.launch, runner._run):
//   reset=1 start=0 dcwe=0 | edge | edge | reset=0 dcwe=1 dcd=threads | edge | dcwe=0 start=1, then each edge:
//   stop if done, else run the data and program memory models on the pre-edge outputs; cycles counts those edges.
//
// Usage: Vgpu <job.txt> <result.json>. job.txt: mem_latency timeout ndata d... nlaunch {threads nprog p...}...
// Compile with -DADDR_BITS=<DATA_MEM_ADDR_BITS> -DDATA_BITS=<DATA_MEM_DATA_BITS>.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "Vgpu.h"
#include "verilated.h"

#ifndef ADDR_BITS
#error "define ADDR_BITS"
#endif
#ifndef DATA_BITS
#error "define DATA_BITS"
#endif

static constexpr int CH = 4;   // DATA_MEM_NUM_CHANNELS
static inline uint64_t mask(int w) { return w >= 64 ? ~0ull : ((1ull << w) - 1); }

// Bit fields of a Verilator port: scalar (CData..QData) or VlWide.
template <typename T> static inline uint64_t get(const T& v, int lsb, int w) { return (uint64_t(v) >> lsb) & mask(w); }
template <std::size_t N> static inline uint64_t get(const VlWide<N>& v, int lsb, int w) {
    uint64_t r = 0;
    for (int b = 0; b < w;) {
        int i = (lsb + b) / 32, o = (lsb + b) % 32, n = std::min(32 - o, w - b);
        r |= (uint64_t((v[i] >> o) & mask(n))) << b;
        b += n;
    }
    return r;
}
template <typename T> static inline void put(T& v, int lsb, int w, uint64_t x) {
    v = T((uint64_t(v) & ~(mask(w) << lsb)) | ((x & mask(w)) << lsb));
}
template <std::size_t N> static inline void put(VlWide<N>& v, int lsb, int w, uint64_t x) {
    for (int b = 0; b < w;) {
        int i = (lsb + b) / 32, o = (lsb + b) % 32, n = std::min(32 - o, w - b);
        uint32_t m = uint32_t(mask(n) << o);
        v[i] = (v[i] & ~m) | (uint32_t(((x >> b) & mask(n)) << o) & m);
        b += n;
    }
}

struct Sample {   // DUT outputs before an edge
    bool done;
    uint32_t p_valid, p_addr;
    uint32_t r_valid, w_valid, r_addr[CH], w_addr[CH];
    uint64_t w_data[CH];
    uint32_t counters[8];
};

static const char* COUNTER_NAMES[8] = {"data_mem_requests", "data_mem_requests_coalesced", "icache_hits",
                                       "icache_misses", "dcache_hits", "dcache_misses", "warp_issues_total",
                                       "bar_stall_cycles_total"};

struct Harness {
    std::unique_ptr<VerilatedContext> ctx{new VerilatedContext};
    std::unique_ptr<Vgpu> top{new Vgpu{ctx.get()}};

    Sample sample() {
        Sample s;
        s.done = top->done & 1;
        s.p_valid = top->program_mem_read_valid & 1;
        s.p_addr = uint32_t(get(top->program_mem_read_address, 0, 8));
        s.r_valid = top->data_mem_read_valid;
        s.w_valid = top->data_mem_write_valid;
        for (int c = 0; c < CH; c++) {
            s.r_addr[c] = uint32_t(get(top->data_mem_read_address, c * ADDR_BITS, ADDR_BITS));
            s.w_addr[c] = uint32_t(get(top->data_mem_write_address, c * ADDR_BITS, ADDR_BITS));
            s.w_data[c] = get(top->data_mem_write_data, c * DATA_BITS, DATA_BITS);
        }
        uint32_t cs[8] = {top->data_mem_requests, top->data_mem_requests_coalesced, top->icache_hits_total,
                          top->icache_misses_total, top->dcache_hits_total, top->dcache_misses_total,
                          top->warp_issues_total, top->bar_stall_cycles_total};
        for (int i = 0; i < 8; i++) s.counters[i] = cs[i];
        return s;
    }
    // One rising edge: returns the pre-edge outputs; the inputs set before it are the ones the DUT samples.
    Sample edge() {
        Sample s = sample();
        top->clk = 1;
        top->eval();
        top->clk = 0;
        top->eval();
        return s;
    }
    void settle() { top->eval(); }
};

// memory.py's Memory: per channel, a transaction is serviced after `latency` waiting edges; a still-high valid
// after service is not serviced again until valid drops.
struct Chan { int pending_r = 0, pending_w = 0; bool served_r = false, served_w = false; };

struct DataMemory {
    std::vector<uint64_t> mem = std::vector<uint64_t>(1u << ADDR_BITS, 0);
    Chan ch[CH];
    int latency = 0;
    uint64_t reads = 0, writes = 0;

    void reset_channels() { for (auto& c : ch) c = Chan{}; }
    void run(const Sample& s, Vgpu* top) {
        uint32_t ready = 0;
        for (int c = 0; c < CH; c++) {
            uint64_t data = 0;
            bool v = (s.r_valid >> c) & 1;
            if (v && ch[c].served_r) {
            } else if (v) {
                if (ch[c].pending_r < latency) ch[c].pending_r++;
                else {
                    data = mem[s.r_addr[c]];
                    ready |= 1u << c;
                    ch[c].pending_r = 0;
                    ch[c].served_r = true;
                    reads++;
                }
            } else {
                ch[c].pending_r = 0;
                ch[c].served_r = false;
            }
            put(top->data_mem_read_data, c * DATA_BITS, DATA_BITS, data);
        }
        top->data_mem_read_ready = ready;
        uint32_t wready = 0;
        for (int c = 0; c < CH; c++) {   // channel 0 first, so the highest channel wins (ADR-016)
            bool v = (s.w_valid >> c) & 1;
            if (v && ch[c].served_w) {
            } else if (v) {
                if (ch[c].pending_w < latency) ch[c].pending_w++;
                else {
                    mem[s.w_addr[c]] = s.w_data[c] & mask(DATA_BITS);
                    wready |= 1u << c;
                    ch[c].pending_w = 0;
                    ch[c].served_w = true;
                    writes++;
                }
            } else {
                ch[c].pending_w = 0;
                ch[c].served_w = false;
            }
        }
        top->data_mem_write_ready = wready;
    }
};

struct ProgramMemory {
    std::vector<uint32_t> mem = std::vector<uint32_t>(256, 0);
    Chan ch;
    int latency = 0;
    void run(const Sample& s, Vgpu* top) {
        uint32_t data = 0, ready = 0;
        if (s.p_valid && ch.served_r) {
        } else if (s.p_valid) {
            if (ch.pending_r < latency) ch.pending_r++;
            else {
                data = mem[s.p_addr & 0xFF];
                ready = 1;
                ch.pending_r = 0;
                ch.served_r = true;
            }
        } else {
            ch.pending_r = 0;
            ch.served_r = false;
        }
        put(top->program_mem_read_data, 0, 16, data);
        top->program_mem_read_ready = ready;
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s job.txt result.json\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1]);
    long latency, timeout, ndata, nlaunch;
    in >> latency >> timeout >> ndata;
    Harness h;
    Vgpu* top = h.top.get();
    DataMemory dm;
    dm.latency = int(latency);
    for (long i = 0; i < ndata; i++) {
        uint64_t w;
        in >> w;
        dm.mem[i] = w & mask(DATA_BITS);
    }
    in >> nlaunch;
    std::string out = "{\"launches\": [";
    top->clk = 0;
    top->data_mem_read_ready = 0;
    top->data_mem_write_ready = 0;
    top->program_mem_read_ready = 0;
    for (long l = 0; l < nlaunch; l++) {
        long threads, nprog;
        in >> threads >> nprog;
        ProgramMemory pm;
        pm.latency = int(latency);
        for (long i = 0; i < nprog; i++) in >> pm.mem[i];
        dm.reset_channels();
        top->program_mem_read_ready = 0;   // Memory(...) for the program writes ready = 0
        // setup.launch
        top->reset = 1;
        top->start = 0;
        top->device_control_write_enable = 0;
        top->device_control_data = 0;
        h.settle();
        h.edge();
        h.edge();
        top->reset = 0;
        top->device_control_write_enable = 1;
        top->device_control_data = uint8_t(threads);
        h.settle();
        Sample s = h.edge();
        top->device_control_write_enable = 0;
        top->start = 1;
        // runner._run
        long cycles = 0;
        while (!s.done) {
            dm.run(s, top);
            pm.run(s, top);
            h.settle();
            s = h.edge();
            if (++cycles >= timeout) {
                std::fprintf(stderr, "TimeoutError: Kernel did not complete in %ld cycles\n", timeout);
                return 3;
            }
        }
        if (s.counters[0] != dm.reads + dm.writes) {
            std::fprintf(stderr, "data_mem_requests %u != %llu reads + %llu writes\n", s.counters[0],
                         (unsigned long long)dm.reads, (unsigned long long)dm.writes);
            return 4;
        }
        dm.reads = dm.writes = 0;
        out += l ? ", " : "";
        out += "{\"cycles\": " + std::to_string(cycles) + ", \"counters\": {";
        for (int i = 0; i < 8; i++)
            out += std::string(i ? ", " : "") + "\"" + COUNTER_NAMES[i] + "\": " + std::to_string(s.counters[i]);
        out += "}}";
    }
    top->start = 0;
    out += "], \"memory\": [";
    for (size_t i = 0; i < dm.mem.size(); i++) out += (i ? "," : "") + std::to_string(dm.mem[i]);
    out += "]}\n";
    h.top->final();
    std::ofstream(argv[2]) << out;
    return 0;
}
