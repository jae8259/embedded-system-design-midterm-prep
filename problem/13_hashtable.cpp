#include <cstdio>
#include <vector>
#include <list>
#include <mutex>
#include <thread>
#include <chrono>

static const int BUCKETS  = 1024;
static const int NSTRIPS  = 64;
static const int NTHREADS = 8;
static const int NOPS     = 1 << 20; // 1M inserts

struct HashTable {
    std::vector<std::list<int>> buckets;
    std::mutex                  global_mtx;
    std::vector<std::mutex>     bucket_mtxs;
    std::vector<std::mutex>     strip_mtxs;

    HashTable() : buckets(BUCKETS), bucket_mtxs(BUCKETS), strip_mtxs(NSTRIPS) {}

    void reset() { for (auto& b : buckets) b.clear(); }

    int hash(int key) const { return (unsigned)key % BUCKETS; }

    // TODO: lock global_mtx, push key into buckets[hash(key)]
    void insert_coarse(int key) {
        // TODO
    }

    // TODO: lock bucket_mtxs[hash(key)], push key into buckets[hash(key)]
    void insert_fine(int key) {
        // TODO
    }

    // TODO: lock strip_mtxs[hash(key) % NSTRIPS], push key into buckets[hash(key)]
    void insert_strip(int key) {
        // TODO
    }

    int size() const {
        int s = 0;
        for (auto& b : buckets) s += (int)b.size();
        return s;
    }
};

static double run_bench(HashTable& ht, void (HashTable::*fn)(int)) {
    ht.reset();
    std::vector<std::thread> workers;
    int chunk = NOPS / NTHREADS;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < NTHREADS; t++) {
        int lo = t * chunk, hi = lo + chunk;
        workers.emplace_back([&ht, fn, lo, hi]() {
            for (int k = lo; k < hi; k++) (ht.*fn)(k);
        });
    }
    for (auto& w : workers) w.join();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    HashTable ht;

    // Single-threaded correctness: each method should insert NOPS keys
    printf("TEST01: hashtable insertion correctness\n");

    ht.reset();
    for (int k = 0; k < NOPS; k++) ht.insert_coarse(k);
    bool ok_coarse = (ht.size() == NOPS);

    ht.reset();
    for (int k = 0; k < NOPS; k++) ht.insert_fine(k);
    bool ok_fine = (ht.size() == NOPS);

    ht.reset();
    for (int k = 0; k < NOPS; k++) ht.insert_strip(k);
    bool ok_strip = (ht.size() == NOPS);

    printf("coarse: %s | fine: %s | strip: %s\n",
           ok_coarse ? "SUCCESS" : "FAIL",
           ok_fine   ? "SUCCESS" : "FAIL",
           ok_strip  ? "SUCCESS" : "FAIL");

    // Concurrent insertion benchmark (NTHREADS threads, NOPS total inserts)
    printf("TEST02:\n");
    double ms_coarse = run_bench(ht, &HashTable::insert_coarse);
    double ms_fine   = run_bench(ht, &HashTable::insert_fine);
    double ms_strip  = run_bench(ht, &HashTable::insert_strip);
    printf("AVG %.2fms (coarse) vs AVG %.2fms (fine) vs AVG %.2fms (strip)\n",
           ms_coarse, ms_fine, ms_strip);

    return 0;
}
