#include <atomic>
#include <cstdint>

// a lockless ring queue
// safe with one producer and multiple consumer
// it's recomended that T be within 64 bytes.
template <typename T>
class LLRQ {
private:

    T* data;
    size_t capacity;

    size_t prod_head;
    std::atomic<size_t> cons_head;
    std::atomic<size_t> cons_tail;

    size_t nextof(size_t curr_pos) {
        return (curr_pos + 1) % capacity;
    }

public:

    LLRQ(size_t cap): capacity(cap), prod_head(0), cons_head(0), cons_tail(0) {
        data = new T[cap + 1];
    }

    bool push(const T& elem) {
        size_t nextof_prod_head = nextof(prod_head);
        if (cons_tail.load(std::memory_order_relaxed) == nextof_prod_head) {
            return false;
        }
        data[prod_head] = elem;
        ++prod_head;
        return true;
    }

    bool pop(T& output) {
        bool success = false;
        size_t nextof_curr_cons_head;

        size_t curr_cons_head = cons_head.load(std::memory_order_relaxed);
        size_t curr_prod_head = prod_head.load(std::memory_order_relaxed);
        if (curr_cons_head != curr_prod_head) {
            nextof_curr_cons_head = nextof(curr_cons_head);
            success = cons_head.compare_exchange_weak(curr_cons_head, nextof_curr_cons_head, std::memory_order_relaxed);
        }
        if (!success) return false;

        output = data[curr_cons_head];

        while(!cons_tail.compare_exchange_weak(curr_cons_head, nextof_curr_cons_head, std::memory_order_relaxed));

        return true;
    }
};