#pragma once
#include <thread>
#include <functional>
#include <mutex>
#include <vector>
#include <queue>
#include <atomic>

class ThreadPool {
private:
    size_t threadCounts;
    std::vector<std::thread> threads{};
    std::queue<std::function<void()>> tasks{};

    std::mutex tasksMutex;
    std::condition_variable cv;
    std::atomic_bool stop{false};
public:
    inline ThreadPool(size_t threadCounts);

    template <class F, class... Args>
    void addTasks(F&& f, Args&&... args);

    inline size_t getThreadCounts() const noexcept;
    inline size_t getWaitingTaskCounts() const noexcept;

    inline ~ThreadPool() noexcept;
};

// ==============================
//         Definition
// ==============================

template <class F, class... Args>
void ThreadPool::addTasks(F&& f, Args&&... args) {
    {
        std::unique_lock<std::mutex> lock(tasksMutex);
        tasks.emplace(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    }
    cv.notify_one();
}

inline size_t ThreadPool::getThreadCounts() const noexcept {
    return threadCounts;
}

inline size_t ThreadPool::getWaitingTaskCounts() const noexcept {
    return tasks.size();
}

inline ThreadPool::ThreadPool(size_t threadCounts): threadCounts{threadCounts} {
    for (int i = 0; i < threadCounts; ++i) {
        threads.emplace_back([this](){
            std::function<void()> task;
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(this->tasksMutex);
                    this->cv.wait(lock, [this](){
                        return this->stop || !this->tasks.empty();
                    });
                    if (this->stop && this->tasks.empty()) return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
            }
        });
    }
}

inline ThreadPool::~ThreadPool() noexcept {
    stop = true;
    cv.notify_all();
    for (auto& _thread: threads)
        _thread.join();
}

