#pragma once
//
// Created by Yngvi Björnsson on 7.5.2022.
//

#include <vector>
#include <chrono>
#include <memory>
#include <limits>


namespace utils {
    /*
     * Manually non dereferenceable iterator (except with type as()).
    */
    struct [[maybe_unused]] void_ptr_iterator {
        using iterator_category [[maybe_unused]] = std::forward_iterator_tag;
        using difference_type [[maybe_unused]] = std::ptrdiff_t;
        using value_type [[maybe_unused]] = void;
        using pointer           = void*;

        // using reference       = void&;
        // reference operator*() const { return *ptr_; }
        // pointer operator->() { return ptr_; }

        void_ptr_iterator(void* ptr, int size) : ptr_(ptr), size_(size) {}

        template<class TT>
        TT* as() { return static_cast<TT*>(ptr_); }

        void_ptr_iterator& operator++() {
            ptr_ = static_cast<void*>(static_cast<char*>(ptr_) + size_);
            return *this;
        }

        void_ptr_iterator operator++(int) {
            void_ptr_iterator tmp(*this);
            ++(*this);
            return tmp;
        }

        void_ptr_iterator& operator+=(int n) {
            ptr_ = static_cast<void*>(static_cast<char*>(ptr_) + n * size_);
            return *this;
        }

        void_ptr_iterator operator+(int n) {
            void_ptr_iterator tmp(*this);
            tmp += n;
            return tmp;
        }

        friend bool operator== (const void_ptr_iterator& a, const void_ptr_iterator& b) { return a.ptr_ == b.ptr_; };
        friend bool operator!= (const void_ptr_iterator& a, const void_ptr_iterator& b) { return a.ptr_ != b.ptr_; };

    private:
        pointer ptr_;
        int size_;
    };


    /*
     * Simple object pool.
     */
    template<class T>
    class ObjectPool {
    public:

        class Deleter {
        public:
            explicit Deleter(ObjectPool<T>& op) : op_(op) {}
            void operator()(void* obj) {
                op_.dispose(static_cast<T*>(obj));
            }
        private:
            ObjectPool<T>& op_;
        };

        using smart_ptr = std::unique_ptr<T, Deleter>;

        ObjectPool() : deleter_(*this) {}

        template<typename ... Args>
        T* make(Args&& ... args) {
            T* obj;
            if (reusable_objects_.empty()) {
                obj = new T(std::forward<Args>(args) ...);
            }
            else {
                obj = reusable_objects_.back();
                reusable_objects_.pop_back();
                *obj = T(std::forward<Args>(args) ...);
            }
            return obj;
        }

        template<typename ... Args>
        smart_ptr make_managed(Args&& ... args) {
            return smart_ptr(make(std::forward<Args>(args) ...), deleter_);;
        }

        void dispose(T* obj) {
            reusable_objects_.push_back(obj);
        }

        void clear() {
            for (auto obj : reusable_objects_) {
                delete obj;
            }
            reusable_objects_.clear();
        }

        ~ObjectPool() {
            clear();
        }

    private:
        Deleter deleter_;
        std::vector<T*> reusable_objects_;
    };


    /*
     * Timing.
     */
    template<typename F, typename... Args>
    [[maybe_unused]] auto time_function(F func, Args&&... args){
        using namespace std::chrono;
        high_resolution_clock::time_point time_start = high_resolution_clock::now();
        func(std::forward<Args>(args)...);
        high_resolution_clock::time_point time_end = high_resolution_clock::now();
        return duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
    }

    class Timer {
    public:
        void start() { stop_ = start_ = std::chrono::high_resolution_clock::now(); }
        void stop() { stop_ = std::chrono::high_resolution_clock::now(); }
        [[nodiscard]] uint64_t duration_ms() const {
            return std::chrono::duration_cast<std::chrono::milliseconds>(stop_ - start_).count();
        }
    private:
        std::chrono::high_resolution_clock::time_point start_;
        std::chrono::high_resolution_clock::time_point stop_;
    };

    template <typename T, typename Func>
    int arg_max(const std::vector<T>& vec, Func f) {
        return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end(),
                                                                       [&f](const T& a, const T& b){
                                                                           return f(a) < f(b);
                                                                       })));
    }

    template <typename T, typename Func>
    int arg_min(const std::vector<T>& vec, Func f) {
        return static_cast<int>(std::distance(vec.begin(), min_element(vec.begin(), vec.end(),
                                                                       [&f](const T& a, const T& b){
                                                                           return f(a) < f(b);
                                                                       })));
    }
}




