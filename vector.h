#ifndef __VECTOR_H
#define __VECTOR_H

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <chrono>

namespace utils {

    template<typename _T>
    class Vector {
    private:
        _T* data_;
        size_t size_;
        size_t capacity_;

    public:
        // Constructor
        Vector() : data_(nullptr), size_(0), capacity_(0) {}

        // Copy constructor
        Vector(const Vector& ot_) : size_(ot_.size_), capacity_(ot_.capacity_) {
            data_ = static_cast<_T*>(::operator new(sizeof(_T) * capacity_));
            for (size_t i = 0; i < size_; i++) {
                new(&data_[i]) _T(ot_.data_[i]);
            }
        }

        // Move constructor
        Vector(Vector&& ot_) noexcept : data_(ot_.data_), size_(ot_.size_), capacity_(ot_.capacity_) {
            ot_.data_ = nullptr;
            ot_.size_ = 0;
            ot_.capacity_ = 0;
        }

        // Destructor
        ~Vector() {
            clear();
            ::operator delete(data_);
        }

        void push(const _T& __val) {
            if (size_ == capacity_) {
                reserve(capacity_ == 0 ? 1 : capacity_ * 2);
            }
            new(&data_[size_++]) _T(__val);
        }

        void pop() {
            if (size_ > 0) {
                data_[--size_].~_T();
            }
        }
        void reserve(size_t _new_size) {
            if (_new_size <= capacity_) return;

            _T* new_data = static_cast<_T*>(::operator new(sizeof(_T) * _new_size));
            for (size_t i = 0; i < size_; i++) {
                new(&new_data[i]) _T(std::move(data_[i]));
                data_[i].~_T();
            }
            ::operator delete(data_);
            data_ = new_data;
            capacity_ = _new_size;
        }

        void clear() {
            for (size_t i = 0; i < size_; i++) {
                data_[i].~_T();
            }
            size_ = 0;
        }

        _T& operator[](size_t __index) {
            if (__index >= size_) throw std::out_of_range("Index out of range");
            return data_[__index];
        }

        const _T& operator[](size_t __index) const {
            if (__index >= size_) throw std::out_of_range("Index out of range");
            return data_[__index];
        }

        size_t size() const { return size_; }
        size_t capacity() const { return capacity_; }
        bool empty() const { return size_ == 0; }
    };



} // namespace utils

#endif