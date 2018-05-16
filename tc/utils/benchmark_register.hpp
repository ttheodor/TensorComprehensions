#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <vector>

class Register {
 public:
  using KType = std::chrono::high_resolution_clock::duration(
      const std::vector<const void*>&,
      std::vector<void*>&);

  static Register& get() {
    static Register r;
    return r;
  }
  void registerBenchmark(std::function<KType> impl, uint64_t id);

 private:
  Register();
};
