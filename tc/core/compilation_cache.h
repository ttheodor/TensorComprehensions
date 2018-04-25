/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include "tc/proto/compcache.pb.h"

#include "tc/core/utils/time.h"

namespace tc {

namespace detail {
/**
 * TensorInfo wraps the necessary bits of DLTensor that are used as part of the
 * CompilationCache's entry keys.
 *
 * It is serializable to protobuf and stored directly in the cache.
 */
struct TensorInfo {
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  uint64_t alignment;
  DLDataType dType;

  TensorInfo(const DLTensor* t);
  TensorInfo(const TensorInfoProto& buf);

  bool operator==(const DLTensor* t) const;
  bool operator==(const TensorInfo& t) const;
  bool operator<(const TensorInfo& t) const;
  TensorInfoProto toProtobuf() const;
};
} // namespace detail

template <typename CC, typename CachedEntryType>
class Cache {
 public:
  static void enableCache();
  static void disableCache();
  static void dumpCacheToProtobuf(const std::string& filename);
  static void loadCacheFromProtobuf(const std::string& filename);
  template <typename Protobuf>
  static void loadCacheFromProtobuf(const Protobuf& buf);
  static std::shared_ptr<CC> getCache();
  static bool cacheEnabled();

  typename std::vector<CachedEntryType>::const_iterator begin() const {
    return entries_.begin();
  }
  typename std::vector<CachedEntryType>::const_iterator end() const {
    return entries_.end();
  }
  size_t size() const;
  void clear();

  mutable int numberAttemptedRetrievals = 0;
  mutable int numberSuccessfulRetrievals = 0;
  mutable int numberCacheAttemps = 0;

  Cache() = default;
  Cache(Cache&& other) : entries_(std::move(other.entries_)) {}

 protected:
  // XXX:this should be a std or boost shared_mutex
  mutable std::mutex mtx_;

  std::vector<CachedEntryType> entries_;
};

class CacheEntrySameKeyDifferentValue : public std::invalid_argument {
 public:
  explicit CacheEntrySameKeyDifferentValue(const std::string& what_arg)
      : invalid_argument(what_arg) {}
  explicit CacheEntrySameKeyDifferentValue(const char* what_arg)
      : invalid_argument(what_arg) {}
};

bool operator==(
    const std::vector<const DLTensor*>& inputsTensor,
    const std::vector<detail::TensorInfo>& inputsInfo);

inline std::string makeOptionsFilename(const std::string& filename) {
  return filename + ".options";
}

inline std::string makeCudaFilename(const std::string& filename) {
  return filename + ".cuda";
}
} // namespace tc
