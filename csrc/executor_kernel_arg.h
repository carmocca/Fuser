// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/util/Exception.h>
#include <serde/fusion_cache_generated.h>
#include <serde/utils.h>
#include <torch/csrc/jit/ir/ir.h>
#include <type.h>
#include <array>
#include <optional>

namespace nvfuser {

// This should match the tensor used in the code generation (almost exactly)
template <typename T, int N, typename nvfuser_index_t>
struct TensorArgCodegen {
  using data_type = T;
  using index_type = nvfuser_index_t;
  static constexpr int ndims = N;

  T& operator[](nvfuser_index_t ind) {
    return data[ind];
  };

  T* data;
  std::array<nvfuser_index_t, N> size;
  std::array<nvfuser_index_t, N> stride;
  constexpr int nDims() const {
    return N;
  }
  void setSize(int64_t i, nvfuser_index_t s) {
    size[i] = s;
  }
  void setStride(int64_t i, nvfuser_index_t s) {
    stride[i] = s;
  }
  nvfuser_index_t getSize(int64_t i) const {
    return size[i];
  }
  nvfuser_index_t getStride(int64_t i) const {
    return stride[i];
  }
  constexpr bool isInt32IndexMode() const {
    return std::is_same_v<nvfuser_index_t, int>;
  }
};

// 0-Dim GPU based tensor
template <typename T, typename nvfuser_index_t>
struct TensorArgCodegen<T, 0, nvfuser_index_t> {
  using data_type = T;
  using index_type = nvfuser_index_t;
  static constexpr int ndims = 0;

  T& operator[](nvfuser_index_t ind) {
    return data[ind];
  };

  T* data;
  constexpr int nDims() const {
    return 0;
  }
  void setSize(int64_t, nvfuser_index_t) {
    TORCH_INTERNAL_ASSERT(false, "Tried to set size of a 0-dim tensor");
  }
  void setStride(int64_t, nvfuser_index_t) {
    TORCH_INTERNAL_ASSERT(false, "Tried to set stride of a 0-dim tensor");
  }
  nvfuser_index_t getSize(int64_t i) const {
    TORCH_INTERNAL_ASSERT(false, "Tried to get size of a 0-dim tensor");
  }
  nvfuser_index_t getStride(int64_t i) const {
    TORCH_INTERNAL_ASSERT(false, "Tried to get stride of a 0-dim tensor");
  }
  constexpr bool isInt32IndexMode() const {
    return std::is_same_v<nvfuser_index_t, int>;
  }
};

// Specialization for 0-dim case that's easy to pass in a CPU based tensor
// without memcpy
template <typename T>
struct CpuScalarTensorCodegen {
  T& operator[](int) {
    return data;
  };

  T data;
};

// TODO: macro this and the printer below
enum class ArgType {
  PhiloxCudaState,
  Long,
  Double,
  ComplexDouble,
  Bool,
  Tensor,
  CpuScalarTensor
};

inline std::string argTypeToString(ArgType type) {
  std::string ret;
  switch (type) {
    case ArgType::PhiloxCudaState:
      ret = "PhiloxCudaState";
      break;
    case ArgType::Long:
      ret = "Long";
      break;
    case ArgType::Double:
      ret = "Double";
      break;
    case ArgType::ComplexDouble:
      ret = "ComplexDouble";
      break;
    case ArgType::Bool:
      ret = "Bool";
      break;
    case ArgType::Tensor:
      ret = "Tensor";
      break;
    case ArgType::CpuScalarTensor:
      ret = "CpuScalarTensor";
      break;
  }
  return ret;
}

struct ArgAbstract {
  virtual ~ArgAbstract() = default;
  virtual const void* arg() const = 0;
  virtual void* arg() = 0;
  virtual bool isType(ArgType type) const = 0;
  virtual ArgType type() const = 0;
  virtual std::unique_ptr<ArgAbstract> copy_unique_ptr() const = 0;
  virtual std::string toString() const {
    return "input type: " + argTypeToString(type());
  };
  virtual flatbuffers::Offset<serde::ArgAbstract> serialize(
      flatbuffers::FlatBufferBuilder& builder) const = 0;
};

#define DEF_HELPEE_FUNC(TARGET_TYPE, ARG_NAME)                    \
  bool isType(ArgType type) const override {                      \
    return ArgType::TARGET_TYPE == type;                          \
  }                                                               \
  ArgType type() const override {                                 \
    return ArgType::TARGET_TYPE;                                  \
  }                                                               \
  const void* arg() const override {                              \
    return &ARG_NAME;                                             \
  }                                                               \
  void* arg() override {                                          \
    return &ARG_NAME;                                             \
  }                                                               \
  std::unique_ptr<ArgAbstract> copy_unique_ptr() const override { \
    return std::make_unique<TARGET_TYPE##Arg>(*this);             \
  }

#define DEF_TOSTRING_FUNC                 \
  std::string toString() const override { \
    std::stringstream ss;                 \
    ss << val_;                           \
    return ss.str();                      \
  }

struct PhiloxCudaStateArg : public ArgAbstract {
  at::PhiloxCudaState val_;
  PhiloxCudaStateArg(at::PhiloxCudaState _val) : val_(_val){};
  DEF_HELPEE_FUNC(PhiloxCudaState, val_)

  flatbuffers::Offset<serde::ArgAbstract> serialize(
      flatbuffers::FlatBufferBuilder& builder) const override {
    auto data =
        serde::CreatePhiloxCudaState(builder, val_.seed_.val, val_.offset_.val);
    return serde::CreateArgAbstract(
        builder, serde::ArgAbstractData_PhiloxCudaState, data.Union());
  }
};

struct LongArg : public ArgAbstract {
  int64_t val_;
  explicit LongArg(int64_t _val) : val_(_val) {}
  DEF_HELPEE_FUNC(Long, val_)
  DEF_TOSTRING_FUNC

  flatbuffers::Offset<serde::ArgAbstract> serialize(
      flatbuffers::FlatBufferBuilder& builder) const override {
    auto data = serde::CreateLong(builder, val_);
    return serde::CreateArgAbstract(
        builder, serde::ArgAbstractData_Long, data.Union());
  }
};

struct DoubleArg : public ArgAbstract {
  double val_;
  explicit DoubleArg(double _val) : val_(_val) {}
  DEF_HELPEE_FUNC(Double, val_)
  DEF_TOSTRING_FUNC

  flatbuffers::Offset<serde::ArgAbstract> serialize(
      flatbuffers::FlatBufferBuilder& builder) const override {
    auto data = serde::CreateDouble(builder, val_);
    return serde::CreateArgAbstract(
        builder, serde::ArgAbstractData_Double, data.Union());
  }
};

struct ComplexDoubleArg : public ArgAbstract {
  c10::complex<double> val_;
  explicit ComplexDoubleArg(c10::complex<double> _val) : val_(_val) {}
  DEF_HELPEE_FUNC(ComplexDouble, val_)
  DEF_TOSTRING_FUNC

  flatbuffers::Offset<serde::ArgAbstract> serialize(
      flatbuffers::FlatBufferBuilder& builder) const override {
    auto data = serde::CreateComplexDouble(builder, val_.real(), val_.imag());
    return serde::CreateArgAbstract(
        builder, serde::ArgAbstractData_ComplexDouble, data.Union());
  }
};

struct BoolArg : public ArgAbstract {
  bool val_;
  explicit BoolArg(bool _val) : val_(_val) {}
  DEF_HELPEE_FUNC(Bool, val_)
  DEF_TOSTRING_FUNC

  flatbuffers::Offset<serde::ArgAbstract> serialize(
      flatbuffers::FlatBufferBuilder& builder) const override {
    auto data = serde::CreateBool(builder, val_);
    return serde::CreateArgAbstract(
        builder, serde::ArgAbstractData_Bool, data.Union());
  }
};

struct TensorArgAbstract : ArgAbstract {
  virtual int64_t getRank() const = 0;
  virtual int64_t getSize(int64_t i) const = 0;
  virtual int64_t getStride(int64_t i) const = 0;
  virtual void* getPointer() const = 0;
  virtual DataType getDataType() const = 0;
  virtual int64_t numel() const = 0;
  virtual at::Tensor getTensor() const = 0;

  std::string toString() const override;
};

template <typename TENSOR_TYPE>
struct TensorArg : public TensorArgAbstract {
  TENSOR_TYPE instance_;
  at::Tensor tensor_;

  TensorArg(const at::Tensor& tensor) : tensor_(tensor) {
    setPointer(tensor.data_ptr());
    for (const auto i : c10::irange(tensor.ndimension())) {
      setSize(i, tensor.sizes()[i]);
      setStride(i, tensor.strides()[i]);
    }
  }

  // Create Metadata TensorArg using Flatbuffers
  TensorArg(const serde::TensorArg* tensor) {
    using pointer_type = std::add_pointer_t<typename TENSOR_TYPE::data_type>;
    instance_.data = (pointer_type)tensor->ptr();
    for (auto dim : c10::irange(instance_.nDims())) {
      setSize(dim, tensor->size()->Get(dim));
      setStride(dim, tensor->stride()->Get(dim));
    }
  }

  void setSize(int64_t i, int64_t size) {
    instance_.setSize(i, (typename TENSOR_TYPE::index_type)size);
  }
  void setStride(int64_t i, int64_t stride) {
    instance_.setStride(i, (typename TENSOR_TYPE::index_type)stride);
  }
  void setPointer(void* ptr) {
    instance_.data = static_cast<decltype(TENSOR_TYPE::data)>(ptr);
  }
  void setTensor(at::Tensor tensor) {
    tensor_ = tensor;
  }

  int64_t getSize(int64_t i) const override {
    return instance_.getSize(i);
  }
  int64_t getStride(int64_t i) const override {
    return instance_.getStride(i);
  }
  int64_t getRank() const override {
    return instance_.nDims();
  }
  void* getPointer() const override {
    return instance_.data;
  }
  DataType getDataType() const override {
    return NativeTypeWithC10ComplexToDataType<
        typename TENSOR_TYPE::data_type>::type;
  }
  at::Tensor getTensor() const override {
    return tensor_;
  }
  int64_t numel() const override {
    int64_t ret = 1;
    for (auto i : c10::irange(instance_.nDims())) {
      ret *= instance_.getSize(i);
    }
    return ret;
  }
  DEF_HELPEE_FUNC(Tensor, instance_)

  flatbuffers::Offset<serde::ArgAbstract> serialize(
      flatbuffers::FlatBufferBuilder& builder) const override {
    std::vector<int64_t> sizes_fb;
    std::vector<int64_t> strides_fb;
    sizes_fb.reserve(getRank());
    strides_fb.reserve(getRank());
    for (auto dim : c10::irange(instance_.nDims())) {
      sizes_fb.push_back(getSize(dim));
      strides_fb.push_back(getStride(dim));
    }

    auto data = serde::CreateTensorArg(
        builder,
        (size_t)getPointer(),
        builder.CreateVector(sizes_fb),
        builder.CreateVector(strides_fb),
        instance_.nDims(),
        serde::mapToSerdeDtype(getDataType()),
        instance_.isInt32IndexMode());
    return serde::CreateArgAbstract(
        builder, serde::ArgAbstractData_TensorArg, data.Union());
  }
};

template <typename CPU_TENSOR_TYPE>
struct CpuScalarTensorArg : public ArgAbstract {
  CPU_TENSOR_TYPE instance_;
  using DATATYPE = decltype(CPU_TENSOR_TYPE::data);

  CpuScalarTensorArg() = delete;

  explicit CpuScalarTensorArg(DATATYPE _data) {
    instance_.data = _data;
  }

  DEF_HELPEE_FUNC(CpuScalarTensor, instance_)

  flatbuffers::Offset<serde::ArgAbstract> serialize(
      flatbuffers::FlatBufferBuilder& builder) const override {
    flatbuffers::Offset<void> value = 0;
    serde::ScalarCpuData data_type = serde::ScalarCpuData_NONE;

    if constexpr (std::is_same_v<DATATYPE, bool>) {
      data_type = serde::ScalarCpuData_Bool;
      value = serde::CreateBool(builder, instance_.data).Union();

    } else if constexpr (std::is_same_v<DATATYPE, double>) {
      data_type = serde::ScalarCpuData_Double;
      value = serde::CreateDouble(builder, instance_.data).Union();

    } else if constexpr (std::is_same_v<DATATYPE, float>) {
      data_type = serde::ScalarCpuData_Float;
      value = serde::CreateFloat(builder, instance_.data).Union();

    } else if constexpr (std::is_same_v<DATATYPE, at::Half>) {
      data_type = serde::ScalarCpuData_Half;
      value = serde::CreateHalf(builder, instance_.data.x).Union();

    } else if constexpr (std::is_same_v<DATATYPE, at::BFloat16>) {
      data_type = serde::ScalarCpuData_BFloat16;
      value = serde::CreateBFloat16(builder, instance_.data.x).Union();

    } else if constexpr (std::is_same_v<DATATYPE, int32_t>) {
      data_type = serde::ScalarCpuData_Int;
      value = serde::CreateInt(builder, instance_.data).Union();

    } else if constexpr (std::is_same_v<DATATYPE, int64_t>) {
      data_type = serde::ScalarCpuData_Long;
      value = serde::CreateLong(builder, instance_.data).Union();

    } else if constexpr (std::is_same_v<DATATYPE, c10::complex<float>>) {
      data_type = serde::ScalarCpuData_ComplexFloat;
      value = serde::CreateComplexFloat(
                  builder, instance_.data.real(), instance_.data.imag())
                  .Union();

    } else if constexpr (std::is_same_v<DATATYPE, c10::complex<double>>) {
      data_type = serde::ScalarCpuData_ComplexDouble;
      value = serde::CreateComplexDouble(
                  builder, instance_.data.real(), instance_.data.imag())
                  .Union();
    }
    auto data = serde::CreateScalarCpu(builder, data_type, value).Union();
    return serde::CreateArgAbstract(
        builder, serde::ArgAbstractData_ScalarCpu, data);
  }
};

// TODO: This class needs some further clean up and refactor
//! KernelArgumentHolder copies meta information from kernel inputs, including
//! tensor sizes/shapes/dtype/memory_ptr and copies scalar inputs. It is used
//! for both compilation as well as kernel execution. The important thing is to
//! strip ownership of tensor from KernelArgumentHolder, so that during async
//! compilation, we are not unnecessarily holding memory that is not needed.
class TORCH_CUDA_CU_API KernelArgumentHolder {
 public:
  //! create KernelArgumentHolder from c10 inputs. Note that we we not taking
  //! the ownership of the memory from the original inputs, but just recording
  //! its meta data for kernel execution/compilation.
  static KernelArgumentHolder createKernelArgumentHolder(
      const c10::ArrayRef<c10::IValue>& inputs,
      const std::optional<KernelIndexMode>& index_mode = std::nullopt);

  KernelIndexMode getIndexMode() const {
    return index_mode_;
  }

  void setIndexMode(KernelIndexMode mode) {
    index_mode_ = mode;
  }

  PrimDataType getIndexType() const {
    return indexModeToDtype(index_mode_);
  }

  KernelArgumentHolder() = default;

  explicit KernelArgumentHolder(KernelIndexMode index_mode)
      : index_mode_(index_mode) {}

  explicit KernelArgumentHolder(PrimDataType index_type)
      : index_mode_(indexTypeToMode(index_type)) {}

  KernelArgumentHolder(const KernelArgumentHolder& self)
      : device_index_(self.getDeviceIndex()),
        cache_id_(self.getCacheId()),
        index_mode_(self.getIndexMode()) {
    for (const auto& arg : self.arguments_) {
      push(arg.get());
    }
  }

  KernelArgumentHolder& operator=(const KernelArgumentHolder& self) {
    device_index_ = self.getDeviceIndex();
    index_mode_ = self.getIndexMode();
    for (const auto& arg : self.arguments_) {
      push(arg.get());
    }
    return *this;
  }

  // Push a tensor to the arguments
  void push(const at::Tensor& tensor);

  // Push a scalar or integer to the arguments
  void push(const c10::IValue& val);

  void push(const at::PhiloxCudaState& val);

  // Create buffer, flatten arguments into it, align by 8 Bytes, return pointers
  // in the buffer
  void** getBuffer();

  void push(const c10::ArrayRef<c10::IValue>& args);

  void push(const std::vector<at::Tensor>& tensors);

  void push(const ArgAbstract* arg);

  void swap(int i, const ArgAbstract* arg);

  // push int64
  void push(int64_t val);

  const ArgAbstract* back() const {
    return arguments_.back().get();
  }

  void appendPhiloxRNGSeed(uint64_t rand_offset);

  const ArgAbstract* at(size_t ind) const {
    return arguments_.at(ind).get();
  };

  const ArgAbstract* operator[](size_t ind) const {
    return at(ind);
  };

  size_t size() const {
    return arguments_.size();
  }

  bool empty() const {
    return arguments_.empty();
  }

  void setDeviceIndex(int8_t index) {
    device_index_ = index;
  }

  int8_t getDeviceIndex() const {
    return device_index_;
  }

  void setCacheId(size_t id) {
    cache_id_ = id;
  }

  c10::optional<size_t> getCacheId() const {
    return cache_id_;
  }

  std::string toString() const;

  flatbuffers::Offset<serde::KernelArgumentHolder> serialize(
      flatbuffers::FlatBufferBuilder& builder) const;

  void deserialize(const serde::KernelArgumentHolder* buffer);

 private:
  std::vector<std::unique_ptr<ArgAbstract>> arguments_;
  std::vector<void*> void_ptrs_;
  bool changed_ = true;

  int8_t device_index_ = 0;
  c10::optional<size_t> cache_id_ = c10::nullopt;
  KernelIndexMode index_mode_ = KernelIndexMode::INT64;
};

} // namespace nvfuser
