# Qwen38's SM90 module is independent of common_ops. This also permits focused
# compilation/testing on a Hopper machine without rebuilding unrelated kernels.
add_library(w4a8_sm_ops MODULE
    csrc/moe/cutlass_moe/w4a8_sm/qwen38_mm.cu
    csrc/moe/cutlass_moe/w4a8_sm/qwen38_swiglu.cu
)
target_include_directories(w4a8_sm_ops PRIVATE
    ${PROJECT_SOURCE_DIR}/csrc
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${repo-cutlass_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/tools/util/include
)
target_compile_options(w4a8_sm_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-DNDEBUG;-gencode=arch=compute_90a,code=sm_90a;--expt-relaxed-constexpr;--expt-extended-lambda;--threads=2>
)
separate_arguments(W4A8_SM_TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(w4a8_sm_ops PRIVATE ${W4A8_SM_TORCH_FLAGS})
target_compile_definitions(w4a8_sm_ops PRIVATE
    CUTE_USE_PACKED_TUPLE=1 CUTLASS_ENABLE_TENSOR_CORE_MMA=1
    CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED CUTLASS_DEBUG_TRACE_LEVEL=0
)
target_link_libraries(w4a8_sm_ops PRIVATE ${TORCH_LIBRARIES} c10 CUDA::cuda_driver CUDA::cudart)
set_target_properties(w4a8_sm_ops PROPERTIES PREFIX "" CUDA_ARCHITECTURES OFF)
install(TARGETS w4a8_sm_ops LIBRARY DESTINATION sgl_kernel)
