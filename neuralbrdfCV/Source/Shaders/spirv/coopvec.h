// Inline-SPIR-V cooperative vectors (SPV_NV_cooperative_vector) for the DXC -spirv backend.
//
// DXC has no language-level support for cooperative vectors when targeting SPIR-V
// (dx::linalg's __builtin_MatVecMulAdd only lowers to DXIL). This header reproduces the
// cooperative-vector instructions by hand using the inline-SPIR-V mechanism, exactly like
// DXC's own vk/khr/cooperative_matrix.h does for cooperative matrices.
//
// Only meaningful under DXC -spirv (the whole file is guarded by __spirv__ at the include site).
// Requires SPIR-V 1.6 (Vulkan 1.3) + the VulkanMemoryModel capability; both are arranged by the
// engine's Vulkan shader-compiler coopvec path (-fspv-target-env=vulkan1.3 + a memory-model patch).
#ifndef NB_SPIRV_COOPVEC_H
#define NB_SPIRV_COOPVEC_H

// SPV_NV_cooperative_vector "Component Type" interpretation values.
#define NB_CV_FLOAT16 0

// SPV_NV_cooperative_vector "Cooperative Vector Matrix Layout" values.
#define NB_CV_ROW_MAJOR 0
#define NB_CV_COL_MAJOR 1
#define NB_CV_INFERENCING_OPTIMAL 2 // == dx::linalg MATRIX_LAYOUT_MUL_OPTIMAL
#define NB_CV_TRAINING_OPTIMAL 3

namespace nbcv
{
    // Cooperative vector types: OpTypeCooperativeVectorNV <component type> <component count>.
    // NOTE: vk::ext_extension / vk::ext_capability attributes are NOT permitted on a `using`
    // type-alias declaration (DXC: "an attribute list cannot appear here"). The required
    // SPV_NV_cooperative_vector extension + CooperativeVectorNV / VulkanMemoryModel capabilities are
    // instead declared on the MatVecMulAdd function below, which always runs and pulls them into the
    // module, so the type aliases must stay plain.
    using CoopVecHalf30 = vk::SpirvOpaqueType</* OpTypeCooperativeVectorNV */ 5288, half, vk::integral_constant<uint, 30> >;
    using CoopVecHalf32 = vk::SpirvOpaqueType</* OpTypeCooperativeVectorNV */ 5288, half, vk::integral_constant<uint, 32> >;
    using CoopVecHalf4 = vk::SpirvOpaqueType</* OpTypeCooperativeVectorNV */ 5288, half, vk::integral_constant<uint, 4> >;

    // Pointer to the raw weight/bias storage (ByteAddressBuffer == StorageBuffer struct { runtime array of uint }).
    // OpCooperativeVectorMatrixMulAddNV wants a pointer to an array-of-scalar; the byte offset is a separate operand
    // and the element interpretation comes from MatrixInterpretation/BiasInterpretation, so a uint array works.
    using NB_U32Array = vk::SpirvOpaqueType</* OpTypeRuntimeArray */ 29, uint>;
    using NB_U32ArrayPtr = vk::SpirvOpaqueType</* OpTypePointer */ 32, /* StorageBuffer */ vk::Literal<vk::integral_constant<uint, 12> >, NB_U32Array>;

    // Build a cooperative vector with every component equal to v (single-operand OpCompositeConstruct == splat).
    template <typename CoopVecT, typename T>
    [[vk::ext_instruction(/* OpCompositeConstruct */ 80)]]
    CoopVecT Splat(T v);

    // Read/replace one component. Cooperative vectors explicitly support the dynamic vector ops.
    template <typename T, typename CoopVecT>
    [[vk::ext_instruction(/* OpVectorExtractDynamic */ 77)]]
    T Extract(CoopVecT v, uint index);

    template <typename CoopVecT, typename T>
    [[vk::ext_instruction(/* OpVectorInsertDynamic */ 78)]]
    CoopVecT Insert(CoopVecT v, T component, uint index);

    // Component-wise max (ReLU uses max(x, 0)); FMax is allowed on cooperative vectors.
    template <typename CoopVecT>
    [[vk::ext_instruction(/* FMax */ 40, "GLSL.std.450")]]
    CoopVecT Max(CoopVecT a, CoopVecT b);

    // OpAccessChain to member 0 (the runtime array) of a ByteAddressBuffer, yielding a StorageBuffer
    // pointer to an array-of-scalar suitable for the cooperative-vector matrix-multiply operands.
    // The result is held in a local (Function-storage) variable, so the module must declare the
    // VariablePointersStorageBuffer capability (core in SPIR-V 1.6, so no extension is required).
    [[vk::ext_capability(/* VariablePointersStorageBuffer */ 4441)]]
    [[vk::ext_instruction(/* OpAccessChain */ 65)]]
    NB_U32ArrayPtr ArrayPtr([[vk::ext_reference]] ByteAddressBuffer buffer, uint memberIndex);

    // OpCooperativeVectorMatrixMulAddNV: result = Matrix(MxK) * Input + Bias.
    // Matrix/Bias are pointers; MatrixOffset/BiasOffset are byte offsets; interpretations/M/K/layout/transpose
    // must be constants (passed as plain values so DXC emits OpConstant ids for them).
    template <typename ResultCoopVecT, typename InputCoopVecT>
    [[vk::ext_extension("SPV_NV_cooperative_vector")]]
    [[vk::ext_capability(/* CooperativeVectorNV */ 5394)]]
    [[vk::ext_capability(/* VulkanMemoryModel */ 5345)]]
    [[vk::ext_instruction(/* OpCooperativeVectorMatrixMulAddNV */ 5292)]]
    ResultCoopVecT MatVecMulAdd(
        InputCoopVecT input, uint inputInterpretation,
        NB_U32ArrayPtr matrix, uint matrixOffset, uint matrixInterpretation,
        NB_U32ArrayPtr bias, uint biasOffset, uint biasInterpretation,
        uint M, uint K, uint memoryLayout, bool transpose);
}

#endif // NB_SPIRV_COOPVEC_H
