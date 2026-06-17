// dx::linalg - HLSL cooperative-vector linear-algebra helper library.
//
// This is a vendored, self-contained implementation of the Shader Model 6.9
// "dx/linalg.h" helper that ships with the DirectX Shader Compiler. It wraps the
// compiler cooperative-vector builtins (__builtin_MatVecMul, __builtin_MatVecMulAdd,
// __builtin_OuterProductAccumulate, __builtin_VectorAccumulate) in the documented
// dx::linalg API (see microsoft/hlsl-specs proposal 0031 and the DirectX
// "Cooperative Vector" developer blog).
//
// Flax routes every #include through its own include handler (which only sees the
// project shader tree), so DXC's built-in copy of this header is never found - we
// vendor it here and include it explicitly. Requires: -T cs_6_9 and -enable-16bit-types.

#ifndef DX_LINALG_H
#define DX_LINALG_H

namespace dx {
namespace linalg {

// Data type of the elements stored in a matrix/vector in memory. Values match the
// underlying DXIL component-type encoding.
enum DataType {
    DATA_TYPE_SINT16 = 2,
    DATA_TYPE_UINT16 = 3,
    DATA_TYPE_SINT32 = 4,
    DATA_TYPE_UINT32 = 5,
    DATA_TYPE_FLOAT16 = 8,
    DATA_TYPE_FLOAT32 = 9,
    DATA_TYPE_SINT8_T4_PACKED = 17,
    DATA_TYPE_UINT8_T4_PACKED = 18,
    DATA_TYPE_UINT8 = 19,
    DATA_TYPE_SINT8 = 20,
    DATA_TYPE_FLOAT8_E4M3 = 21,
    DATA_TYPE_FLOAT8_E5M2 = 22,
};

// Physical layout of a matrix in memory.
enum MatrixLayout {
    MATRIX_LAYOUT_ROW_MAJOR = 0,
    MATRIX_LAYOUT_COLUMN_MAJOR = 1,
    MATRIX_LAYOUT_MUL_OPTIMAL = 2,
    MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL = 3,
};

namespace details {
// Compile-time "is this an unsigned integer element type" trait. Cooperative-vector
// builtins take explicit signed/unsigned flags for the input/output vectors.
template <typename T> struct IsUnsignedT { static const bool value = false; };
template <> struct IsUnsignedT<uint16_t> { static const bool value = true; };
template <> struct IsUnsignedT<uint32_t> { static const bool value = true; };
template <> struct IsUnsignedT<uint64_t> { static const bool value = true; };
} // namespace details

// Reference to a matrix stored in a (RW)ByteAddressBuffer. Dimensions/layout/format are
// compile-time template parameters; offset/stride are runtime members.
template <typename BufferTy, DataType DT, uint M, uint K, MatrixLayout ML, bool Transpose>
struct MatrixRefImpl {
    BufferTy Buffer;
    uint StartOffset;
    uint Stride; // must be 0 for MUL_OPTIMAL / OUTER_PRODUCT_OPTIMAL layouts
};

template <DataType DT, uint M, uint K, MatrixLayout ML, bool Transpose = false>
using MatrixRef = MatrixRefImpl<ByteAddressBuffer, DT, M, K, ML, Transpose>;

template <DataType DT, uint M, uint K, MatrixLayout ML, bool Transpose = false>
using RWMatrixRef = MatrixRefImpl<RWByteAddressBuffer, DT, M, K, ML, Transpose>;

// Reference to a vector (e.g. a bias vector) stored in a (RW)ByteAddressBuffer.
template <typename BufferTy, DataType DT> struct VectorRefImpl {
    BufferTy Buffer;
    uint StartOffset;
};

template <DataType DT> using VectorRef = VectorRefImpl<ByteAddressBuffer, DT>;
template <DataType DT> using RWVectorRef = VectorRefImpl<RWByteAddressBuffer, DT>;

// A vector value plus the interpretation (DataType) the hardware should apply to it.
template <typename T, int N, DataType DT> struct InterpretedVector {
    vector<T, N> Data;
};

template <DataType DT, typename T, int N>
InterpretedVector<T, N, DT> MakeInterpretedVector(vector<T, N> Vec) {
    InterpretedVector<T, N, DT> IV = {Vec};
    return IV;
}

// OutputVector = Matrix * InputVector
template <typename OutputElTy, typename InputElTy, int InputElCount,
          typename MatrixBufferTy, DataType InputDT, DataType MatrixDT,
          uint MatrixM, uint MatrixK, MatrixLayout MatrixLayoutT,
          bool MatrixTranspose>
vector<OutputElTy, MatrixM>
Mul(MatrixRefImpl<MatrixBufferTy, MatrixDT, MatrixM, MatrixK, MatrixLayoutT, MatrixTranspose> Matrix,
    InterpretedVector<InputElTy, InputElCount, InputDT> InputVector) {
    vector<OutputElTy, MatrixM> OutputVector;
    __builtin_MatVecMul(
        /*out*/ OutputVector, details::IsUnsignedT<OutputElTy>::value, InputVector.Data,
        details::IsUnsignedT<InputElTy>::value, InputDT, Matrix.Buffer,
        Matrix.StartOffset, MatrixDT, MatrixM, MatrixK, MatrixLayoutT,
        MatrixTranspose, Matrix.Stride);
    return OutputVector;
}

// OutputVector = Matrix * InputVector + BiasVector
template <typename OutputElTy, typename InputElTy, int InputElCount,
          typename MatrixBufferTy, DataType InputDT, DataType MatrixDT,
          uint MatrixM, uint MatrixK, MatrixLayout MatrixLayoutT,
          bool MatrixTranspose, typename BiasVectorBufferTy, DataType BiasVectorDT>
vector<OutputElTy, MatrixM>
MulAdd(MatrixRefImpl<MatrixBufferTy, MatrixDT, MatrixM, MatrixK, MatrixLayoutT, MatrixTranspose> Matrix,
       InterpretedVector<InputElTy, InputElCount, InputDT> InputVector,
       VectorRefImpl<BiasVectorBufferTy, BiasVectorDT> BiasVector) {
    vector<OutputElTy, MatrixM> OutputVector;
    __builtin_MatVecMulAdd(
        /*out*/ OutputVector, details::IsUnsignedT<OutputElTy>::value, InputVector.Data,
        details::IsUnsignedT<InputElTy>::value, InputDT, Matrix.Buffer,
        Matrix.StartOffset, MatrixDT, MatrixM, MatrixK, MatrixLayoutT,
        MatrixTranspose, Matrix.Stride, BiasVector.Buffer, BiasVector.StartOffset,
        BiasVectorDT);
    return OutputVector;
}

// Matrix += InputVector1 * Transpose(InputVector2), accumulated atomically (device scope).
// The destination matrix must use MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL with stride 0.
template <typename ElTy, int MatrixM, int MatrixN, DataType MatrixDT, MatrixLayout MatrixLayoutT>
void OuterProductAccumulate(
    vector<ElTy, MatrixM> InputVector1, vector<ElTy, MatrixN> InputVector2,
    RWMatrixRef<MatrixDT, MatrixM, MatrixN, MatrixLayoutT, false> Matrix) {
    __builtin_OuterProductAccumulate(InputVector1, InputVector2, Matrix.Buffer,
                                     Matrix.StartOffset, MatrixDT, MatrixLayoutT,
                                     Matrix.Stride);
}

// Array[Offset + i] += InputVector[i], accumulated atomically (device scope).
template <typename ElTy, int ElCount>
void VectorAccumulate(vector<ElTy, ElCount> InputVector, RWByteAddressBuffer Buffer, uint Offset) {
    __builtin_VectorAccumulate(InputVector, Buffer, Offset);
}

} // namespace linalg
} // namespace dx

#endif // DX_LINALG_H
