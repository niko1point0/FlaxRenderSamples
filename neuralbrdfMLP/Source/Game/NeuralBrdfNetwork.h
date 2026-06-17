// Host-side neural network storage for the Neural BRDF sample.
//
// Holds the MLP parameters (row-major fp32, contiguous) and provides random
// initialization plus binary load/save. This mirrors RTXNS HostNetwork but for the
// portable fp32 / row-major layout used by this sample (no cooperative-vector
// device-optimal layout conversion). The .bin format is self-describing so trained
// models can be reused across projects.

#pragma once

#include "Engine/Core/Types/String.h"
#include "Engine/Core/Log.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Serialization/FileWriteStream.h"
#include "Engine/Serialization/FileReadStream.h"
#include "../Shaders/NeuralBrdfConfig.h"

class NeuralBrdfNetwork
{
public:
    // File header (self-describing so models are portable across projects).
    struct Header
    {
        uint32 Magic;       // 'NBRD'
        uint32 Version;     // 1
        uint32 InputFeatures;
        uint32 InputNeurons;
        uint32 HiddenNeurons;
        uint32 OutputNeurons;
        uint32 NumHiddenLayers;
        uint32 ParamCount;
    };

    static const uint32 MAGIC = 0x4452424E; // 'NBRD' little-endian
    static const uint32 VERSION = 1;

    float Params[NB_PARAM_COUNT];

    NeuralBrdfNetwork()
    {
        Initialise();
    }

    // Per-layer geometry helper.
    static void GetLayer(int32 layer, int32& inN, int32& outN, int32& wOff, int32& bOff)
    {
        switch (layer)
        {
        case 0: inN = NB_INPUT_NEURONS;  outN = NB_HIDDEN_NEURONS; wOff = NB_L0_W_OFF; bOff = NB_L0_B_OFF; break;
        case 1: inN = NB_HIDDEN_NEURONS; outN = NB_HIDDEN_NEURONS; wOff = NB_L1_W_OFF; bOff = NB_L1_B_OFF; break;
        case 2: inN = NB_HIDDEN_NEURONS; outN = NB_HIDDEN_NEURONS; wOff = NB_L2_W_OFF; bOff = NB_L2_B_OFF; break;
        default: inN = NB_HIDDEN_NEURONS; outN = NB_OUTPUT_NEURONS; wOff = NB_L3_W_OFF; bOff = NB_L3_B_OFF; break;
        }
    }

    // Xavier-style random init (matches RTXNS ResetParameters: U(-1,1) * sqrt(6/(in+out))).
    void Initialise()
    {
        uint32 rng = 0x12345678u;
        auto nextFloat = [&rng]() -> float
        {
            // xorshift32 -> [-1,1)
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            return ((rng >> 8) * (1.0f / 16777216.0f)) * 2.0f - 1.0f;
        };

        for (int32 layer = 0; layer < NB_NUM_LAYERS; ++layer)
        {
            int32 inN, outN, wOff, bOff;
            GetLayer(layer, inN, outN, wOff, bOff);
            const float kW = Math::Sqrt(6.0f / (float)(inN + outN));
            const float kB = Math::Sqrt(6.0f / (float)outN);
            for (int32 i = 0; i < inN * outN; ++i)
                Params[wOff + i] = nextFloat() * kW;
            for (int32 i = 0; i < outN; ++i)
                Params[bOff + i] = nextFloat() * kB;
        }
    }

    bool Save(const StringView& path) const
    {
        FileWriteStream* stream = FileWriteStream::Open(path);
        if (!stream)
        {
            LOG(Error, "NeuralBrdf: failed to open '{0}' for writing.", String(path));
            return false;
        }
        Header header;
        header.Magic = MAGIC;
        header.Version = VERSION;
        header.InputFeatures = NB_INPUT_FEATURES;
        header.InputNeurons = NB_INPUT_NEURONS;
        header.HiddenNeurons = NB_HIDDEN_NEURONS;
        header.OutputNeurons = NB_OUTPUT_NEURONS;
        header.NumHiddenLayers = NB_NUM_HIDDEN_LAYERS;
        header.ParamCount = NB_PARAM_COUNT;
        stream->WriteBytes(&header, sizeof(header));
        stream->WriteBytes(Params, sizeof(Params));
        stream->Close();
        Delete(stream);
        return true;
    }

    bool Load(const StringView& path)
    {
        FileReadStream* stream = FileReadStream::Open(path);
        if (!stream)
        {
            LOG(Error, "NeuralBrdf: failed to open '{0}' for reading.", String(path));
            return false;
        }
        Header header;
        stream->ReadBytes(&header, sizeof(header));
        bool ok = header.Magic == MAGIC && header.Version == VERSION && header.ParamCount == (uint32)NB_PARAM_COUNT &&
                  header.InputNeurons == NB_INPUT_NEURONS && header.HiddenNeurons == NB_HIDDEN_NEURONS &&
                  header.OutputNeurons == NB_OUTPUT_NEURONS && header.NumHiddenLayers == NB_NUM_HIDDEN_LAYERS;
        if (!ok)
        {
            LOG(Error, "NeuralBrdf: model '{0}' does not match the network architecture in the shaders.", String(path));
            stream->Close();
            Delete(stream);
            return false;
        }
        stream->ReadBytes(Params, sizeof(Params));
        stream->Close();
        Delete(stream);
        return true;
    }
};
