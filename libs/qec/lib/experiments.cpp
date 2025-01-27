/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/qec/experiments.h"

#include "device/memory_circuit.h"
#include "cudaq/simulators.h"

using namespace cudaqx;

namespace cudaq::qec {

namespace details {
auto __sample_code_capacity(const cudaqx::tensor<uint8_t> &H,
                            std::size_t nShots, double error_probability,
                            unsigned seed) {
  // init RNG
  std::mt19937 rng(seed);
  std::bernoulli_distribution dist(error_probability);

  // Each row is a shot
  // Each row elem is a 1 if error, 0 else.
  cudaqx::tensor<uint8_t> data({nShots, H.shape()[1]});
  cudaqx::tensor<uint8_t> syndromes({nShots, H.shape()[0]});

  std::vector<uint8_t> bits(nShots * H.shape()[1]);
  std::generate(bits.begin(), bits.end(), [&]() { return dist(rng); });

  data.copy(bits.data(), data.shape());

  // Syn = D * H^T
  // [n,s] = [n,d]*[d,s]
  syndromes = data.dot(H.transpose()) % 2;

  return std::make_tuple(syndromes, data);
}
} // namespace details

// Single shot version
cudaqx::tensor<uint8_t> generate_random_bit_flips(size_t numBits,
                                                  double error_probability) {
  // init RNG
  std::random_device rd;
  std::mt19937 rng(rd());
  std::bernoulli_distribution dist(error_probability);

  // Each row is a shot
  // Each row elem is a 1 if error, 0 else.
  cudaqx::tensor<uint8_t> data({numBits});
  std::vector<uint8_t> bits(numBits);
  std::generate(bits.begin(), bits.end(), [&]() { return dist(rng); });

  data.copy(bits.data(), data.shape());
  return data;
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const cudaqx::tensor<uint8_t> &H, std::size_t nShots,
                     double error_probability, unsigned seed) {
  return details::__sample_code_capacity(H, nShots, error_probability, seed);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const cudaqx::tensor<uint8_t> &H, std::size_t nShots,
                     double error_probability) {
  return details::__sample_code_capacity(H, nShots, error_probability,
                                         std::random_device()());
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const code &code, std::size_t nShots,
                     double error_probability) {
  return sample_code_capacity(code.get_parity(), nShots, error_probability);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const code &code, std::size_t nShots,
                     double error_probability, unsigned seed) {
  return sample_code_capacity(code.get_parity(), nShots, error_probability,
                              seed);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, operation statePrep,
                      std::size_t numShots, std::size_t numRounds,
                      cudaq::noise_model &noise) {
  if (!code.contains_operation(statePrep))
    throw std::runtime_error(
        "sample_memory_circuit_error - requested state prep kernel not found.");

  auto &prep = code.get_operation<code::one_qubit_encoding>(statePrep);

  if (!code.contains_operation(operation::stabilizer_round))
    throw std::runtime_error("sample_memory_circuit error - no stabilizer "
                             "round kernel for this code.");

  auto &stabRound =
      code.get_operation<code::stabilizer_round>(operation::stabilizer_round);

  auto parity_x = code.get_parity_x();
  auto parity_z = code.get_parity_z();
  auto numData = code.get_num_data_qubits();
  auto numAncx = code.get_num_ancilla_x_qubits();
  auto numAncz = code.get_num_ancilla_z_qubits();

  std::vector<std::size_t> xVec(parity_x.data(),
                                parity_x.data() + parity_x.size());
  std::vector<std::size_t> zVec(parity_z.data(),
                                parity_z.data() + parity_z.size());

  std::size_t numRows = numShots * numRounds;
  std::size_t numCols = numAncx + numAncz;
  std::size_t numMeasRows = numShots * numRounds;
  std::size_t numSyndRows = numShots * (numRounds - 1);

  // Allocate the tensor data for the syndromes and data.
  cudaqx::tensor<uint8_t> syndromeTensor({numShots * (numRounds - 1), numCols});
  cudaqx::tensor<uint8_t> dataResults({numShots, numData});
  cudaqx::tensor<uint8_t> measuresTensor({numMeasRows, numCols});
  std::vector<uint8_t> measurements, dataMeasures;

  // Get the right memory circuit impl
  std::function<void(const code::stabilizer_round &,
                     const code::one_qubit_encoding &, std::size_t, std::size_t,
                     std::size_t, std::size_t, const std::vector<std::size_t> &,
                     const std::vector<std::size_t>)>
      memCircKernel =
          statePrep == operation::prepp || statePrep == operation::prepm
              ? memory_circuit_mx
              : memory_circuit_mz;

  // Run the memory circuit experiment
  if (cudaq::get_simulator()->name() == "stim") {
    // Stim has performant support for sampling when we have
    // circuits with mid circuit measurements but no conditional feedback
    auto counts = cudaq::sample({.shots = numShots, .noise = noise},
                                memCircKernel, stabRound, prep, numData,
                                numAncx, numAncz, numRounds, xVec, zVec);

    for (auto &s : counts.sequential_data()) {
      for (int i = 0; i < numRounds * (numAncx + numAncz); i++)
        measurements.push_back(s[i] == '0' ? 0 : 1);
      for (int i = numRounds * (numAncx + numAncz); i < s.size(); i++)
        dataMeasures.push_back(s[i] == '0' ? 0 : 1);
      printf("cudaq::sample bitstring - %s\n", s.c_str());
    }

    counts.dump();

    dataResults.copy(dataMeasures.data());
    measuresTensor.borrow(measurements.data());
  } else {

    cudaq::ExecutionContext ctx("");
    ctx.noiseModel = &noise;
    auto &platform = cudaq::get_platform();

    for (std::size_t shot = 0; shot < numShots; shot++) {
      platform.set_exec_ctx(&ctx);
      memCircKernel(stabRound, prep, numData, numAncx, numAncz, numRounds, xVec,
                    zVec);
      platform.reset_exec_ctx();
    }

    auto &dataMeasures = getMemoryCircuitDataMeasurements();
    dataResults.copy(dataMeasures.data());

    // Get the raw ancilla measurments
    auto &measurements = getMemoryCircuitAncillaMeasurements();
    measuresTensor.borrow(measurements.data());
  }

  // Convert to Syndromes

  // #pragma omp parallel for collapse(2)
  for (std::size_t shot = 0; shot < numShots; ++shot)
    for (std::size_t round = 1; round < numRounds; ++round)
      for (std::size_t col = 0; col < numCols; ++col) {
        std::size_t measIdx = shot * numRounds + round;
        std::size_t prevMeasIdx = shot * numRounds + (round - 1);
        std::size_t syndIdx = shot * (numRounds - 1) + (round - 1);
        syndromeTensor.at({syndIdx, col}) =
            measuresTensor.at({measIdx, col}) ^
            measuresTensor.at({prevMeasIdx, col});
      }

  clearRawMeasurements();

  // Return the data.
  return std::make_tuple(syndromeTensor, dataResults);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, operation op, std::size_t numShots,
                      std::size_t numRounds) {
  cudaq::noise_model noise;
  return sample_memory_circuit(code, op, numShots, numRounds, noise);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, std::size_t numShots,
                      std::size_t numRounds) {
  return sample_memory_circuit(code, operation::prep0, numShots, numRounds);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, std::size_t numShots,
                      std::size_t numRounds, cudaq::noise_model &noise) {
  return sample_memory_circuit(code, operation::prep0, numShots, numRounds,
                               noise);
}

} // namespace cudaq::qec
