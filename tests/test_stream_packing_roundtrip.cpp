// test_stream_packing_roundtrip.cpp
//
// The full seeded protocol, with EVERYTHING that crosses the network sent as
// bit-packed byte streams (see stream_packing.hpp), and every receiving side
// working ONLY on what it unpacked:
//
//   client --[1 eval key stream]---------------> server   (one-time)
//   client --[r query streams, one per ring]----> server
//   server --[r response streams, one per ring]-> client   (s RLWE cts each)
//
// Same protocol and same final check as test_full_scoring_with_splits_seeded.cpp:
// the client decrypts the UNPACKED response streams, recomposes the CRT
// components and compares against the plaintext ground truth
// sum_j query[j] * db[j] for every split and coefficient.
//
// In addition, every unpack is checked field-by-field against the original
// (canonicalized to [0, q)), and every stream's byte length is checked
// against ceil(bits / 8) of the layout documented in stream_packing.hpp.
// A second test checks that truncated / extended / corrupted streams are
// rejected instead of silently decoded.
//
// Parameterized over TWO Params factories, run via the SAME test body.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_stream_packing_roundtrip

#include <gtest/gtest.h>

#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "crt.hpp"
#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "seeded_distribution.hpp"
#include "seeded_eval_keys.hpp"
#include "seeded_query.hpp"
#include "stream_packing.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

class StreamPackingRoundtrip : public ::testing::TestWithParam<Params (*)()> {};

int64_t canonical(int64_t v, int64_t q) {
    int64_t m = v % q;
    return m < 0 ? m + q : m;
}

std::vector<int64_t> canonical(const std::vector<int64_t>& v, int64_t q) {
    std::vector<int64_t> out;
    out.reserve(v.size());
    for (int64_t x : v) {
        out.push_back(canonical(x, q));
    }
    return out;
}

std::vector<std::vector<int64_t>> canonical(const std::vector<std::vector<int64_t>>& v, int64_t q) {
    std::vector<std::vector<int64_t>> out;
    out.reserve(v.size());
    for (const auto& x : v) {
        out.push_back(canonical(x, q));
    }
    return out;
}

/// Expected byte length of a stream: seed bytes + ceil(coefficients * bits / 8).
size_t expected_stream_bytes(size_t seed_bytes, int64_t coefficients, int bits) {
    return seed_bytes + static_cast<size_t>((coefficients * bits + 7) / 8);
}

/// Same scoring as test_full_scoring_with_splits_seeded.cpp.
RLWECT compute_split_score(const CryptoContext& ctx, const std::vector<RLWECTEvalForm>& query_eval,
                           const std::vector<DatabasePolynomialEvalForm>& db_split) {
    RLWECTEvalForm score_eval(ctx.rlwe_param);
    for (size_t j = 0; j < query_eval.size(); ++j) {
        RLWECTEvalForm product_eval(ctx.rlwe_param);
        query_eval[j].mul(product_eval, *db_split[j].poly_eval);
        score_eval.add(score_eval, product_eval);
    }
    return RLWECT(score_eval);
}

} // namespace

TEST_P(StreamPackingRoundtrip, FullProtocolOverPackedStreamsDecryptsToGroundTruth) {
    Params params = GetParam()();

    // num_clusters == 1 is a valid parameter set (e.g. Mimir-III). The body
    // handles it: the cluster loop runs once on cluster 0, whose RGSW selector
    // bit is 1, and every correctness check still applies. Only the
    // "other clusters are masked to zero" property is not exercised then;
    // every set with num_clusters > 1 still covers it.
    ASSERT_GE(params.num_clusters, 1);
    if (params.num_clusters == 1) {
        std::cout << "[info] num_clusters == 1: running on the single cluster; "
                  << "cluster masking of other clusters is not exercised.\n";
    }
    ASSERT_GE(params.desired_cluster_index, 0);
    ASSERT_LT(params.desired_cluster_index, params.num_clusters);
    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params));

    CryptoContext ctx = CryptoContext::from_params(params);
    ASSERT_EQ(static_cast<int64_t>(ctx.component_encodings.size()), params.num_component_rings);

    const int64_t r = params.num_component_rings;
    const int64_t q = params.q;
    const int64_t combined_modulus = (r == 1) ? params.plaintext_modulus : params.combined_component_ring_modulus;
    const WireShape shape = wire_shape(params);
    const size_t seed_bytes = SeededUniformDistribution::kSeedBytes;

    // Independent check of the bit width: 2^(bits-1) < q <= 2^bits.
    ASSERT_GE(shape.bits, 1);
    ASSERT_LE(shape.bits, 62);
    EXPECT_LE(static_cast<uint64_t>(q), uint64_t(1) << shape.bits);
    EXPECT_GT(static_cast<uint64_t>(q), uint64_t(1) << (shape.bits - 1));

    constexpr int kNumIterations = 2;
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);

        // =====================================================================
        // 1. Eval keys: client packs ONE stream, server unpacks + reconstructs.
        // =====================================================================
        SeededClientPublicMaterial eval_wire = build_seeded_public_material(ctx, secret);
        ByteStream eval_key_stream = pack_public_material(params, eval_wire);

        EXPECT_EQ(eval_key_stream.size(),
                  expected_stream_bytes(seed_bytes,
                                        shape.automorphism_levels * shape.ksk_digits * shape.n +
                                            2 * shape.prime_digits * shape.n,
                                        shape.bits));
        EXPECT_EQ(eval_key_stream.size(), public_material_stream_bytes(params));

        SeededClientPublicMaterial received_eval_wire = unpack_public_material(params, eval_key_stream);
        EXPECT_EQ(received_eval_wire.eval_key_seed, eval_wire.eval_key_seed);
        ASSERT_EQ(received_eval_wire.automorphism_b_values.size(), eval_wire.automorphism_b_values.size());
        for (size_t lvl = 0; lvl < eval_wire.automorphism_b_values.size(); ++lvl) {
            EXPECT_EQ(received_eval_wire.automorphism_b_values[lvl], canonical(eval_wire.automorphism_b_values[lvl], q))
                << "automorphism level " << lvl;
        }
        EXPECT_EQ(received_eval_wire.rgsw_message_row_b_values, canonical(eval_wire.rgsw_message_row_b_values, q));
        EXPECT_EQ(received_eval_wire.rgsw_message_sk_row_b_values,
                  canonical(eval_wire.rgsw_message_sk_row_b_values, q));

        ClientPublicMaterial pub = reconstruct_public_material(ctx, params, received_eval_wire);

        // =====================================================================
        // 2. Query: client packs one stream PER RING, server unpacks all.
        // =====================================================================
        std::vector<SignedValue> messages(static_cast<size_t>(params.embedding_length));
        std::vector<int64_t> embedding_values(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            messages[static_cast<size_t>(j)] = sample_signed_value(params, rng);
            // Canonical mod the COMBINED modulus -- see test_full_scoring_with_splits_seeded.cpp.
            embedding_values[static_cast<size_t>(j)] =
                reduce_mod(messages[static_cast<size_t>(j)].raw, combined_modulus);
        }

        SeededQuery query_wire =
            build_seeded_query(ctx, params, secret, embedding_values, params.desired_cluster_index);

        std::vector<ByteStream> upload_streams;
        for (int64_t ring = 0; ring < r; ++ring) {
            upload_streams.push_back(pack_query_ring(params, query_wire, ring));
        }
        ASSERT_EQ(static_cast<int64_t>(upload_streams.size()), r);

        for (int64_t ring = 0; ring < r; ++ring) {
            int64_t coeffs = params.embedding_length + (ring == 0 ? params.num_clusters * shape.prime_digits : 0);
            EXPECT_EQ(upload_streams[static_cast<size_t>(ring)].size(),
                      expected_stream_bytes(seed_bytes, coeffs, shape.bits))
                << "query stream of ring " << ring;
            EXPECT_EQ(upload_streams[static_cast<size_t>(ring)].size(), query_stream_bytes(params, ring));
        }

        SeededQuery received_query = unpack_query(params, upload_streams);
        EXPECT_EQ(received_query.seeds, query_wire.seeds);
        EXPECT_EQ(received_query.embedding_b_values, canonical(query_wire.embedding_b_values, q));
        EXPECT_EQ(received_query.selector_b_values, canonical(query_wire.selector_b_values, q));

        ReconstructedQuery query = reconstruct_query(ctx, params, received_query);
        ASSERT_EQ(static_cast<int64_t>(query.embedding_cts.size()), r);

        // =====================================================================
        // 3. Server: scoring on the reconstructed (received) material only.
        // =====================================================================
        std::vector<std::vector<RLWECTEvalForm>> query_eval(static_cast<size_t>(r)); // [ring][j]
        for (int64_t ring = 0; ring < r; ++ring) {
            for (const auto& lwe_ct : query.embedding_cts[static_cast<size_t>(ring)]) {
                RLWECT rlwe_ct(ctx.rlwe_param);
                pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
                query_eval[static_cast<size_t>(ring)].emplace_back(rlwe_ct);
            }
        }

        std::vector<RLWEGadgetCT> rgsw_ct;
        rgsw_ct.reserve(query.selector_cts.size());
        for (const auto& gadget_ct : query.selector_cts) {
            rgsw_ct.push_back(pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct));
        }

        std::vector<std::vector<std::vector<int64_t>>> desired_cluster_raw_values( // [s][j][i]
            static_cast<size_t>(params.splits_per_cluster));

        std::vector<std::vector<RLWECT>> final_result(static_cast<size_t>(r)); // [ring][s]
        for (int64_t ring = 0; ring < r; ++ring) {
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                final_result[static_cast<size_t>(ring)].emplace_back(ctx.rlwe_param);
            }
        }

        for (int64_t c = 0; c < params.num_clusters; ++c) {
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                std::vector<std::vector<int64_t>> split_raw_values(static_cast<size_t>(params.embedding_length));
                std::vector<std::vector<DatabasePolynomialEvalForm>> db_eval_per_j( // [j][ring]
                    static_cast<size_t>(params.embedding_length));

                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    std::vector<int64_t> raw(static_cast<size_t>(params.n));
                    for (int64_t i = 0; i < params.n; ++i) {
                        raw[static_cast<size_t>(i)] = sample_signed_value(params, rng).raw;
                    }
                    db_eval_per_j[static_cast<size_t>(j)] = crt_split_database_polynomial_eval_form(ctx, params, raw);
                    split_raw_values[static_cast<size_t>(j)] = std::move(raw);
                }

                if (c == params.desired_cluster_index) {
                    desired_cluster_raw_values[static_cast<size_t>(s)] = split_raw_values;
                }

                for (int64_t ring = 0; ring < r; ++ring) {
                    std::vector<DatabasePolynomialEvalForm> db_split_ring;
                    db_split_ring.reserve(static_cast<size_t>(params.embedding_length));
                    for (int64_t j = 0; j < params.embedding_length; ++j) {
                        db_split_ring.push_back(
                            std::move(db_eval_per_j[static_cast<size_t>(j)][static_cast<size_t>(ring)]));
                    }

                    RLWECT score = compute_split_score(ctx, query_eval[static_cast<size_t>(ring)], db_split_ring);
                    RLWECT masked(ctx.rlwe_param);
                    rgsw_ct[static_cast<size_t>(c)].mul(masked, score);
                    final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)].add(
                        final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)], masked);
                }
            }
        }

        // =====================================================================
        // 4. Response: server packs one stream PER RING, client unpacks all.
        // =====================================================================
        std::vector<ByteStream> download_streams;
        for (int64_t ring = 0; ring < r; ++ring) {
            download_streams.push_back(pack_response_ring(params, final_result[static_cast<size_t>(ring)]));
        }
        ASSERT_EQ(static_cast<int64_t>(download_streams.size()), r);
        for (int64_t ring = 0; ring < r; ++ring) {
            EXPECT_EQ(download_streams[static_cast<size_t>(ring)].size(),
                      expected_stream_bytes(0, params.splits_per_cluster * 2 * params.n, shape.bits))
                << "response stream of ring " << ring;
            EXPECT_EQ(download_streams[static_cast<size_t>(ring)].size(), response_stream_bytes(params));
        }

        std::vector<std::vector<RLWECT>> received_result; // [ring][s]
        for (int64_t ring = 0; ring < r; ++ring) {
            received_result.push_back(unpack_response_ring(ctx, params, download_streams[static_cast<size_t>(ring)]));
        }

        for (int64_t ring = 0; ring < r; ++ring) {
            ASSERT_EQ(static_cast<int64_t>(received_result[static_cast<size_t>(ring)].size()),
                      params.splits_per_cluster);
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                const RLWECT& sent = final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)];
                const RLWECT& got = received_result[static_cast<size_t>(ring)][static_cast<size_t>(s)];
                int64_t mismatches = 0;
                for (int64_t i = 0; i < params.n; ++i) {
                    mismatches += (got.a()[static_cast<size_t>(i)] != canonical(sent.a()[static_cast<size_t>(i)], q));
                    mismatches += (got.b()[static_cast<size_t>(i)] != canonical(sent.b()[static_cast<size_t>(i)], q));
                }
                EXPECT_EQ(mismatches, 0) << "response coefficients differ, ring " << ring << ", split " << s;
            }
        }

        // =====================================================================
        // 5. Client: decrypt the RECEIVED ciphertexts, compare to ground truth.
        // =====================================================================
        for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
            std::vector<Vector> decrypted_per_ring;
            for (int64_t ring = 0; ring < r; ++ring) {
                decrypted_per_ring.push_back(
                    secret.rlwe_sk->decrypt_vector(received_result[static_cast<size_t>(ring)][static_cast<size_t>(s)],
                                                   ctx.component_encodings[static_cast<size_t>(ring)]));
            }

            const auto& raw_values_for_split = desired_cluster_raw_values[static_cast<size_t>(s)];
            for (int64_t i = 0; i < params.n; ++i) {
                int64_t recomposed = (r == 1) ? decrypted_per_ring[0][i]
                                              : crt_recompose(decrypted_per_ring[0][i], decrypted_per_ring[1][i],
                                                              params.comp_ring_modulus);

                int64_t true_sum = 0;
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    true_sum += messages[static_cast<size_t>(j)].raw *
                                raw_values_for_split[static_cast<size_t>(j)][static_cast<size_t>(i)];
                }

                EXPECT_EQ(centered_residue(recomposed, combined_modulus), true_sum)
                    << "Wrong score after stream roundtrip on iteration " << iter << ", split " << s
                    << ", coefficient " << i << " (r=" << r << ")";
            }
        }
    }
}

TEST_P(StreamPackingRoundtrip, MalformedStreamsAreRejected) {
    Params params = GetParam()();
    CryptoContext ctx = CryptoContext::from_params(params);
    ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
    std::mt19937_64 rng(std::random_device{}());

    std::vector<int64_t> embedding_values;
    for (int64_t j = 0; j < params.embedding_length; ++j) {
        embedding_values.push_back(0);
    }
    SeededQuery query_wire = build_seeded_query(ctx, params, secret, embedding_values, params.desired_cluster_index);
    std::vector<ByteStream> upload_streams = pack_query(params, query_wire);

    // Truncated stream.
    {
        auto bad = upload_streams;
        bad[0].pop_back();
        EXPECT_THROW(unpack_query(params, bad), std::runtime_error);
    }
    // Extra trailing byte.
    {
        auto bad = upload_streams;
        bad.back().push_back(0);
        EXPECT_THROW(unpack_query(params, bad), std::runtime_error);
    }
    // Missing ring stream (r - 1 streams instead of r).
    {
        auto bad = upload_streams;
        bad.pop_back();
        EXPECT_THROW(unpack_query(params, bad), std::runtime_error);
    }

    // A response stream whose coefficient bits are all ones decodes to
    // 2^bits - 1 >= q (q is not a power of two here) and must be rejected.
    ByteStream all_ones(response_stream_bytes(params), 0xFF);
    int bits = coefficient_bit_width(params.q);
    if ((uint64_t(1) << bits) - 1 >= static_cast<uint64_t>(params.q)) {
        EXPECT_THROW(unpack_response_ring(ctx, params, all_ones), std::runtime_error);
    }

    // Packing a response with the wrong number of ciphertexts is refused.
    std::vector<RLWECT> too_few;
    too_few.emplace_back(ctx.rlwe_param);
    if (params.splits_per_cluster != 1) {
        EXPECT_THROW(pack_response_ring(params, too_few), std::invalid_argument);
    }

    // Raw bit-packer roundtrip at the boundary widths.
    for (int width : {1, 7, 8, 9, 31, 32, 33, 52, 53, 62, 63}) {
        std::uniform_int_distribution<uint64_t> dist(0, (uint64_t(1) << width) - 1);
        std::vector<uint64_t> values(257);
        for (auto& v : values) {
            v = dist(rng);
        }
        values[0] = 0;
        values[1] = (uint64_t(1) << width) - 1;

        BitPacker packer;
        for (uint64_t v : values) {
            packer.put(v, width);
        }
        ByteStream stream = packer.finish();
        EXPECT_EQ(stream.size(), (values.size() * static_cast<size_t>(width) + 7) / 8) << "width " << width;

        BitUnpacker unpacker(stream);
        for (size_t i = 0; i < values.size(); ++i) {
            ASSERT_EQ(unpacker.get(width), values[i]) << "width " << width << ", index " << i;
        }
        EXPECT_NO_THROW(unpacker.expect_end());
    }
}

INSTANTIATE_TEST_SUITE_P(SingleAndTwoComponentRings, StreamPackingRoundtrip,
                         ::testing::Values(&Params::make_test_params, &Params::make_test_params_component_rings),
                         [](const ::testing::TestParamInfo<Params (*)()>& info) {
                             return info.param == &Params::make_test_params ? "SingleRing" : "TwoComponentRings";
                         });
