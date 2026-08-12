#include <iostream>
#include <cassert>
#include <chrono>

#include <omp.h>

#include "fhe_deck.h" 

 
using namespace FHEDeck;
 
void perf_test_sequential(); 
  
int main(){  

    perf_test_sequential(); 

}

 
void perf_test_sequential(){
    std::cout << "======== Performance Test ===========" << std::endl; 
    int64_t plaintext_space = 31; 
    int64_t ciphertext_modulus = 281474976694273;
    int64_t degree = 2048;
    int64_t decomposition_base = 256;
    double noise = 3.2;

    PlaintextEncoding encoding(PlaintextEncodingType::full_domain, plaintext_space, ciphertext_modulus);
    auto rlwe_param = std::make_shared<const RLWEParam>(RingType::negacyclic, degree, ciphertext_modulus, PolynomialArithmetic::ntt64);
    auto gadget = std::make_shared<SignedDecompositionGadget>(degree, ciphertext_modulus, decomposition_base);

    auto rlwe_sk = std::make_shared<RLWESK>(rlwe_param, KeyDistribution::ternary, noise);
    auto gadget_sk = std::make_shared<RLWEGadgetSK>(gadget, rlwe_sk);

    std::shared_ptr<FHEDeck::LWESK> lwe_sk = rlwe_sk->extract_lwe_key(); 
    LWEGadgetSK lwe_gadget_sk(lwe_sk, decomposition_base);

    StandardRoundedGaussianDistribution msg_dist(0, noise);

    /// Size of the query vector
    int32_t inner_product_size = 1024;

    /// The database is made of random polynomials (Gaussian)
    std::vector<std::shared_ptr<FHEDeck::PolynomialEvalForm>> vector_database;
    int32_t num_of_clusters = 200;


    // Create the database of random polynomials
    for(int i = 0; i < inner_product_size * num_of_clusters; ++i){ 
        Polynomial database_entry(degree, plaintext_space);
        msg_dist.fill(database_entry);
        
        std::shared_ptr<FHEDeck::PolynomialEvalForm> database_entry_eval_form = rlwe_param->mul_engine()->init_polynomial_eval_form();
        database_entry.to_eval(*database_entry_eval_form, rlwe_param->mul_engine());
        /// Get Eval Form
        vector_database.push_back(database_entry_eval_form); 
    }

    LWEToRLWEKeySwitchKey lwe_to_rlwe_ksk(*lwe_sk, *gadget_sk);

    /// Building the LWE query vector (just encryption of zero here)
    std::vector<LWECT> query_lwe;
    for(int i = 0; i < inner_product_size; ++i){
        query_lwe.push_back(lwe_sk->encrypt(0));
    }

    std::cout << "Starting the score computation" << std::endl;

    /// Swith the LWE from the query, to RLWE and get the Eval Form of the RLWE ciphertexts.
    std::vector<RLWECTEvalForm> vector_query_rlwe;  
    // Encrypt all encodings
    for(int i = 0; i < inner_product_size; ++i){ 
        RLWECT ct(rlwe_param);
        lwe_to_rlwe_ksk.lwe_to_rlwe_key_switch(ct, query_lwe[i]);
        /// Get to Eval Form
        vector_query_rlwe.push_back(RLWECTEvalForm(ct)); 
    }


    /// Compute the scores (inner product with a cluster vector)
    /// NOTE: Here a cluster has only one vector. We actually can use larger clusters, that could contain many vectors.
    // The downside is, that we need to send back more RLWE ciphertext to the client (one RLWECT per vector in a cluster).
    // But we deal with the computational bottleneck and reduce the numer of clusters that we need to choose later.
    std::vector<RLWECT> score_ciphertexts;
    for(int32_t t = 0; t < num_of_clusters; ++t){    
 
        RLWECTEvalForm out_ct(rlwe_param); 
        for(int32_t i = 0; i < inner_product_size; ++i){
            RLWECTEvalForm product(rlwe_param);
            vector_query_rlwe[i].mul(product, *vector_database[i + (inner_product_size * t)]);
            out_ct.add(out_ct, product);
        } 
        score_ciphertexts.push_back(RLWECT(out_ct));
    }  

    /// Prepare the LWE for the PIR (To choose the cluster) 
    LWEToRGSWKeySwitchKey lwe_to_rgsw_ksk(*lwe_sk, *gadget_sk);

    // The client prepares the cluster choice query, that consists of GadgetLWE ciphertexts that encrypt 0, 
    // except for the one that corresponds to the cluster that the client wants to choose.
    std::vector<LWEGadgetCT> cluster_choise_query;
    for(int32_t it = 0; it < num_of_clusters; ++it){ 
        cluster_choise_query.push_back(lwe_gadget_sk.gadget_encrypt(0));
    }

    
    std::cout << "Key switching the cluster choice query" << std::endl;
    /// Key Switching to RGSW
    /// Now all but one of these RGSW's has the message 1, and everything else is 0. The one that has the message 1 is the one that corresponds to the cluster that we want to choose.
    std::vector<RLWEGadgetCT> cluster_choise_query_rgsw;
    for(int32_t it = 0; it < num_of_clusters; ++it){
        cluster_choise_query_rgsw.push_back(lwe_to_rgsw_ksk.lwe_to_rlwe_key_switch(cluster_choise_query[it]));
    }

    std::cout << "Choosing the Cluster" << std::endl;

     auto start = std::chrono::high_resolution_clock::now(); 

    /// What is left is to multiply the score_ciphertexts by the cluster_choise_query_rgsw, and then sum all of them together. 
    ///The result will be a single RLWE ciphertext that corresponds to the inner product of the query with the database.
    RLWECT final_result(rlwe_param);
    cluster_choise_query_rgsw[0].mul(final_result, score_ciphertexts[0]); 
    for(int32_t it = 1; it < num_of_clusters; ++it){
        RLWECT product(rlwe_param);
        cluster_choise_query_rgsw[it].mul(product, score_ciphertexts[it]);
        
        final_result.add(final_result, product);
    }


      auto end = std::chrono::high_resolution_clock::now();  
    // 3. Calculate the duration (difference) [2]
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 

    // 4. Extract the numeric value using .count() [2]
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl; 
    

    std::cout << "Finished" << std::endl;
    /// Now the client knows the scores and chooses the element to download.
    /// To download this element we can use conventional PIR.
}
  
 