#include "nlohmann/json.hpp"
#include "ort_genai.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

static void print_usage(int /*argc*/, char **argv)
{
    std::cerr << "usage: " << argv[0] << " model_path" << std::endl;
}

bool load_search_options(const fs::path& dirpath, std::unique_ptr<OgaGeneratorParams> &params)
{
    const fs::path config_filepath = dirpath / "genai_config.json";
    std::ifstream istrm(config_filepath);
    if (!istrm.is_open()) return false;

    const nlohmann::json j = nlohmann::json::parse(istrm);
    if (auto k = j.find("search"); k != j.end())
    {
        if (auto it = k->find("diversity_penalty"); it != k->end()) params->SetSearchOption("diversity_penalty", *it);
        if (auto it = k->find("do_sample"); it != k->end()) params->SetSearchOptionBool("do_sample", *it);
        if (auto it = k->find("early_stopping"); it != k->end()) params->SetSearchOptionBool("early_stopping", *it);
        if (auto it = k->find("length_penalty"); it != k->end()) params->SetSearchOption("length_penalty", *it);
        if (auto it = k->find("max_length"); it != k->end()) params->SetSearchOption("max_length", *it);
        if (auto it = k->find("min_length"); it != k->end()) params->SetSearchOption("min_length", *it);
        if (auto it = k->find("no_repeat_ngram_size"); it != k->end()) params->SetSearchOption("no_repeat_ngram_size", *it);
        if (auto it = k->find("num_beams"); it != k->end()) params->SetSearchOption("num_beams", *it);
        if (auto it = k->find("num_return_sequences"); it != k->end()) params->SetSearchOption("num_return_sequences", *it);
        if (auto it = k->find("past_present_share_buffer"); it != k->end()) params->SetSearchOptionBool("past_present_share_buffer", *it);
        if (auto it = k->find("repetition_penalty"); it != k->end()) params->SetSearchOption("repetition_penalty", *it);
        if (auto it = k->find("temperature"); it != k->end()) params->SetSearchOption("temperature", *it);
        if (auto it = k->find("top_k"); it != k->end()) params->SetSearchOption("top_k", *it);
        if (auto it = k->find("top_p"); it != k->end()) params->SetSearchOption("top_p", *it);
    }
    istrm.close();
    return true;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        print_usage(argc, argv);
        return -1;
    }

    const char *const model_path = argv[1];

    std::cout << "Loading model ..." << std::endl;
    auto model = OgaModel::Create(model_path);

    std::cout << "Creating tokenizer ..." << std::endl;
    auto tokenizer = OgaTokenizer::Create(*model);

    std::cout << "Loading genai_config.json ..." << std::endl;
    auto params = OgaGeneratorParams::Create(*model);

    std::cout << "Evaluating generator params and search options ..." << std::endl;
    load_search_options(model_path, params);

    const char* const prompt = "Who is Albert Einstein?";
    auto sequences = OgaSequences::Create();

    std::cout << "Encoding prompt ..." << std::endl;
    tokenizer->Encode(prompt, *sequences);
    params->SetInputSequences(*sequences);

    std::cout << "Generating tokens ..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto output_sequences = model->Generate(*params);
    auto run_time = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start);

    std::cout << "Decoding generated tokens ..." << std::endl;
    auto out_sequences = output_sequences->Get(0);
    auto out_string = tokenizer->Decode(out_sequences);

    std::cout << "Prompt: " << std::endl
              << prompt << std::endl << std::endl;
    std::cout << "Output: " << std::endl
              << out_string << std::endl << std::endl;

    std::cout << std::setprecision(2)
              << "Tokens: " << out_sequences.size()
              << ", run_time: " << run_time.count() << " seconds"
              << ", Tokens/sec: " << std::setprecision(2) << out_sequences.size() / (double)run_time.count()
              << std::endl;

    return 0;
}
