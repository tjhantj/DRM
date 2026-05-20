#include "core.h"
#include "mga.h"
#include "bgd.h"

#include <string>
#include <iostream>
#include <chrono>

using namespace drm;
using Clock = std::chrono::high_resolution_clock;

struct Args {
    std::string algorithm = "MGA";
    std::string network   = "";
    double gamma    = 1.0;
    double sampling = 0.3;
    int    g        = 1;
    std::string phi    = "cosine";
    std::string output = "";
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        if (i + 1 >= argc) break;
        std::string val = argv[++i];
        if      (key == "--algorithm") a.algorithm = val;
        else if (key == "--network")   a.network = val;
        else if (key == "--gamma")     a.gamma = std::stod(val);
        else if (key == "--sampling")  a.sampling = std::stod(val);
        else if (key == "--g")         a.g = std::stoi(val);
        else if (key == "--phi")       a.phi = val;
        else if (key == "--output")    a.output = val;
    }
    if (a.network.empty() || a.output.empty() || a.g < 1 || a.g > 5) {
        std::cerr << "Usage: ./drm --network <name> --output <dir> "
                  << "[--algorithm MGA|BGD] [--gamma 1.0] "
                  << "[--sampling 0.3] [--g 1|2|3|4|5] "
                  << "[--phi cosine|sigmoid|pearson]\n"
                  << "  --g 1=kmeans (default), 2=nog, 3=nokk2m, "
                  << "4=nophi, 5=gaverage\n";
        exit(1);
    }
    return a;
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    std::string base = "../datasets/" + args.network;
    Graph G = load_network(base + "/network.dat");
    auto [Z, dim] = load_representations(base + "/embedding.dat", G);

    // nophi (4) reuses the main kmeans calibration internally; only compute_w_uv differs.
    int internal_method = (args.g == 4) ? 1 : args.g;
    int internal_g_mode = (args.g == 4) ? 0 : args.g;

    auto t0 = Clock::now();
    compute_structural_properties(G);

    int actual_k = 0;  // overwritten by silhouette / forced to 5 for const g
    auto [bins, bin_averages] = precompute_gd_distribution(
        Z.data(), G.n, dim, actual_k,
        args.sampling, internal_method, args.phi);

    compute_w_uv(G, Z.data(), dim, args.phi, args.g);
    double pre_time = std::chrono::duration<double>(Clock::now() - t0).count();

    auto t1 = Clock::now();
    std::vector<int> partition;
    if (args.algorithm == "MGA")
        partition = run_mga(G, Z.data(), dim, bins, bin_averages, args.gamma, internal_g_mode);
    else if (args.algorithm == "BGD")
        partition = run_bgd(G, Z.data(), dim, bins, bin_averages, args.gamma, internal_g_mode);
    else {
        std::cerr << "Unknown algorithm: " << args.algorithm << "\n";
        return 1;
    }
    double algo_time = std::chrono::duration<double>(Clock::now() - t1).count();

    static const char* method_map[] = {
        "", "kmeans", "nog", "nokk2m", "nophi", "gaverage"
    };
    std::string bin_method = method_map[args.g];

    save_partition(args.algorithm, args.output, args.network, args.phi,
                   bin_method, actual_k, args.gamma, args.sampling,
                   pre_time, algo_time, partition, G);
    return 0;
}
