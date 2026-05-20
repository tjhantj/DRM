#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>

namespace drm {

constexpr int DIM = 128;

// ── Distances and similarities ───────────────────────────────────────────

inline double dist_sq(const double* Z, int u, int v, int dim) {
    const double* zu = Z + (size_t)u * dim;
    const double* zv = Z + (size_t)v * dim;
    double s = 0;
    for (int d = 0; d < dim; d++) {
        double diff = zu[d] - zv[d];
        s += diff * diff;
    }
    return s;
}

inline double stable_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double z = std::exp(x);
    return z / (1.0 + z);
}

inline double cosine_sim_01(const double* a, const double* b, int dim) {
    double dot = 0.0, na2 = 0.0, nb2 = 0.0;
    for (int d = 0; d < dim; d++) {
        dot += a[d] * b[d];
        na2 += a[d] * a[d];
        nb2 += b[d] * b[d];
    }
    double denom = std::sqrt(na2) * std::sqrt(nb2);
    double cs = (denom > 0.0) ? dot / denom : -1.0;
    cs = std::clamp(cs, -1.0, 1.0);
    return (cs + 1.0) / 2.0;
}

inline double pearson_sim_01(const double* a, const double* b, int dim) {
    double ma = 0.0, mb = 0.0;
    for (int d = 0; d < dim; d++) { ma += a[d]; mb += b[d]; }
    ma /= dim; mb /= dim;
    double cov = 0.0, va = 0.0, vb = 0.0;
    for (int d = 0; d < dim; d++) {
        double ca = a[d] - ma, cb = b[d] - mb;
        cov += ca * cb; va += ca * ca; vb += cb * cb;
    }
    double denom = std::sqrt(va) * std::sqrt(vb);
    double corr = (denom > 0.0) ? cov / denom : -1.0;
    corr = std::clamp(corr, -1.0, 1.0);
    return (corr + 1.0) / 2.0;
}

inline double vector_similarity(const double* a, const double* b, int dim,
                                const std::string& phi_type) {
    if (phi_type == "sigmoid") {
        double dot = 0.0;
        for (int d = 0; d < dim; d++) dot += a[d] * b[d];
        return stable_sigmoid(dot);
    }
    if (phi_type == "pearson") return pearson_sim_01(a, b, dim);
    return cosine_sim_01(a, b, dim);
}

inline double embedding_similarity(const double* Z, int u, int v, int dim,
                                   const std::string& phi_type) {
    return vector_similarity(Z + (size_t)u * dim, Z + (size_t)v * dim, dim, phi_type);
}

inline double vec_dist(const double* a, const double* b, int dim) {
    double s = 0;
    for (int d = 0; d < dim; d++) {
        double diff = a[d] - b[d];
        s += diff * diff;
    }
    return std::sqrt(s);
}

// Piecewise-linear interpolation over sorted anchors.
inline double g_interp(double d,
                       const std::vector<double>& centers,
                       const std::vector<double>& means) {
    int k = (int)centers.size();
    if (k == 0) return 0.0;
    if (k == 1) return means[0];
    if (d <= centers.front()) return means.front();
    if (d >= centers.back())  return means.back();
    auto it = std::upper_bound(centers.begin(), centers.end(), d);
    int hi = (int)(it - centers.begin());
    int lo = hi - 1;
    double denom = centers[hi] - centers[lo];
    double t = (denom > 1e-18) ? (d - centers[lo]) / denom : 0.0;
    return means[lo] + t * (means[hi] - means[lo]);
}

// ── Graph ────────────────────────────────────────────────────────────────

struct Edge {
    int to;
    double weight;
    double w_uv;
};

struct Graph {
    int n = 0;
    std::vector<std::string> names;
    std::unordered_map<std::string, int> id_map;
    std::vector<std::vector<Edge>> adj;
    std::vector<double> deg;
    double two_m  = 0;
    double two_m_w = 0;
    double g_average = 1.0;

    int add_node(const std::string& name) {
        auto it = id_map.find(name);
        if (it != id_map.end()) return it->second;
        int idx = n++;
        names.push_back(name);
        id_map[name] = idx;
        adj.emplace_back();
        return idx;
    }
};

inline Graph load_network(const std::string& path) {
    Graph G;
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open: " + path);
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string su, sv;
        double w = 1.0;
        iss >> su >> sv;
        if (!(iss >> w)) w = 1.0;
        int u = G.add_node(su);
        int v = G.add_node(sv);
        G.adj[u].push_back({v, w, 0.0});
        G.adj[v].push_back({u, w, 0.0});
    }
    return G;
}

inline std::pair<std::vector<double>, int> load_representations(
    const std::string& path, const Graph& G)
{
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open: " + path);
    std::unordered_map<std::string, std::vector<double>> rep_map;
    int dim = -1;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string node_id;
        iss >> node_id;
        std::vector<double> vec;
        double val;
        while (iss >> val) vec.push_back(val);
        if (vec.empty()) continue;
        if (dim < 0) dim = (int)vec.size();
        if ((int)vec.size() == dim) rep_map[node_id] = std::move(vec);
    }
    std::vector<double> Z((size_t)G.n * dim, 0.0);
    for (int i = 0; i < G.n; i++) {
        auto it = rep_map.find(G.names[i]);
        if (it != rep_map.end())
            for (int d = 0; d < dim; d++)
                Z[(size_t)i * dim + d] = it->second[d];
    }
    return {std::move(Z), dim};
}

inline void compute_structural_properties(Graph& G) {
    G.deg.assign(G.n, 0.0);
    for (int u = 0; u < G.n; u++)
        for (auto& e : G.adj[u]) G.deg[u] += e.weight;
    G.two_m = 0;
    for (int u = 0; u < G.n; u++) G.two_m += G.deg[u];
    if (G.two_m == 0) G.two_m = 1.0;
}

inline void compute_w_uv(Graph& G, const double* Z, int dim,
                         const std::string& phi_type, int g_mode = 0)
{
    G.two_m_w = 0;
    double sum_phi = 0.0;
    long long n_edges = 0;
    for (int u = 0; u < G.n; u++) {
        for (auto& e : G.adj[u]) {
            double sim = embedding_similarity(Z, u, e.to, dim, phi_type);
            e.w_uv = (g_mode == 4) ? e.weight : e.weight * sim;  // nophi: drop phi
            G.two_m_w += e.w_uv;
            sum_phi += sim;
            n_edges++;
        }
    }
    G.g_average = (n_edges > 0) ? sum_phi / (double)n_edges : 1.0;
}

// ── 1D k-means + silhouette ──────────────────────────────────────────────

struct Kmeans1DResult {
    std::vector<double> centers;
    std::vector<int>    assignments;
    double              wcss;
};

inline Kmeans1DResult kmeans_1d(const std::vector<double>& data_sorted,
                                int k, std::mt19937& rng, int max_iter = 50)
{
    int N = (int)data_sorted.size();
    Kmeans1DResult R;
    R.centers.assign(k, 0.0);
    R.assignments.assign(N, -1);

    // k-means++ init
    std::uniform_int_distribution<int> uni(0, N - 1);
    R.centers[0] = data_sorted[uni(rng)];
    std::vector<double> d2(N, 0.0);
    for (int j = 1; j < k; j++) {
        double total = 0.0;
        for (int i = 0; i < N; i++) {
            double best = std::numeric_limits<double>::max();
            for (int c = 0; c < j; c++) {
                double diff = data_sorted[i] - R.centers[c];
                double dd = diff * diff;
                if (dd < best) best = dd;
            }
            d2[i] = best;
            total += best;
        }
        if (total <= 0.0) { R.centers[j] = data_sorted[uni(rng)]; continue; }
        std::uniform_real_distribution<double> ur(0.0, total);
        double r = ur(rng);
        double acc = 0.0;
        int chosen = N - 1;
        for (int i = 0; i < N; i++) {
            acc += d2[i];
            if (acc >= r) { chosen = i; break; }
        }
        R.centers[j] = data_sorted[chosen];
    }
    std::sort(R.centers.begin(), R.centers.end());

    std::vector<double> mids(std::max(0, k - 1));
    for (int it = 0; it < max_iter; it++) {
        for (int c = 0; c + 1 < k; c++)
            mids[c] = 0.5 * (R.centers[c] + R.centers[c + 1]);
        bool changed = false;
        for (int i = 0; i < N; i++) {
            int a = (k == 1) ? 0
                : (int)(std::upper_bound(mids.begin(), mids.end(), data_sorted[i]) - mids.begin());
            if (R.assignments[i] != a) { R.assignments[i] = a; changed = true; }
        }
        if (!changed && it > 0) break;
        std::vector<double> sum(k, 0.0);
        std::vector<int>    cnt(k, 0);
        for (int i = 0; i < N; i++) {
            int a = R.assignments[i];
            sum[a] += data_sorted[i];
            cnt[a] += 1;
        }
        for (int c = 0; c < k; c++)
            if (cnt[c] > 0) R.centers[c] = sum[c] / cnt[c];
        std::sort(R.centers.begin(), R.centers.end());
    }

    double wcss = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = data_sorted[i] - R.centers[R.assignments[i]];
        wcss += diff * diff;
    }
    R.wcss = wcss;
    return R;
}

inline double silhouette_score_1d(const std::vector<double>& data,
                                  const std::vector<int>& assignments, int k)
{
    int N = (int)data.size();
    if (k <= 1 || k >= N) return -1.0;
    std::vector<std::vector<int>> clusters(k);
    for (int i = 0; i < N; i++) clusters[assignments[i]].push_back(i);
    double total = 0.0;
    long long count = 0;
    for (int c = 0; c < k; c++) {
        const auto& Ic = clusters[c];
        int nc = (int)Ic.size();
        if (nc <= 1) continue;
        for (int idx : Ic) {
            double sum_a = 0.0;
            for (int j : Ic) if (j != idx) sum_a += std::abs(data[idx] - data[j]);
            double a = sum_a / (nc - 1);
            double b = std::numeric_limits<double>::infinity();
            for (int cc = 0; cc < k; cc++) {
                if (cc == c) continue;
                const auto& Icc = clusters[cc];
                int ncc = (int)Icc.size();
                if (ncc == 0) continue;
                double sum_b = 0.0;
                for (int j : Icc) sum_b += std::abs(data[idx] - data[j]);
                double mean = sum_b / ncc;
                if (mean < b) b = mean;
            }
            double denom = std::max(a, b);
            if (denom > 0) total += (b - a) / denom;
            count++;
        }
    }
    return (count > 0) ? total / (double)count : -1.0;
}

inline int silhouette_best_k(const std::vector<double>& data_sorted,
                             int k_min, int k_max, std::mt19937& rng, int fallback_k)
{
    double best_score = -std::numeric_limits<double>::infinity();
    int best_k = fallback_k;
    for (int k = k_min; k <= k_max; k++) {
        auto R = kmeans_1d(data_sorted, k, rng, 30);
        double s = silhouette_score_1d(data_sorted, R.assignments, k);
        if (s > best_score) { best_score = s; best_k = k; }
    }
    return best_k;
}

// ── g(d) estimation ──────────────────────────────────────────────────────

inline std::pair<std::vector<double>, std::vector<double>>
precompute_gd_distribution(const double* Z, int n, int dim,
                           int& num_bins, double sampling,
                           int method, const std::string& phi_type)
{
    // nog (2) and gaverage (5) use a constant g(d); skip distribution estimation.
    if (method == 2 || method == 5) {
        num_bins = 5;
        return {std::vector<double>(), std::vector<double>()};
    }

    long long total_pairs = (long long)n * (n - 1) / 2;
    long long num_samples = std::min((long long)(total_pairs * sampling), 5000000LL);

    std::mt19937 rng(std::random_device{}());
    std::vector<double> distances, similarities;

    if (total_pairs <= num_samples) {
        distances.reserve(total_pairs);
        similarities.reserve(total_pairs);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                distances.push_back(std::sqrt(dist_sq(Z, i, j, dim)));
                similarities.push_back(embedding_similarity(Z, i, j, dim, phi_type));
            }
        num_samples = (long long)distances.size();
    } else {
        distances.resize(num_samples);
        similarities.resize(num_samples);
        std::uniform_int_distribution<int> uni(0, n - 1);
        long long count = 0;
        while (count < num_samples) {
            int u = uni(rng), v = uni(rng);
            if (u == v) continue;
            distances[count] = std::sqrt(dist_sq(Z, u, v, dim));
            similarities[count] = embedding_similarity(Z, u, v, dim, phi_type);
            count++;
        }
    }

    // kmeans + silhouette (method == 1 or 3)
    const int k_min_search = 2;
    const int k_max_search = 15;
    const int fallback_k = 5;
    const long long subsample_size = std::min((long long)5000, num_samples);

    std::vector<double> sub_dist;
    sub_dist.reserve((size_t)subsample_size);
    if (subsample_size == num_samples) {
        sub_dist = distances;
    } else {
        std::vector<long long> idx(num_samples);
        std::iota(idx.begin(), idx.end(), 0LL);
        std::shuffle(idx.begin(), idx.end(), rng);
        for (long long i = 0; i < subsample_size; i++)
            sub_dist.push_back(distances[idx[i]]);
    }
    std::sort(sub_dist.begin(), sub_dist.end());

    int est_k = silhouette_best_k(sub_dist, k_min_search, k_max_search, rng, fallback_k);
    if (est_k < 1) est_k = fallback_k;

    std::vector<double> sorted_d = distances;
    std::sort(sorted_d.begin(), sorted_d.end());
    auto final_R = kmeans_1d(sorted_d, est_k, rng, 50);

    std::vector<double> mids(std::max(0, est_k - 1));
    for (int c = 0; c + 1 < est_k; c++)
        mids[c] = 0.5 * (final_R.centers[c] + final_R.centers[c + 1]);

    std::vector<double> sim_sum(est_k, 0.0);
    std::vector<long long> sim_cnt(est_k, 0);
    for (long long i = 0; i < num_samples; i++) {
        int a = (est_k == 1) ? 0
            : (int)(std::upper_bound(mids.begin(), mids.end(), distances[i]) - mids.begin());
        sim_sum[a] += similarities[i];
        sim_cnt[a] += 1;
    }
    std::vector<double> cluster_means(est_k, 0.0);
    for (int c = 0; c < est_k; c++)
        if (sim_cnt[c] > 0) cluster_means[c] = sim_sum[c] / sim_cnt[c];
    for (int c = 1; c < est_k; c++)
        if (sim_cnt[c] == 0) cluster_means[c] = cluster_means[c - 1];

    num_bins = est_k;
    return {final_R.centers, cluster_means};
}

// ── Output ───────────────────────────────────────────────────────────────

inline void save_partition(const std::string& algorithm, const std::string& output_dir,
                           const std::string& network_name, const std::string& phi,
                           const std::string& bin_method, int bin_num,
                           double gamma, double sampling,
                           double precompute_time, double algo_time,
                           const std::vector<int>& partition, const Graph& G,
                           const std::string& base_dir = "../output")
{
    namespace fs = std::filesystem;
    std::string algo_dir = base_dir + "/" + output_dir;
    fs::create_directories(algo_dir);

    std::ostringstream ss;
    ss << network_name << "_" << algorithm
       << "_" << phi << "_g" << bin_method << "_bin" << bin_num
       << "_gamma" << gamma << "_samp" << sampling;
    std::string run_folder = ss.str();
    std::string run_dir = algo_dir + "/" + run_folder;
    fs::create_directories(run_dir);

    int run_idx = 1;
    for (auto& entry : fs::directory_iterator(run_dir)) {
        std::string fname = entry.path().filename().string();
        auto dp = fname.rfind('.');
        if (dp != std::string::npos && fname.substr(dp) == ".txt") {
            try { run_idx = std::max(run_idx, std::stoi(fname.substr(0, dp)) + 1); }
            catch (...) {}
        }
    }
    std::string txt_path;
    while (true) {
        txt_path = run_dir + "/" + std::to_string(run_idx) + ".txt";
        int fd = ::open(txt_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (fd >= 0) { ::close(fd); break; }
        if (errno != EEXIST) throw std::runtime_error("Cannot create: " + txt_path);
        run_idx++;
    }

    {
        std::vector<std::pair<std::string, int>> sorted_p;
        for (int i = 0; i < G.n; i++) sorted_p.emplace_back(G.names[i], partition[i]);
        std::sort(sorted_p.begin(), sorted_p.end());
        std::ofstream fout(txt_path);
        for (auto& [name, cid] : sorted_p) fout << name << " " << cid << "\n";
    }

    std::string csv_path = algo_dir + "/summary.csv";
    bool exists = fs::exists(csv_path);
    {
        std::ofstream fout(csv_path, std::ios::app);
        if (!exists) {
            fout << "Dataset,Algorithm,Phi,Gamma,Sampling,"
                 << "Bin_Method,Bin_Num,Idx,Precompute_Time,Algorithm_Time,Total_Time,Run_Dir\n";
        }
        fout << network_name << "," << algorithm << ","
             << phi << "," << gamma << "," << sampling << ","
             << bin_method << "," << bin_num << "," << run_idx << ","
             << std::fixed << std::setprecision(4)
             << precompute_time << "," << algo_time << ","
             << (precompute_time + algo_time) << "," << run_folder << "\n";
    }

    std::cout << "[" << algorithm << "] " << run_folder << "/" << run_idx << ".txt\n";
}

} // namespace drm
