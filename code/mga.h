#pragma once
#include "core.h"
#include <unordered_set>

namespace drm {

// Multi-level greedy aggregation with centroid approximation.
// One node moves at most once per level (no pingpong).
inline std::vector<int> run_mga(
    const Graph& G, const double* Z, int dim,
    const std::vector<double>& bins,
    const std::vector<double>& bin_averages,
    double gamma, int g_mode = 0)
{
    int n = G.n;
    if (G.two_m_w == 0) return std::vector<int>(n, 0);

    const bool skip_dist = (g_mode == 2 || g_mode == 5);   // nog or gaverage
    const double const_g = (g_mode == 5) ? G.g_average : 1.0;  // gaverage

    struct AdjE { int to; double w; };
    std::vector<std::vector<AdjE>> curr_adj(n);
    for (int u = 0; u < n; u++)
        for (auto& e : G.adj[u])
            curr_adj[u].push_back({e.to, e.w_uv});

    std::vector<std::vector<double>> node_sum_z;
    if (!skip_dist) node_sum_z.assign(n, std::vector<double>(dim));
    std::vector<double> node_sum_k(n);
    std::vector<int> node_count(n, 1);

    for (int i = 0; i < n; i++) {
        if (!skip_dist) {
            const double* zi = Z + (size_t)i * dim;
            for (int d = 0; d < dim; d++) node_sum_z[i][d] = zi[d];
        }
        node_sum_k[i] = G.deg[i];
    }

    int curr_n = n;
    std::vector<int> final_partition(n);
    std::iota(final_partition.begin(), final_partition.end(), 0);
    int level = 0;

    std::vector<double> cent_u(skip_dist ? 0 : dim);
    std::vector<double> cent_c(skip_dist ? 0 : dim);

    while (true) {
        std::cout << "Level " << level << " (nodes=" << curr_n << ")\n";

        std::vector<int> partition(curr_n);
        std::iota(partition.begin(), partition.end(), 0);
        std::vector<std::unordered_set<int>> communities(curr_n);
        for (int i = 0; i < curr_n; i++) communities[i].insert(i);

        std::vector<std::vector<double>> comm_sum_z;
        if (!skip_dist) comm_sum_z.assign(curr_n, std::vector<double>(dim));
        std::vector<double> comm_sum_k(curr_n);
        std::vector<int> comm_count(curr_n);

        for (int i = 0; i < curr_n; i++) {
            if (!skip_dist) comm_sum_z[i] = node_sum_z[i];
            comm_sum_k[i] = node_sum_k[i];
            comm_count[i] = node_count[i];
        }

        std::vector<bool> locked(curr_n, false);

        auto p_approx = [&](int u, int cid) -> double {
            if (comm_count[cid] == 0) return 0.0;
            if (skip_dist)
                return gamma / G.two_m * node_sum_k[u] * comm_sum_k[cid] * const_g;
            double inv_u = 1.0 / node_count[u];
            for (int d = 0; d < dim; d++) cent_u[d] = node_sum_z[u][d] * inv_u;
            double inv_c = 1.0 / comm_count[cid];
            for (int d = 0; d < dim; d++) cent_c[d] = comm_sum_z[cid][d] * inv_c;
            double dist = vec_dist(cent_u.data(), cent_c.data(), dim);
            double g_d = g_interp(dist, bins, bin_averages);
            if (g_mode == 3) return gamma * comm_count[cid] * g_d;  // nokk2m
            return gamma / G.two_m * node_sum_k[u] * comm_sum_k[cid] * g_d;
        };

        auto p_approx_exclude = [&](int u, int cid) -> double {
            int cnt = comm_count[cid] - node_count[u];
            if (cnt <= 0) return 0.0;
            double sum_k_excl = comm_sum_k[cid] - node_sum_k[u];
            if (skip_dist)
                return gamma / G.two_m * node_sum_k[u] * sum_k_excl * const_g;
            double inv_u = 1.0 / node_count[u];
            for (int d = 0; d < dim; d++) cent_u[d] = node_sum_z[u][d] * inv_u;
            double inv_c = 1.0 / cnt;
            for (int d = 0; d < dim; d++)
                cent_c[d] = (comm_sum_z[cid][d] - node_sum_z[u][d]) * inv_c;
            double dist = vec_dist(cent_u.data(), cent_c.data(), dim);
            double g_d = g_interp(dist, bins, bin_averages);
            if (g_mode == 3) return gamma * cnt * g_d;  // nokk2m
            return gamma / G.two_m * node_sum_k[u] * sum_k_excl * g_d;
        };

        std::mt19937 rng(std::random_device{}());
        std::vector<int> order(curr_n);
        std::iota(order.begin(), order.end(), 0);
        int iteration = 0;

        while (true) {
            int moves = 0;
            iteration++;
            std::shuffle(order.begin(), order.end(), rng);

            for (int u : order) {
                if (locked[u]) continue;
                int A_id = partition[u];

                std::unordered_map<int, double> w_u_comm;
                for (auto& [v, w] : curr_adj[u]) w_u_comm[partition[v]] += w;

                double gain_A = (w_u_comm.count(A_id) ? w_u_comm[A_id] : 0.0)
                              - p_approx_exclude(u, A_id);
                int best_B = A_id;
                double max_gain = gain_A;

                for (auto& [B_id, w_uB] : w_u_comm) {
                    if (B_id == A_id) continue;
                    double gain_B = w_uB - p_approx(u, B_id);
                    if (gain_B > max_gain) { max_gain = gain_B; best_B = B_id; }
                }

                if (best_B != A_id) {
                    if (!skip_dist)
                        for (int d = 0; d < dim; d++)
                            comm_sum_z[A_id][d] -= node_sum_z[u][d];
                    comm_sum_k[A_id] -= node_sum_k[u];
                    comm_count[A_id] -= node_count[u];
                    communities[A_id].erase(u);

                    if (!skip_dist)
                        for (int d = 0; d < dim; d++)
                            comm_sum_z[best_B][d] += node_sum_z[u][d];
                    comm_sum_k[best_B] += node_sum_k[u];
                    comm_count[best_B] += node_count[u];
                    communities[best_B].insert(u);

                    partition[u] = best_B;
                    locked[u] = true;
                    moves++;
                }
            }
            std::cout << "  iter " << iteration << ": " << moves << " moves\n";
            if (moves == 0) break;
        }

        std::set<int> active;
        for (int i = 0; i < curr_n; i++)
            if (!communities[partition[i]].empty()) active.insert(partition[i]);
        std::vector<int> unique_comms(active.begin(), active.end());
        int m = (int)unique_comms.size();
        std::unordered_map<int, int> id_map;
        for (int i = 0; i < m; i++) id_map[unique_comms[i]] = i;

        std::vector<std::vector<double>> new_node_sum_z;
        if (!skip_dist) new_node_sum_z.assign(m, std::vector<double>(dim, 0.0));
        std::vector<double> new_node_sum_k(m, 0.0);
        std::vector<int> new_node_count(m, 0);

        for (int i = 0; i < m; i++) {
            int old_cid = unique_comms[i];
            if (!skip_dist) new_node_sum_z[i] = comm_sum_z[old_cid];
            new_node_sum_k[i] = comm_sum_k[old_cid];
            new_node_count[i] = comm_count[old_cid];
        }

        if (level == 0) {
            for (int i = 0; i < curr_n; i++)
                final_partition[i] = id_map[partition[i]];
        } else {
            std::unordered_map<int, int> old_to_new;
            for (int i = 0; i < curr_n; i++) old_to_new[i] = id_map[partition[i]];
            for (int orig = 0; orig < n; orig++)
                final_partition[orig] = old_to_new[final_partition[orig]];
        }

        if (m == curr_n) { std::cout << "  converged.\n"; break; }
        std::cout << "  aggregation: " << curr_n << " -> " << m << "\n";

        std::unordered_map<long long, double> edge_w;
        for (int u = 0; u < curr_n; u++) {
            int uc = id_map[partition[u]];
            for (auto& [v, w] : curr_adj[u]) {
                int vc = id_map[partition[v]];
                if (uc != vc) {
                    long long key = (long long)std::min(uc, vc) * m + std::max(uc, vc);
                    edge_w[key] += w;
                }
            }
        }

        std::vector<std::vector<AdjE>> new_adj(m);
        for (auto& [key, w] : edge_w) {
            int u = (int)(key / m), v = (int)(key % m);
            double half = w / 2.0;
            new_adj[u].push_back({v, half});
            new_adj[v].push_back({u, half});
        }

        node_sum_z = std::move(new_node_sum_z);
        node_sum_k = std::move(new_node_sum_k);
        node_count = std::move(new_node_count);
        curr_adj   = std::move(new_adj);
        curr_n     = m;
        level++;
    }

    std::set<int> unique(final_partition.begin(), final_partition.end());
    std::unordered_map<int, int> remap;
    int nid = 0;
    for (int uid : unique) remap[uid] = nid++;
    for (int& p : final_partition) p = remap[p];
    return final_partition;
}

} // namespace drm
