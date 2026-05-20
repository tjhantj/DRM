#pragma once
#include "core.h"
#include "dynamic_connectivity.h"
#include <queue>
#include <unordered_set>
#include <functional>
#include <limits>

namespace drm {

// Working graph with edge removal.
struct BGDGraph {
    int n;
    std::vector<std::unordered_map<int, double>> adj;
    int edge_count = 0;

    BGDGraph(const Graph& G) : n(G.n), adj(G.n) {
        double eps = 1e-8;
        for (int u = 0; u < n; u++)
            for (auto& e : G.adj[u])
                if (u < e.to) {
                    double len = 1.0 / (eps + e.w_uv);
                    adj[u][e.to] = len;
                    adj[e.to][u] = len;
                    edge_count++;
                }
    }

    void remove_edge(int u, int v) {
        adj[u].erase(v);
        adj[v].erase(u);
        edge_count--;
    }
};

inline std::vector<std::vector<int>> connected_components(const BGDGraph& G) {
    std::vector<bool> visited(G.n, false);
    std::vector<std::vector<int>> comps;
    for (int i = 0; i < G.n; i++) {
        if (visited[i]) continue;
        std::vector<int> comp;
        std::queue<int> q;
        q.push(i); visited[i] = true;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            comp.push_back(u);
            for (auto& [v, _] : G.adj[u])
                if (!visited[v]) { visited[v] = true; q.push(v); }
        }
        comps.push_back(std::move(comp));
    }
    return comps;
}

// Brandes-style sampled edge betweenness on weighted shortest paths.
struct EdgeKey {
    int u, v;
    bool operator==(const EdgeKey& o) const { return u == o.u && v == o.v; }
};
struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const {
        return std::hash<long long>()(((long long)k.u << 32) | k.v);
    }
};
using BetMap = std::unordered_map<EdgeKey, double, EdgeKeyHash>;

inline BetMap compute_edge_betweenness(const BGDGraph& G, int k_samples) {
    BetMap bet;
    int n = G.n;
    std::vector<int> active;
    for (int i = 0; i < n; i++) if (!G.adj[i].empty()) active.push_back(i);
    if (active.empty()) return bet;

    std::mt19937 rng(std::random_device{}());
    int k = std::min(k_samples, (int)active.size());
    std::shuffle(active.begin(), active.end(), rng);

    const double INF = std::numeric_limits<double>::infinity();
    for (int si = 0; si < k; si++) {
        int s = active[si];
        std::vector<double> dist(n, INF);
        std::vector<double> sigma(n, 0.0);
        std::vector<std::vector<int>> pred(n);
        std::vector<int> order;
        order.reserve(n);

        using PQ = std::priority_queue<std::pair<double,int>,
                   std::vector<std::pair<double,int>>, std::greater<>>;
        PQ pq;
        dist[s] = 0; sigma[s] = 1.0;
        pq.push({0, s});

        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;
            order.push_back(u);
            for (auto& [v, length] : G.adj[u]) {
                double nd = dist[u] + length;
                if (nd < dist[v] - 1e-12) {
                    dist[v] = nd;
                    sigma[v] = sigma[u];
                    pred[v].clear();
                    pred[v].push_back(u);
                    pq.push({nd, v});
                } else if (std::abs(nd - dist[v]) < 1e-12) {
                    sigma[v] += sigma[u];
                    pred[v].push_back(u);
                }
            }
        }

        std::vector<double> delta(n, 0.0);
        for (int i = (int)order.size() - 1; i >= 0; i--) {
            int v = order[i];
            for (int p : pred[v]) {
                double c = (sigma[p] / sigma[v]) * (1.0 + delta[v]);
                EdgeKey ek = {std::min(p, v), std::max(p, v)};
                bet[ek] += c;
                delta[p] += c;
            }
        }
    }
    return bet;
}

// BGD with incremental Q tracking and HLT dynamic connectivity for bridge tests.
inline std::vector<int> run_bgd(
    const Graph& G, const double* Z, int dim,
    const std::vector<double>& bins,
    const std::vector<double>& bin_averages,
    double gamma, int g_mode = 0, int k_samples = -1)
{
    int n = G.n;
    if (G.two_m_w == 0) return std::vector<int>(n, 0);

    const bool skip_dist = (g_mode == 2 || g_mode == 5);   // nog or gaverage
    const double const_g = (g_mode == 5) ? G.g_average : 1.0;  // gaverage

    if (k_samples < 0)
        k_samples = std::min(100, std::max(1, (int)std::sqrt(n)));

    BGDGraph Gt(G);

    FullyDynamicConnectivity dc(n);
    for (int u = 0; u < n; ++u)
        for (auto& e : G.adj[u])
            if (u < e.to) dc.insert_edge(u, e.to);

    auto init_comps = connected_components(Gt);
    int num_comms = (int)init_comps.size();

    std::vector<int> labels(n, 0);
    std::vector<std::vector<int>> comm_members(num_comms);
    for (int c = 0; c < num_comms; c++)
        for (int v : init_comps[c]) {
            labels[v] = c;
            comm_members[c].push_back(v);
        }
    std::vector<int> C_star = labels;

    double total_w_curr = 0;
    for (int u = 0; u < n; u++)
        for (auto& e : G.adj[u])
            if (labels[u] == labels[e.to]) total_w_curr += e.w_uv;

    double total_p_curr = 0;
    for (int c = 0; c < num_comms; c++) {
        const auto& members = comm_members[c];
        int nc = (int)members.size();
        if (nc < 2) continue;
        for (int i = 0; i < nc; i++) {
            int u = members[i];
            for (int j = i + 1; j < nc; j++) {
                int v = members[j];
                double g_d = skip_dist ? const_g
                           : g_interp(std::sqrt(dist_sq(Z, u, v, dim)), bins, bin_averages);
                if (g_mode == 3) total_p_curr += 2.0 * gamma * g_d;  // nokk2m
                else total_p_curr += 2.0 * gamma / G.two_m * G.deg[u] * G.deg[v] * g_d;
            }
        }
    }

    double Q_star = -std::numeric_limits<double>::infinity();
    int num_splits = 0;

    std::cout << "BGD: N=" << n << ", E=" << Gt.edge_count << ", k=" << k_samples << "\n";

    while (Gt.edge_count > 0) {
        BetMap bet = compute_edge_betweenness(Gt, k_samples);
        std::vector<std::pair<EdgeKey, double>> sorted_edges(bet.begin(), bet.end());
        std::sort(sorted_edges.begin(), sorted_edges.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });

        bool split_found = false;
        for (auto& [ek, _] : sorted_edges) {
            if (Gt.adj[ek.u].find(ek.v) == Gt.adj[ek.u].end()) continue;
            Gt.remove_edge(ek.u, ek.v);
            dc.delete_edge(ek.u, ek.v);
            if (dc.connected(ek.u, ek.v)) continue;

            num_splits++;
            int L = labels[ek.u];
            std::vector<int> C1, C2;
            C1.reserve(comm_members[L].size());
            C2.reserve(comm_members[L].size());
            for (int v : comm_members[L]) {
                if (dc.connected(ek.u, v)) C1.push_back(v);
                else                       C2.push_back(v);
            }

            const std::vector<int>& smaller = (C1.size() <= C2.size()) ? C1 : C2;
            std::vector<bool> in_other(n, false);
            if (&smaller == &C1) for (int v : C2) in_other[v] = true;
            else                 for (int v : C1) in_other[v] = true;
            double dw = 0;
            for (int u : smaller)
                for (auto& e : G.adj[u])
                    if (in_other[e.to]) dw += e.w_uv;
            total_w_curr -= 2.0 * dw;

            double dp_sum = 0;
            for (int u : C1) {
                for (int v : C2) {
                    double g_d = skip_dist ? const_g
                               : g_interp(std::sqrt(dist_sq(Z, u, v, dim)), bins, bin_averages);
                    if (g_mode == 3) dp_sum += g_d;  // nokk2m
                    else dp_sum += G.deg[u] * G.deg[v] * g_d;
                }
            }
            if (g_mode == 3) total_p_curr -= 2.0 * gamma * dp_sum;  // nokk2m
            else total_p_curr -= 2.0 * gamma / G.two_m * dp_sum;

            int new_label = num_comms++;
            for (int v : C2) labels[v] = new_label;
            comm_members[L] = std::move(C1);
            comm_members.push_back(std::move(C2));

            double Q_curr = (total_w_curr - total_p_curr) / G.two_m_w;
            if (Q_curr > Q_star) { Q_star = Q_curr; C_star = labels; }
            split_found = true;
            break;
        }
        if (!split_found) break;
    }

    std::set<int> unique_labels(C_star.begin(), C_star.end());
    std::unordered_map<int, int> remap;
    int nid = 0;
    for (int uid : unique_labels) remap[uid] = nid++;
    for (int& p : C_star) p = remap[p];

    std::cout << "BGD done (splits=" << num_splits << ")\n";
    return C_star;
}

} // namespace drm
