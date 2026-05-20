#pragma once
// Holm-Lichtenberg-Thorup fully dynamic connectivity.
// Amortized: insert O(log n), delete O(log^2 n), connected O(log n).

#include "euler_tour_tree.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <cmath>

namespace drm {

struct DCEdgeKey {
    int u, v;
    bool operator==(const DCEdgeKey& o) const noexcept { return u == o.u && v == o.v; }
};
struct DCEdgeHash {
    std::size_t operator()(const DCEdgeKey& e) const noexcept {
        return std::hash<long long>()(
            (static_cast<long long>(e.u) << 32) | static_cast<unsigned>(e.v));
    }
};

class FullyDynamicConnectivity {
public:
    int n;
    int comp_count;
    int L;

    std::vector<EulerTourTree> forests;
    std::vector<std::vector<std::unordered_set<int>>> incident_nontree;
    std::unordered_map<DCEdgeKey, int,  DCEdgeHash> edge_level;
    std::unordered_map<DCEdgeKey, bool, DCEdgeHash> is_tree_edge;

    explicit FullyDynamicConnectivity(int n_, int max_edges_hint = 0)
        : n(n_), comp_count(n_)
    {
        L = (n_ > 1) ? static_cast<int>(std::ceil(std::log2((double)n_))) : 1;
        if (L < 1) L = 1;
        int me = std::max(max_edges_hint, n_);
        forests.reserve(L + 1);
        for (int i = 0; i <= L; ++i) forests.emplace_back(n_, me);
        incident_nontree.assign(L + 1, std::vector<std::unordered_set<int>>(n_));
    }

    static DCEdgeKey canon(int u, int v) {
        if (u > v) std::swap(u, v);
        return {u, v};
    }

    void insert_edge(int u, int v) {
        if (u == v) return;
        DCEdgeKey e = canon(u, v);
        if (edge_level.count(e)) return;
        edge_level[e] = 0;
        if (!forests[0].connected(u, v)) {
            forests[0].link(u, v);
            is_tree_edge[e] = true;
            comp_count--;
        } else {
            add_nontree_edge(u, v, 0);
            is_tree_edge[e] = false;
        }
    }

    bool connected(int u, int v) {
        if (u == v) return true;
        return forests[0].connected(u, v);
    }

    int get_component_count() const { return comp_count; }

    void delete_edge(int u, int v) {
        if (u == v) return;
        DCEdgeKey e = canon(u, v);
        auto it_l = edge_level.find(e);
        if (it_l == edge_level.end()) return;
        int level = it_l->second;
        edge_level.erase(it_l);

        bool tree = false;
        auto it_t = is_tree_edge.find(e);
        if (it_t != is_tree_edge.end()) {
            tree = it_t->second;
            is_tree_edge.erase(it_t);
        }

        if (!tree) {
            remove_nontree_edge(u, v, level);
            return;
        }

        for (int i = 0; i <= level; ++i) forests[i].cut(u, v);

        for (int i = level; i >= 0; --i) {
            int sz_u = forests[i].size(u);
            int sz_v = forests[i].size(v);
            int smaller_root = (sz_u <= sz_v) ? u : v;

            if (i + 1 <= L) {
                std::vector<std::pair<int,int>> to_promote_tree;
                forests[i].iterate_tree_edges(smaller_root,
                    [&](int a, int b) {
                        DCEdgeKey ek{a, b};
                        auto it = edge_level.find(ek);
                        if (it != edge_level.end() && it->second == i)
                            to_promote_tree.push_back({a, b});
                    });
                for (auto& pr : to_promote_tree) {
                    int a = pr.first, b = pr.second;
                    DCEdgeKey ek{a, b};
                    edge_level[ek] = i + 1;
                    if (!forests[i + 1].connected(a, b))
                        forests[i + 1].link(a, b);
                }
            }

            DCEdgeKey replacement{-1, -1};
            bool found = false;

            while (true) {
                int x = forests[i].find_marked(smaller_root);
                if (x < 0) break;

                std::vector<int> incs(incident_nontree[i][x].begin(),
                                      incident_nontree[i][x].end());
                for (int y : incs) {
                    if (forests[i].connected(x, y)) {
                        if (i + 1 <= L) {
                            remove_nontree_edge(x, y, i);
                            add_nontree_edge(x, y, i + 1);
                            edge_level[canon(x, y)] = i + 1;
                        } else {
                            break;
                        }
                    } else {
                        replacement = canon(x, y);
                        remove_nontree_edge(x, y, i);
                        is_tree_edge[replacement] = true;
                        edge_level[replacement] = i;
                        for (int j = 0; j <= i; ++j)
                            if (!forests[j].connected(replacement.u, replacement.v))
                                forests[j].link(replacement.u, replacement.v);
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }

            if (found) return;
        }

        comp_count++;
    }

private:
    void add_nontree_edge(int u, int v, int level) {
        incident_nontree[level][u].insert(v);
        incident_nontree[level][v].insert(u);
        forests[level].add_mark(u, 1);
        forests[level].add_mark(v, 1);
    }

    void remove_nontree_edge(int u, int v, int level) {
        if (incident_nontree[level][u].erase(v)) forests[level].add_mark(u, -1);
        if (incident_nontree[level][v].erase(u)) forests[level].add_mark(v, -1);
    }
};

} // namespace drm
