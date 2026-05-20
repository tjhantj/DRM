#pragma once
// Treap-based Euler Tour Tree used by the HLT dynamic connectivity layer.
//
// One self-loop occurrence per graph node and two directed occurrences
// per tree edge. In-order traversal of the treap yields the Euler tour.

#include <vector>
#include <deque>
#include <unordered_map>
#include <utility>
#include <random>
#include <cassert>
#include <cstdint>

namespace drm {

struct ETTreapNode {
    uint32_t priority = 0;
    ETTreapNode* l = nullptr;
    ETTreapNode* r = nullptr;
    ETTreapNode* p = nullptr;
    int  size      = 1;
    int  mark      = 0;
    int  mark_sub  = 0;
    int  u_id      = -1;
    int  v_id      = -1;
};

inline void ettrp_update(ETTreapNode* x) {
    x->size     = 1;
    x->mark_sub = x->mark;
    if (x->l) { x->size += x->l->size; x->mark_sub += x->l->mark_sub; }
    if (x->r) { x->size += x->r->size; x->mark_sub += x->r->mark_sub; }
}

inline ETTreapNode* ettrp_merge(ETTreapNode* a, ETTreapNode* b) {
    if (!a) { if (b) b->p = nullptr; return b; }
    if (!b) { a->p = nullptr; return a; }
    ETTreapNode* result;
    if (a->priority > b->priority) {
        a->r = ettrp_merge(a->r, b);
        if (a->r) a->r->p = a;
        ettrp_update(a);
        a->p = nullptr;
        result = a;
    } else {
        b->l = ettrp_merge(a, b->l);
        if (b->l) b->l->p = b;
        ettrp_update(b);
        b->p = nullptr;
        result = b;
    }
    return result;
}

inline std::pair<ETTreapNode*, ETTreapNode*>
ettrp_split(ETTreapNode* root, int k) {
    if (!root) return {nullptr, nullptr};
    int ls = root->l ? root->l->size : 0;
    ETTreapNode *L, *R;
    if (k <= ls) {
        auto pr = ettrp_split(root->l, k);
        L = pr.first;
        root->l = pr.second;
        if (root->l) root->l->p = root;
        ettrp_update(root);
        R = root;
    } else {
        auto pr = ettrp_split(root->r, k - ls - 1);
        root->r = pr.first;
        if (root->r) root->r->p = root;
        ettrp_update(root);
        L = root;
        R = pr.second;
    }
    if (L) L->p = nullptr;
    if (R) R->p = nullptr;
    return {L, R};
}

inline ETTreapNode* ettrp_root(ETTreapNode* x) {
    while (x->p) x = x->p;
    return x;
}

inline int ettrp_pos(ETTreapNode* x) {
    int pos = x->l ? x->l->size : 0;
    ETTreapNode* c = x;
    while (c->p) {
        if (c->p->r == c) pos += 1 + (c->p->l ? c->p->l->size : 0);
        c = c->p;
    }
    return pos;
}

inline ETTreapNode* ettrp_find_marked(ETTreapNode* x) {
    if (!x || x->mark_sub == 0) return nullptr;
    if (x->l && x->l->mark_sub > 0) return ettrp_find_marked(x->l);
    if (x->mark > 0) return x;
    return ettrp_find_marked(x->r);
}

class EulerTourTree {
public:
    int n;
    std::deque<ETTreapNode> pool;
    std::vector<ETTreapNode*> self_loop;

    struct DirKey {
        int u, v;
        bool operator==(const DirKey& o) const noexcept { return u == o.u && v == o.v; }
    };
    struct DirHash {
        std::size_t operator()(const DirKey& k) const noexcept {
            return std::hash<long long>()(
                (static_cast<long long>(k.u) << 32) | static_cast<unsigned>(k.v));
        }
    };
    std::unordered_map<DirKey, ETTreapNode*, DirHash> edge_occ;
    std::mt19937 rng;

    explicit EulerTourTree(int n_, int = 0)
        : n(n_), rng(0xC0FFEEu)
    {
        self_loop.assign(n_, nullptr);
        for (int i = 0; i < n_; ++i) self_loop[i] = alloc_occ(i, i);
    }

    ETTreapNode* alloc_occ(int u, int v) {
        pool.emplace_back();
        ETTreapNode* nd = &pool.back();
        nd->u_id = u;
        nd->v_id = v;
        nd->priority = (uint32_t)rng();
        nd->size = 1;
        return nd;
    }

    void reroot(int u) {
        ETTreapNode* sl = self_loop[u];
        int pos = ettrp_pos(sl);
        if (pos == 0) return;
        ETTreapNode* r = ettrp_root(sl);
        auto pr = ettrp_split(r, pos);
        ettrp_merge(pr.second, pr.first);
    }

    bool connected(int u, int v) {
        if (u == v) return true;
        return ettrp_root(self_loop[u]) == ettrp_root(self_loop[v]);
    }

    int size(int u) {
        ETTreapNode* r = ettrp_root(self_loop[u]);
        return (r->size + 2) / 3;
    }

    void link(int u, int v) {
        assert(!connected(u, v));
        reroot(u);
        reroot(v);
        ETTreapNode* uv = alloc_occ(u, v);
        ETTreapNode* vu = alloc_occ(v, u);
        ETTreapNode* ru = ettrp_root(self_loop[u]);
        ETTreapNode* rv = ettrp_root(self_loop[v]);
        ETTreapNode* m1 = ettrp_merge(ru, uv);
        ETTreapNode* m2 = ettrp_merge(m1, rv);
        ettrp_merge(m2, vu);
        edge_occ[{u, v}] = uv;
        edge_occ[{v, u}] = vu;
    }

    void cut(int u, int v) {
        auto it_uv = edge_occ.find({u, v});
        auto it_vu = edge_occ.find({v, u});
        assert(it_uv != edge_occ.end() && it_vu != edge_occ.end());
        ETTreapNode* uv = it_uv->second;
        ETTreapNode* vu = it_vu->second;

        int p_uv = ettrp_pos(uv);
        int p_vu = ettrp_pos(vu);
        if (p_uv > p_vu) {
            std::swap(p_uv, p_vu);
            std::swap(uv, vu);
        }

        ETTreapNode* root = ettrp_root(uv);
        auto p1 = ettrp_split(root, p_uv);
        auto p2 = ettrp_split(p1.second, 1);
        int len_mid = p_vu - p_uv - 1;
        auto p3 = ettrp_split(p2.second, len_mid);
        auto p4 = ettrp_split(p3.second, 1);

        ettrp_merge(p1.first, p4.second);
        if (p3.first) p3.first->p = nullptr;
        edge_occ.erase(it_uv);
        edge_occ.erase(it_vu);
    }

    void add_mark(int u, int delta) {
        ETTreapNode* sl = self_loop[u];
        sl->mark += delta;
        ETTreapNode* c = sl;
        while (c) { c->mark_sub += delta; c = c->p; }
    }

    int total_mark(int u) { return ettrp_root(self_loop[u])->mark_sub; }

    int find_marked(int u) {
        ETTreapNode* root = ettrp_root(self_loop[u]);
        ETTreapNode* found = ettrp_find_marked(root);
        return found ? found->u_id : -1;
    }

    bool has_edge(int u, int v) { return edge_occ.count({u, v}) > 0; }

    template<typename F>
    void iterate_tree_edges(int u, F&& cb) {
        ETTreapNode* root = ettrp_root(self_loop[u]);
        std::vector<ETTreapNode*> stk;
        ETTreapNode* c = root;
        while (c || !stk.empty()) {
            while (c) { stk.push_back(c); c = c->l; }
            c = stk.back(); stk.pop_back();
            if (c->u_id < c->v_id) cb(c->u_id, c->v_id);
            c = c->r;
        }
    }

    template<typename F>
    void iterate_nodes(int u, F&& cb) {
        ETTreapNode* root = ettrp_root(self_loop[u]);
        std::vector<ETTreapNode*> stk;
        ETTreapNode* c = root;
        while (c || !stk.empty()) {
            while (c) { stk.push_back(c); c = c->l; }
            c = stk.back(); stk.pop_back();
            if (c->u_id == c->v_id) cb(c->u_id);
            c = c->r;
        }
    }
};

} // namespace drm
