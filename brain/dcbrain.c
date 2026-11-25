#include "dcbrain.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- loading ---------------- */
void (*brain_progress)(long done, long total);
static char *cur;
static Q8 take_q8(int n, int gs) {
    Q8 r;
    r.q = (int8_t *)cur; cur += n;
    r.s = (float *)cur;  cur += (n / gs) * sizeof(float);
    return r;
}
static float *take_f(int n) { float *r = (float *)cur; cur += n * sizeof(float); return r; }

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if(!p) { printf("dcbrain: out of memory (%u bytes)\n", (unsigned)(n * sz)); exit(1); }
    return p;
}

int brain_load(Brain *b, const char *path) {
    FILE *f = fopen(path, "rb");
    if(!f) return -1;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    b->blob = malloc(size);
    if(!b->blob) { fclose(f); return -2; }
    for(long done = 0; done < size;) {             /* read in chunks so the UI can show progress */
        long n = size - done < 256 * 1024 ? size - done : 256 * 1024;
        if(fread((char *)b->blob + done, 1, n, f) != (size_t)n) { fclose(f); return -2; }
        done += n;
        if(brain_progress) brain_progress(done, size);
    }
    fclose(f);
    char *p = b->blob;
    if(memcmp(p, "DCT1", 4)) return -3;
    int32_t *h = (int32_t *)(p + 4);
    b->dim = h[0]; b->hidden_dim = h[1]; b->n_layers = h[2]; b->n_heads = h[3];
    b->n_kv_heads = h[4]; b->vocab_size = h[5]; b->seq_len = h[6]; b->gs = h[7];
    cur = p + 36;
    int L = b->n_layers, d = b->dim, hd = b->hidden_dim, gs = b->gs;
    int kv = d * b->n_kv_heads / b->n_heads;
    b->tok_emb = take_q8(b->vocab_size * d, gs);
    b->att_norm = xcalloc(L, sizeof(float *)); b->ffn_norm = xcalloc(L, sizeof(float *));
    b->wq = xcalloc(L, sizeof(Q8)); b->wk = xcalloc(L, sizeof(Q8)); b->wv = xcalloc(L, sizeof(Q8));
    b->wo = xcalloc(L, sizeof(Q8)); b->w1 = xcalloc(L, sizeof(Q8)); b->w2 = xcalloc(L, sizeof(Q8));
    b->w3 = xcalloc(L, sizeof(Q8));
    for(int l = 0; l < L; l++) {
        b->att_norm[l] = take_f(d);
        b->wq[l] = take_q8(d * d, gs); b->wk[l] = take_q8(kv * d, gs);
        b->wv[l] = take_q8(kv * d, gs); b->wo[l] = take_q8(d * d, gs);
        b->ffn_norm[l] = take_f(d);
        b->w1[l] = take_q8(hd * d, gs); b->w2[l] = take_q8(d * hd, gs); b->w3[l] = take_q8(hd * d, gs);
    }
    b->norm = take_f(d);
    if(cur - p != size) { printf("dcbrain: size mismatch %ld vs %ld\n", (long)(cur - p), size); return -4; }

    b->x = xcalloc(d, 4); b->xb = xcalloc(d, 4); b->xb2 = xcalloc(d, 4);
    b->hb = xcalloc(hd, 4); b->hb2 = xcalloc(hd, 4);
    b->q = xcalloc(d, 4); b->k = xcalloc(kv, 4); b->v = xcalloc(kv, 4);
    b->att = xcalloc(b->n_heads * b->seq_len, 4);
    b->logits = xcalloc(b->vocab_size, 4);
    b->key_cache = xcalloc((size_t)L * b->seq_len * kv, 4);
    b->value_cache = xcalloc((size_t)L * b->seq_len * kv, 4);
    if(gs != 32) { printf("dcbrain: group size must be 32\n"); return -5; }
    return 0;
}

/* ---------------- math ---------------- */
static void rmsnorm(float *o, const float *x, const float *w, int n) {
    float ss = 0;
    for(int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-5f);
    for(int i = 0; i < n; i++) o[i] = w[i] * (ss * x[i]);
}

static void softmax(float *x, int n) {
    float m = x[0];
    for(int i = 1; i < n; i++) if(x[i] > m) m = x[i];
    float s = 0;
    for(int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for(int i = 0; i < n; i++) x[i] /= s;
}

/* 4-wide dot product: the SH-4's FIPR instruction on the Dreamcast, plain C elsewhere. */
#ifdef _arch_dreamcast
#include <dc/fmath.h>
#define DOT4(a, b0, b1, b2, b3) fipr((a)[0], (a)[1], (a)[2], (a)[3], (b0), (b1), (b2), (b3))
#else
#define DOT4(a, b0, b1, b2, b3) ((a)[0] * (b0) + (a)[1] * (b1) + (a)[2] * (b2) + (a)[3] * (b3))
#endif

/* xout[d] = W[d][n] @ x[n]: float activations, int8 weights with per-group (32) scales.
   Four independent FIPR accumulators per group hide FIPR's 4-cycle latency. */
static void matmul(float *xout, const float *x, const Q8 *w, int n, int d, int gs) {
    for(int i = 0; i < d; i++) {
        const int8_t *wr = w->q + (size_t)i * n;
        const float *ws = w->s + (size_t)i * n / gs;
        if(i + 1 < d) __builtin_prefetch(wr + n);
        float val = 0;
        for(int j = 0, g = 0; j < n; j += 32, g++) {
            const float *xg = x + j;
            const int8_t *wg = wr + j;
            float a0 = DOT4(xg,      wg[0],  wg[1],  wg[2],  wg[3]);
            float a1 = DOT4(xg + 4,  wg[4],  wg[5],  wg[6],  wg[7]);
            float a2 = DOT4(xg + 8,  wg[8],  wg[9],  wg[10], wg[11]);
            float a3 = DOT4(xg + 12, wg[12], wg[13], wg[14], wg[15]);
            a0 += DOT4(xg + 16, wg[16], wg[17], wg[18], wg[19]);
            a1 += DOT4(xg + 20, wg[20], wg[21], wg[22], wg[23]);
            a2 += DOT4(xg + 24, wg[24], wg[25], wg[26], wg[27]);
            a3 += DOT4(xg + 28, wg[28], wg[29], wg[30], wg[31]);
            val += ((a0 + a1) + (a2 + a3)) * ws[g];
        }
        xout[i] = val;
    }
}

float *brain_forward(Brain *b, int token, int pos) {
    int d = b->dim, hd = b->hidden_dim, gs = b->gs, nh = b->n_heads;
    int kv = d * b->n_kv_heads / nh, kvmul = nh / b->n_kv_heads, hs = d / nh;
    float *x = b->x;
    const int8_t *er = b->tok_emb.q + (size_t)token * d;
    const float *es = b->tok_emb.s + (size_t)token * d / gs;
    for(int i = 0; i < d; i++) x[i] = er[i] * es[i / gs];

    /* RoPE rotation for this position, once per token (same for every layer and head) */
    static float rcos[128], rsin[128];
    for(int i = 0; i < hs / 2; i++) {
        float freq = 1.0f / powf(10000.0f, (2 * i) / (float)hs);
        rcos[i] = cosf(pos * freq);
        rsin[i] = sinf(pos * freq);
    }

    for(int l = 0; l < b->n_layers; l++) {
        rmsnorm(b->xb, x, b->att_norm[l], d);
        size_t off = (size_t)l * b->seq_len * kv;
        float *kc = b->key_cache + off + (size_t)pos * kv, *vc = b->value_cache + off + (size_t)pos * kv;
        matmul(b->q, b->xb, &b->wq[l], d, d, gs);
        matmul(kc, b->xb, &b->wk[l], d, kv, gs);
        matmul(vc, b->xb, &b->wv[l], d, kv, gs);
        /* RoPE on pairs (i, i+1) */
        for(int i = 0; i < d; i += 2) {
            int hdim = i % hs;
            float fcr = rcos[hdim / 2], fci = rsin[hdim / 2];
            float *vecs[2] = { b->q, kc };
            for(int v = 0; v < (i < kv ? 2 : 1); v++) {
                float *vec = vecs[v], v0 = vec[i], v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
        /* attention */
        float scale = 1.0f / sqrtf((float)hs);
        for(int h = 0; h < nh; h++) {
            float *q = b->q + h * hs, *att = b->att + h * b->seq_len;
            for(int t = 0; t <= pos; t++) {
                float *k = b->key_cache + off + (size_t)t * kv + (h / kvmul) * hs, s = 0;
                for(int i = 0; i < hs; i += 4) s += DOT4(q + i, k[i], k[i + 1], k[i + 2], k[i + 3]);
                att[t] = s * scale;
            }
            softmax(att, pos + 1);
            float *o = b->xb + h * hs;
            memset(o, 0, hs * sizeof(float));
            for(int t = 0; t <= pos; t++) {
                float *v = b->value_cache + off + (size_t)t * kv + (h / kvmul) * hs, a = att[t];
                for(int i = 0; i < hs; i++) o[i] += a * v[i];
            }
        }
        matmul(b->xb2, b->xb, &b->wo[l], d, d, gs);
        for(int i = 0; i < d; i++) x[i] += b->xb2[i];
        /* SwiGLU FFN */
        rmsnorm(b->xb, x, b->ffn_norm[l], d);
        matmul(b->hb, b->xb, &b->w1[l], d, hd, gs);
        matmul(b->hb2, b->xb, &b->w3[l], d, hd, gs);
        for(int i = 0; i < hd; i++) { float v = b->hb[i]; b->hb[i] = v / (1.0f + expf(-v)) * b->hb2[i]; }
        matmul(b->xb, b->hb, &b->w2[l], hd, d, gs);
        for(int i = 0; i < d; i++) x[i] += b->xb[i];
    }
    rmsnorm(x, x, b->norm, d);
    matmul(b->logits, x, &b->tok_emb, d, b->vocab_size, gs);
    return b->logits;
}

/* ---------------- tokenizer ---------------- */
static Tokenizer *g_tok;
static int cmp_ids(const void *a, const void *b) {
    return strcmp(g_tok->vocab[*(const int *)a], g_tok->vocab[*(const int *)b]);
}

int tok_load(Tokenizer *t, const char *path, int vocab_size) {
    FILE *f = fopen(path, "rb");
    if(!f) return -1;
    int maxlen;
    if(fread(&maxlen, 4, 1, f) != 1) return -2;
    t->vocab_size = vocab_size;
    t->vocab = xcalloc(vocab_size, sizeof(char *));
    t->len = xcalloc(vocab_size, sizeof(int));
    t->score = xcalloc(vocab_size, sizeof(float));
    t->sorted = xcalloc(vocab_size, sizeof(int));
    t->nsorted = 0;
    for(int i = 0; i < vocab_size; i++) {
        if(fread(&t->score[i], 4, 1, f) != 1 || fread(&t->len[i], 4, 1, f) != 1) return -3;
        t->vocab[i] = xcalloc(t->len[i] + 1, 1);
        if(t->len[i] && fread(t->vocab[i], 1, t->len[i], f) != (size_t)t->len[i]) return -4;
        if(i > TOK_BYTE0 + 255) t->sorted[t->nsorted++] = i;   /* normal pieces only */
    }
    fclose(f);
    g_tok = t;
    qsort(t->sorted, t->nsorted, sizeof(int), cmp_ids);
    return 0;
}

static int lookup(Tokenizer *t, const char *s) {
    int lo = 0, hi = t->nsorted - 1;
    while(lo <= hi) {
        int mid = (lo + hi) / 2, c = strcmp(s, t->vocab[t->sorted[mid]]);
        if(!c) return t->sorted[mid];
        if(c < 0) hi = mid - 1; else lo = mid + 1;
    }
    return -1;
}

/* SentencePiece-BPE style: start from characters, greedily merge the best-scoring pair. */
int tok_encode(Tokenizer *t, const char *text, int *out, int max) {
    int n = 0;
    char buf[256];
    for(const unsigned char *c = (const unsigned char *)text; *c && n < max;) {
        int clen = 1;
        if(*c >= 0xC0) { while(clen < 4 && (c[clen] & 0xC0) == 0x80) clen++; }
        memcpy(buf, c, clen); buf[clen] = 0;
        int id = lookup(t, buf);
        if(id >= 0) out[n++] = id;
        else for(int i = 0; i < clen && n < max; i++) out[n++] = TOK_BYTE0 + c[i];
        c += clen;
    }
    for(;;) {
        float best = -1e30f; int bi = -1, bid = -1;
        for(int i = 0; i + 1 < n; i++) {
            if(out[i] < TOK_BYTE0 + 256 || out[i + 1] < TOK_BYTE0 + 256) continue;
            if(t->len[out[i]] + t->len[out[i + 1]] >= (int)sizeof(buf)) continue;
            snprintf(buf, sizeof(buf), "%s%s", t->vocab[out[i]], t->vocab[out[i + 1]]);
            int id = lookup(t, buf);
            if(id >= 0 && t->score[id] > best) { best = t->score[id]; bi = i; bid = id; }
        }
        if(bi < 0) break;
        out[bi] = bid;
        memmove(out + bi + 1, out + bi + 2, (n - bi - 2) * sizeof(int));
        n--;
    }
    return n;
}

int tok_piece(Tokenizer *t, int id, char *buf) {
    if(id < TOK_BYTE0) return 0;
    if(id < TOK_BYTE0 + 256) { buf[0] = (char)(id - TOK_BYTE0); return 1; }
    memcpy(buf, t->vocab[id], t->len[id]);
    return t->len[id];
}

/* ---------------- sampling ---------------- */
static float rnd(uint32_t *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s >> 8) / 16777216.0f;
}

int sample(float *logits, int n, float temp, float topp, uint32_t *rng) {
    if(temp <= 0) {
        int best = 0;
        for(int i = 1; i < n; i++) if(logits[i] > logits[best]) best = i;
        return best;
    }
    for(int i = 0; i < n; i++) logits[i] /= temp;
    softmax(logits, n);
    /* top-p: repeatedly take the max until mass >= topp (vocab is small, n=4096) */
    static int idx[8192];
    float mass = 0; int k = 0;
    static float p[8192];
    memcpy(p, logits, n * sizeof(float));
    while(mass < topp && k < 64) {
        int best = 0;
        for(int i = 1; i < n; i++) if(p[i] > p[best]) best = i;
        idx[k++] = best; mass += p[best]; p[best] = -1;
    }
    float r = rnd(rng) * mass, c = 0;
    for(int i = 0; i < k; i++) { c += logits[idx[i]]; if(r < c) return idx[i]; }
    return idx[k - 1];
}
