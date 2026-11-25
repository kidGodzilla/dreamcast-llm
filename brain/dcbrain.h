/* dcbrain: llama2.c-style int8 inference for the Dreamcast (portable C, also builds on the Mac). */
#ifndef DCBRAIN_H
#define DCBRAIN_H
#include <stdint.h>

#define TOK_BOS 1
#define TOK_U 3
#define TOK_DC 4
#define TOK_BOOK 5
#define TOK_E 6
#define TOK_LOOK 7
#define TOK_ENDLOOK 8
#define TOK_BYTE0 9          /* <0x00> .. <0xFF> are ids 9..264 */

typedef struct { int8_t *q; float *s; } Q8;

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, gs;
    Q8 tok_emb;
    float **att_norm, **ffn_norm, *norm;
    Q8 *wq, *wk, *wv, *wo, *w1, *w2, *w3;
    /* run state */
    float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits, *key_cache, *value_cache;
    Q8 xq, hq;
    void *blob;
} Brain;

typedef struct {
    char **vocab;
    int *len;
    float *score;
    int vocab_size;
    int *sorted;              /* ids sorted by string, normal pieces only */
    int nsorted;
} Tokenizer;

int brain_load(Brain *b, const char *path);
extern void (*brain_progress)(long done, long total);   /* optional, called while loading */
float *brain_forward(Brain *b, int token, int pos);
int tok_load(Tokenizer *t, const char *path, int vocab_size);
int tok_encode(Tokenizer *t, const char *text, int *out, int max);
/* writes the bytes for a token into buf, returns length (0 for special tokens) */
int tok_piece(Tokenizer *t, int id, char *buf);
int sample(float *logits, int n, float temp, float topp, uint32_t *rng);

#endif
