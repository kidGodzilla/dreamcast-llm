/* Mac-side check: encode prompts and greedy-decode; compare with PyTorch's dc7m.test.txt.
   cc -O2 -o host_test host_test.c dcbrain.c -lm && ./host_test ../train/out/dc7m.bin ../train/out/tok.bin ../train/out/dc7m.test.txt */
#include "dcbrain.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    Brain b; Tokenizer t;
    if(brain_load(&b, argv[1])) { printf("model load failed\n"); return 1; }
    if(tok_load(&t, argv[2], b.vocab_size)) { printf("tok load failed\n"); return 1; }
    FILE *f = fopen(argv[3], "r");
    char line[4096];
    int ok = 0, total = 0;
    while(fgets(line, sizeof(line), f)) {
        char *prompt = strtok(line, "\t"), *xin = strtok(NULL, "\t"), *yref = strtok(NULL, "\t");
        int ref_in[256], nin = 0, ref_out[256], nout = 0;
        for(char *s = strtok(xin, " "); s; s = strtok(NULL, " ")) ref_in[nin++] = atoi(s);
        for(char *s = strtok(yref, " "); s; s = strtok(NULL, " ")) ref_out[nout++] = atoi(s);
        int x[256], n = 0;
        x[n++] = TOK_BOS; x[n++] = TOK_U;
        n += tok_encode(&t, prompt, x + n, 200);
        x[n++] = TOK_DC;
        int enc_ok = n == nin && !memcmp(x, ref_in, n * sizeof(int));
        clock_t c0 = clock();
        float *logits = NULL;
        for(int i = 0; i < n; i++) logits = brain_forward(&b, x[i], i);
        int match = 0, pos = n;
        printf("> %s\nDC: ", prompt);
        for(int k = 0; k < nout; k++) {
            int tk = sample(logits, b.vocab_size, 0, 0, NULL);
            if(tk == ref_out[k]) match++;
            char buf[64]; int l = tok_piece(&t, tk, buf); fwrite(buf, 1, l, stdout);
            if(tk == TOK_E || tk == TOK_U) break;
            logits = brain_forward(&b, tk, pos++);
        }
        double s = (double)(clock() - c0) / CLOCKS_PER_SEC;
        printf("\n   encode %s, %d/%d tokens match PyTorch, %.0f tok/s on this Mac\n",
               enc_ok ? "OK" : "MISMATCH", match, nout, pos / s);
        ok += enc_ok && match == nout; total++;
    }
    printf("%d/%d prompts identical\n", ok, total);
    return ok == total ? 0 : 1;
}
