/* dreamcast_thinking — milestone 0
 *
 * 1. Benchmarks a float mat-vec (the inner loop of LLM inference), naive C vs
 *    the SH-4 FIPR instruction, and prints projected tokens/sec for a few
 *    model sizes.
 * 2. A keyboard console: type a line, press Enter, get a placeholder reply.
 *    The model will replace the placeholder.
 */
#include <kos.h>
#include <dc/fmath.h>
#include <string.h>
#include <stdio.h>

KOS_INIT_FLAGS(INIT_DEFAULT);

#define W 640
#define H 480
#define COLS (W / BFONT_THIN_WIDTH)      /* 53 */
#define ROWS (H / BFONT_HEIGHT)          /* 20 */

/* ---------- text console ---------- */
static char screen[ROWS][COLS + 1];
static int cur_row = 0;

static void redraw(void) {
    vid_clear(0, 0, 0);
    for(int r = 0; r < ROWS; r++)
        bfont_draw_str(vram_s + r * BFONT_HEIGHT * W, W, 1, screen[r]);
}

static void newline(void) {
    if(++cur_row >= ROWS) {
        memmove(screen[0], screen[1], sizeof(screen[0]) * (ROWS - 1));
        cur_row = ROWS - 1;
    }
    screen[cur_row][0] = 0;
}

/* Word-wrapped print. '\n' starts a new line. */
static void put(const char *s) {
    for(; *s; s++) {
        if(*s == '\n') { newline(); continue; }
        int len = strlen(screen[cur_row]);
        if(len >= COLS) { newline(); len = 0; }
        screen[cur_row][len] = *s;
        screen[cur_row][len + 1] = 0;
    }
}

/* ---------- mat-vec benchmark ---------- */
#define N 512                             /* 512x512 floats = 1 MB of weights */
static float Wm[N * N] __attribute__((aligned(32)));
static float xv[N] __attribute__((aligned(32)));
static float yv[N] __attribute__((aligned(32)));

static void matvec_naive(void) {
    for(int i = 0; i < N; i++) {
        const float *row = Wm + i * N;
        float acc = 0.0f;
        for(int j = 0; j < N; j++) acc += row[j] * xv[j];
        yv[i] = acc;
    }
}

static void matvec_fipr(void) {
    for(int i = 0; i < N; i++) {
        const float *r = Wm + i * N;
        float acc = 0.0f;
        for(int j = 0; j < N; j += 4)
            acc += fipr(r[j], r[j + 1], r[j + 2], r[j + 3],
                        xv[j], xv[j + 1], xv[j + 2], xv[j + 3]);
        yv[i] = acc;
    }
}

/* Returns mega-multiply-accumulates per second. */
static double bench(void (*fn)(void)) {
    const int iters = 8;
    fn(); /* warm up */
    uint64_t t0 = timer_us_gettime64();
    for(int k = 0; k < iters; k++) fn();
    uint64_t dt = timer_us_gettime64() - t0;
    return (double)N * N * iters / (double)dt;  /* MACs per microsecond == M/s */
}

static void run_benchmarks(void) {
    char buf[128];
    for(int i = 0; i < N * N; i++) Wm[i] = (float)((i * 7) % 13) * 0.01f;
    for(int i = 0; i < N; i++) xv[i] = (float)(i % 5) * 0.1f;

    put("Benchmarking 512x512 fp32 mat-vec...\n");
    redraw();
    double naive = bench(matvec_naive);
    double fip = bench(matvec_fipr);

    snprintf(buf, sizeof(buf), "naive C: %6.1f MMAC/s\n", naive);  put(buf);
    snprintf(buf, sizeof(buf), "FIPR:    %6.1f MMAC/s\n", fip);    put(buf);

    /* One token needs roughly one MAC per parameter. */
    static const int sizes_m[] = { 1, 3, 5, 10 };
    for(unsigned k = 0; k < sizeof(sizes_m) / sizeof(sizes_m[0]); k++) {
        snprintf(buf, sizeof(buf), "  ~%2dM params -> %5.1f tok/s\n",
                 sizes_m[k], fip / sizes_m[k]);
        put(buf);
    }
    snprintf(buf, sizeof(buf), "(checksum %d)\n", (int)yv[N / 2]);
    put(buf);
    redraw();
}

/* ---------- main ---------- */
int main(int argc, char **argv) {
    char line[COLS * 4];
    int len = 0;

    vid_set_mode(DM_640x480, PM_RGB565);
    put("DREAMCAST THINKING  milestone 0\n\n");
    redraw();
    run_benchmarks();
    put("\nPlug in a keyboard and type. Enter sends.\n> ");
    redraw();

    maple_device_t *kbd = NULL;
    for(;;) {
        if(!kbd || !kbd->valid) {
            kbd = maple_enum_type(0, MAPLE_FUNC_KEYBOARD);
            if(!kbd) { thd_sleep(100); continue; }
        }

        int k, dirty = 0;
        while((k = kbd_queue_pop(kbd, true)) != KBD_QUEUE_END) {
            int raw = (k >> 8) & 0xff;
            if(k == '\n' || k == '\r' || raw == KBD_KEY_ENTER) {
                line[len] = 0;
                put("\n");
                if(len) {
                    put("dc: I heard \"");
                    put(line);
                    put("\". (No brain yet.)\n");
                }
                put("> ");
                len = 0;
            }
            else if(k == '\b' || raw == KBD_KEY_BACKSPACE) {
                int l = strlen(screen[cur_row]);
                if(len > 0 && l > 2) { len--; screen[cur_row][l - 1] = 0; }
            }
            else if(k >= 32 && k < 127 && len < (int)sizeof(line) - 1) {
                char c[2] = { (char)k, 0 };
                line[len++] = (char)k;
                put(c);
            }
            dirty = 1;
        }
        if(dirty) redraw();
        thd_sleep(10);
    }
    return 0;
}
