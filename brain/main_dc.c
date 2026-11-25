/* DREAMCAST THINKING: chat with the Dreamcast on its keyboard. */
#include <kos.h>
#include <dc/maple/vmu.h>
#include <dc/sq.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "dcbrain.h"

KOS_INIT_FLAGS(INIT_DEFAULT);

#define W 640
#define H 480
#define RGB(r, g, b) ((uint16_t)((((r) >> 3) << 11) | (((g) >> 2) << 5) | ((b) >> 3)))

static const uint16_t C_ORANGE = RGB(255, 122, 20), C_WHITE = RGB(236, 240, 255),
    C_USER = RGB(255, 205, 130), C_NOTE = RGB(120, 140, 185), C_HEAD = RGB(16, 18, 34),
    C_BOX = RGB(24, 30, 58), C_DIM = RGB(90, 100, 140);

static uint16_t fb[W * H] __attribute__((aligned(32)));

/* ---------- drawing ---------- */
static void fill(int x, int y, int w, int h, uint16_t c) {
    for(int j = y; j < y + h; j++)
        for(int i = x; i < x + w; i++) fb[j * W + i] = c;
}

static void text(int x, int y, uint16_t c, const char *s) {
    bfont_draw_str_ex(fb + y * W + x, W, c, 0, 16, false, s);
}

/* the background is a vertical gradient: one color per row, computed once */
static uint16_t row_color[H];

static void clear_rows(int y0, int y1) {
    if(!row_color[H - 1]) {
        for(int y = 0; y < H; y++) {
            int t = y * 255 / H;
            row_color[y] = RGB(8 + t * 12 / 255, 10 + t * 16 / 255, 24 + t * 30 / 255);
        }
    }
    for(int y = y0; y < y1; y++) {
        uint32_t two = row_color[y] | (uint32_t)row_color[y] << 16;   /* two pixels per write */
        uint32_t *p = (uint32_t *)(fb + y * W);
        for(int x = 0; x < W / 2; x++) p[x] = two;
    }
}

static void background(void) { clear_rows(0, H); }

/* the thinking swirl: a 3-turn spiral, rotated by `rot` */
static void swirl(int cx, int cy, int R, float rot, uint16_t c) {
    for(float t = 0.4f; t < 6.0f * F_PI; t += 0.04f) {
        float r = R * t / (6.0f * F_PI);
        int x = cx + (int)(r * fcos(t + rot)), y = cy + (int)(r * fsin(t + rot));
        int d = 1 + (int)(r * 3 / R);
        for(int j = -d; j <= d; j++)
            for(int i = -d; i <= d; i++)
                if(i * i + j * j <= d * d && x + i >= 0 && x + i < W && y + j >= 0 && y + j < H)
                    fb[(y + j) * W + x + i] = c;
    }
}

/* copy rows [y0, y1) of our buffer to the screen using the SH-4 store queues */
static void copy_rows(int y0, int y1) {
    sq_cpy(vram_s + y0 * W, fb + y0 * W, (y1 - y0) * W * sizeof(uint16_t));
}

static void present(void) {
    vid_waitvbl();
    copy_rows(0, H);
}

/* ---------- VMU screen: the swirl, spinning while DC thinks ---------- */
static void vmu_swirl(float rot) {
    static char bits[48 * 32 / 8];
    memset(bits, 0, sizeof(bits));
    for(float t = 0.4f; t < 6.0f * F_PI; t += 0.05f) {
        float r = 15.0f * t / (6.0f * F_PI);
        int x = 24 + (int)(r * fcos(t + rot)), y = 16 + (int)(r * fsin(t + rot));
        if(x >= 0 && x < 48 && y >= 0 && y < 32) bits[y * 6 + x / 8] |= 1 << (x % 8);
    }
    maple_device_t *dev;
    for(int i = 0; (dev = maple_enum_type(i, MAPLE_FUNC_LCD)); i++) vmu_draw_lcd_xbm(dev, bits);
}

/* ---------- chat log (word-wrapped) ---------- */
#define LOG_ROWS 64
#define SHOW_ROWS 13
#define TEXT_X 72
#define COLS ((W - TEXT_X - 20) / BFONT_THIN_WIDTH)   /* 45 */
enum { K_DC, K_USER, K_NOTE };
static char rows[LOG_ROWS][COLS + 1];
static uint8_t rkind[LOG_ROWS], rtag[LOG_ROWS];
static int nrows = 0;

static void start_line(int kind, int tag) {
    if(nrows == LOG_ROWS) {
        memmove(rows, rows[1], sizeof(rows[0]) * (LOG_ROWS - 1));
        memmove(rkind, rkind + 1, LOG_ROWS - 1);
        memmove(rtag, rtag + 1, LOG_ROWS - 1);
        nrows--;
    }
    rows[nrows][0] = 0;
    rkind[nrows] = kind;
    rtag[nrows] = tag;
    nrows++;
}

static void append(const char *s) {
    for(; *s; s++) {
        char ch = *s;
        if(ch == '\n') { start_line(rkind[nrows - 1], 0); continue; }
        if(ch < 32 || ch > 126) continue;
        char *line = rows[nrows - 1];
        int len = strlen(line);
        if(len >= COLS) {
            if(ch == ' ') { start_line(rkind[nrows - 1], 0); continue; }
            char carry[COLS + 1] = "";
            char *sp = strrchr(line, ' ');
            if(sp && sp != line) { strcpy(carry, sp + 1); *sp = 0; }
            start_line(rkind[nrows - 1], 0);
            strcpy(rows[nrows - 1], carry);
            len = strlen(carry);
        }
        if(len == 0 && ch == ' ') continue;
        line = rows[nrows - 1];
        line[len] = ch;
        line[len + 1] = 0;
    }
}

/* ---------- screen ---------- */
static char input[160];
static int input_len = 0, thinking = 0, blink = 0;
static float spin = 0;
static char status[48] = "";

/* ---------- on-screen keyboard (controller only, no keyboard plugged in) ---------- */
static const char *osk_rows[4]  = { "1234567890", "qwertyuiop", "asdfghjkl'", "zxcvbnm,.?" };
static const char *osk_shift[4] = { "!@#$%^&*()", "QWERTYUIOP", "ASDFGHJKL\"", "ZXCVBNM-!:" };
static const char *osk_special[4] = { "SHIFT", "SPACE", "DEL", "SEND" };
static int osk_on = 0, osk_row = 1, osk_col = 0, osk_caps = 0;
#define OSK_X 60
#define OSK_Y 246
#define OSK_KW 52
#define OSK_KH 30

static void draw_osk(void) {
    for(int r = 0; r < 5; r++) {
        int y = OSK_Y + r * (OSK_KH + 3);
        int nkeys = r < 4 ? 10 : 4, kw = r < 4 ? OSK_KW : OSK_KW * 10 / 4;
        for(int c = 0; c < nkeys; c++) {
            int x = OSK_X + c * kw, sel = r == osk_row && c == osk_col;
            int lit = r == 4 && c == 0 && osk_caps;
            fill(x, y, kw - 4, OSK_KH, sel ? C_ORANGE : lit ? C_DIM : C_BOX);
            char label[8];
            if(r < 4) { label[0] = (osk_caps ? osk_shift : osk_rows)[r][c]; label[1] = 0; }
            else strcpy(label, osk_special[c]);
            int tw = strlen(label) * BFONT_THIN_WIDTH;
            text(x + (kw - 4 - tw) / 2, y + 3, sel ? C_HEAD : C_WHITE, label);
        }
    }
}

/* Redraws the chat screen. While DC is thinking, only three bands change from token to token
   (the header swirl, the last chat lines, and the input box), so we clear and copy just those
   and skip waiting for vblank. Anything else (idle, scrolling, keyboard shown) redraws it all. */
#define CHAT_Y 76
#define ROW_H (BFONT_HEIGHT + 2)
static void draw(void) {
    static int last_first = -1, last_show = -1;
    int show = osk_on && !thinking ? 6 : SHOW_ROWS;
    int first = nrows > show ? nrows - show : 0;
    int full = !thinking || first != last_first || show != last_show;
    last_first = first;
    last_show = show;
    int band_top = CHAT_Y + ((nrows - 2 > first ? nrows - 2 : first) - first) * ROW_H;
    int band_bottom = CHAT_Y + (nrows - first) * ROW_H;
    if(full) background();
    else {
        clear_rows(0, 62);
        clear_rows(band_top, band_bottom);
        clear_rows(412, 466);
    }
    /* header */
    fill(0, 0, W, 60, C_HEAD);
    fill(0, 60, W, 2, C_ORANGE);
    swirl(36, 30, 22, spin, C_ORANGE);
    text(72, 18, C_ORANGE, "DREAMCAST");
    text(72 + 10 * BFONT_THIN_WIDTH, 18, C_WHITE, "THINKING");
    text(W - 16 - strlen(status) * BFONT_THIN_WIDTH, 18, C_DIM, status);
    /* chat */
    for(int r = first, y = CHAT_Y; r < nrows; r++, y += ROW_H) {
        uint16_t c = rkind[r] == K_USER ? C_USER : rkind[r] == K_NOTE ? C_NOTE : C_WHITE;
        if(rtag[r] == 1) text(20, y, C_USER, "YOU");
        if(rtag[r] == 2) { swirl(34, y + 12, 9, 0, C_ORANGE); text(48, y, C_ORANGE, ""); }
        text(TEXT_X, y, c, rows[r]);
    }
    if(osk_on && !thinking) draw_osk();
    /* input box */
    fill(12, 414, W - 24, 50, C_ORANGE);
    fill(14, 416, W - 28, 46, C_BOX);
    if(thinking) {
        static const char *dots[] = { "", ".", "..", "..." };
        char t[40];
        snprintf(t, sizeof(t), "It's thinking%s", dots[(thinking / 2) % 4]);
        text(28, 427, C_DIM, t);
    } else {
        char shown[COLS + 4];
        const char *s = input_len > 46 ? input + input_len - 46 : input;
        snprintf(shown, sizeof(shown), "> %s%s", s, (blink / 25) % 2 ? "" : "_");
        text(28, 427, C_USER, shown);
    }
    if(full) present();
    else {
        copy_rows(0, 62);
        copy_rows(band_top, band_bottom);
        copy_rows(412, 466);
    }
}

/* ---------- fact book (on the CD; titles indexed in RAM) ---------- */
typedef struct { char *title; uint32_t off; } Fact;
static Fact *facts;
static int nfacts;
static FILE *fbf;

static void factbook_load(const char *path) {
    fbf = fopen(path, "r");
    if(!fbf) return;
    /* read the whole file once in big chunks and index line starts */
    fseek(fbf, 0, SEEK_END);
    long size = ftell(fbf);
    fseek(fbf, 0, SEEK_SET);
    char *buf = malloc(64 * 1024);
    facts = malloc(16384 * sizeof(Fact));
    char title[128];
    int tl = 0, in_title = 1;
    uint32_t line_start = 0;
    for(long base = 0; base < size;) {
        int n = fread(buf, 1, 64 * 1024, fbf);
        if(n <= 0) break;
        for(int i = 0; i < n; i++) {
            char c = buf[i];
            if(c == '\n') { in_title = 1; tl = 0; line_start = base + i + 1; continue; }
            if(!in_title) continue;
            if(c == '\t') {
                title[tl] = 0;
                in_title = 0;
                if(nfacts < 16384) { facts[nfacts].title = strdup(title); facts[nfacts].off = line_start; nfacts++; }
            } else if(tl < 127) title[tl++] = tolower((unsigned char)c);
        }
        base += n;
    }
    free(buf);
}

/* exact (case-insensitive) title, else first title that starts with the query */
static const char *factbook_find(const char *q, char *out, int max) {
    char key[128];
    int n = 0;
    for(; q[n] && n < 127; n++) key[n] = tolower((unsigned char)q[n]);
    key[n] = 0;
    while(n && key[n - 1] == ' ') key[--n] = 0;
    int lo = 0, hi = nfacts - 1, hit = -1;
    while(lo <= hi) {
        int mid = (lo + hi) / 2, c = strcmp(key, facts[mid].title);
        if(!c) { hit = mid; break; }
        if(c < 0) hi = mid - 1; else lo = mid + 1;
    }
    if(hit < 0 && lo < nfacts && !strncmp(facts[lo].title, key, strlen(key))) hit = lo;
    if(hit < 0 || !fbf) return NULL;
    fseek(fbf, facts[hit].off, SEEK_SET);
    if(!fgets(out, max, fbf)) return NULL;
    char *tab = strchr(out, '\t');
    if(!tab) return NULL;
    char *nl = strchr(tab, '\n');
    if(nl) *nl = 0;
    return tab + 1;
}

/* ---------- chat ---------- */
static Brain brain;
static Tokenizer tok;
static int pos = 0;
static uint32_t rng;
static maple_device_t *kbd;

static void tick(void) {          /* animate one step while thinking */
    thinking++;
    spin += 0.35f;
    if(kbd) while(kbd_queue_pop(kbd, true) != KBD_QUEUE_END) {}
    if(thinking % 3 == 0) vmu_swirl(spin);
}

static float *feed(const int *t, int n, int animate) {
    float *logits = NULL;
    for(int i = 0; i < n; i++) {
        logits = brain_forward(&brain, t[i], pos++);
        if(animate) { tick(); draw(); }
    }
    return logits;
}

static void reply(const char *user) {
    int t[300], n = 0, enc[256];
    int ne = tok_encode(&tok, user, enc, 200);
    if(pos + ne + 160 > brain.seq_len) {           /* context full: start a fresh conversation */
        pos = 0;
        start_line(K_NOTE, 0);
        append("(my VMU is full, so I'm starting fresh)");
    }
    if(pos == 0) t[n++] = TOK_BOS;
    t[n++] = TOK_U;
    memcpy(t + n, enc, ne * sizeof(int)); n += ne;
    t[n++] = TOK_DC;

    thinking = 1;
    uint64_t t0 = timer_us_gettime64();
    float *logits = feed(t, n, 1);
    start_line(K_DC, 2);
    int generated = 0, looking = 0, lookups = 0, tn = 0;
    char topic[128], piece[64];
    while(generated < 120 && pos < brain.seq_len - 1) {
        int id = sample(logits, brain.vocab_size, 0.7f, 0.9f, &rng);
        generated++;
        if(id == TOK_E || id == TOK_U) break;
        if(id == TOK_LOOK) { looking = 1; tn = 0; logits = feed(&id, 1, 1); continue; }
        if(looking && id == TOK_ENDLOOK) {
            looking = 0;
            topic[tn] = 0;
            logits = feed(&id, 1, 1);
            static char line[1024];
            const char *entry = lookups++ < 2 ? factbook_find(topic, line, sizeof(line)) : NULL;
            /* show the lookup above DC's reply line */
            int dc_row = nrows - 1;
            char saved[COLS + 1];
            strcpy(saved, rows[dc_row]);
            rkind[dc_row] = K_NOTE; rtag[dc_row] = 0;
            snprintf(rows[dc_row], COLS + 1, "  ...looking up \"%s\" in my fact book", topic);
            start_line(K_DC, 2);
            append(saved);
            draw();
            int bt[200], nb = 0;
            bt[nb++] = TOK_BOOK;
            nb += tok_encode(&tok, entry ? entry : "(no entry)", bt + nb, 180);
            bt[nb++] = TOK_DC;
            if(pos + nb + 60 > brain.seq_len) break;
            logits = feed(bt, nb, 1);
            continue;
        }
        int l = tok_piece(&tok, id, piece);
        if(looking) {
            for(int i = 0; i < l && tn < 127; i++) topic[tn++] = piece[i];
        } else {
            piece[l] = 0;
            append(piece);
        }
        tick();
        draw();
        logits = feed(&id, 1, 0);
    }
    double secs = (timer_us_gettime64() - t0) / 1e6;
    snprintf(status, sizeof(status), "%.1f tok/s", (generated + n) / secs);
    thinking = 0;
    vmu_swirl(0);
}

/* ---------- input shared by the keyboard and the on-screen keyboard ---------- */
static void show_credits(void);
static maple_device_t *ctrl;

static void type_char(char c) {
    if(input_len < (int)sizeof(input) - 1) { input[input_len++] = c; input[input_len] = 0; }
}
static void backspace(void) { if(input_len > 0) input[--input_len] = 0; }
static void submit(void) {
    if(!input_len) return;
    if(!strcmp(input, "/credits")) { input_len = 0; input[0] = 0; show_credits(); draw(); return; }
    start_line(K_USER, 1);
    append(input);
    char msg[sizeof(input)];
    strcpy(msg, input);
    input_len = 0; input[0] = 0;
    draw();
    reply(msg);
}

/* twice a second: notice keyboards and controllers being plugged in or pulled out */
static void scan_devices(void) {
    maple_device_t *k = maple_enum_type(0, MAPLE_FUNC_KEYBOARD);
    maple_device_t *c = maple_enum_type(0, MAPLE_FUNC_CONTROLLER);
    int had_kbd = kbd != NULL, had_ctrl = ctrl != NULL;
    kbd = k; ctrl = c;
    int was = osk_on;
    osk_on = !kbd && ctrl;
    if((kbd != NULL) != had_kbd) {
        start_line(K_NOTE, 0);
        append(kbd ? "(keyboard connected)" : "(keyboard unplugged)");
    }
    if(osk_on && !was) {
        start_line(K_NOTE, 0);
        append("(controller: A types, B deletes, X space, Y shift, Start sends)");
    } else if(!kbd && !ctrl && (had_kbd || had_ctrl)) {
        start_line(K_NOTE, 0);
        append("(plug in a keyboard or controller to talk to me)");
    }
}

static void osk_press(void) {
    if(osk_row < 4) {
        type_char((osk_caps ? osk_shift : osk_rows)[osk_row][osk_col]);
        osk_caps = 0;                                   /* one-shot shift, like a phone */
    } else if(osk_col == 0) osk_caps = !osk_caps;
    else if(osk_col == 1) type_char(' ');
    else if(osk_col == 2) backspace();
    else submit();
}

static void osk_move(int dx, int dy) {
    if(dy) {
        int nr = osk_row + dy;
        if(nr < 0 || nr > 4) return;
        if(nr == 4) osk_col = osk_col * 4 / 10;          /* into the wide bottom row */
        else if(osk_row == 4) osk_col = osk_col * 10 / 4 + 1;
        osk_row = nr;
    }
    if(dx) {
        int n = osk_row < 4 ? 10 : 4;
        osk_col = (osk_col + dx + n) % n;
    }
}

/* returns 1 if anything changed; D-pad/stick repeat after 300 ms, then every 80 ms */
static int controller_input(void) {
    static uint32_t prev;
    static int held;
    cont_state_t *st = ctrl ? (cont_state_t *)maple_dev_status(ctrl) : NULL;
    if(!st) return 0;
    uint32_t b = st->buttons;
    if(st->joyx < -64) b |= CONT_DPAD_LEFT;
    if(st->joyx > 64) b |= CONT_DPAD_RIGHT;
    if(st->joyy < -64) b |= CONT_DPAD_UP;
    if(st->joyy > 64) b |= CONT_DPAD_DOWN;
    uint32_t pressed = b & ~prev, dirs = CONT_DPAD_UP | CONT_DPAD_DOWN | CONT_DPAD_LEFT | CONT_DPAD_RIGHT;
    int changed = 0;
    uint32_t move = pressed & dirs;
    if(b & dirs) {
        held++;
        if(!move && held > 15 && held % 4 == 0) move = b & dirs;
    } else held = 0;
    if(move & CONT_DPAD_UP) { osk_move(0, -1); changed = 1; }
    if(move & CONT_DPAD_DOWN) { osk_move(0, 1); changed = 1; }
    if(move & CONT_DPAD_LEFT) { osk_move(-1, 0); changed = 1; }
    if(move & CONT_DPAD_RIGHT) { osk_move(1, 0); changed = 1; }
    if(pressed & CONT_A) { osk_press(); changed = 1; }
    if(pressed & CONT_B) { backspace(); changed = 1; }
    if(pressed & CONT_X) { type_char(' '); changed = 1; }
    if(pressed & CONT_Y) { osk_caps = !osk_caps; changed = 1; }
    prev = b;
    if(pressed & CONT_START) { submit(); prev = 0; changed = 1; }
    return changed;
}

/* ---------- boot splash ---------- */
static void centered(int y, uint16_t c, const char *s) {
    text(W / 2 - (int)strlen(s) * BFONT_THIN_WIDTH / 2, y, c, s);
}

/* progress < 0: loading done, show the prompt (blink toggles "press any key") */
static float splash_rot;
/* ---------- credits (CC BY-SA attribution for the fact book) ---------- */
static const char *credits[] = {
    "#CREDITS",
    "",
    "DC's fact book uses text from Wikipedia and",
    "StrategyWiki contributors, under the Creative",
    "Commons Attribution-ShareAlike 4.0 license:",
    "@creativecommons.org/licenses/by-sa/4.0",
    "",
    "Conversations written with Claude (Anthropic).",
    "Model design follows llama2.c (Andrej Karpathy).",
    "Dreamcast llama2.c port by gameblabla (MIT).",
    "Built with KallistiOS and mkdcdisc.",
    "",
    "Sega and Dreamcast are trademarks of Sega.",
    "An unofficial, non-commercial fan project.",
};

static int any_input(void) {          /* edge-triggered: a new key or button press */
    static uint32_t prev = ~0u;
    int hit = 0;
    maple_device_t *k = maple_enum_type(0, MAPLE_FUNC_KEYBOARD);
    if(k) while(kbd_queue_pop(k, true) != KBD_QUEUE_END) hit = 1;
    maple_device_t *c = maple_enum_type(0, MAPLE_FUNC_CONTROLLER);
    cont_state_t *st = c ? (cont_state_t *)maple_dev_status(c) : NULL;
    uint32_t b = st ? st->buttons : 0;
    if(b & ~prev & (CONT_A | CONT_B | CONT_X | CONT_Y | CONT_START)) hit = 1;
    prev = b;
    return hit;
}

static void show_credits(void) {
    any_input();                       /* swallow the press that opened us */
    for(int t = 0;; t++) {
        background();
        swirl(W / 2, 46, 26, (timer_ms_gettime64() % 2500) * (2.0f * F_PI / 2500.0f), C_ORANGE);
        int y = 92;
        for(unsigned i = 0; i < sizeof(credits) / sizeof(credits[0]); i++, y += 24) {
            const char *l = credits[i];
            uint16_t c = C_WHITE;
            if(*l == '#') { c = C_ORANGE; l++; }
            else if(*l == '@') { c = C_USER; l++; }
            text(W / 2 - (int)strlen(l) * BFONT_THIN_WIDTH / 2, y, c, l);
        }
        if((t / 30) % 2 == 0) {
            const char *m = "press any key";
            text(W / 2 - (int)strlen(m) * BFONT_THIN_WIDTH / 2, 444, C_DIM, m);
        }
        present();
        if(t > 10 && any_input()) return;
    }
}

static void splash(float progress, const char *label, int blink_on) {
    background();
    swirl(W / 2, 140, 80, splash_rot, C_ORANGE);
    int tx = W / 2 - 18 * BFONT_THIN_WIDTH / 2;
    text(tx, 248, C_ORANGE, "DREAMCAST");
    text(tx + 10 * BFONT_THIN_WIDTH, 248, C_WHITE, "THINKING");
    centered(286, C_NOTE, "a 7M-parameter language model");
    centered(312, C_NOTE, "running on the SH-4 at 200 MHz");
    if(progress >= 0) {
        fill(120, 360, 400, 16, C_ORANGE);
        fill(122, 362, 396, 12, C_BOX);
        fill(122, 362, (int)(396 * progress), 12, C_ORANGE);
        centered(384, C_DIM, label);
    } else {
        centered(364, C_ORANGE, "It's thinking.");
        if(blink_on) centered(398, C_DIM, label);
    }
    centered(446, C_DIM, "SH-4 200 MHz \xb7 16 MB RAM \xb7 int8 \xb7 runs offline");
    present();
}

/* Loading runs on its own thread; the main thread animates at a steady pace. */
static volatile float load_p = 0;
static volatile int load_state = 0;          /* 0 loading, 1 done, -1 failed */
static char load_label[48] = "reading the disc";

static void load_progress(long done, long total) {
    snprintf(load_label, sizeof(load_label), "loading brain %.1f / %.1f MB", done / 1e6, total / 1e6);
    load_p = 0.9f * done / total;
}

static void *loader(void *arg) {
    (void)arg;
    brain_progress = load_progress;
    if(brain_load(&brain, "/cd/brain.bin") || tok_load(&tok, "/cd/tok.bin", brain.vocab_size)) {
        load_state = -1;
        return NULL;
    }
    strcpy(load_label, "opening the fact book");
    load_p = 0.95f;
    factbook_load("/cd/factbook.txt");
    load_p = 1.0f;
    load_state = 1;
    return NULL;
}

static float clock_rot(void) {               /* one turn every 2.5 seconds */
    return (timer_ms_gettime64() % 2500) * (2.0f * F_PI / 2500.0f);
}

static void boot_screen(void) {
    thd_create(true, loader, NULL);
    float shown = 0;
    for(int frame = 0; load_state == 0 || shown < 0.999f; frame++) {
        if(load_state == -1) {
            splash_rot = clock_rot();
            splash(-1, "Could not load my brain from the disc.", 1);
            for(;;) thd_sleep(1000);
        }
        shown += (load_p - shown) * 0.15f;     /* ease the bar toward real progress */
        if(load_state == 1 && load_p - shown < 0.01f) shown = 1.0f;
        splash_rot = clock_rot();
        splash(shown, load_label, 1);
        if(frame % 8 == 0) vmu_swirl(splash_rot);
    }
}

static void wait_for_key(void) {
    uint32_t prev = ~0u;
    for(int t = 0;; t++) {
        splash_rot = clock_rot();
        splash(-1, "press any key  \xb7  C for credits", (t / 30) % 2 == 0);
        if(t % 8 == 0) vmu_swirl(splash_rot);
        maple_device_t *k = maple_enum_type(0, MAPLE_FUNC_KEYBOARD);
        int key = k ? kbd_queue_pop(k, true) : KBD_QUEUE_END;
        if(key != KBD_QUEUE_END) {
            while(kbd_queue_pop(k, true) != KBD_QUEUE_END) {}
            if(key == 'c' || key == 'C') { show_credits(); prev = ~0u; continue; }
            return;
        }
        maple_device_t *c = maple_enum_type(0, MAPLE_FUNC_CONTROLLER);
        cont_state_t *st = c ? (cont_state_t *)maple_dev_status(c) : NULL;
        uint32_t b = st ? st->buttons : 0, pressed = b & ~prev;
        prev = b;
        if(pressed & CONT_Y) { show_credits(); prev = ~0u; continue; }
        if(pressed & (CONT_START | CONT_A)) return;
    }
}

int main(int argc, char **argv) {
    vid_set_mode(DM_640x480, PM_RGB565);
    boot_screen();
    wait_for_key();
    vmu_swirl(0);
    rng = (uint32_t)timer_us_gettime64() | 1;
    snprintf(status, sizeof(status), "%d facts", nfacts);
    start_line(K_NOTE, 0);
    append("Hi! I'm your Dreamcast. Type to me and press Enter.");

    for(int tick_n = 0;; tick_n++) {
        if(tick_n % 25 == 0) scan_devices();
        int k, dirty = 0;
        while(kbd && (k = kbd_queue_pop(kbd, true)) != KBD_QUEUE_END) {
            int raw = (k >> 8) & 0xff;
            if(k == '\n' || k == '\r' || raw == KBD_KEY_ENTER) submit();
            else if(k == '\b' || raw == KBD_KEY_BACKSPACE) backspace();
            else if(k >= 32 && k < 127) type_char((char)k);
            dirty = 1;
        }
        if(osk_on && controller_input()) dirty = 1;
        blink++;
        if(dirty || blink % 25 == 0) draw();
        thd_sleep(20);
    }
    return 0;
}
