/*
 * c2f.c
 *
 * Converts Chrome Trace Event Format (JSON) on stdin to
 * Fuchsia Trace Format (FXT) binary on stdout.
 *
 * Build:
 *   cc -O2 -o c2f c2f.c -lyajl -lm
 *
 * Usage:
 *   ./c2f < trace.json > trace.fxt
 *
 * Fuchsia Trace Format reference:
 *   https://fuchsia.dev/fuchsia-src/reference/tracing/trace-format
 *
 * Chrome Trace Event Format reference:
 *   https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
 */

#define _POSIX_C_SOURCE 200809L   /* for strdup */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <yajl/yajl_parse.h>

/* -----------------------------------------------------------------------
 * Compile-time constants
 * --------------------------------------------------------------------- */

#define MAX_STRING_TABLE 32767   /* 0x0001 .. 0x7fff */
#define MAX_THREAD_TABLE 255     /* 0x01   .. 0xff   */

/* -----------------------------------------------------------------------
 * Output buffer (we write 8-byte aligned words; flush at the end)
 * --------------------------------------------------------------------- */

#define OUTBUF_INITIAL (4 * 1024 * 1024)  /* 4 MB initial */

static uint8_t *g_outbuf  = NULL;
static size_t   g_outsize = 0;
static size_t   g_outcap  = 0;

/* Streaming output: flush everything written so far to stdout,
 * then reset g_outsize to 0 so the buffer is reused as a ring.
 * Called periodically and once at the end.  */
#define FLUSH_THRESHOLD (1u << 20)  /* flush every 1 MB */

static void out_flush(void) {
    if (g_outsize == 0) return;
    size_t written = fwrite(g_outbuf, 1, g_outsize, stdout);
    if (written != g_outsize) { fprintf(stderr, "Write error\n"); exit(1); }
    g_outsize = 0;
}

/* Grow the buffer so that at least `extra` more bytes can be appended.
 * Called only on the slow path (capacity exceeded). */
static void out_grow(size_t extra) {
    /* First try flushing: if the buffer has data we can flush it and reuse */
    if (g_outsize >= FLUSH_THRESHOLD) {
        out_flush();
        if (g_outsize + extra <= g_outcap) return;
    }
    size_t newcap = g_outcap ? g_outcap * 2 : OUTBUF_INITIAL;
    while (newcap < g_outsize + extra) newcap *= 2;
    g_outbuf = (uint8_t *)realloc(g_outbuf, newcap);
    if (!g_outbuf) { fprintf(stderr, "OOM\n"); exit(1); }
    g_outcap = newcap;
}

/* Reserve at least `extra` bytes – inline fast path. */
static inline void out_reserve(size_t extra) {
    if (__builtin_expect(g_outsize + extra > g_outcap, 0))
        out_grow(extra);
}

/* Pre-reserve a large chunk before emitting many words in one record.
 * max_record_bytes is a conservative upper bound for one event record.
 * With args<=15, inline strings, and a generous arg payload the maximum
 * a single event can produce is well under 8 KB. */
#define MAX_RECORD_BYTES 8192

/* Write a 64-bit word (little-endian) – no individual bounds check needed
 * after an out_reserve(MAX_RECORD_BYTES) call at the start of each record. */
static inline void out_word_unsafe(uint64_t w) {
    memcpy(g_outbuf + g_outsize, &w, 8);
    g_outsize += 8;
}

static inline void out_word(uint64_t w) {
    out_reserve(8);
    out_word_unsafe(w);
}

/* Return the current write position (word index).  Used to patch a header
 * word in-place after we know the final record size. */
static inline size_t out_tell(void) { return g_outsize; }

/* Overwrite a previously written word at byte offset `pos`. */
static inline void out_patch(size_t pos, uint64_t w) {
    memcpy(g_outbuf + pos, &w, 8);
}

/*
 * Write a UTF-8 string as a "stream" atom: bytes followed by zero-padding
 * to the next 8-byte boundary.  Returns the number of 8-byte words written.
 */
static inline size_t out_stream(const char *s, size_t len) {
    size_t padded = (len + 7) & ~(size_t)7;
    /* out_reserve already called by the record-level pre-reservation */
    memcpy(g_outbuf + g_outsize, s, len);
    memset(g_outbuf + g_outsize + len, 0, padded - len);
    g_outsize += padded;
    return padded / 8;
}

/* -----------------------------------------------------------------------
 * String table  –  open-addressed hash map for O(1) average intern
 * --------------------------------------------------------------------- */

/* Hash table size must be a power of two and larger than MAX_STRING_TABLE. */
#define STRHASH_SIZE  65536u   /* 2^16, load factor ≤ 0.5 for 32767 entries */
#define STRHASH_MASK  (STRHASH_SIZE - 1u)

typedef struct {
    char    *str;      /* NULL = empty slot */
    size_t   len;      /* cached strlen    */
    uint16_t idx;      /* FXT string index (1-based) */
} StringEntry;

static StringEntry g_strtab[STRHASH_SIZE];  /* hash table, zero-initialised */
static int         g_strtab_count = 0;

/* Statistics */
static long g_strref_count = 0;

/* FNV-1a 32-bit – fast, low collision rate for short strings */
static inline uint32_t fnv1a(const char *s, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++)
        h = (h ^ (uint8_t)s[i]) * 16777619u;
    return h;
}

/* Emit a String Record (record type 2) for a newly registered string. */
static void emit_string_record(uint16_t idx, const char *s, size_t len) {
    size_t str_words  = ((len + 7) >> 3);
    uint64_t size_words = 1 + str_words;
    uint64_t header =
          (uint64_t)2
        | (size_words    << 4)
        | ((uint64_t)idx << 16)
        | ((uint64_t)len << 32);
    out_word(header);
    out_stream(s, len);
}

/*
 * intern_string_n: look up or register `s` (with known length `len`).
 * Returns an FXT 16-bit string ref (indexed or inline).
 * Avoids a second strlen() call since callers often already know len.
 */
static uint16_t intern_string_n(const char *s, size_t len) {
    if (!len) return 0;

    uint32_t h    = fnv1a(s, len);
    uint32_t slot = h & STRHASH_MASK;

    /* Linear probing */
    for (;;) {
        StringEntry *e = &g_strtab[slot];
        if (!e->str) break;                          /* empty – not found */
        if (e->len == len && memcmp(e->str, s, len) == 0) {
            g_strref_count++;
            return e->idx;
        }
        slot = (slot + 1) & STRHASH_MASK;
    }

    /* Not found – register, resetting the table first if it is full */
    if (g_strtab_count >= MAX_STRING_TABLE) {
        /* Free all strdup'd strings and wipe the hash table. */
        for (uint32_t i = 0; i < STRHASH_SIZE; i++) {
            if (g_strtab[i].str) { free(g_strtab[i].str); }
        }
        memset(g_strtab, 0, sizeof(g_strtab));
        g_strtab_count = 0;
        /* Re-probe for the now-empty slot. */
        slot = fnv1a(s, len) & STRHASH_MASK;
        while (g_strtab[slot].str) slot = (slot + 1) & STRHASH_MASK;
    }

    uint16_t new_idx = (uint16_t)(g_strtab_count + 1);
    StringEntry *e   = &g_strtab[slot];
    e->str = strdup(s);
    e->len = len;
    e->idx = new_idx;
    g_strtab_count++;
    emit_string_record(new_idx, s, len);
    g_strref_count++;
    return new_idx;
}

static inline uint16_t intern_string(const char *s) {
    if (!s || !*s) return 0;
    return intern_string_n(s, strlen(s));
}

/*
 * StrRef: resolved string reference + metadata for deferred inline emission.
 */
typedef struct { uint16_t ref; int is_inline; const char *ptr; size_t len; } StrRef;

static inline StrRef make_strref(const char *s) {
    StrRef r;
    if (!s || !*s) {
        r.ref = 0; r.is_inline = 0; r.ptr = NULL; r.len = 0;
        return r;
    }
    r.ptr = s;
    r.len = strlen(s);
    r.ref = intern_string_n(s, r.len);  /* reuse already-computed len */
    r.is_inline = (r.ref & 0x8000u) ? 1 : 0;
    return r;
}

static inline void emit_inline_str(const StrRef *sr) {
    if (sr->is_inline) out_stream(sr->ptr, sr->len);
}

/* -----------------------------------------------------------------------
 * Kernel Object Records  (record type 7)
 *
 * Perfetto needs these to resolve koid (pid/tid) -> display name.
 * We emit one PROCESS record per unique pid and one THREAD record per
 * unique (pid, tid) pair, using names collected from Chrome "M" events.
 *
 * Spec layout:
 *   header word:
 *     [0..3]   record type (7)
 *     [4..15]  record size in 8-byte words
 *     [16..23] kernel object type  (1=process, 2=thread)
 *     [24..39] name (string ref, 16 bits)
 *     [40..43] number of arguments
 *     [44..63] reserved (must be zero)
 *   koid word: the pid or tid
 *   [optional inline name stream]
 *   [argument data]
 *
 * For threads we include one koid argument named "process" (type 8)
 * pointing to the owning process pid, as required by the convention
 * described in the Fuchsia trace-format spec.
 * --------------------------------------------------------------------- */

#define ZX_OBJ_TYPE_PROCESS 1u
#define ZX_OBJ_TYPE_THREAD  2u

/* Name lookup tables populated during the pre-pass over "M" events. */
#define MAX_PROC_NAMES  4096
#define MAX_THR_NAMES   8192

typedef struct { uint64_t pid; char *name; } ProcName;
typedef struct { uint64_t pid; uint64_t tid; char *name; } ThrName;

static ProcName g_proc_names[MAX_PROC_NAMES];
static int      g_proc_names_count = 0;
static ThrName  g_thr_names[MAX_THR_NAMES];
static int      g_thr_names_count  = 0;

/* Track which pids/tids have already had a KObj record emitted. */
static uint64_t g_kobj_pids[MAX_PROC_NAMES];
static int      g_kobj_pids_count = 0;

static void register_proc_name(uint64_t pid, const char *name) {
    for (int i = 0; i < g_proc_names_count; i++)
        if (g_proc_names[i].pid == pid) return; /* already have one */
    if (g_proc_names_count >= MAX_PROC_NAMES) return;
    g_proc_names[g_proc_names_count].pid  = pid;
    g_proc_names[g_proc_names_count].name = strdup(name);
    g_proc_names_count++;
}

static void register_thr_name(uint64_t pid, uint64_t tid, const char *name) {
    for (int i = 0; i < g_thr_names_count; i++)
        if (g_thr_names[i].pid == pid && g_thr_names[i].tid == tid) return;
    if (g_thr_names_count >= MAX_THR_NAMES) return;
    g_thr_names[g_thr_names_count].pid  = pid;
    g_thr_names[g_thr_names_count].tid  = tid;
    g_thr_names[g_thr_names_count].name = strdup(name);
    g_thr_names_count++;
}

static const char *lookup_proc_name(uint64_t pid) {
    for (int i = 0; i < g_proc_names_count; i++)
        if (g_proc_names[i].pid == pid) return g_proc_names[i].name;
    return NULL;
}

static const char *lookup_thr_name(uint64_t pid, uint64_t tid) {
    for (int i = 0; i < g_thr_names_count; i++)
        if (g_thr_names[i].pid == pid && g_thr_names[i].tid == tid)
            return g_thr_names[i].name;
    return NULL;
}

/*
 * emit_kobj_process: emit a Kernel Object Record for a process.
 * Called once per unique pid, before any events for that pid.
 */
static void emit_kobj_process(uint64_t pid, const char *name) {
    StrRef nref = make_strref(name);
    /* size = 1 (hdr) + 1 (koid) + inline_name_words */
    size_t name_words = nref.is_inline ? ((nref.len + 7) >> 3) : 0;
    uint64_t size_words = 2 + name_words;
    uint64_t header =
          (uint64_t)7                        /* record type */
        | (size_words              << 4)
        | ((uint64_t)ZX_OBJ_TYPE_PROCESS << 16)
        | ((uint64_t)nref.ref      << 24)
        | ((uint64_t)0             << 40);   /* num args = 0 */
    out_word(header);
    out_word(pid);                           /* koid = pid */
    emit_inline_str(&nref);
}

/*
 * emit_kobj_thread: emit a Kernel Object Record for a thread.
 * Includes a "process" koid argument (arg type 8) per the spec convention.
 *
 * Argument layout for the "process" koid arg:
 *   arg header:
 *     [0..3]   arg type = 8 (koid)
 *     [4..15]  arg size = 2 words
 *     [16..31] name string ref  (for the key string "process")
 *     [32..63] reserved = 0
 *   arg value word: the pid (koid of parent process)
 */
static void emit_kobj_thread(uint64_t pid, uint64_t tid, const char *name) {
    StrRef nref    = make_strref(name);
    StrRef procref = make_strref("process"); /* arg key "process" */

    size_t name_words = nref.is_inline ? ((nref.len + 7) >> 3) : 0;
    size_t key_words  = procref.is_inline ? ((procref.len + 7) >> 3) : 0;
    /* 1 arg: [arg_hdr] + [inline key] + [koid value word] */
    size_t arg_words  = 1 + key_words + 1;
    uint64_t size_words = 2 + name_words + arg_words;

    uint64_t header =
          (uint64_t)7
        | (size_words              << 4)
        | ((uint64_t)ZX_OBJ_TYPE_THREAD << 16)
        | ((uint64_t)nref.ref      << 24)
        | ((uint64_t)1             << 40);   /* num args = 1 */
    out_word(header);
    out_word(tid);                           /* koid = tid */
    emit_inline_str(&nref);

    /* "process" koid argument */
    uint64_t arg_size = 2 + key_words;      /* hdr + [inline key] + value */
    uint64_t arg_hdr  =
          (uint64_t)8                        /* arg type 8 = koid */
        | (arg_size                << 4)
        | ((uint64_t)procref.ref   << 16);
    out_word(arg_hdr);
    emit_inline_str(&procref);
    out_word(pid);                           /* value = parent pid */
}

/*
 * ensure_kobj_process: emit a process KObj record the first time we see a pid.
 * Falls back to "pid/<number>" if no name was registered via an "M" event.
 */
static void ensure_kobj_process(uint64_t pid) {
    for (int i = 0; i < g_kobj_pids_count; i++)
        if (g_kobj_pids[i] == pid) return;
    if (g_kobj_pids_count < MAX_PROC_NAMES)
        g_kobj_pids[g_kobj_pids_count++] = pid;

    const char *name = lookup_proc_name(pid);
    char fallback[32];
    if (!name) {
        snprintf(fallback, sizeof(fallback), "%" PRIu64, pid);
        name = fallback;
    }
    emit_kobj_process(pid, name);
}

/* -----------------------------------------------------------------------
 * Thread table
 * --------------------------------------------------------------------- */

typedef struct {
    uint64_t pid;
    uint64_t tid;
    uint8_t  idx;
} ThreadEntry;

static ThreadEntry g_thrtab[MAX_THREAD_TABLE];
static int         g_thrtab_count = 0;

/*
 * intern_thread: register a (pid, tid) pair in the FXT thread table and
 * emit the necessary FXT records so Perfetto can display the timeline.
 *
 * On first encounter of a (pid, tid):
 *   1. Emit a Kernel Object Record for the process (once per pid).
 *   2. Emit a Kernel Object Record for the thread (with "process" arg).
 *   3. Emit the FXT Thread Record (assigns the 8-bit thread_ref index).
 *
 * Returns the 8-bit thread ref (non-zero = indexed, 0 = inline fallback).
 */
static uint8_t intern_thread(uint64_t pid, uint64_t tid) {
    for (int i = 0; i < g_thrtab_count; i++) {
        if (g_thrtab[i].pid == pid && g_thrtab[i].tid == tid)
            return g_thrtab[i].idx;
    }
    if (g_thrtab_count >= MAX_THREAD_TABLE) {
        /* Thread table full – wipe it and start over. */
        memset(g_thrtab, 0, sizeof(g_thrtab));
        g_thrtab_count = 0;
    }

    uint8_t new_idx = (uint8_t)(g_thrtab_count + 1);
    g_thrtab[g_thrtab_count].pid = pid;
    g_thrtab[g_thrtab_count].tid = tid;
    g_thrtab[g_thrtab_count].idx = new_idx;
    g_thrtab_count++;

    /* 1. Kernel Object Record: process (once per pid) */
    ensure_kobj_process(pid);

    /* 2. Kernel Object Record: thread */
    {
        const char *tname = lookup_thr_name(pid, tid);
        char fallback[32];
        if (!tname) {
            snprintf(fallback, sizeof(fallback), "%" PRIu64, tid);
            tname = fallback;
        }
        emit_kobj_thread(pid, tid, tname);
    }

    /* 3. FXT Thread Record (record type 3) */
    uint64_t header =
          (uint64_t)3
        | ((uint64_t)3       << 4)
        | ((uint64_t)new_idx << 16);
    out_word(header);
    out_word(pid);
    out_word(tid);

    return new_idx;
}

/* -----------------------------------------------------------------------
 * Magic number + Initialization record
 * --------------------------------------------------------------------- */

static void emit_magic(void) {
    /*
     * Magic number record (trace info type = 0):
     *   [0..3]   = record type 0
     *   [4..15]  = size 1
     *   [16..19] = metadata type 4
     *   [20..23] = trace info type 0
     *   [24..55] = 0x16547846
     *   [56..63] = 0
     *
     * The whole 8-byte value must equal 0x0016547846040010.
     */
    uint64_t magic_word = (uint64_t)0x0016547846040010ULL;
    out_word(magic_word);
}

static void emit_init(void) {
    /*
     * Initialization record (record type 1):
     *   header word [0..3]=1, [4..15]=2 (2 words total)
     *   tick multiplier = 1000000000 (1 tick = 1 ns)
     */
    uint64_t header = (uint64_t)1 | ((uint64_t)2 << 4);
    out_word(header);
    out_word(1000000000ULL); /* 1 billion ticks per second = 1 ns per tick */
}

/* -----------------------------------------------------------------------
 * Argument emission helpers
 * -----------------------------------------------------------------------
 * We emit arguments from Chrome's "args" object as FXT arguments.
 * Supported FXT arg types we use:
 *   0  = null
 *   4  = int64
 *   6  = double
 *   7  = string
 */

/*
 * Emit one argument.
 * Returns the number of 8-byte words written for this argument.
 */
/*
 * Arg value kinds used by the streaming parser.
 */
typedef enum { ARG_NULL=0, ARG_BOOL, ARG_INT64, ARG_UINT64, ARG_DOUBLE, ARG_STRING } ArgKind;

#define MAX_ARG_KEY_LEN  128
#define MAX_ARG_STR_LEN  512

typedef struct {
    char    key[MAX_ARG_KEY_LEN];
    ArgKind kind;
    union {
        int      b;
        int64_t  i64;
        uint64_t u64;
        double   f64;
        char     str[MAX_ARG_STR_LEN];
    } v;
} EventArg;

/*
 * Resolved argument: key StrRef + pre-resolved value refs.
 * Fully independent of the JSON DOM.
 */
typedef struct {
    StrRef  kref;
    ArgKind kind;
    union {
        int64_t  i64;
        uint64_t u64;
        double   f64;
        StrRef   sref;   /* for ARG_STRING: value already interned */
    } v;
} ResolvedArg;

/*
 * emit_resolved_arg: write one FXT argument word sequence.
 * All StrRefs already resolved; no intern calls happen here.
 *
 * FXT argument type numbers:
 *   0=null  3=int64  4=uint64  5=double  6=string  9=bool
 */
static void emit_resolved_arg(const ResolvedArg *ra) {
    const StrRef *kref    = &ra->kref;
    size_t        key_extra = kref->is_inline ? ((kref->len + 7) >> 3) : 0;

    switch (ra->kind) {
    case ARG_UINT64: {
        uint64_t sz  = 2 + key_extra;
        uint64_t hdr = (uint64_t)4 | (sz << 4) | ((uint64_t)kref->ref << 16);
        out_word_unsafe(hdr);
        emit_inline_str(kref);
        out_word_unsafe(ra->v.u64);
        break; }
    case ARG_INT64: {
        uint64_t sz  = 2 + key_extra;
        uint64_t hdr = (uint64_t)3 | (sz << 4) | ((uint64_t)kref->ref << 16);
        out_word_unsafe(hdr);
        emit_inline_str(kref);
        out_word_unsafe((uint64_t)ra->v.i64);
        break; }
    case ARG_DOUBLE: {
        uint64_t sz  = 2 + key_extra;
        uint64_t hdr = (uint64_t)5 | (sz << 4) | ((uint64_t)kref->ref << 16);
        out_word_unsafe(hdr);
        emit_inline_str(kref);
        uint64_t dword; memcpy(&dword, &ra->v.f64, 8);
        out_word_unsafe(dword);
        break; }
    case ARG_STRING: {
        const StrRef *vref     = &ra->v.sref;
        size_t        val_extra = vref->is_inline ? ((vref->len + 7) >> 3) : 0;
        uint64_t sz  = 1 + key_extra + val_extra;
        uint64_t hdr = (uint64_t)6 | (sz << 4)
                     | ((uint64_t)kref->ref << 16)
                     | ((uint64_t)vref->ref << 32);
        out_word_unsafe(hdr);
        emit_inline_str(kref);
        emit_inline_str(vref);
        break; }
    case ARG_BOOL: {
        uint64_t sz  = 1 + key_extra;
        uint64_t hdr = (uint64_t)9 | (sz << 4)
                     | ((uint64_t)kref->ref << 16)
                     | ((uint64_t)(ra->v.i64 ? 1 : 0) << 32);
        out_word_unsafe(hdr);
        emit_inline_str(kref);
        break; }
    default: /* ARG_NULL */ {
        uint64_t sz  = 1 + key_extra;
        uint64_t hdr = (uint64_t)0 | (sz << 4) | ((uint64_t)kref->ref << 16);
        out_word_unsafe(hdr);
        emit_inline_str(kref);
        break; }
    }
}

/* -----------------------------------------------------------------------
 * FxtEvent (streaming)
 * --------------------------------------------------------------------- */

#define MAX_EVENT_ARGS 15

typedef struct {
    int       etype;      /* FXT event type 0..10; -1=M metadata; -2=skip */
    uint64_t  ts_ns;
    uint64_t  pid;
    uint64_t  tid;
    char      cat[MAX_ARG_KEY_LEN];
    char      name[MAX_ARG_KEY_LEN];
    uint64_t  extra_word;
    int       has_extra;
    EventArg  args[MAX_EVENT_ARGS];
    size_t    num_args;
} FxtEvent;

/* -----------------------------------------------------------------------
 * emit_event
 * --------------------------------------------------------------------- */

static void emit_event(const FxtEvent *ev) {
    /* Step 1: resolve cat and name (may emit String Records) */
    StrRef cat_ref  = make_strref(ev->cat[0]  ? ev->cat  : "");
    StrRef name_ref = make_strref(ev->name[0] ? ev->name : "");

    /* Step 2: resolve thread (may emit Thread + KObj records) */
    uint8_t tref = intern_thread(ev->pid, ev->tid);

    /* Step 3: resolve all arg string refs before hdr_pos */
    ResolvedArg rargs[MAX_EVENT_ARGS];
    size_t num_args = ev->num_args < MAX_EVENT_ARGS ? ev->num_args : MAX_EVENT_ARGS;

    for (size_t i = 0; i < num_args; i++) {
        const EventArg *a = &ev->args[i];
        ResolvedArg    *r = &rargs[i];
        r->kref = make_strref(a->key);
        r->kind = a->kind;
        switch (a->kind) {
        case ARG_UINT64: r->v.u64  = a->v.u64; break;
        case ARG_INT64:  r->v.i64  = a->v.i64; break;
        case ARG_DOUBLE: r->v.f64  = a->v.f64; break;
        case ARG_BOOL:   r->v.i64  = a->v.b;   break;
        case ARG_STRING:
            intern_string(a->v.str);          /* pre-flush before hdr_pos */
            r->v.sref = make_strref(a->v.str);
            break;
        default: break; /* ARG_NULL */
        }
    }

    /* ALL intern calls done. Write event body. */
    out_reserve(MAX_RECORD_BYTES);

    size_t hdr_pos = out_tell();
    out_word_unsafe(0);          /* placeholder – patched below */
    out_word_unsafe(ev->ts_ns);

    if (tref == 0) {
        out_word_unsafe(ev->pid);
        out_word_unsafe(ev->tid);
    }

    emit_inline_str(&cat_ref);
    emit_inline_str(&name_ref);

    for (size_t i = 0; i < num_args; i++)
        emit_resolved_arg(&rargs[i]);

    if (ev->has_extra)
        out_word_unsafe(ev->extra_word);

    size_t total_words = (out_tell() - hdr_pos) / 8;
    uint64_t header =
          (uint64_t)4
        | ((uint64_t)total_words   << 4)
        | ((uint64_t)ev->etype     << 16)
        | ((uint64_t)num_args      << 20)
        | ((uint64_t)tref          << 24)
        | ((uint64_t)cat_ref.ref   << 32)
        | ((uint64_t)name_ref.ref  << 48);
    out_patch(hdr_pos, header);
}

/* -----------------------------------------------------------------------
 * Chrome → FXT event type mapping
 * -----------------------------------------------------------------------
 *
 * Chrome phase → FXT event type:
 *   'B'  Begin      → 2 (Duration begin)
 *   'E'  End        → 3 (Duration end)
 *   'X'  Complete   → 4 (Duration complete, needs "dur" field)
 *   'i'/'I' Instant → 0 (Instant)
 *   'C'  Counter    → 1 (Counter, counter_id from "id" field)
 *   'b'  Async begin→ 5
 *   'n'  Async inst → 6
 *   'e'  Async end  → 7
 *   's'  Flow begin → 8
 *   't'  Flow step  → 9
 *   'f'  Flow end   → 10
 *
 * All others (M=metadata, etc.) we emit as Instant events so the count
 * remains consistent.
 */

static long g_events_read    = 0;
static long g_events_written = 0;

/* -----------------------------------------------------------------------
 * Per-thread call-stack depth tracker
 *
 * Tracks the B/E nesting depth for each (pid, tid) pair using an
 * open-addressed hash map.  Each entry holds a stack of B event names so
 * that on every E event we can verify the name matches the innermost B.
 * Depth must never go below zero.  At end of conversion every entry must
 * be zero (no unmatched B events).
 * --------------------------------------------------------------------- */
#define DEPTH_HASH_SIZE  4096u   /* power of two */
#define DEPTH_HASH_MASK  (DEPTH_HASH_SIZE - 1u)
#define DEPTH_STACK_MAX  256     /* max nesting depth tracked per thread */

typedef struct {
    uint64_t pid;
    uint64_t tid;
    long     depth;
    int      used;
    /* Stack of B event names, one per nesting level */
    char     names[DEPTH_STACK_MAX][MAX_ARG_KEY_LEN];
} DepthEntry;

static DepthEntry g_depth_map[DEPTH_HASH_SIZE];

static DepthEntry *depth_find_or_insert(uint64_t pid, uint64_t tid) {
    uint64_t h = 2166136261u;
    for (int b = 0; b < 8; b++) h = (h ^ ((pid >> (b*8)) & 0xff)) * 16777619u;
    for (int b = 0; b < 8; b++) h = (h ^ ((tid >> (b*8)) & 0xff)) * 16777619u;
    uint32_t slot = (uint32_t)(h & DEPTH_HASH_MASK);
    for (;;) {
        DepthEntry *e = &g_depth_map[slot];
        if (!e->used) {
            e->pid = pid; e->tid = tid; e->depth = 0; e->used = 1;
            return e;
        }
        if (e->pid == pid && e->tid == tid) return e;
        slot = (slot + 1) & DEPTH_HASH_MASK;
    }
}

static void depth_push(uint64_t pid, uint64_t tid, const char *name) {
    DepthEntry *e = depth_find_or_insert(pid, tid);
    if (e->depth < DEPTH_STACK_MAX) {
        size_t _nl = strlen(name);
        if (_nl >= MAX_ARG_KEY_LEN) _nl = MAX_ARG_KEY_LEN - 1;
        memcpy(e->names[e->depth], name, _nl);
        e->names[e->depth][_nl] = 0;
    }
    e->depth++;
}


/* -----------------------------------------------------------------------
 * Command-line options
 * --------------------------------------------------------------------- */


static int g_opt_verbose  = 0;   /* --verbose : print stats to stderr */
static int g_opt_debug    = 0;   /* --debug   : print every event as JSON to stderr */

/* -----------------------------------------------------------------------
 * yajl streaming parser
 *
 * Parses Chrome Trace JSON incrementally from stdin, building one
 * FxtEvent at a time.  When an event object closes, dispatch_event()
 * processes and emits it immediately.  The full JSON tree is never kept
 * in memory; peak allocation is O(one event).
 *
 * State machine:
 *   PS_ROOT         – before any token (root may be [ or {)
 *   PS_ROOT_OBJ     – inside root object, scanning for "traceEvents"
 *   PS_ROOT_SKIP    – skipping a non-traceEvents value in root object
 *   PS_EVENTS_ARRAY – between event objects in the events array
 *   PS_EVENT_OBJ    – inside one event object, scanning field keys
 *   PS_EVENT_FIELD  – expecting the scalar value of a known field
 *   PS_EVENT_SKIP   – skipping a nested value inside an event
 *   PS_ARGS_OBJ     – inside the "args" object of an event
 *   PS_ARG_VAL      – expecting one arg value
 *   PS_ARG_SKIP     – skipping nested object/array inside args
 * --------------------------------------------------------------------- */


typedef enum {
    PS_ROOT, PS_ROOT_OBJ, PS_ROOT_SKIP,
    PS_TRACE_EVENTS_VALUE,  /* expecting the [ that opens traceEvents */
    PS_EVENTS_ARRAY,
    PS_EVENT_OBJ, PS_EVENT_FIELD, PS_EVENT_SKIP,
    PS_ARGS_OBJ, PS_ARG_VAL, PS_ARG_SKIP
} ParseState;

typedef enum {
    EF_NONE, EF_PH, EF_CAT, EF_NAME, EF_TS, EF_DUR, EF_PID, EF_TID, EF_ID
} EventField;

typedef struct {
    ParseState state;
    EventField cur_field;
    int        skip_depth;
    FxtEvent   ev;
    char       arg_key[MAX_ARG_KEY_LEN];
} ParseCtx;

static void reset_event(FxtEvent *ev) {
    memset(ev, 0, sizeof(*ev));
    ev->etype = -2;   /* default: unknown phase – skip */
}

static void set_phase(FxtEvent *ev, char ph) {
    switch (ph) {
    case 'B': ev->etype = 2;  ev->has_extra = 0; break;
    case 'E': ev->etype = 3;  ev->has_extra = 0; break;
    case 'X': ev->etype = 4;  ev->has_extra = 1; break;
    case 'i':
    case 'I': ev->etype = 0;  ev->has_extra = 0; break;
    case 'C': ev->etype = 1;  ev->has_extra = 1; break;
    case 'b': ev->etype = 5;  ev->has_extra = 1; break;
    case 'n': ev->etype = 6;  ev->has_extra = 1; break;
    case 'e': ev->etype = 7;  ev->has_extra = 1; break;
    case 's': ev->etype = 8;  ev->has_extra = 1; break;
    case 't': ev->etype = 9;  ev->has_extra = 1; break;
    case 'f': ev->etype = 10; ev->has_extra = 1; break;
    case 'M': ev->etype = -1; ev->has_extra = 0; break;
    default:   ev->etype = -2; ev->has_extra = 0; break;
    }
}



static void debug_print_event(const FxtEvent *ev, long depth) {
    const char *ph;
    switch (ev->etype) {
    case  2: ph = "B"; break;  case  3: ph = "E"; break;
    case  4: ph = "X"; break;  case  0: ph = "i"; break;
    case  1: ph = "C"; break;  case  5: ph = "b"; break;
    case  6: ph = "n"; break;  case  7: ph = "e"; break;
    case  8: ph = "s"; break;  case  9: ph = "t"; break;
    case 10: ph = "f"; break;  case -1: ph = "M"; break;
    default: ph = "?"; break;
    }
    double ts_us = (double)ev->ts_ns / 1000.0;
    /* Build the JSON object field by field to avoid PRIu64 string concat issues */
    fprintf(stderr, "{\"ph\":\"%s\",\"cat\":\"%s\",\"name\":\"%s\","
                    "\"ts\":%.3f,\"pid\":", ph, ev->cat, ev->name, ts_us);
    fprintf(stderr, "%" PRIu64 ",\"tid\":", ev->pid);
    fprintf(stderr, "%" PRIu64, ev->tid);
    if (ev->etype == 4)
        fprintf(stderr, ",\"dur\":%.3f",
                (double)(ev->extra_word - ev->ts_ns) / 1000.0);
    if (ev->has_extra && ev->etype != 4)
        fprintf(stderr, ",\"id\":%" PRIu64, ev->extra_word);
    if (ev->num_args > 0) {
        fprintf(stderr, ",\"args\":{");
        for (size_t i = 0; i < ev->num_args; i++) {
            const EventArg *a = &ev->args[i];
            if (i) fputc(',', stderr);
            fprintf(stderr, "\"%s\":", a->key);
            switch (a->kind) {
            case ARG_NULL:   fprintf(stderr, "null"); break;
            case ARG_BOOL:   fprintf(stderr, "%s", a->v.b ? "true" : "false"); break;
            case ARG_INT64:  fprintf(stderr, "%" PRId64, a->v.i64); break;
            case ARG_UINT64: fprintf(stderr, "%" PRIu64, a->v.u64); break;
            case ARG_DOUBLE: fprintf(stderr, "%g",  a->v.f64); break;
            case ARG_STRING: fprintf(stderr, "\"%s\"", a->v.str); break;
            }
        }
        fputc('}', stderr);
    }
    fprintf(stderr, ",\"depth\":%ld}\n", depth);
}

/*
 * emit_synthetic_e: emit a synthetic Duration-end event for `name` on the
 * given thread, at timestamp `ts_ns`.  Used to close orphaned B events when
 * a real E arrives for a function deeper in the stack.
 */
static void emit_synthetic_e(uint64_t pid, uint64_t tid, uint64_t ts_ns,
                              const char *cat, const char *name) {
    FxtEvent fake;
    memset(&fake, 0, sizeof(fake));
    fake.etype     = 3;   /* Duration end */
    fake.has_extra = 0;
    fake.pid       = pid;
    fake.tid       = tid;
    fake.ts_ns     = ts_ns;
    size_t cl = strlen(cat);
    if (cl >= MAX_ARG_KEY_LEN) cl = MAX_ARG_KEY_LEN - 1;
    memcpy(fake.cat,  cat,  cl);  fake.cat[cl]  = 0;
    size_t nl = strlen(name);
    if (nl >= MAX_ARG_KEY_LEN) nl = MAX_ARG_KEY_LEN - 1;
    memcpy(fake.name, name, nl);  fake.name[nl] = 0;

    fprintf(stderr,
            "WARNING: synthesising E event '%s' on pid=%" PRIu64 " tid=%" PRIu64
            " to close orphaned B event\n", name, pid, tid);

    /* Update depth stack */
    DepthEntry *e = depth_find_or_insert(pid, tid);
    if (e->depth > 0) e->depth--;

    if (g_opt_debug) {
        long d = depth_find_or_insert(pid, tid)->depth;
        debug_print_event(&fake, d + 1);
    }
    emit_event(&fake);
    g_events_read++;
    g_events_written++;
}

static void dispatch_event(FxtEvent *ev) {
    if (ev->etype == -1) {
        /* Metadata 'M': register process/thread name */
        const char *label = NULL;
        for (size_t i = 0; i < ev->num_args; i++)
            if (strcmp(ev->args[i].key, "name") == 0 &&
                ev->args[i].kind == ARG_STRING)
                label = ev->args[i].v.str;
        if (label) {
            if      (strcmp(ev->name, "process_name") == 0)
                register_proc_name(ev->pid, label);
            else if (strcmp(ev->name, "thread_name") == 0)
                register_thr_name(ev->pid, ev->tid, label);
        }
        if (g_opt_debug) debug_print_event(ev, 0);
        return;   /* M events do not count toward read/written */
    }
    if (ev->etype == -2) return;  /* unknown phase – skip silently */

    /* Track B/E call-stack depth per thread */
    if (ev->etype == 2) {      /* Duration begin */
        depth_push(ev->pid, ev->tid, ev->name);
    } else if (ev->etype == 3) {  /* Duration end */
        DepthEntry *_de = depth_find_or_insert(ev->pid, ev->tid);
        if (_de->depth <= 0) {
            /* Stack already empty – warn and ignore */
            fprintf(stderr,
                    "WARNING: unmatched E event '%s' on pid=%" PRIu64
                    " tid=%" PRIu64 " (depth would go negative)\n",
                    ev->name, ev->pid, ev->tid);
            return;
        } else {
            /* Check if the top of stack matches */
            long _top = _de->depth - 1;
            if (_top < DEPTH_STACK_MAX &&
                strcmp(_de->names[_top], ev->name) == 0) {
                /* Perfect match – normal pop */
                _de->depth--;
            } else {
                /* Mismatch: search downward for a matching frame */
                long _found = -1;
                for (long _i = _top - 1; _i >= 0; _i--) {
                    if (_i < DEPTH_STACK_MAX &&
                        strcmp(_de->names[_i], ev->name) == 0) {
                        _found = _i;
                        break;
                    }
                }
                if (_found >= 0) {
                    /* Synthesise E events for all frames above the match */
                    for (long _i = _top; _i > _found; _i--) {
                        const char *_orphan = (_i < DEPTH_STACK_MAX)
                                              ? _de->names[_i] : "?";
                        emit_synthetic_e(ev->pid, ev->tid, ev->ts_ns,
                                         ev->cat, _orphan);
                    }
                    /* Now pop the matching frame */
                    _de->depth--;
                } else {
                    /* Not found anywhere in the stack – warn and ignore */
                    fprintf(stderr,
                            "WARNING: E event '%s' on pid=%" PRIu64
                            " tid=%" PRIu64
                            " does not match any open B event\n",
                            ev->name, ev->pid, ev->tid);
                    return;
                }
            }
        }
    }

    if (g_opt_debug) {
        long d = depth_find_or_insert(ev->pid, ev->tid)->depth;
        /* For E events show depth+1 (pre-pop depth) to match the B entry */
        debug_print_event(ev, ev->etype == 3 ? d + 1 : d);
    }
    g_events_read++;
    emit_event(ev);
    g_events_written++;

}

/* ---- helpers ---- */
#define SCOPY(dst, src, srclen, maxlen) do { \
    size_t _n = (srclen) < (maxlen)-1 ? (srclen) : (maxlen)-1; \
    memcpy((dst), (src), _n); (dst)[_n] = '\0'; } while(0)

static EventArg *next_arg(ParseCtx *ctx) {
    if (ctx->ev.num_args >= MAX_EVENT_ARGS) return NULL;
    EventArg *a = &ctx->ev.args[ctx->ev.num_args++];
    memset(a, 0, sizeof(*a));
    SCOPY(a->key, ctx->arg_key, strlen(ctx->arg_key), MAX_ARG_KEY_LEN);
    return a;
}

/* ---- yajl callbacks ---- */

static int cb_null(void *vctx) {
    ParseCtx *ctx = vctx;
    if (ctx->state == PS_ARG_VAL) {
        EventArg *a = next_arg(ctx);
        if (a) a->kind = ARG_NULL;
        ctx->state = PS_ARGS_OBJ;
    } else if (ctx->state == PS_EVENT_FIELD) {
        ctx->state = PS_EVENT_OBJ;
    }
    return 1;
}

static int cb_boolean(void *vctx, int val) {
    ParseCtx *ctx = vctx;
    if (ctx->state == PS_ARG_VAL) {
        EventArg *a = next_arg(ctx);
        if (a) { a->kind = ARG_BOOL; a->v.b = val; }
        ctx->state = PS_ARGS_OBJ;
    } else if (ctx->state == PS_EVENT_FIELD) {
        ctx->state = PS_EVENT_OBJ;
    }
    return 1;
}

static int cb_integer(void *vctx, long long val) {
    ParseCtx *ctx = vctx;
    if (ctx->state == PS_EVENT_FIELD) {
        switch (ctx->cur_field) {
        case EF_TS:  ctx->ev.ts_ns = (uint64_t)((double)val * 1000.0); break;
        case EF_DUR: ctx->ev.extra_word = ctx->ev.ts_ns
                                        + (uint64_t)((double)val * 1000.0); break;
        case EF_PID: ctx->ev.pid = (uint64_t)val; break;
        case EF_TID: ctx->ev.tid = (uint64_t)val; break;
        case EF_ID:
            if (ctx->ev.has_extra && ctx->ev.etype != 4)
                ctx->ev.extra_word = (uint64_t)val;
            break;
        default: break;
        }
        ctx->state = PS_EVENT_OBJ;
    } else if (ctx->state == PS_ARG_VAL) {
        EventArg *a = next_arg(ctx);
        if (a) {
            if (val < 0) { a->kind = ARG_INT64;  a->v.i64 = (int64_t)val; }
            else         { a->kind = ARG_UINT64; a->v.u64 = (uint64_t)val; }
        }
        ctx->state = PS_ARGS_OBJ;
    }
    return 1;
}

static int cb_double(void *vctx, double val) {
    ParseCtx *ctx = vctx;
    if (ctx->state == PS_EVENT_FIELD) {
        switch (ctx->cur_field) {
        case EF_TS:  ctx->ev.ts_ns      = (uint64_t)(val * 1000.0); break;
        case EF_DUR: ctx->ev.extra_word = ctx->ev.ts_ns
                                        + (uint64_t)(val * 1000.0); break;
        default: break;
        }
        ctx->state = PS_EVENT_OBJ;
    } else if (ctx->state == PS_ARG_VAL) {
        EventArg *a = next_arg(ctx);
        if (a) { a->kind = ARG_DOUBLE; a->v.f64 = val; }
        ctx->state = PS_ARGS_OBJ;
    }
    return 1;
}

static int cb_string(void *vctx, const unsigned char *s, size_t l) {
    ParseCtx *ctx = vctx;
    if (ctx->state == PS_EVENT_FIELD) {
        switch (ctx->cur_field) {
        case EF_PH:
            set_phase(&ctx->ev, l > 0 ? (char)s[0] : 'i');
            break;
        case EF_CAT:  SCOPY(ctx->ev.cat,  s, l, MAX_ARG_KEY_LEN); break;
        case EF_NAME: SCOPY(ctx->ev.name, s, l, MAX_ARG_KEY_LEN); break;
        case EF_ID: {
            char tmp[64]; size_t tl = l < 63 ? l : 63;
            memcpy(tmp, s, tl); tmp[tl] = '\0';
            if (ctx->ev.has_extra && ctx->ev.etype != 4)
                ctx->ev.extra_word = strtoull(tmp, NULL, 0);
            break; }
        default: break;
        }
        ctx->state = PS_EVENT_OBJ;
    } else if (ctx->state == PS_ARG_VAL) {
        EventArg *a = next_arg(ctx);
        if (a) { a->kind = ARG_STRING; SCOPY(a->v.str, s, l, MAX_ARG_STR_LEN); }
        ctx->state = PS_ARGS_OBJ;
    }
    return 1;
}

static int cb_map_key(void *vctx, const unsigned char *s, size_t l) {
    ParseCtx *ctx = vctx;
    switch (ctx->state) {
    case PS_ROOT_OBJ:
        if (l == 11 && memcmp(s, "traceEvents", 11) == 0)
            ctx->state = PS_TRACE_EVENTS_VALUE;
        else {
            ctx->state = PS_ROOT_SKIP;
            ctx->skip_depth = 0;
        }
        break;
    case PS_EVENT_OBJ:
        ctx->cur_field = EF_NONE;
        ctx->state     = PS_EVENT_FIELD;
        if      (l==2 && memcmp(s,"ph",2)==0)   ctx->cur_field = EF_PH;
        else if (l==3 && memcmp(s,"cat",3)==0)  ctx->cur_field = EF_CAT;
        else if (l==4 && memcmp(s,"name",4)==0) ctx->cur_field = EF_NAME;
        else if (l==2 && memcmp(s,"ts",2)==0)   ctx->cur_field = EF_TS;
        else if (l==3 && memcmp(s,"dur",3)==0)  ctx->cur_field = EF_DUR;
        else if (l==3 && memcmp(s,"pid",3)==0)  ctx->cur_field = EF_PID;
        else if (l==3 && memcmp(s,"tid",3)==0)  ctx->cur_field = EF_TID;
        else if (l==2 && memcmp(s,"id",2)==0)   ctx->cur_field = EF_ID;
        else if (l==4 && memcmp(s,"args",4)==0) { ctx->state = PS_ARGS_OBJ; }
        else { ctx->state = PS_EVENT_SKIP; ctx->skip_depth = 0; }
        break;
    case PS_ARGS_OBJ: {
        size_t kl = l < MAX_ARG_KEY_LEN-1 ? l : MAX_ARG_KEY_LEN-1;
        memcpy(ctx->arg_key, s, kl); ctx->arg_key[kl] = '\0';
        if (strcmp(ctx->arg_key, "srcline") == 0)
            strcpy(ctx->arg_key, "args.srcline");
        ctx->state = PS_ARG_VAL;
        break; }
    default: break;
    }
    return 1;
}

static int cb_start_map(void *vctx) {
    ParseCtx *ctx = vctx;
    switch (ctx->state) {
    case PS_ROOT:         ctx->state = PS_ROOT_OBJ; break;
    case PS_EVENTS_ARRAY:
        reset_event(&ctx->ev);
        ctx->state = PS_EVENT_OBJ;
        break;
    case PS_EVENT_FIELD:
        ctx->state = PS_EVENT_SKIP; ctx->skip_depth = 1; break;
    case PS_EVENT_SKIP:   ctx->skip_depth++; break;
    case PS_ROOT_SKIP:    ctx->skip_depth++; break;
    case PS_ARG_VAL: {
        EventArg *a = next_arg(ctx);
        if (a) a->kind = ARG_NULL;   /* nested map -> null arg */
        ctx->state = PS_ARG_SKIP; ctx->skip_depth = 1; break; }
    case PS_ARG_SKIP:     ctx->skip_depth++; break;
    default: break;
    }
    return 1;
}

static int cb_end_map(void *vctx) {
    ParseCtx *ctx = vctx;
    switch (ctx->state) {
    case PS_EVENT_OBJ:
        dispatch_event(&ctx->ev);
        reset_event(&ctx->ev);
        ctx->state = PS_EVENTS_ARRAY;
        break;
    case PS_ARGS_OBJ:     ctx->state = PS_EVENT_OBJ; break;
    case PS_ROOT_OBJ:     /* root object closed */ break;
    case PS_EVENT_SKIP:
        if (--ctx->skip_depth <= 0) ctx->state = PS_EVENT_OBJ;
        break;
    case PS_ARG_SKIP:
        if (--ctx->skip_depth <= 0) ctx->state = PS_ARGS_OBJ;
        break;
    case PS_ROOT_SKIP:
        if (--ctx->skip_depth <= 0) ctx->state = PS_ROOT_OBJ;
        break;
    default: break;
    }
    return 1;
}

static int cb_start_array(void *vctx) {
    ParseCtx *ctx = vctx;
    switch (ctx->state) {
    case PS_ROOT:                ctx->state = PS_EVENTS_ARRAY; break;
    case PS_TRACE_EVENTS_VALUE:  ctx->state = PS_EVENTS_ARRAY; break;
    case PS_EVENTS_ARRAY: ctx->state = PS_EVENT_SKIP; ctx->skip_depth = 1; break;
    case PS_EVENT_FIELD:  ctx->state = PS_EVENT_SKIP; ctx->skip_depth = 1; break;
    case PS_EVENT_SKIP:   ctx->skip_depth++; break;
    case PS_ROOT_SKIP:    ctx->skip_depth++; break;
    case PS_ARG_VAL: {
        EventArg *a = next_arg(ctx);
        if (a) a->kind = ARG_NULL;
        ctx->state = PS_ARG_SKIP; ctx->skip_depth = 1; break; }
    case PS_ARG_SKIP:     ctx->skip_depth++; break;
    default: break;
    }
    return 1;
}

static int cb_end_array(void *vctx) {
    ParseCtx *ctx = vctx;
    switch (ctx->state) {
    case PS_EVENTS_ARRAY: ctx->state = PS_ROOT_OBJ; break;
    case PS_EVENT_SKIP:
        if (--ctx->skip_depth <= 0) ctx->state = PS_EVENT_OBJ;
        break;
    case PS_ARG_SKIP:
        if (--ctx->skip_depth <= 0) ctx->state = PS_ARGS_OBJ;
        break;
    case PS_ROOT_SKIP:
        if (--ctx->skip_depth <= 0) ctx->state = PS_ROOT_OBJ;
        break;
    default: break;
    }
    return 1;
}



/* -----------------------------------------------------------------------
 * Main
 * --------------------------------------------------------------------- */

static void print_help(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [OPTIONS] < input.json > output.fxt\n"
        "\n"
        "Convert a Chrome Trace Event Format (JSON) file to Fuchsia Trace Format (FXT).\n"
        "Reads from stdin, writes binary FXT to stdout.\n"
        "\n"
        "Options:\n"
        "  --help      Show this help message and exit.\n"
        "  --verbose   Print conversion statistics to stderr when done.\n"
        "  --debug     Print every event as JSON to stderr.\n"
        "\n"
        "Examples:\n"
        "  %s < trace.json > trace.fxt\n"
        "  %s --verbose < trace.json > trace.fxt\n"
        "  %s --debug   < trace.json > trace.fxt\n",
        argv0, argv0, argv0, argv0);
}

int main(int argc, char **argv) {
    /* Parse command-line options */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            g_opt_verbose = 1;
        } else if (strcmp(argv[i], "--debug") == 0) {
            g_opt_debug = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n"
                            "Run '%s --help' for usage.\n",
                    argv[i], argv[0]);
            return 1;
        }
    }

    /* Allocate output buffer (pre-sized generously; out_grow expands as needed) */
    g_outcap = OUTBUF_INITIAL * 4;
    g_outbuf = (uint8_t *)malloc(g_outcap);
    if (!g_outbuf) { fprintf(stderr, "OOM\n"); return 1; }

    /* Emit mandatory header records */
    emit_magic();
    emit_init();

    /* ---- yajl streaming parse ---- */
    yajl_callbacks cbs = {
        cb_null, cb_boolean, cb_integer, cb_double,
        NULL,  /* number: use integer+double */
        cb_string,
        cb_start_map, cb_map_key, cb_end_map,
        cb_start_array, cb_end_array
    };
    ParseCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.state = PS_ROOT;
    reset_event(&ctx.ev);

    yajl_handle yh = yajl_alloc(&cbs, NULL, &ctx);
    yajl_config(yh, yajl_allow_comments, 1);


    /*
     * Read stdin through a trailing-comma filter before feeding yajl.
     *
     * Some Chrome trace writers emit invalid JSON with a trailing comma after
     * the last element of an array or object, e.g.:
     *   {"traceEvents": [ {...}, {...}, ]}
     *                                  ^-- illegal
     *
     * The filter uses a one-byte look-ahead: when a ',' is seen, it is held
     * back.  If the next non-whitespace character is ']' or '}' the comma is
     * silently dropped; otherwise the comma is flushed before the new byte.
     * Whitespace between a held comma and the closing bracket is passed
     * through normally (it goes to yajl after the comma is dropped).
     */
    unsigned char ibuf[65536];
    unsigned char fbuf[65536]; /* filtered output fed to yajl */
    size_t got = 0;
    yajl_status yst = yajl_status_ok;
    int comma_held = 0; /* 1 if a ',' is buffered */

    for (;;) {
        got = fread(ibuf, 1, sizeof(ibuf), stdin);
        if (got == 0) break;

        size_t flen = 0;
        for (size_t i = 0; i < got; i++) {
            unsigned char ch = ibuf[i];
            if (comma_held) {
                if (ch == ']' || ch == '}') {
                    /* Drop the held comma – it was trailing */
                    comma_held = 0;
                    fbuf[flen++] = ch;
                } else if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
                    /* Whitespace after comma: pass through but keep holding comma */
                    fbuf[flen++] = ch;
                } else if (ch == ',') {
                    /* Two commas in a row: flush the first, hold the new one */
                    fbuf[flen++] = ',';
                    /* comma_held stays 1 */
                } else {
                    /* Non-whitespace, non-bracket: flush comma then the char */
                    fbuf[flen++] = ',';
                    comma_held = 0;
                    fbuf[flen++] = ch;
                }
            } else {
                if (ch == ',') comma_held = 1;
                else           fbuf[flen++] = ch;
            }
        }

        if (flen > 0) {
            yst = yajl_parse(yh, fbuf, flen);
            if (yst != yajl_status_ok) break;
        }
    }
    /* Flush any held comma at EOF (it would be trailing – drop it) */
    /* comma_held at EOF means trailing comma: silently discard */
    if (yst == yajl_status_ok)
        yst = yajl_complete_parse(yh);


    if (yst != yajl_status_ok) {
        unsigned char *emsg = yajl_get_error(yh, 1, ibuf, got);
        fprintf(stderr, "JSON parse error: %s\n", emsg);
        yajl_free_error(yh, emsg);
        yajl_free(yh);
        return 1;
    }
    yajl_free(yh);


    /* Flush any remaining buffered output */
    out_flush();

    free(g_outbuf);

    /* Free string table */
    for (int i = 0; i < g_strtab_count; i++) {
        free(g_strtab[i].str);
    }

    /* Stats: only printed when --verbose is given */
    if (g_opt_verbose) {
        fprintf(stderr, "Events read:          %ld\n", g_events_read);
        fprintf(stderr, "Events written:       %ld\n", g_events_written);
        fprintf(stderr, "String refs used:     %ld\n", g_strref_count);
        fprintf(stderr, "String table size:    %d / %d\n", g_strtab_count, MAX_STRING_TABLE);
        fprintf(stderr, "Thread table size:    %d / %d\n", g_thrtab_count, MAX_THREAD_TABLE);
        /* g_outsize is 0 after streaming flush; total output bytes not tracked */
    }

    if (g_events_read != g_events_written) {
        fprintf(stderr,
                "WARNING: events_read (%ld) != events_written (%ld)!\n",
                g_events_read, g_events_written);
        return 1;
    }

    /* Check that every thread's B/E call stack is balanced. */
    {
        int unbalanced = 0;
        for (uint32_t i = 0; i < DEPTH_HASH_SIZE; i++) {
            DepthEntry *e = &g_depth_map[i];
            if (e->used && e->depth != 0) {
                fprintf(stderr,
                        "WARNING: unmatched B events on pid=%" PRIu64
                        " tid=%" PRIu64 " (depth=%ld)\n",
                        e->pid, e->tid, e->depth);
                unbalanced = 1;
            }
        }
        if (unbalanced) return 1;
    }

    return 0;
}
