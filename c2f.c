/*
 * c2f.c
 *
 * Converts Chrome Trace Event Format (JSON) on stdin to
 * Fuchsia Trace Format (FXT) binary on stdout.
 *
 * Build:
 *   cc -O2 -o c2f c2f.c -lyyjson -lm
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

#include "yyjson.h"

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

    /* Not found – register if table has room */
    if (g_strtab_count < MAX_STRING_TABLE) {
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

    /* Table full – inline ref */
    size_t clamped = len > 32767 ? 32767 : len;
    return (uint16_t)(0x8000u | (uint16_t)clamped);
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
    if (g_thrtab_count < MAX_THREAD_TABLE) {
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

        /* 3. FXT Thread Record (record type 3):
         *     [0..3]=3, [4..15]=3 (words), [16..23]=thread_idx
         *     process id word
         *     thread id word
         */
        uint64_t header =
              (uint64_t)3
            | ((uint64_t)3       << 4)
            | ((uint64_t)new_idx << 16);
        out_word(header);
        out_word(pid);
        out_word(tid);

        return new_idx;
    }
    return 0; /* inline fallback: thread table full */
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
 * Resolved argument: key StrRef + yyjson value, pre-resolved before event body.
 * String records for new strings have already been flushed by the time
 * emit_resolved_arg is called, so no intern calls happen here.
 */
typedef struct {
    StrRef      kref;
    yyjson_val *val;
} ResolvedArg;

/*
 * emit_resolved_arg: write one argument word sequence.
 * All StrRefs are already resolved (intern calls done before hdr_pos).
 * BUG B fix: compute sizes dynamically from kref.is_inline / vref.is_inline.
 */
static void emit_resolved_arg(const ResolvedArg *ra) {
    const StrRef *kref = &ra->kref;
    yyjson_val   *val  = ra->val;
    (void)val;  /* used in every branch below; suppresses stub-induced warning */
    /* extra words consumed by an inline key name stream */
    size_t key_extra = kref->is_inline ? ((kref->len + 7) >> 3) : 0;

    /*
     * FXT argument type numbers (from the Fuchsia trace-format spec):
     *   0 = null
     *   1 = int32   (value in header [32..63], size=1)
     *   2 = uint32  (value in header [32..63], size=1)
     *   3 = int64   (separate value word,       size=2)
     *   4 = uint64  (separate value word,        size=2)
     *   5 = double  (separate value word,        size=2)
     *   6 = string  (string refs in header,      size=1+inline)
     *   7 = pointer (separate value word,        size=2)
     *   8 = koid    (separate value word,        size=2)
     *   9 = bool    (value in header [32],        size=1)
     */

    if (yyjson_is_uint(val)) {
        /* arg type 4 = uint64
         * layout: [hdr] [optional inline key] [uint64 value]
         * size = 1 + key_extra + 1
         */
        uint64_t sz  = 2 + key_extra;
        uint64_t hdr = (uint64_t)4
                     | (sz << 4)
                     | ((uint64_t)kref->ref << 16);
        out_word_unsafe(hdr);
        emit_inline_str(kref);
        out_word_unsafe(yyjson_get_uint(val));

    } else if (yyjson_is_sint(val) || yyjson_is_int(val)) {
        /* arg type 3 = int64
         * layout: [hdr] [optional inline key] [int64 value]
         * size = 1 + key_extra + 1
         */
        uint64_t sz  = 2 + key_extra;
        uint64_t hdr = (uint64_t)3
                     | (sz << 4)
                     | ((uint64_t)kref->ref << 16);
        out_word_unsafe(hdr);
        emit_inline_str(kref);
        out_word_unsafe((uint64_t)yyjson_get_sint(val));

    } else if (yyjson_is_real(val)) {
        /* arg type 5 = double
         * layout: [hdr] [optional inline key] [double value]
         * size = 1 + key_extra + 1
         */
        uint64_t sz  = 2 + key_extra;
        uint64_t hdr = (uint64_t)5
                     | (sz << 4)
                     | ((uint64_t)kref->ref << 16);
        out_word_unsafe(hdr);
        emit_inline_str(kref);
        double dv = yyjson_get_real(val);
        uint64_t dword;
        memcpy(&dword, &dv, 8);
        out_word_unsafe(dword);

    } else if (yyjson_is_str(val)) {
        /* arg type 6 = string
         * layout: [hdr] [optional inline key] [optional inline value]
         * header [32..47] = value string ref
         * All string refs already resolved in the pre-pass (BUG A fix):
         * make_strref here hits the hash table without emitting new records.
         */
        const char *sv   = yyjson_get_str(val);
        StrRef      vref = make_strref(sv);
        size_t val_extra = vref.is_inline ? ((vref.len + 7) >> 3) : 0;
        uint64_t sz  = 1 + key_extra + val_extra;
        uint64_t hdr = (uint64_t)6
                     | (sz                  << 4)
                     | ((uint64_t)kref->ref << 16)
                     | ((uint64_t)vref.ref  << 32);
        out_word_unsafe(hdr);
        emit_inline_str(kref);
        emit_inline_str(&vref);

    } else if (yyjson_is_bool(val)) {
        /* arg type 9 = boolean
         * value packed into header word bit [32]; size = 1 + key_extra
         */
        uint64_t sz  = 1 + key_extra;
        int bv = yyjson_get_bool(val) ? 1 : 0;
        uint64_t hdr = (uint64_t)9
                     | (sz                  << 4)
                     | ((uint64_t)kref->ref << 16)
                     | ((uint64_t)bv        << 32);
        out_word_unsafe(hdr);
        emit_inline_str(kref);

    } else {
        /* null / array / object -> arg type 0 (null)
         * size = 1 + key_extra
         */
        uint64_t sz  = 1 + key_extra;
        uint64_t hdr = (uint64_t)0
                     | (sz                  << 4)
                     | ((uint64_t)kref->ref << 16);
        out_word_unsafe(hdr);
        emit_inline_str(kref);
    }
}

/* -----------------------------------------------------------------------
 * Event record emission
 * -----------------------------------------------------------------------
 *
 * FXT Event record layout (record type 4):
 *
 *   header word:
 *     [0..3]  = 4
 *     [4..15] = total size in 8-byte words
 *     [16..19]= event type
 *     [20..23]= num args
 *     [24..31]= thread ref (8 bits)
 *     [32..47]= category string ref (16 bits)
 *     [48..63]= name string ref (16 bits)
 *
 *   timestamp word (ticks = microseconds, since init says 1e9 ticks/sec
 *                   but Chrome uses microseconds; we convert µs -> ns below)
 *
 *   process id word  (if thread ref == 0)
 *   thread id word   (if thread ref == 0)
 *
 *   category stream  (if category string ref has bit15 set)
 *   name stream      (if name string ref has bit15 set)
 *
 *   argument data    (repeated)
 *
 *   event-type specific data
 */

typedef struct {
    int      etype;         /* FXT event type 0..10                  */
    uint64_t ts_ns;         /* timestamp in nanoseconds               */
    uint64_t pid;
    uint64_t tid;
    const char *cat;
    const char *name;
    yyjson_val *args;       /* JSON object of args, may be NULL       */
    uint64_t extra_word;    /* counter id / async id / flow id / end_ts */
    int      has_extra;     /* 1 if extra_word should be written      */
} FxtEvent;

static void emit_event(const FxtEvent *ev) {
    /*
     * BUG A FIX: Resolve ALL string references (and register any new strings)
     * BEFORE recording hdr_pos.  intern_string may call emit_string_record
     * which writes bytes to the output buffer.  Those String Record bytes must
     * appear BEFORE the Event Record in the stream, not inside its body.
     *
     * Order: cat, name, then every arg key and string arg value.
     * Thread intern only writes Thread Records, same concern applies.
     */

    /* Step 1: resolve cat and name (may emit String Records) */
    StrRef cat_ref  = make_strref(ev->cat  ? ev->cat  : "");
    StrRef name_ref = make_strref(ev->name ? ev->name : "");

    /* Step 2: resolve thread (may emit a Thread Record) */
    uint8_t tref = intern_thread(ev->pid, ev->tid);

    /* Step 3: resolve all argument string refs and cache them.
     * We also resolve string arg values here so their String Records are
     * flushed before hdr_pos. */
#define MAX_ARGS 15
    ResolvedArg rargs[MAX_ARGS];
    size_t num_args = 0;

    if (ev->args && yyjson_is_obj(ev->args)) {
        yyjson_obj_iter iter;
        yyjson_obj_iter_init(ev->args, &iter);
        yyjson_val *key_v;
        while ((key_v = yyjson_obj_iter_next(&iter)) != NULL
               && num_args < MAX_ARGS) {
            yyjson_val *val_v   = yyjson_obj_iter_get_val(key_v);
            const char *key_str = yyjson_get_str(key_v);
            memset(&rargs[num_args], 0, sizeof(rargs[num_args]));
            rargs[num_args].kref = make_strref(key_str);
            /* Pre-intern string values so their String Records appear before hdr_pos */
            if (yyjson_is_str(val_v))
                intern_string(yyjson_get_str(val_v));
            rargs[num_args].val = val_v;
            num_args++;
        }
    }


    /*
     * ALL intern calls are done. From here on, no new String/Thread Records
     * will be written.  Now record hdr_pos and write the event body.
     */
    out_reserve(MAX_RECORD_BYTES);

    size_t hdr_pos = out_tell();
    out_word_unsafe(0);                   /* placeholder header – patched below */
    out_word_unsafe(ev->ts_ns);

    if (tref == 0) {
        out_word_unsafe(ev->pid);
        out_word_unsafe(ev->tid);
    }

    emit_inline_str(&cat_ref);
    emit_inline_str(&name_ref);

    for (size_t i = 0; i < num_args; i++) {
        emit_resolved_arg(&rargs[i]);
    }

    if (ev->has_extra) {
        out_word_unsafe(ev->extra_word);
    }

    /* Compute actual record size and patch the placeholder header */
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
#undef MAX_ARGS
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
 * Command-line options
 * --------------------------------------------------------------------- */

#include <unistd.h>   /* isatty, STDERR_FILENO */

static int g_opt_verbose  = 0;   /* --verbose : print stats to stderr    */
static int g_opt_progress = 0;   /* --progress: show progress bar        */

/* -----------------------------------------------------------------------
 * Progress bar
 *
 * Active only when --progress is given.
 * Uses \r to overwrite in-place; redraws at most every PROGRESS_INTERVAL
 * events to avoid becoming the bottleneck on huge traces.
 * A final call with force=1 always redraws and appends a newline.
 * ----------------------------------------------------------------------- */

#define PROGRESS_BAR_WIDTH  40
#define PROGRESS_INTERVAL   500   /* redraw every N events */

static long g_total_events = 0;   /* set before the conversion loop */

static void progress_init(long total) {
    g_total_events = total;
}

static void progress_draw(long done, int force) {
    if (!g_opt_progress) return;
    if (!force && (done % PROGRESS_INTERVAL) != 0) return;

    long total = g_total_events;
    double frac = (total > 0) ? (double)done / (double)total : 0.0;
    if (frac > 1.0) frac = 1.0;

    int filled = (int)(frac * PROGRESS_BAR_WIDTH);

    char bar[PROGRESS_BAR_WIDTH + 1];
    for (int i = 0; i < PROGRESS_BAR_WIDTH; i++)
        bar[i] = (i < filled) ? '#' : '-';
    bar[PROGRESS_BAR_WIDTH] = '\0';

    fprintf(stderr, "\r  Converting  [%s]  %ld / %ld  (%3.0f%%)   ",
            bar, done, total, frac * 100.0);

    if (force) fputc('\n', stderr);
    fflush(stderr);
}

/*
 * collect_metadata_events: first pass over the events array.
 * Scans for Chrome 'M' (metadata) phase events and registers any
 * process_name / thread_name values BEFORE any events are emitted.
 *
 * This guarantees that intern_thread() always finds a real name in the
 * lookup tables, even when M events appear after the first event that
 * references a given pid/tid.
 */
static void collect_metadata_events(yyjson_val *events) {
    size_t idx, max;
    yyjson_val *ev;
    yyjson_arr_foreach(events, idx, max, ev) {
        if (!yyjson_is_obj(ev)) continue;

        const char *ph = yyjson_get_str(yyjson_obj_get(ev, "ph"));
        if (!ph || ph[0] != 'M') continue;

        const char *mname = yyjson_get_str(yyjson_obj_get(ev, "name"));
        if (!mname) continue;

        yyjson_val *pid_v = yyjson_obj_get(ev, "pid");
        yyjson_val *tid_v = yyjson_obj_get(ev, "tid");
        yyjson_val *margs = yyjson_obj_get(ev, "args");
        const char *label = margs
            ? yyjson_get_str(yyjson_obj_get(margs, "name")) : NULL;
        if (!label) continue;

        uint64_t pid = 0, tid = 0;
        if (pid_v) {
            if (yyjson_is_uint(pid_v)) pid = yyjson_get_uint(pid_v);
            else if (yyjson_is_int(pid_v)) pid = (uint64_t)yyjson_get_sint(pid_v);
        }
        if (tid_v) {
            if (yyjson_is_uint(tid_v)) tid = yyjson_get_uint(tid_v);
            else if (yyjson_is_int(tid_v)) tid = (uint64_t)yyjson_get_sint(tid_v);
        }

        if (strcmp(mname, "process_name") == 0) {
            register_proc_name(pid, label);
        } else if (strcmp(mname, "thread_name") == 0) {
            register_thr_name(pid, tid, label);
        }
    }
}

static void process_event(yyjson_val *ev_obj) {
    g_events_read++;

    const char *ph   = yyjson_get_str(yyjson_obj_get(ev_obj, "ph"));
    const char *cat  = yyjson_get_str(yyjson_obj_get(ev_obj, "cat"));
    const char *name = yyjson_get_str(yyjson_obj_get(ev_obj, "name"));
    yyjson_val *ts_v  = yyjson_obj_get(ev_obj, "ts");
    yyjson_val *dur_v = yyjson_obj_get(ev_obj, "dur");
    yyjson_val *pid_v = yyjson_obj_get(ev_obj, "pid");
    yyjson_val *tid_v = yyjson_obj_get(ev_obj, "tid");
    yyjson_val *id_v  = yyjson_obj_get(ev_obj, "id");
    yyjson_val *args  = yyjson_obj_get(ev_obj, "args");

    /* Timestamps in Chrome are in microseconds (floating point allowed).
     * FXT init record says 1e9 ticks/sec = 1 tick per nanosecond.
     * Convert: ts_ns = ts_us * 1000.
     */
    double ts_us = 0.0;
    if (ts_v) {
        if (yyjson_is_real(ts_v))      ts_us = yyjson_get_real(ts_v);
        else if (yyjson_is_int(ts_v))  ts_us = (double)yyjson_get_sint(ts_v);
        else if (yyjson_is_uint(ts_v)) ts_us = (double)yyjson_get_uint(ts_v);
    }
    uint64_t ts_ns = (uint64_t)(ts_us * 1000.0);

    uint64_t pid = 0, tid = 0;
    if (pid_v) {
        if (yyjson_is_int(pid_v))  pid = (uint64_t)yyjson_get_sint(pid_v);
        else if (yyjson_is_uint(pid_v)) pid = yyjson_get_uint(pid_v);
    }
    if (tid_v) {
        if (yyjson_is_int(tid_v))  tid = (uint64_t)yyjson_get_sint(tid_v);
        else if (yyjson_is_uint(tid_v)) tid = yyjson_get_uint(tid_v);
    }

    uint64_t id = 0;
    if (id_v) {
        if (yyjson_is_int(id_v))       id = (uint64_t)yyjson_get_sint(id_v);
        else if (yyjson_is_uint(id_v)) id = yyjson_get_uint(id_v);
        else if (yyjson_is_str(id_v)) {
            /* IDs can be hex strings like "0x1a2b" */
            const char *id_str = yyjson_get_str(id_v);
            if (id_str) id = (uint64_t)strtoull(id_str, NULL, 0);
        }
    }

    FxtEvent fev;
    memset(&fev, 0, sizeof(fev));
    fev.ts_ns = ts_ns;
    fev.pid   = pid;
    fev.tid   = tid;
    fev.cat   = cat  ? cat  : "";
    fev.name  = name ? name : "";
    fev.args  = (args && yyjson_is_obj(args)) ? args : NULL;

    if (!ph) ph = "i";  /* default to instant */
    char p = ph[0];

    switch (p) {
    case 'B':
        fev.etype     = 2;
        fev.has_extra = 0;
        break;

    case 'E':
        fev.etype     = 3;
        fev.has_extra = 0;
        break;

    case 'X': {
        /* Duration complete: needs end_time word */
        double dur_us = 0.0;
        if (dur_v) {
            if (yyjson_is_real(dur_v))      dur_us = yyjson_get_real(dur_v);
            else if (yyjson_is_int(dur_v))  dur_us = (double)yyjson_get_sint(dur_v);
            else if (yyjson_is_uint(dur_v)) dur_us = (double)yyjson_get_uint(dur_v);
        }
        fev.etype      = 4;
        fev.has_extra  = 1;
        fev.extra_word = ts_ns + (uint64_t)(dur_us * 1000.0);
        break;
    }

    case 'i':
    case 'I':
        fev.etype     = 0;
        fev.has_extra = 0;
        break;

    case 'C':
        fev.etype      = 1;
        fev.has_extra  = 1;
        fev.extra_word = id;
        break;

    case 'b':
        fev.etype      = 5;
        fev.has_extra  = 1;
        fev.extra_word = id;
        break;

    case 'n':
        fev.etype      = 6;
        fev.has_extra  = 1;
        fev.extra_word = id;
        break;

    case 'e':
        fev.etype      = 7;
        fev.has_extra  = 1;
        fev.extra_word = id;
        break;

    case 's':
        fev.etype      = 8;
        fev.has_extra  = 1;
        fev.extra_word = id;
        break;

    case 't':
        fev.etype      = 9;
        fev.has_extra  = 1;
        fev.extra_word = id;
        break;

    case 'f':
        fev.etype      = 10;
        fev.has_extra  = 1;
        fev.extra_word = id;
        break;

    case 'M':
        /*
         * Metadata events were already handled in collect_metadata_events().
         * Skip here without emitting any FXT record.
         */
        g_events_read--; /* un-count: M events are not trace events */
        return;

    default:
        /* Object ('O'), Sample ('P'), etc. – emit as Instant. */
        fev.etype     = 0;
        fev.has_extra = 0;
        break;
    }

    emit_event(&fev);
    g_events_written++;
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
        "  --progress  Show an animated progress bar on stderr during conversion.\n"
        "\n"
        "Examples:\n"
        "  %s < trace.json > trace.fxt\n"
        "  %s --verbose < trace.json > trace.fxt\n"
        "  %s --progress --verbose < trace.json > trace.fxt\n",
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
        } else if (strcmp(argv[i], "--progress") == 0) {
            g_opt_progress = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n"
                            "Run '%s --help' for usage.\n",
                    argv[i], argv[0]);
            return 1;
        }
    }

    /* Read all of stdin with yyjson_read_fp */
    yyjson_read_err err;
    yyjson_doc *doc = yyjson_read_fp(stdin, 0, NULL, &err);
    if (!doc) {
        fprintf(stderr, "JSON parse error at position %zu: %s\n",
                err.pos, err.msg ? err.msg : "unknown error");
        return 1;
    }

    yyjson_val *root = yyjson_doc_get_root(doc);

    /* Pre-size output buffer based on JSON input size to avoid realloc chain.
     * FXT output is typically ~25-30% of JSON input. We use 40% as a safe margin.
     * yyjson_doc_get_read_size() returns the number of bytes read from the source. */
    {
        size_t json_bytes = yyjson_doc_get_read_size(doc);
        g_outcap = json_bytes * 2 / 5;  /* 40% of input size */
        if (g_outcap < OUTBUF_INITIAL) g_outcap = OUTBUF_INITIAL;
    }
    g_outbuf = (uint8_t *)malloc(g_outcap);
    if (!g_outbuf) { fprintf(stderr, "OOM\n"); return 1; }

    /* Emit mandatory header records */
    emit_magic();
    emit_init();

    /*
     * Chrome trace format root can be:
     *   1. An object  { "traceEvents": [...], ... }
     *   2. An array   [ event, event, ... ]
     */
    yyjson_val *events = NULL;

    if (yyjson_is_obj(root)) {
        events = yyjson_obj_get(root, "traceEvents");
    } else if (yyjson_is_arr(root)) {
        events = root;
    }

    if (events && yyjson_is_arr(events)) {
        /*
         * Pass 1: collect all 'M' metadata events to populate the
         * process/thread name tables before any records are emitted.
         * This ensures intern_thread() always has real names available,
         * regardless of where in the array the M events appear.
         */
        collect_metadata_events(events);

        /* Pass 2: emit all FXT records */
        long total = (long)yyjson_arr_size(events);
        progress_init(total);
        if (g_opt_progress) {
            fprintf(stderr, "  Parsing done. Converting %ld events...\n", total);
        }

        size_t idx, max;
        yyjson_val *ev;
        yyjson_arr_foreach(events, idx, max, ev) {
            if (yyjson_is_obj(ev)) {
                process_event(ev);
                progress_draw(g_events_written, 0);
            }
        }
        progress_draw(g_events_written, 1);   /* final, forced redraw */
    }

    yyjson_doc_free(doc);

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

    return 0;
}
