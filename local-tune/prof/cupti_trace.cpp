// CUPTI activity tracer, loaded through CUDA_INJECTION64_PATH. No root needed.
// Writes one tab-separated line per record to $CUPTI_TRACE_OUT (default cupti-trace.tsv).
// K kernel, C memcpy, S memset, Y synchronization, M memory alloc/free, R runtime API call.
// Build: local-tune/prof/build.sh
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static FILE * out;
static const size_t BUF_SIZE = 8 * 1024 * 1024;

static void CUPTIAPI buffer_requested(uint8_t ** buffer, size_t * size, size_t * max_records) {
    *buffer = (uint8_t *) aligned_alloc(8, BUF_SIZE);
    *size = BUF_SIZE;
    *max_records = 0;
}

static void write_record(CUpti_Activity * r) {
    switch (r->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            CUpti_ActivityKernel13 * k = (CUpti_ActivityKernel13 *) r;
            fprintf(out, "K\t%llu\t%llu\t%u\t%u\t%d,%d,%d\t%d,%d,%d\t%d\t%d\t%u\t%u\t%llu\t%s\n",
                (unsigned long long) k->start, (unsigned long long) k->end, k->correlationId, k->streamId,
                k->gridX, k->gridY, k->gridZ, k->blockX, k->blockY, k->blockZ,
                k->staticSharedMemory, k->dynamicSharedMemory, k->localMemoryPerThread, k->graphId,
                (unsigned long long) k->queued, k->name ? k->name : "?");
        } break;
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            CUpti_ActivityMemcpy7 * m = (CUpti_ActivityMemcpy7 *) r;
            fprintf(out, "C\t%llu\t%llu\t%u\t%u\t%u\t%llu\t%u\n",
                (unsigned long long) m->start, (unsigned long long) m->end, m->correlationId, m->streamId,
                (unsigned) m->copyKind, (unsigned long long) m->bytes, m->graphId);
        } break;
        case CUPTI_ACTIVITY_KIND_MEMSET: {
            CUpti_ActivityMemset5 * m = (CUpti_ActivityMemset5 *) r;
            fprintf(out, "S\t%llu\t%llu\t%u\t%u\t%llu\n",
                (unsigned long long) m->start, (unsigned long long) m->end, m->correlationId, m->streamId,
                (unsigned long long) m->bytes);
        } break;
        case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
            CUpti_ActivitySynchronization2 * s = (CUpti_ActivitySynchronization2 *) r;
            fprintf(out, "Y\t%llu\t%llu\t%u\t%u\t%u\n",
                (unsigned long long) s->start, (unsigned long long) s->end, s->correlationId, s->streamId,
                (unsigned) s->type);
        } break;
        case CUPTI_ACTIVITY_KIND_MEMORY2: {
            CUpti_ActivityMemory4 * m = (CUpti_ActivityMemory4 *) r;
            fprintf(out, "M\t%llu\t%u\t%u\t%u\t%llu\n",
                (unsigned long long) m->timestamp, m->correlationId,
                (unsigned) m->memoryOperationType, (unsigned) m->memoryKind, (unsigned long long) m->bytes);
        } break;
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
            CUpti_ActivityAPI * a = (CUpti_ActivityAPI *) r;
            fprintf(out, "R\t%llu\t%llu\t%u\t%u\t%u\n",
                (unsigned long long) a->start, (unsigned long long) a->end, a->correlationId,
                (unsigned) a->cbid, a->threadId);
        } break;
        default:
            break;
    }
}

static void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t * buffer, size_t, size_t valid_size) {
    CUpti_Activity * r = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &r) == CUPTI_SUCCESS) {
        write_record(r);
    }
    size_t dropped = 0;
    cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped);
    if (dropped) {
        fprintf(out, "D\t%zu\n", dropped);
    }
    free(buffer);
}

static void at_exit() {
    cuptiActivityFlushAll(1);
    if (out) {
        fclose(out);
    }
}

extern "C" int InitializeInjection(void) {
    const char * path = getenv("CUPTI_TRACE_OUT");
    out = fopen(path ? path : "cupti-trace.tsv", "w");
    if (!out) {
        return 0;
    }
    setvbuf(out, nullptr, _IOFBF, 1 << 20);
    atexit(at_exit);
    cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2);
    // CUPTI_TRACE_API=0 leaves runtime API records off, for a lower-overhead run.
    const char * api = getenv("CUPTI_TRACE_API");
    if (!api || strcmp(api, "0") != 0) {
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    }
    return 1;
}
