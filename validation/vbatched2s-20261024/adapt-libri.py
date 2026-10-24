#!/usr/bin/env python3
"""Test-only adapter to the DDLA public handle API; shared by both variants."""
from pathlib import Path
import difflib
import re
import sys

root = Path(sys.argv[1])
patch = []


def save(path, before, after):
    patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                    fromfile='a/' + str(path), tofile='b/' + str(path)))
    (root / path).write_text(after)


path = Path('include/RI/global/gpu/GPU_Backend.h')
before = text = (root / path).read_text()
replacements = {
    '#include <ddla/ddla_connector.h>': '#include <ddla/ddla.h>',
    '    #include <ddla/gemmVbatched.h>\n': '',
    'using Transpose = ddla::deblasOperation_t;': 'using Transpose = char;',
    'ddla::DEBLAS_OP_N': "'N'", 'ddla::DEBLAS_OP_T': "'T'", 'ddla::DEBLAS_OP_C': "'C'",
    '''    inline ddla::deviceStream_t stream(Queue queue)
    {
        return static_cast<ddla::deviceStream_t>(ddla::ddla_get_stream(queue));
    }
''': '',
    'validate_local_gpu_count(const MPI_Comm& communicator)':
        'validate_local_gpu_count(const MPI_Comm& communicator, Queue queue)',
    'ddla::DEVICE_CHECK(ddla::deviceGetDeviceCount(&device_count));':
        'ddla::DDLA_CHECK(ddla::ddlaGetDeviceCount(queue, &device_count));',
    '''        validate_local_gpu_count(communicator);
        ddla::ddla_init(queue_);
        ddla::ddla_set(queue_, communicator);''':
    '''        ddla::DDLA_CHECK(ddla::ddlaInit(queue_, ddla::DdlaBackend::GPU));
        try
        {
            validate_local_gpu_count(communicator, queue_);
            ddla::DDLA_CHECK(ddla::ddlaSet(queue_, communicator));
        }
        catch (...)
        {
            ddla::ddlaDestroy(queue_);
            throw;
        }''',
    'validate_local_gpu_count(communicator);': 'validate_local_gpu_count(communicator, queue_);',
    'ddla::ddla_destroy(queue_);': 'ddla::ddlaDestroy(queue_);',
    'allocate(T** pointer, std::size_t count)': 'allocate(T** pointer, std::size_t count, Queue queue)',
    'ddla::DEVICE_CHECK(ddla::deviceMalloc(pointer, count * sizeof(T)));':
        'ddla::DDLA_CHECK(ddla::ddlaMalloc(queue, pointer, count * sizeof(T)));',
    'free(T* pointer)': 'free(T* pointer, Queue queue)',
    'ddla::DEVICE_CHECK(ddla::deviceFree(pointer));': 'ddla::DDLA_CHECK(ddla::ddlaFree(queue, pointer));',
    '''ddla::DEVICE_CHECK(deviceMemcpyAsync(
        device_data, host_data, count * sizeof(T),
        ddla::deviceMemcpyHostToDevice, stream(queue)));''':
    '''ddla::DDLA_CHECK(ddla::ddlaMemcpy(
        queue, device_data, host_data, count * sizeof(T)));''',
    '''ddla::DEVICE_CHECK(deviceMemcpyAsync(
        host_data, device_data, count * sizeof(T),
        ddla::deviceMemcpyDeviceToHost, stream(queue)));''':
    '''ddla::DDLA_CHECK(ddla::ddlaMemcpy(
        queue, host_data, device_data, count * sizeof(T)));''',
    '''ddla::DEVICE_CHECK(deviceMemsetAsync(
        pointer, value, bytes, stream(queue)));''':
        'ddla::DDLA_CHECK(ddla::ddlaMemset(queue, pointer, value, bytes));',
    '''ddla::DEVICE_CHECK(
        ddla::deviceStreamSynchronize(stream(queue)));''':
        'ddla::DDLA_CHECK(ddla::ddlaSynchronize(queue));',
    'ddla::gemmVbatched(\n        transB': 'ddla::gemmVbatched(\n        queue, transB',
    'beta, dC_array, ldc, batch_count, queue);': 'beta, dC_array, ldc, batch_count);',
    'ddla::gemmVbatched2s(\n        transB_0': 'ddla::gemmVbatched2s(\n        queue, transB_0',
    'batch_count, segment_sizes.data(), segment_sizes.size(), queue);':
        'batch_count, segment_sizes.data(), segment_sizes.size());',
}
for old, new in replacements.items():
    assert old in text, old
    text = text.replace(old, new)
save(path, before, text)

for name in ['GPU_Data_Input.h', 'GPU_Data_Mul.h', 'GPU_Data_Output.h', 'GPU_Data_Tmp.h', 'Dim.h']:
    path = Path('include/RI/ri/gpu') / name
    before = text = (root / path).read_text()
    text, allocations = re.subn(r'GPU_Backend::allocate\(([^\n]+)\);',
                                r'GPU_Backend::allocate(\1, queue);', text)
    text, frees = re.subn(r'GPU_Backend::free\(([^\n]+)\);',
                         r'GPU_Backend::free(\1, this->queue_);', text)
    assert allocations and allocations == frees, (name, allocations, frees)
    # The outer GPU context outlives these data holders.
    position = text.index('  public:') + len('  public:')
    text = text[:position] + '\n\tGPU_Backend::Queue queue_ = nullptr;' + text[position:]
    text = re.sub(r'(\b(?:upload|upload_1st|upload_2nd)\([^{}]*GPU_Backend::Queue queue\)\s*\{)',
                  r'\1\n\t\tthis->queue_ = queue;', text, count=1 if name == 'Dim.h' else 0)
    save(path, before, text)

(root.parent / 'libri-public-api.patch').write_text(''.join(patch))
print('Prepared test-only public API adapter:', root)
