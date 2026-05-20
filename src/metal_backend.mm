#include <Metal/Metal.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace {

using Array = nb::ndarray<nb::numpy>;

id<MTLDevice> get_device() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        throw std::runtime_error("Metal is not available on this machine");
    }
    return device;
}

id<MTLCommandQueue> get_queue() {
    static id<MTLCommandQueue> queue = [get_device() newCommandQueue];
    if (queue == nil) {
        throw std::runtime_error("Metal command queue creation failed");
    }
    return queue;
}

std::string ns_error_message(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    return std::string([[error localizedDescription] UTF8String]);
}

MTLSize tuple_to_size(nb::tuple value) {
    if (nb::len(value) != 3) {
        throw std::runtime_error("grid and threads must be 3-tuples");
    }
    return MTLSizeMake(
        nb::cast<NSUInteger>(value[0]),
        nb::cast<NSUInteger>(value[1]),
        nb::cast<NSUInteger>(value[2])
    );
}

// Persistent buffer registry. Handles are opaque uint64 IDs handed back to
// Python; the registry owns the MTLBuffer until release_buffer is called.
struct BufferRegistry {
    std::mutex mu;
    std::unordered_map<uint64_t, id<MTLBuffer>> buffers;
    std::atomic<uint64_t> next_id{1};

    uint64_t create(size_t nbytes) {
        id<MTLBuffer> buffer = [get_device() newBufferWithLength:nbytes
                                                         options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw std::runtime_error("Metal buffer allocation failed");
        }
        uint64_t id = next_id.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mu);
        buffers.emplace(id, buffer);
        return id;
    }

    id<MTLBuffer> get(uint64_t handle) {
        std::lock_guard<std::mutex> lock(mu);
        auto it = buffers.find(handle);
        if (it == buffers.end()) {
            throw std::runtime_error("Unknown buffer handle: " + std::to_string(handle));
        }
        return it->second;
    }

    bool release(uint64_t handle) {
        std::lock_guard<std::mutex> lock(mu);
        return buffers.erase(handle) > 0;
    }
};

BufferRegistry& registry() {
    static BufferRegistry r;
    return r;
}

uint64_t create_buffer(size_t nbytes) {
    if (nbytes == 0) {
        throw std::runtime_error("create_buffer: nbytes must be > 0");
    }
    return registry().create(nbytes);
}

bool release_buffer(uint64_t handle) {
    return registry().release(handle);
}

size_t buffer_size(uint64_t handle) {
    id<MTLBuffer> buffer = registry().get(handle);
    return static_cast<size_t>([buffer length]);
}

void write_buffer(uint64_t handle, Array data) {
    id<MTLBuffer> buffer = registry().get(handle);
    size_t nbytes = data.size() * data.itemsize();
    if (nbytes > [buffer length]) {
        throw std::runtime_error("write_buffer: source larger than buffer");
    }
    std::memcpy([buffer contents], data.data(), nbytes);
}

void read_buffer(uint64_t handle, Array out) {
    id<MTLBuffer> buffer = registry().get(handle);
    size_t nbytes = out.size() * out.itemsize();
    if (nbytes > [buffer length]) {
        throw std::runtime_error("read_buffer: destination larger than buffer");
    }
    std::memcpy(out.data(), [buffer contents], nbytes);
}

void fill_buffer(uint64_t handle, uint8_t value) {
    @autoreleasepool {
        id<MTLBuffer> buffer = registry().get(handle);
        id<MTLCommandBuffer> command_buffer = [get_queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
        [blit fillBuffer:buffer range:NSMakeRange(0, [buffer length]) value:value];
        [blit endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] == MTLCommandBufferStatusError) {
            throw std::runtime_error("fill_buffer failed: " + ns_error_message([command_buffer error]));
        }
    }
}

// A bound slot for a kernel call. Either a persistent buffer (handle != 0,
// referencing the registry) or a transient buffer wrapping an ndarray for
// back-compat with the old run_kernel call style.
struct Slot {
    id<MTLBuffer> buffer;
    // For transient slots only: where to copy data back after the kernel runs.
    void* writeback_ptr;
    size_t writeback_bytes;
};

Slot resolve_slot(nb::handle item) {
    // Persistent buffer: caller passes int handle, or any object exposing
    // .handle (int). Try int first since Buffer wrappers will be int-castable
    // via their handle attribute.
    if (nb::hasattr(item, "handle")) {
        uint64_t h = nb::cast<uint64_t>(item.attr("handle"));
        return Slot{registry().get(h), nullptr, 0};
    }
    if (PyLong_Check(item.ptr())) {
        uint64_t h = nb::cast<uint64_t>(item);
        return Slot{registry().get(h), nullptr, 0};
    }
    // Transient ndarray: allocate a one-shot shared buffer and remember the
    // writeback target so output data lands back in the caller's numpy array.
    Array array = nb::cast<Array>(item);
    size_t nbytes = array.size() * array.itemsize();
    id<MTLBuffer> buffer = [get_device() newBufferWithBytes:array.data()
                                                     length:nbytes
                                                    options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        throw std::runtime_error("Metal buffer allocation failed");
    }
    return Slot{buffer, array.data(), nbytes};
}

nb::dict run_kernel(
    const std::string& source,
    const std::string& function_name,
    nb::list buffers,
    nb::tuple grid,
    nb::tuple threads
) {
    @autoreleasepool {
        id<MTLDevice> device = get_device();
        NSError* error = nil;
        NSString* metal_source = [[NSString alloc] initWithBytes:source.data()
                                                         length:source.size()
                                                       encoding:NSUTF8StringEncoding];
        id<MTLLibrary> library = [device newLibraryWithSource:metal_source options:nil error:&error];
        if (library == nil) {
            throw std::runtime_error("Metal library compile failed: " + ns_error_message(error));
        }

        NSString* name = [[NSString alloc] initWithBytes:function_name.data()
                                                  length:function_name.size()
                                                encoding:NSUTF8StringEncoding];
        id<MTLFunction> function = [library newFunctionWithName:name];
        if (function == nil) {
            throw std::runtime_error("Metal function was not found: " + function_name);
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            throw std::runtime_error("Metal pipeline creation failed: " + ns_error_message(error));
        }

        std::vector<Slot> slots;
        slots.reserve(nb::len(buffers));
        for (size_t i = 0; i < nb::len(buffers); i++) {
            slots.push_back(resolve_slot(buffers[i]));
        }

        id<MTLCommandBuffer> command_buffer = [get_queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];

        for (NSUInteger i = 0; i < slots.size(); i++) {
            [encoder setBuffer:slots[i].buffer offset:0 atIndex:i];
        }

        auto start = std::chrono::steady_clock::now();
        [encoder dispatchThreadgroups:tuple_to_size(grid) threadsPerThreadgroup:tuple_to_size(threads)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        auto end = std::chrono::steady_clock::now();

        if ([command_buffer status] == MTLCommandBufferStatusError) {
            throw std::runtime_error("Metal command buffer failed: " + ns_error_message([command_buffer error]));
        }

        for (const Slot& slot : slots) {
            if (slot.writeback_ptr != nullptr) {
                std::memcpy(slot.writeback_ptr, [slot.buffer contents], slot.writeback_bytes);
            }
        }

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        if ([command_buffer GPUStartTime] > 0.0 && [command_buffer GPUEndTime] >= [command_buffer GPUStartTime]) {
            time_ms = ([command_buffer GPUEndTime] - [command_buffer GPUStartTime]) * 1000.0;
        }

        nb::dict result;
        result["time_ms"] = time_ms;
        return result;
    }
}

}

NB_MODULE(metal_backend, m) {
    m.def("run_kernel", &run_kernel, nb::arg("source"), nb::arg("function_name"), nb::arg("buffers"), nb::arg("grid"), nb::arg("threads"));
    m.def("create_buffer", &create_buffer, nb::arg("nbytes"));
    m.def("release_buffer", &release_buffer, nb::arg("handle"));
    m.def("buffer_size", &buffer_size, nb::arg("handle"));
    m.def("write_buffer", &write_buffer, nb::arg("handle"), nb::arg("data"));
    m.def("read_buffer", &read_buffer, nb::arg("handle"), nb::arg("out"));
    m.def("fill_buffer", &fill_buffer, nb::arg("handle"), nb::arg("value"));
}
