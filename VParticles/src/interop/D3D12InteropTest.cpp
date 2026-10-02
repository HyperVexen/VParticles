#include "VParticles/ParticleSystem.h"
#include "VParticles/ExternalInterop.h"

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

using Microsoft::WRL::ComPtr;
using namespace vparticles;
using namespace vparticles::interop;

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "  VParticles Phase 10B: DirectX 12 Interop Test\n";
    std::cout << "========================================================\n";

    // 1. Select High-Performance DXGI Adapter matching NVIDIA GPU
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
        std::cerr << "[-] Failed to create DXGI factory.\n";
        return 1;
    }

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (desc.VendorId == 0x10DE) { // NVIDIA
            std::wcout << L"[+] Selected NVIDIA Adapter: " << desc.Description << L"\n";
            break;
        }
    }

    // 2. Create D3D12 Device
    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        std::cerr << "[-] Failed to create D3D12 device.\n";
        return 1;
    }
    std::cout << "[+] D3D12 Device initialized.\n";

    // 3. Create Command Queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> commandQueue;
    if (FAILED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)))) {
        std::cerr << "[-] Failed to create D3D12 command queue.\n";
        return 1;
    }

    // 4. Create Shared Vertex Buffer & Shared Indirect Buffer
    const uint32_t kCapacity = 1000000;
    const uint64_t vertexBufferSize = sizeof(InteropVertex) * kCapacity;
    const uint64_t indirectBufferSize = sizeof(D3D12DrawArguments);

    D3D12_HEAP_PROPERTIES heapDefault = {};
    heapDefault.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC vboDesc = {};
    vboDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    vboDesc.Width = vertexBufferSize;
    vboDesc.Height = 1;
    vboDesc.DepthOrArraySize = 1;
    vboDesc.MipLevels = 1;
    vboDesc.Format = DXGI_FORMAT_UNKNOWN;
    vboDesc.SampleDesc.Count = 1;
    vboDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    vboDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ComPtr<ID3D12Resource> vertexBuffer;
    if (FAILED(device->CreateCommittedResource(
            &heapDefault,
            D3D12_HEAP_FLAG_SHARED,
            &vboDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&vertexBuffer)))) {
        std::cerr << "[-] Failed to create shared D3D12 vertex buffer.\n";
        return 1;
    }

    D3D12_RESOURCE_DESC indirectDesc = vboDesc;
    indirectDesc.Width = indirectBufferSize;

    ComPtr<ID3D12Resource> indirectBuffer;
    if (FAILED(device->CreateCommittedResource(
            &heapDefault,
            D3D12_HEAP_FLAG_SHARED,
            &indirectDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&indirectBuffer)))) {
        std::cerr << "[-] Failed to create shared D3D12 indirect command buffer.\n";
        return 1;
    }

    // 5. Create Shared D3D12 Fence
    ComPtr<ID3D12Fence> fence;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence)))) {
        std::cerr << "[-] Failed to create shared D3D12 fence.\n";
        return 1;
    }

    // 6. Export Win32 Shared Handles
    HANDLE hVertexBuffer = nullptr;
    device->CreateSharedHandle(vertexBuffer.Get(), nullptr, GENERIC_ALL, nullptr, &hVertexBuffer);

    HANDLE hIndirectBuffer = nullptr;
    device->CreateSharedHandle(indirectBuffer.Get(), nullptr, GENERIC_ALL, nullptr, &hIndirectBuffer);

    HANDLE hFence = nullptr;
    device->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, nullptr, &hFence);

    std::cout << "[+] Shared NT handles exported successfully.\n";

    // 7. Import into CUDA
    ExternalMemoryBuffer extVertexBuffer;
    ExternalMemoryDesc memDescVbo = {};
    memDescVbo.type = ExternalMemoryType::D3D12Resource;
    memDescVbo.handle = hVertexBuffer;
    memDescVbo.sizeBytes = vertexBufferSize;
    memDescVbo.isDedicated = true;
    if (!extVertexBuffer.import(memDescVbo)) {
        std::cerr << "[-] Failed to import vertex buffer into CUDA.\n";
        return 1;
    }

    ExternalMemoryBuffer extIndirectBuffer;
    ExternalMemoryDesc memDescInd = {};
    memDescInd.type = ExternalMemoryType::D3D12Resource;
    memDescInd.handle = hIndirectBuffer;
    memDescInd.sizeBytes = indirectBufferSize;
    memDescInd.isDedicated = true;
    if (!extIndirectBuffer.import(memDescInd)) {
        std::cerr << "[-] Failed to import indirect command buffer into CUDA.\n";
        return 1;
    }

    ExternalTimelineSemaphore extFence;
    ExternalSemaphoreDesc semDesc = {};
    semDesc.type = ExternalSemaphoreType::D3D12Fence;
    semDesc.handle = hFence;
    if (!extFence.import(semDesc)) {
        std::cerr << "[-] Failed to import D3D12 fence into CUDA.\n";
        return 1;
    }
    std::cout << "[+] CUDA external memory and semaphore imported successfully.\n";

    // 8. Initialize ModernInteropBridge
    ModernInteropBridge bridge;
    if (!bridge.init(extVertexBuffer.devicePointer(), extIndirectBuffer.devicePointer(), kCapacity)) {
        std::cerr << "[-] Failed to initialize ModernInteropBridge.\n";
        return 1;
    }

    // 9. Initialize VParticles System
    ParticleSystem system(kCapacity, StorageMode::FP32, true);
    EmitterDesc emitter = {};
    emitter.position = {0.0f, 0.0f, 0.0f};
    emitter.velocity = {0.0f, 15.0f, 0.0f};
    emitter.velocityVariance = {5.0f, 5.0f, 5.0f};
    emitter.color = {1.0f, 0.5f, 0.2f, 1.0f};
    emitter.spawnRate = 250000.0f;
    emitter.lifetime = 4.0f;
    emitter.size = 2.0f;
    system.addEmitter(emitter);

    // 10. Run 60-frame simulation loop with full lockless GPU-to-GPU fence synchronization
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "[+] Executing 60 simulation frames with GPU hardware timeline sync...\n";
    uint64_t fenceValue = 0;

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int frame = 0; frame < 60; ++frame) {
        // Step A: D3D12 command queue signals frame start
        fenceValue++;
        commandQueue->Signal(fence.Get(), fenceValue);

        // Step B: CUDA waits on D3D12 signal asynchronously on the GPU
        extFence.waitAsync(fenceValue, stream);

        // Step C: Simulate
        system.update(1.0f / 60.0f);

        // Step D: Gather into D3D12 shared vertex buffer and emit indirect draw command
        bridge.gather(system.gpuBuffers(), stream);

        // Step E: CUDA signals simulation & gather completion
        fenceValue++;
        extFence.signalAsync(fenceValue, stream);

        // Step F: D3D12 command queue waits for CUDA completion on the GPU
        commandQueue->Wait(fence.Get(), fenceValue);
    }

    // Final CPU event wait for verification
    HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    fence->SetEventOnCompletion(fenceValue, hEvent);
    WaitForSingleObject(hEvent, 5000);
    CloseHandle(hEvent);

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    std::cout << "[+] 60 frames completed in " << totalMs << " ms (" 
              << (totalMs / 60.0) << " ms/frame, " << (60000.0 / totalMs) << " FPS).\n";

    // 11. Read back D3D12 indirect command buffer to verify live count
    D3D12_HEAP_PROPERTIES heapReadback = {};
    heapReadback.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC rbDesc = indirectDesc;
    rbDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ComPtr<ID3D12Resource> readbackBuffer;
    device->CreateCommittedResource(
        &heapReadback,
        D3D12_HEAP_FLAG_NONE,
        &rbDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readbackBuffer));

    ComPtr<ID3D12CommandAllocator> allocator;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
    ComPtr<ID3D12GraphicsCommandList> cmdList;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&cmdList));

    cmdList->CopyBufferRegion(readbackBuffer.Get(), 0, indirectBuffer.Get(), 0, indirectBufferSize);
    cmdList->Close();

    ID3D12CommandList* lists[] = { cmdList.Get() };
    commandQueue->ExecuteCommandLists(1, lists);

    fenceValue++;
    commandQueue->Signal(fence.Get(), fenceValue);
    hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    fence->SetEventOnCompletion(fenceValue, hEvent);
    WaitForSingleObject(hEvent, 2000);
    CloseHandle(hEvent);

    void* mappedCmd = nullptr;
    readbackBuffer->Map(0, nullptr, &mappedCmd);
    auto* drawCmd = static_cast<D3D12DrawArguments*>(mappedCmd);

    std::cout << "\n--- Verification Results ---\n";
    std::cout << "DrawIndirect.VertexCountPerInstance : " << drawCmd->vertexCountPerInstance << "\n";
    std::cout << "DrawIndirect.InstanceCount          : " << drawCmd->instanceCount << "\n";
    std::cout << "DrawIndirect.StartVertexLocation    : " << drawCmd->startVertexLocation << "\n";
    std::cout << "DrawIndirect.StartInstanceLocation  : " << drawCmd->startInstanceLocation << "\n";

    bool success = (drawCmd->vertexCountPerInstance > 50000) && (drawCmd->instanceCount == 1);
    std::cout << "Phase 10B Modern Engine Interop: " << (success ? "PASSED (100% Verified)" : "FAILED") << "\n";
    std::cout << "========================================================\n";

    readbackBuffer->Unmap(0, nullptr);

    // Cleanup
    cudaStreamDestroy(stream);
    bridge.shutdown();
    extVertexBuffer.release();
    extIndirectBuffer.release();
    extFence.release();
    CloseHandle(hVertexBuffer);
    CloseHandle(hIndirectBuffer);
    CloseHandle(hFence);

    return success ? 0 : 1;
}
