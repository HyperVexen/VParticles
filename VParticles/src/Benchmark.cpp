#include "Benchmark.h"
#include "GpuMonitor.h"
#include "PerformanceStats.h"
#include "Camera.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>

void BenchmarkRunner::Run(sf::RenderWindow& window, ParticleSystem& particleSystem, GpuRenderer& gpuRenderer, SimulationSettings& settings)
{
    std::cout << "Starting VParticles Stage 3 Benchmark..." << std::endl;
    
    std::ofstream csvFile("benchmark_results.csv");
    if (!csvFile.is_open())
    {
        std::cerr << "Failed to open benchmark_results.csv for writing." << std::endl;
        return;
    }
    
    csvFile << "count,mode,avg_fps,min_fps,avg_frame_ms,gpu_util_pct,vram_used_mb\n";
    
    std::vector<std::size_t> targetCounts = {10000, 50000, 100000, 250000, 500000, 1000000, 2000000, 5000000};
    
    // We only benchmark GPU mode as requested
    settings.computeMode = ComputeMode::GPU;
    settings.paused = false;
    
    for (std::size_t count : targetCounts)
    {
        RunTest(window, particleSystem, gpuRenderer, settings, count, "GPU");
    }
    
    csvFile.close();
    std::cout << "Benchmark complete. Results saved to benchmark_results.csv." << std::endl;
}

void BenchmarkRunner::RunTest(sf::RenderWindow& window, ParticleSystem& particleSystem, GpuRenderer& gpuRenderer, SimulationSettings& settings, std::size_t targetParticles, const std::string& modeName)
{
    std::cout << "Running test: " << targetParticles << " particles (" << modeName << ")... " << std::flush;
    
    // Ensure pool is large enough
    if (particleSystem.maxParticles < targetParticles)
    {
        particleSystem.InitializePool(targetParticles);
        gpuRenderer.Init(window, targetParticles);
    }
    
    particleSystem.Reset(settings);
    
    // Force spawn particles
    settings.particleCount = targetParticles;
    settings.burstCount = targetParticles;
    settings.spawnRate = 0.0f;
    settings.particleLifetime = 10000.0f; // effectively infinite during benchmark
    settings.lifetimeRandomness = 0.0f;
    
    SimulationStats simStats;
    // Spawn frame
    particleSystem.Update(0.016f, settings, simStats);
    
    GpuMonitor gpuMonitor;
    PerformanceStats stats;

    // Default camera for benchmark
    Camera benchCamera;
    float aspect = static_cast<float>(window.getSize().x) / static_cast<float>(window.getSize().y);
    Mat4 viewMat = benchCamera.GetViewMatrix();
    Mat4 projMat = benchCamera.GetProjectionMatrix(aspect);
    
    // Warm up for 1 second
    sf::Clock clock;
    while (clock.getElapsedTime().asSeconds() < 1.0f)
    {
        while (const std::optional event = window.pollEvent()) { if (event->is<sf::Event::Closed>()) return; }
        
        particleSystem.Update(0.016f, settings, simStats);
        
        window.clear();
        if (gpuRenderer.IsInitialized())
            gpuRenderer.Draw(window, particleSystem, viewMat.Ptr(), projMat.Ptr());
        window.display();
    }
    
    // Sample for 5 seconds
    clock.restart();
    
    float timeAccumulator = 0.0f;
    int frameCount = 0;
    float minFps = 99999.0f;
    
    float totalGpuUtil = 0.0f;
    float totalVramMb = 0.0f;
    int gpuSamples = 0;
    sf::Clock gpuPollClock;
    
    while (clock.getElapsedTime().asSeconds() < 5.0f)
    {
        while (const std::optional event = window.pollEvent()) { if (event->is<sf::Event::Closed>()) return; }
        
        sf::Clock frameClock;
        particleSystem.Update(0.016f, settings, simStats);
        
        window.clear();
        if (gpuRenderer.IsInitialized())
            gpuRenderer.Draw(window, particleSystem, viewMat.Ptr(), projMat.Ptr());
        window.display();
        
        float dt = frameClock.getElapsedTime().asSeconds();
        if (dt > 0.0f)
        {
            float fps = 1.0f / dt;
            if (fps < minFps) minFps = fps;
        }
        
        timeAccumulator += dt;
        frameCount++;
        
        if (gpuPollClock.getElapsedTime().asSeconds() >= 0.5f)
        {
            gpuMonitor.Poll(stats);
            totalGpuUtil += stats.gpuUtilPercent;
            totalVramMb += stats.gpuMemUsedMb;
            gpuSamples++;
            gpuPollClock.restart();
        }
    }
    
    float avgFps = static_cast<float>(frameCount) / timeAccumulator;
    float avgFrameMs = (timeAccumulator / static_cast<float>(frameCount)) * 1000.0f;
    float avgGpuUtil = gpuSamples > 0 ? totalGpuUtil / static_cast<float>(gpuSamples) : 0.0f;
    float avgVramMb = gpuSamples > 0 ? totalVramMb / static_cast<float>(gpuSamples) : 0.0f;
    
    std::cout << "Avg FPS: " << avgFps << ", Min FPS: " << minFps << ", VRAM: " << avgVramMb << " MB" << std::endl;
    
    std::ofstream csvFile("benchmark_results.csv", std::ios::app);
    csvFile << targetParticles << ","
            << modeName << ","
            << avgFps << ","
            << minFps << ","
            << avgFrameMs << ","
            << avgGpuUtil << ","
            << avgVramMb << "\n";
}
